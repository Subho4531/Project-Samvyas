"""Emotion-conditioned LoRA adaptation for the pinned IndicF5 flow model.

The base CFM is frozen.  Two small trainable pieces are added:

* low-rank adapters on the DiT attention/MLP projections (``to_q``, ``to_k``,
  ``to_v``, ``to_out.0``, ``ff.ff.0.0``, ``ff.ff.2``);
* a tag embedding added to the DiT time conditioning.  Index 0 is a learned
  "no tag" row, so tag dropout never leaves a parameter unused under DDP.

Training uses the inference layout: ``[prompt frames | target frames]`` with
the prompt visible as conditioning and the loss restricted to target frames.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

EMOTION_TAGS: tuple[str, ...] = ("<none>", "<neutral>", "<happy>", "<sad>", "<angry>")
TAG_INDEX: dict[str, int] = {tag.strip("<>"): index for index, tag in enumerate(EMOTION_TAGS)}
LORA_TARGETS: tuple[str, ...] = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.ff.0.0",
    "ff.ff.2",
)
ADAPTER_FORMAT = "samvyas-indicf5-emotion-lora-v1"


@dataclass(frozen=True)
class LoRAConfig:
    """Low-rank adapter geometry and target module suffixes."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    targets: tuple[str, ...] = LORA_TARGETS

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("LoRA rank must be positive.")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1).")
        if not self.targets:
            raise ValueError("At least one LoRA target module is required.")


class LoRALinear(nn.Module):
    """Frozen linear projection plus a trainable low-rank residual.

    Input and output shapes match the wrapped ``nn.Linear``.  The residual is
    ``dropout(x) @ A.T @ B.T * alpha / rank`` and ``B`` starts at zero, so the
    adapted model is exactly the base model at step zero.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if rank < 1 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1).")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        """Input width of the frozen projection."""
        return self.base.in_features

    @property
    def out_features(self) -> int:
        """Output width of the frozen projection."""
        return self.base.out_features

    def forward(self, x: Tensor) -> Tensor:
        """Apply the frozen projection and the low-rank residual to (..., in)."""
        output = self.base(x)
        residual = F.linear(self.dropout(x), self.lora_a)
        residual = F.linear(residual, self.lora_b)
        return output + residual * self.scaling

    def merged(self) -> nn.Linear:
        """Return an equivalent plain linear with the adapter folded into the weight."""
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        with torch.no_grad():
            linear.weight.copy_(self.base.weight + self.scaling * (self.lora_b @ self.lora_a))
            if self.base.bias is not None:
                linear.bias.copy_(self.base.bias)
        linear.requires_grad_(False)
        return linear

    def extra_repr(self) -> str:
        """Expose adapter geometry in module summaries."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, scaling={self.scaling:.3f}"
        )


def _matches_target(name: str, targets: Sequence[str]) -> bool:
    """Match a qualified module name against LoRA suffixes."""
    return any(name == target or name.endswith(f".{target}") for target in targets)


def apply_lora_to_dit(
    transformer: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    targets: Sequence[str] = LORA_TARGETS,
) -> list[LoRALinear]:
    """Replace matching DiT ``nn.Linear`` modules with frozen-base LoRA layers.

    The traversal is deliberately name-based: F5 nests the MLP inside
    ``ff.ff`` and the output projection inside a ``ModuleList``, so a simple
    ``named_modules`` replacement would miss or mis-target them.
    """
    config = LoRAConfig(rank=rank, alpha=alpha, dropout=dropout, targets=tuple(targets))
    replaced: list[LoRALinear] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            qualified = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear) and _matches_target(qualified, config.targets):
                adapted = LoRALinear(child, config.rank, config.alpha, config.dropout)
                setattr(parent, name, adapted)
                replaced.append(adapted)
            else:
                visit(child, qualified)

    visit(transformer, "")
    if not replaced:
        raise ValueError(f"No LoRA target modules matched: {sorted(config.targets)}")
    return replaced


class TagEmbedding(nn.Module):
    """Categorical emotion conditioning added to the DiT time embedding.

    Row 0 is the learned null tag used for classifier-free-style tag dropout.
    """

    def __init__(self, num_tags: int = len(EMOTION_TAGS), dim: int = 1024) -> None:
        super().__init__()
        if num_tags < len(EMOTION_TAGS):
            raise ValueError(f"At least {len(EMOTION_TAGS)} tag rows are required.")
        if dim < 1:
            raise ValueError("Tag embedding dimension must be positive.")
        self.num_tags = num_tags
        self.dim = dim
        self.embed = nn.Embedding(num_tags, dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        with torch.no_grad():
            self.embed.weight[0].zero_()

    def forward(self, tag_ids: Tensor) -> Tensor:
        """Map (B,) integer tags to (B, dim) conditioning vectors."""
        if tag_ids.ndim != 1:
            raise ValueError("tag_ids must have shape (B,).")
        if tag_ids.min().item() < 0 or tag_ids.max().item() >= self.num_tags:
            raise ValueError("tag_ids contain an out-of-range index.")
        return self.embed(tag_ids.long())


def _lens_to_mask(lens: Tensor, length: int) -> Tensor:
    """Build a boolean (B, length) mask from per-sample valid frame counts."""
    if lens.ndim != 1:
        raise ValueError("lens must have shape (B,).")
    if torch.any(lens < 0) or torch.any(lens > length):
        raise ValueError("lens must be between 0 and the padded sequence length.")
    positions = torch.arange(length, device=lens.device).unsqueeze(0)
    return positions < lens.unsqueeze(1)


def _tokenize(text: Sequence[str] | Tensor, vocab: dict[str, int]) -> Tensor:
    """Convert strings to the pinned character vocabulary, padding with -1."""
    if isinstance(text, Tensor):
        if text.ndim != 2:
            raise ValueError("Token tensor must have shape (B, T).")
        if text.shape[0] < 1:
            raise ValueError("Token tensor must have a batch dimension.")
        return text.long()
    if not text:
        raise ValueError("Text batch cannot be empty.")
    rows: list[list[int]] = []
    for item in text:
        if not isinstance(item, str) or not item:
            raise ValueError("Every text item must be a nonempty string.")
        rows.append([vocab.get(char, 0) for char in item])
    width = max(len(row) for row in rows)
    # F5's TextEmbedding adds one and treats -1 as the filler token.
    tokens = torch.full((len(rows), width), -1, dtype=torch.long)
    for index, row in enumerate(rows):
        tokens[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    # Callers truncate to the actual mel length; inference text may exceed the
    # reference length but still fit the generated target.
    return tokens


class EmotionCFM(nn.Module):
    """IndicF5 CFM wrapper with prompt/target masking and emotion tags.

    The wrapped CFM is frozen except for LoRA layers and ``tag_embed``.  The
    forward pass accepts a concatenated ``(B, T, 100)`` mel where the first
    ``prompt_lengths`` frames are the neutral reference and the remaining valid
    frames are the emotional target.
    """

    def __init__(
        self,
        base_cfm: nn.Module,
        lora_modules: Sequence[LoRALinear] | None = None,
        *,
        num_tags: int = len(EMOTION_TAGS),
        tag_dropout: float = 0.1,
        use_attention_mask: bool = False,
    ) -> None:
        super().__init__()
        if not hasattr(base_cfm, "transformer") or not hasattr(base_cfm, "mel_spec"):
            raise TypeError("base_cfm must expose transformer and mel_spec.")
        if not 0 <= tag_dropout < 1:
            raise ValueError("Tag dropout must be in [0, 1).")
        self.base = base_cfm
        self.lora_modules = list(lora_modules or [])
        self.num_tags = num_tags
        self.tag_dropout = tag_dropout
        self.use_attention_mask = use_attention_mask
        self.tag_embed = TagEmbedding(num_tags, base_cfm.transformer.dim)

    @property
    def device(self) -> torch.device:
        """Device of the trainable tag embedding."""
        return self.tag_embed.embed.weight.device

    @property
    def dtype(self) -> torch.dtype:
        """Parameter dtype used for inference conditioning."""
        return self.tag_embed.embed.weight.dtype

    def configure_trainable(self) -> list[nn.Parameter]:
        """Freeze the CFM and enable only LoRA matrices and the tag embedding."""
        self.base.requires_grad_(False)
        for module in self.lora_modules:
            module.lora_a.requires_grad_(True)
            module.lora_b.requires_grad_(True)
        self.tag_embed.requires_grad_(True)
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        """Currently trainable parameters, for optimizers and reports."""
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def _drop_tags(self, tag_ids: Tensor) -> Tensor:
        """Replace a fraction of tags with the learned null row during training."""
        if not self.training or self.tag_dropout <= 0:
            return tag_ids
        drop = torch.rand(tag_ids.shape, device=tag_ids.device) < self.tag_dropout
        return torch.where(drop, torch.zeros_like(tag_ids), tag_ids)

    def _transformer_forward(
        self,
        *,
        x: Tensor,
        cond: Tensor,
        text: Sequence[str] | Tensor,
        time: Tensor,
        tag_ids: Tensor,
        mask: Tensor | None,
        drop_audio_cond: bool,
        drop_text: bool,
    ) -> Tensor:
        """Run the frozen DiT with a tag vector added to its time embedding."""
        transformer = self.base.transformer
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)
        if tag_ids.shape != (batch,):
            raise ValueError("tag_ids must have shape (B,).")

        t = transformer.time_embed(time)
        t = t + self.tag_embed(tag_ids).to(dtype=t.dtype)
        tokens = _tokenize(text, self.base.vocab_char_map).to(x.device)
        if tokens.shape[0] != batch:
            raise ValueError("text batch size must match the mel batch.")
        if tokens.shape[1] > seq_len:
            tokens = tokens[:, :seq_len]
        text_embed = transformer.text_embed(tokens, seq_len, drop_text=drop_text)
        x = transformer.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        rope = transformer.rotary_embed.forward_from_seq_len(seq_len)
        residual = x if transformer.long_skip_connection is not None else None
        for block in transformer.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)
        if transformer.long_skip_connection is not None:
            x = transformer.long_skip_connection(torch.cat((x, residual), dim=-1))
        x = transformer.norm_out(x, t)
        return transformer.proj_out(x)

    def forward_emotion(
        self,
        mel: Tensor,
        text: Sequence[str] | Tensor,
        tag_ids: Tensor,
        prompt_lengths: Tensor,
        lens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the flow-matching loss on target frames only.

        Args:
            mel: Concatenated prompt/target mel, shape (B, T, 100), 24 kHz.
            text: ``prompt_text + " " + target_text`` per batch item.
            tag_ids: Emotion tag indices, shape (B,).
            prompt_lengths: Number of visible prompt frames, shape (B,).
            lens: Valid total frame counts, shape (B,). Defaults to T.

        Returns:
            ``(loss, cond, prediction)``; ``loss`` is a scalar tensor.
        """
        if mel.ndim == 2:
            mel = self.base.mel_spec(mel).permute(0, 2, 1)
        if mel.ndim != 3 or mel.shape[-1] != self.base.num_channels:
            raise ValueError("mel must have shape (B, T, num_channels).")
        batch, seq_len, _ = mel.shape
        if lens is None:
            lens = torch.full((batch,), seq_len, device=mel.device, dtype=torch.long)
        lens = torch.as_tensor(lens, device=mel.device, dtype=torch.long)
        prompt_lengths = torch.as_tensor(prompt_lengths, device=mel.device, dtype=torch.long)
        tag_ids = torch.as_tensor(tag_ids, device=mel.device, dtype=torch.long)
        if lens.shape != (batch,):
            raise ValueError("lens must have shape (B,).")
        if tag_ids.shape != (batch,):
            raise ValueError("tag_ids must have shape (B,).")
        if prompt_lengths.shape != (batch,):
            raise ValueError("prompt_lengths must have shape (B,).")
        if torch.any(prompt_lengths < 1):
            raise ValueError("Every sample needs at least one prompt frame.")
        if torch.any(prompt_lengths >= lens):
            raise ValueError("Every sample needs at least one target frame.")

        valid = _lens_to_mask(lens, seq_len)
        positions = torch.arange(seq_len, device=mel.device).unsqueeze(0)
        target = valid & (positions >= prompt_lengths.unsqueeze(1))
        if not bool(target.any()):
            raise ValueError("Target mask is empty.")

        x1 = mel
        x0 = torch.randn_like(x1)
        time = torch.rand((batch,), dtype=torch.float32, device=mel.device)
        t = time.view(batch, 1, 1)
        phi = (1 - t) * x0 + t * x1
        flow = x1 - x0
        cond = torch.where(target.unsqueeze(-1), torch.zeros_like(x1), x1)
        cond = torch.where(valid.unsqueeze(-1), cond, torch.zeros_like(cond))

        tag_ids = self._drop_tags(tag_ids)
        if self.training:
            drop_audio_cond = random.random() < self.base.audio_drop_prob
            if random.random() < self.base.cond_drop_prob:
                drop_audio_cond = True
                drop_text = True
            else:
                drop_text = False
        else:
            drop_audio_cond = False
            drop_text = False
        mask = valid if self.use_attention_mask else None
        prediction = self._transformer_forward(
            x=phi,
            cond=cond,
            text=text,
            time=time,
            tag_ids=tag_ids,
            mask=mask,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
        )
        loss = F.mse_loss(prediction.float(), flow.float(), reduction="none")
        loss = loss[target].mean()
        return loss, cond, prediction

    def forward(
        self,
        mel: Tensor,
        text: Sequence[str] | Tensor,
        tag_ids: Tensor,
        prompt_lengths: Tensor,
        lens: Tensor | None = None,
    ) -> Tensor:
        """DDP-friendly forward returning only the scalar training loss."""
        loss, _, _ = self.forward_emotion(
            mel,
            text,
            tag_ids,
            prompt_lengths,
            lens,
        )
        return loss

    @torch.no_grad()
    def sample(
        self,
        cond: Tensor,
        text: Sequence[str] | Tensor,
        duration: int | Tensor,
        tag_ids: Tensor | None = None,
        *,
        lens: Tensor | None = None,
        steps: int = 32,
        cfg_strength: float = 1.0,
        sway_sampling_coef: float | None = None,
        seed: int | None = None,
        max_duration: int = 4096,
        vocoder: Any | None = None,
        no_ref_audio: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Generate a prompt+target mel, mirroring ``CFM.sample`` with tags.

        The returned tensor contains the reference frames followed by generated
        target frames.  Callers should slice ``[:, prompt_frames:, :]`` before
        vocoding when they only want the target utterance.
        """
        from torchdiffeq import odeint

        self.eval()
        if cond.ndim == 2:
            cond = self.base.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
        if cond.shape[-1] != self.base.num_channels:
            raise ValueError("cond has the wrong mel channel count.")
        cond = cond.to(dtype=self.dtype, device=self.device)
        batch, cond_seq_len = cond.shape[:2]
        if lens is None:
            lens = torch.full((batch,), cond_seq_len, device=self.device, dtype=torch.long)
        lens = torch.as_tensor(lens, device=self.device, dtype=torch.long)
        if isinstance(text, Sequence) and not isinstance(text, (Tensor, str)):
            tokens = _tokenize(text, self.base.vocab_char_map).to(self.device)
        elif isinstance(text, str):
            raise TypeError("text must be a batch of strings, not one string.")
        else:
            tokens = text.to(self.device).long()
        if tokens.shape[0] != batch:
            raise ValueError("text batch size must match cond.")
        text_lens = (tokens != -1).sum(dim=-1)
        lens = torch.maximum(text_lens, lens)
        prompt_mask_lens = torch.minimum(lens, torch.full_like(lens, cond_seq_len))
        cond_mask = _lens_to_mask(prompt_mask_lens, cond_seq_len)
        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=self.device, dtype=torch.long)
        duration = torch.as_tensor(duration, device=self.device, dtype=torch.long)
        duration = torch.maximum(lens + 1, duration).clamp(max=max_duration)
        max_total = int(duration.max().item())
        cond = F.pad(cond, (0, 0, 0, max_total - cond_seq_len), value=0.0)
        cond_mask = F.pad(cond_mask, (0, max_total - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
        if no_ref_audio:
            cond = torch.zeros_like(cond)
            step_cond = torch.zeros_like(step_cond)
        mask = _lens_to_mask(duration, max_total) if batch > 1 else None
        if tag_ids is None:
            tag_ids = torch.zeros((batch,), dtype=torch.long, device=self.device)
        tag_ids = torch.as_tensor(tag_ids, device=self.device, dtype=torch.long)
        if tag_ids.shape != (batch,):
            raise ValueError("tag_ids must have shape (B,).")

        def fn(t: Tensor, x: Tensor) -> Tensor:
            pred = self._transformer_forward(
                x=x,
                cond=step_cond,
                text=tokens,
                time=t,
                tag_ids=tag_ids,
                mask=mask,
                drop_audio_cond=False,
                drop_text=False,
            )
            if cfg_strength < 1e-5:
                return pred
            null = self._transformer_forward(
                x=x,
                cond=step_cond,
                text=tokens,
                time=t,
                tag_ids=torch.zeros_like(tag_ids),
                mask=mask,
                drop_audio_cond=True,
                drop_text=True,
            )
            return pred + (pred - null) * cfg_strength

        noise: list[Tensor] = []
        if seed is not None:
            torch.manual_seed(seed)
        for total in duration:
            noise.append(
                torch.randn(
                    int(total),
                    self.base.num_channels,
                    device=self.device,
                    dtype=step_cond.dtype,
                )
            )
        y0 = torch.nn.utils.rnn.pad_sequence(noise, padding_value=0, batch_first=True)
        t = torch.linspace(0, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        trajectory = odeint(fn, y0, t, **self.base.odeint_kwargs)
        sampled = trajectory[-1]
        out = torch.where(cond_mask, cond, sampled)
        if vocoder is not None:
            out = out.permute(0, 2, 1)
            out = vocoder(out)
        return out, trajectory


def load_emotion_model(
    checkpoint: str | Path | None = None,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    num_tags: int = len(EMOTION_TAGS),
    tag_dropout: float = 0.1,
    device: str = "cpu",
    use_attention_mask: bool = False,
) -> tuple[EmotionCFM, list[LoRALinear]]:
    """Load the pinned/adapted base CFM and add frozen-base LoRA + tag layers."""
    from samvyas.models.tts.indic_f5 import load_indic_model

    base = load_indic_model(checkpoint)
    base.requires_grad_(False)
    lora = apply_lora_to_dit(base.transformer, rank=rank, alpha=alpha, dropout=dropout)
    model = EmotionCFM(
        base,
        lora,
        num_tags=num_tags,
        tag_dropout=tag_dropout,
        use_attention_mask=use_attention_mask,
    )
    model.to(device)
    return model, lora


def _adapter_tensors(model: nn.Module) -> dict[str, Tensor]:
    """Collect only LoRA matrices and the tag embedding from a model state dict."""
    tensors: dict[str, Tensor] = {}
    for name, value in model.state_dict().items():
        if "lora_a" in name or "lora_b" in name or name.startswith("tag_embed."):
            tensors[name] = value.detach().cpu().contiguous().clone()
    if not tensors:
        raise ValueError("No adapter tensors found; was LoRA applied?")
    return tensors


def save_adapter(
    model: nn.Module,
    path: str | Path,
    *,
    config: LoRAConfig,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically save only LoRA matrices, tag embedding, and a JSON sidecar."""
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": ADAPTER_FORMAT,
        "lora_rank": config.rank,
        "lora_alpha": config.alpha,
        "lora_dropout": config.dropout,
        "lora_targets": list(config.targets),
        "tag_names": list(EMOTION_TAGS),
        "tag_count": int(getattr(model, "num_tags", len(EMOTION_TAGS))),
    }
    if metadata:
        payload.update(metadata)
    temporary = path.with_suffix(".tmp")
    save_file(_adapter_tensors(model), str(temporary), metadata={"format": ADAPTER_FORMAT})
    temporary.replace(path)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_adapter(
    model: EmotionCFM,
    path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load adapter tensors into an already LoRA-wrapped model."""
    from safetensors.torch import load_file

    path = Path(path)
    sidecar = path.with_suffix(".json")
    metadata: dict[str, Any] = {}
    if sidecar.is_file():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    state = load_file(str(path), device="cpu")
    expected = set(_adapter_tensors(model))
    missing = sorted(expected - set(state))
    unexpected = sorted(set(state) - expected)
    if strict and (missing or unexpected):
        raise ValueError(
            f"Adapter key mismatch; missing={missing[:4]}, unexpected={unexpected[:4]}"
        )
    model.load_state_dict(state, strict=False)
    return metadata


def merge_lora_into_base(model: EmotionCFM) -> None:
    """Fold every adapter into its frozen linear and restore plain F5 modules."""
    for module in model.lora_modules:
        if not isinstance(module, LoRALinear):
            raise TypeError("lora_modules contains a non-LoRALinear module.")
        parent, name = _find_parent(model, module)
        setattr(parent, name, module.merged())
    model.lora_modules.clear()


def _find_parent(root: nn.Module, target: nn.Module) -> tuple[nn.Module, str]:
    """Locate the immediate parent and attribute name of a module."""
    for name, module in root.named_modules():
        if module is target:
            parent_name, _, leaf = name.rpartition(".")
            parent = root.get_submodule(parent_name) if parent_name else root
            return parent, leaf
    raise ValueError("Module is not part of this model.")
