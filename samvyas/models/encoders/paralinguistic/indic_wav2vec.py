"""AI4Bharat IndicWav2Vec paralinguistic encoder (acoustics branch).

Wraps a HuggingFace Wav2Vec2 checkpoint (default: the diagram's
`ai4bharat/indicwav2vec_v1_bengali`) via `transformers` AutoModel and exposes
frozen self-supervised acoustic features for prosody/emotion/imperfection
modeling downstream.

Access note: the AI4Bharat repos are access-gated. Request access at
https://huggingface.co/ai4bharat/indicwav2vec_v1_bengali then run
`huggingface-cli login` once; weights cache to `~/.cache/huggingface`
(git-ignored via `*.bin`/`*.safetensors`). Until then, any `repo_id` with a
Wav2Vec2 config (e.g. a tiny random model) exercises the same code path.

VRAM (RTX 2050 4GB): frozen + eval, fp16 autocast on CUDA, batch-of-1
short clips (<=5s), `empty_cache()` after each forward.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import torch
import torch.nn as nn

from samvyas.models.encoders.paralinguistic.base import (
    ParalinguisticEncoderOutput,
    align_frames,
)
from samvyas.models.encoders.paralinguistic.prosody import ProsodicFeatureExtractor

DEFAULT_REPO_ID = "ai4bharat/indicwav2vec_v1_bengali"
LayerSelection = Literal["last", "mean_last4"]


class IndicWav2VecEncoder(nn.Module):
    """Frozen IndicWav2Vec acoustic feature extractor."""

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        local_path: Optional[Union[str, Path]] = None,
        layer: LayerSelection = "last",
        freeze: bool = True,
        include_prosody: bool = True,
        device: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        if layer not in ("last", "mean_last4"):
            raise ValueError(f"layer must be 'last' or 'mean_last4', got {layer!r}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.repo_id = repo_id
        self.local_path = Path(local_path) if local_path else None
        self.layer = layer
        self.freeze = freeze
        self.include_prosody = include_prosody
        self.trust_remote_code = trust_remote_code
        self._processor = None
        self._model = None
        self._hidden_dim: Optional[int] = None
        self.prosody = ProsodicFeatureExtractor()

    # -- loading -------------------------------------------------------
    def load(self) -> "IndicWav2VecEncoder":
        """Load processor + model (downloads on first call via HF cache)."""
        if self._model is not None:
            return self
        from transformers import AutoModel, AutoProcessor

        src = str(self.local_path) if self.local_path else self.repo_id
        try:
            self._processor = AutoProcessor.from_pretrained(
                src, trust_remote_code=self.trust_remote_code
            )
            self._model = AutoModel.from_pretrained(
                src, trust_remote_code=self.trust_remote_code
            )
        except Exception as e:
            msg = str(e)
            if "gated" in msg.lower() or "403" in msg or "restricted" in msg.lower():
                raise RuntimeError(
                    f"Cannot access '{self.repo_id}': gated repo. Request access at "
                    f"https://huggingface.co/{self.repo_id} then run "
                    f"`huggingface-cli login`. Original error: {e}"
                ) from e
            raise
        self._model.to(self.device)
        if self.freeze:
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
        self._hidden_dim = int(self._model.config.hidden_size)
        return self

    @property
    def hidden_dim(self) -> int:
        """D_acoustic — resolved from the checkpoint config on load."""
        if self._hidden_dim is None:
            self.load()
        assert self._hidden_dim is not None
        return self._hidden_dim

    @staticmethod
    def _to_batch(waveform: torch.Tensor) -> torch.Tensor:
        wav = waveform.float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        elif wav.dim() == 3:
            wav = wav.squeeze(1) if wav.shape[1] == 1 else wav.mean(dim=1)
        return wav.contiguous()

    # -- inference -----------------------------------------------------
    @torch.no_grad()
    def encode_acoustic(self, waveform: torch.Tensor, sample_rate: int = 16_000) -> torch.Tensor:
        """SSL acoustic embeddings (B, T_frames, D_acoustic) on CPU."""
        if sample_rate != 16_000:
            raise ValueError(f"Expected 16000Hz audio, got {sample_rate}Hz.")
        self.load()
        assert self._processor is not None and self._model is not None
        wav = self._to_batch(waveform).cpu().numpy()
        inputs = self._processor(
            list(wav.astype("float32")),
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        use_amp = self.device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = self._model(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states  # (embeddings, layer_1..N)
        assert hidden_states is not None
        if self.layer == "last":
            feats = hidden_states[-1]
        else:  # mean_last4: richer acoustic blend, same shape
            feats = torch.stack(hidden_states[-4:], dim=0).mean(dim=0)

        # Trim padding frames using the attention mask when available.
        if attention_mask is not None:
            ratio = feats.shape[1] / attention_mask.shape[1]
            lengths = (attention_mask.sum(dim=1).float() * ratio).long().clamp_min(1)
        else:
            lengths = torch.full((feats.shape[0],), feats.shape[1])
        self._last_lengths = lengths.cpu()
        out = feats.float().cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    @torch.no_grad()
    def encode(
        self, waveform: torch.Tensor, sample_rate: int = 16_000
    ) -> ParalinguisticEncoderOutput:
        """Full branch output: SSL acoustics + aligned prosody latents."""
        acoustic = self.encode_acoustic(waveform, sample_rate)
        prosody = None
        if self.include_prosody:
            prosody = self.prosody.extract(waveform)
            prosody = align_frames(prosody, acoustic.shape[1])
        return ParalinguisticEncoderOutput(
            acoustic_embeddings=acoustic,
            prosody_latents=prosody,
            lengths=getattr(self, "_last_lengths", None),
        )

    def forward(self, waveform: torch.Tensor, sample_rate: int = 16_000) -> ParalinguisticEncoderOutput:
        return self.encode(waveform, sample_rate)

    def align_to_semantic(
        self, para: ParalinguisticEncoderOutput, semantic_frames: int
    ) -> ParalinguisticEncoderOutput:
        """Align branch output to the semantic branch frame count (fusion-ready)."""
        acoustic = align_frames(para.acoustic_embeddings, semantic_frames)
        prosody = (
            align_frames(para.prosody_latents, semantic_frames)
            if para.prosody_latents is not None
            else None
        )
        return ParalinguisticEncoderOutput(
            acoustic_embeddings=acoustic, prosody_latents=prosody, lengths=para.lengths
        )
