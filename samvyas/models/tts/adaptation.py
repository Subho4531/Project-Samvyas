"""Partial IndicF5 adaptation using its native conditional flow-matching loss."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch import Tensor, nn
from torch.utils.data import Dataset

from samvyas.models.tts.indic_f5 import MODEL_REVISION


class SpeakerDataset(Dataset):
    """Read portable 16 kHz mono records; return native-model 24 kHz (1,T) audio."""

    def __init__(self, root: str | Path, split: str) -> None:
        self.root = Path(root).resolve()
        if split not in ("train", "validation"):
            raise ValueError("Unknown split.")
        self.rows = [
            json.loads(line)
            for line in (self.root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise ValueError("Split is empty.")
        for row in self.rows:
            path = (self.root / row["audio_path"]).resolve()
            path.relative_to(self.root)
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Audio checksum mismatch.")
            if not row["text"].strip():
                raise ValueError("Missing transcript.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        row = self.rows[index]
        audio, rate = sf.read(self.root / row["audio_path"], dtype="float32")
        if rate != 16000 or audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Expected finite mono 16 kHz source audio.")
        audio = resample_poly(audio, 3, 2).astype(np.float32)
        return torch.from_numpy(audio).unsqueeze(0), row["text"]


def configure_trainable_layers(model: nn.Module, last_blocks: int = 2) -> list[nn.Parameter]:
    """Freeze the CFM; enable the final DiT blocks and output modulation/projection."""
    transformer = model.transformer
    blocks = transformer.transformer_blocks
    if not 1 <= last_blocks <= len(blocks):
        raise ValueError("Trainable block count must be between one and model depth.")
    model.requires_grad_(False)
    for module in [*blocks[-last_blocks:], transformer.norm_out, transformer.proj_out]:
        module.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def export_checkpoint(model: nn.Module, path: str | Path) -> None:
    """Atomically save full online weights using the inference loader's EMA-key convention."""
    from safetensors.torch import save_file

    path = Path(path)
    temporary = path.with_suffix(".tmp")
    weights = {
        f"ema_model.{key}": value.detach().cpu().contiguous().clone()
        for key, value in model.state_dict().items()
    }
    save_file(weights, str(temporary), metadata={"weight_type": "online", "format": "IndicF5"})
    temporary.replace(path)


def validation_loss(model: nn.Module, dataset: SpeakerDataset, device: str) -> float:
    """Use fixed validation randomness without changing the next training RNG stream."""
    python_rng = random.getstate()
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            random.seed(1234)
            torch.manual_seed(1234)
            model.eval()
            losses = []
            for audio, text in dataset:
                loss, _, _ = model(audio.to(device), [text])
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite validation loss.")
                losses.append(float(loss))
            return sum(losses) / len(losses)
    finally:
        random.setstate(python_rng)


def train_speaker(
    model: nn.Module,
    data_root: str | Path,
    output: str | Path,
    *,
    device: str = "cuda",
    epochs: int = 40,
    learning_rate: float = 1e-5,
    accumulation: int = 4,
    last_blocks: int = 2,
    patience: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Batch-one partial adaptation; export best and last checkpoints and loss history."""
    if epochs < 1 or accumulation < 1 or patience < 1 or learning_rate <= 0:
        raise ValueError("Training counts and learning rate must be positive.")
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Output must be new or empty.")
    train = SpeakerDataset(data_root, "train")
    valid = SpeakerDataset(data_root, "validation")
    if {r["text"] for r in train.rows} & {r["text"] for r in valid.rows}:
        raise ValueError("Train/validation transcript overlap.")
    vocab = model.vocab_char_map
    unknown = sorted(
        {c for split in (train, valid) for r in split.rows for c in r["text"] if c not in vocab}
    )
    if unknown:
        raise ValueError(
            f"Transcripts contain characters absent from the pinned vocabulary: {unknown}"
        )
    random.seed(seed)
    torch.manual_seed(seed)
    parameters = configure_trainable_layers(model, last_blocks)
    model.to(device)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    amp = device == "cuda"
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)
    root.mkdir(parents=True, exist_ok=True)
    history = []
    best = validation_loss(model, valid, device)
    initial_loss = best
    # Preserve the base model if no epoch improves held-out loss.
    export_checkpoint(model, root / "best.safetensors")
    stale = 0
    steps = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train)))
        random.Random(seed + epoch).shuffle(order)
        losses = []
        for start in range(0, len(order), accumulation):
            group = order[start : start + accumulation]
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                audio, text = train[index]
                with torch.amp.autocast("cuda", dtype=dtype, enabled=amp):
                    loss, _, _ = model(audio.to(device), [text])
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss; checkpoint was not updated.")
                losses.append(float(loss.detach()))
                scaler.scale(loss / len(group)).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Non-finite gradients; checkpoint was not updated.")
            scaler.step(optimizer)
            scaler.update()
            steps += 1
        value = validation_loss(model, valid, device)
        improved = value < best
        if improved:
            best = value
            stale = 0
            export_checkpoint(model, root / "best.safetensors")
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "steps": steps,
                "train_loss": sum(losses) / len(losses),
                "validation_loss": value,
                "improved": improved,
            }
        )
        print(json.dumps(history[-1]), flush=True)
        (root / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stale >= patience:
            break
    export_checkpoint(model, root / "last.safetensors")
    report = {
        "base_model": "ai4bharat/IndicF5",
        "base_revision": MODEL_REVISION,
        "source_sample_rate": 16000,
        "model_sample_rate": 24000,
        "train_clips": len(train),
        "validation_clips": len(valid),
        "method": "partial fine-tuning",
        "last_blocks": last_blocks,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "steps": steps,
        "initial_validation_loss": initial_loss,
        "best_validation_loss": best,
        "improved_over_base": best < initial_loss,
        "weight_type": "online (no EMA averaging)",
        "seed": seed,
        "learning_rate": learning_rate,
        "accumulation": accumulation,
        "listening_review": "pending; validation loss does not measure voice likeness",
    }
    (root / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
