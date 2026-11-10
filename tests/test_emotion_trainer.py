"""CPU smoke tests for the Accelerate emotion trainer and resume path."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("f5_tts")

from samvyas.data.emotion_dataset import EmotionPairDataset, write_manifest
from samvyas.models.tts.emotion_trainer import (
    EmotionTrainingConfig,
    _validate,
    train_emotion,
)
from tests.test_emotion_dataset import _row
from tests.test_emotion_lora import make_tiny_model


def _fake_mel(waveform: torch.Tensor) -> torch.Tensor:
    """Return finite (1, 100, T) features without the real mel frontend."""
    return torch.ones(1, 100, max(1, waveform.shape[-1] // 256))


def _datasets(root: Path) -> tuple[EmotionPairDataset, EmotionPairDataset]:
    """Create two short prompt/target pairs for train and validation."""
    rows = [
        _row(root, "one", "happy", "s1", "a"),
        _row(root, "two", "sad", "s1", "b"),
    ]
    write_manifest(root / "train.jsonl", rows)
    write_manifest(root / "validation.jsonl", rows)
    return (
        EmotionPairDataset(root, "train", mel_fn=_fake_mel),
        EmotionPairDataset(root, "validation", mel_fn=_fake_mel),
    )


def _config(root: Path, output: Path, **kwargs) -> EmotionTrainingConfig:
    """Small CPU config that exercises one optimizer step."""
    values = {
        "data_root": root,
        "output_dir": output,
        "epochs": 1,
        "max_steps": 1,
        "learning_rate": 1e-3,
        "accumulation": 1,
        "max_frames": 1000,
        "max_batch_size": 2,
        "eval_batch_size": 2,
        "num_workers": 0,
        "mixed_precision": "no",
        "tag_dropout": 0.0,
        "save_every": 1,
        "log_every": 1,
        "patience": 0,
    }
    values.update(kwargs)
    return EmotionTrainingConfig(**values)


def test_trainer_writes_adapter_and_resumes(tmp_path: Path) -> None:
    """One step saves adapter/state, and a second run resumes from `last`."""
    train, valid = _datasets(tmp_path)
    model, _ = make_tiny_model()
    report = train_emotion(model, train, valid, _config(tmp_path, tmp_path / "run1"))
    assert report["steps"] == 1
    assert report["trainable_parameters"] > 0
    assert report["world_size"] == 1
    assert (tmp_path / "run1" / "last" / "adapter.safetensors").is_file()
    assert (tmp_path / "run1" / "last" / "adapter.json").is_file()
    assert (tmp_path / "run1" / "last" / "trainer_state.pt").is_file()
    assert (tmp_path / "run1" / "training_report.json").is_file()

    resumed_model, _ = make_tiny_model()
    resumed = train_emotion(
        resumed_model,
        train,
        valid,
        _config(
            tmp_path,
            tmp_path / "run2",
            epochs=2,
            max_steps=2,
            resume=tmp_path / "run1" / "last",
        ),
    )
    assert resumed["steps"] == 2
    assert resumed["initial_validation_loss"] < float("inf")
    assert (tmp_path / "run2" / "last" / "adapter.safetensors").is_file()


def test_validation_gather_sums_all_rank_totals() -> None:
    """Validation averages the gathered loss/count pairs, not rank-zero only."""

    class FakeModel:
        def eval(self) -> None:
            pass

        def train(self) -> None:
            pass

        def __call__(self, *, mel: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
            return mel.mean() * 0 + 2.0

    class FakeAccelerator:
        device = torch.device("cpu")

        def gather(self, tensor: torch.Tensor) -> torch.Tensor:
            extra = torch.tensor([100.0, 2.0], dtype=tensor.dtype)
            return torch.cat([tensor, extra])

    batch = {
        "mel": torch.zeros(2, 10, 4),
        "text": torch.zeros(2, 4, dtype=torch.long),
        "tag_ids": torch.ones(2, dtype=torch.long),
        "prompt_lengths": torch.full((2,), 4),
        "lens": torch.full((2,), 10),
    }
    value = _validate(FakeModel(), [batch], FakeAccelerator())
    # Local total is [4, 2]; the fake peer adds [100, 2].
    assert value == pytest.approx(104.0 / 4.0)
