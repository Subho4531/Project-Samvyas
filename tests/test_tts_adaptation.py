"""Verify speaker splits, native-rate tensors, partial gradients and export compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from samvyas.data.tts_adaptation import prepare_dataset
from samvyas.data.voice_dataset import VoiceDataset, VoicePrompt
from samvyas.models.tts.adaptation import (
    SpeakerDataset,
    configure_trainable_layers,
    export_checkpoint,
    train_speaker,
)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "voice"
    dataset = VoiceDataset(root)
    audio = (0.1 * np.sin(np.arange(16000) * 0.1)).astype(np.float32)
    for i in range(10):
        prompt = VoicePrompt(str(i), f"word {i}", "neutral")
        dataset.accept(prompt, prompt.text, audio)
    return root


def test_portable_split_and_integrity(source: Path, tmp_path: Path) -> None:
    first = tmp_path / "first"
    report = prepare_dataset(source, first)
    prepare_dataset(source, tmp_path / "second")
    assert report["counts"] == {"train": 9, "validation": 1}
    assert (first / "train.jsonl").read_bytes() == (tmp_path / "second/train.jsonl").read_bytes()
    train = SpeakerDataset(first, "train")
    valid = SpeakerDataset(first, "validation")
    assert not {r["clip_id"] for r in train.rows} & {r["clip_id"] for r in valid.rows}
    audio, text = train[0]
    assert audio.shape == (1, 24000) and audio.dtype == torch.float32 and text
    assert sf.info(source / VoiceDataset(source).records[0]["audio_path"]).samplerate == 16000
    path = first / train.rows[0]["audio_path"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        SpeakerDataset(first, "train")


def tiny_cfm():
    from f5_tts.model import CFM, DiT

    vocab = {c: i for i, c in enumerate(" word0123456789")}
    return CFM(
        transformer=DiT(
            dim=32,
            depth=2,
            heads=2,
            dim_head=16,
            ff_mult=2,
            text_dim=16,
            conv_layers=0,
            text_num_embeds=len(vocab),
            mel_dim=100,
        ),
        vocab_char_map=vocab,
        mel_spec_kwargs={"target_sample_rate": 24000, "n_mel_channels": 100},
    )


def test_upstream_flow_gradients_and_export(tmp_path: Path) -> None:
    model = tiny_cfm()
    params = configure_trainable_layers(model, 1)
    before = {k: p.detach().clone() for k, p in model.named_parameters() if not p.requires_grad}
    audio = torch.randn(1, 24000) * 0.05
    loss, cond, pred = model(audio, ["word 1"])
    assert cond.shape == pred.shape and pred.shape[0] == 1 and pred.shape[-1] == 100
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
    assert model.transformer.proj_out.weight.grad.abs().sum() > 0
    torch.optim.AdamW(params, lr=1e-4).step()
    assert all(torch.equal(p, before[k]) for k, p in model.named_parameters() if k in before)
    export_checkpoint(model, tmp_path / "model.safetensors")
    from safetensors.torch import load_file

    restored = tiny_cfm()
    restored.load_state_dict(
        {
            k.removeprefix("ema_model."): v
            for k, v in load_file(str(tmp_path / "model.safetensors")).items()
        },
        strict=True,
    )
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())


def test_training_loop_exports_valid_report(source: Path, tmp_path: Path) -> None:
    root = tmp_path / "prepared"
    prepare_dataset(source, root)
    report = train_speaker(
        tiny_cfm(), root, tmp_path / "run", device="cpu", epochs=1, accumulation=4, last_blocks=1
    )
    assert report["steps"] == 3  # Includes the final partial accumulation group.
    assert (tmp_path / "run/best.safetensors").is_file()
    assert (tmp_path / "run/last.safetensors").is_file()
    history = json.loads((tmp_path / "run/history.json").read_text())
    assert np.isfinite(history[0]["validation_loss"])
