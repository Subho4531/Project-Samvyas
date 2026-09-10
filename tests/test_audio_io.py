"""Tests for samvyas.audio.io (16kHz mono canonical format)."""

from pathlib import Path

import torch

from samvyas.audio.io import TARGET_SR, load_audio, save_audio, to_batch

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "audio"


def test_target_sr_constant():
    assert TARGET_SR == 16_000


def test_load_real_wav_canonical():
    wav = load_audio(ASSETS / "audio_for_inference.wav")
    assert wav.dim() == 1
    assert wav.dtype == torch.float32
    assert wav.numel() > TARGET_SR  # longer than 1s
    assert wav.abs().max() <= 1.0 + 1e-5


def test_load_m4a_resamples_to_16k():
    wav = load_audio(ASSETS / "vocaltest.m4a")
    assert wav.dim() == 1
    assert wav.dtype == torch.float32
    assert wav.numel() > 0


def test_missing_file_raises():
    try:
        load_audio(ASSETS / "does_not_exist.wav")
    except FileNotFoundError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError")


def test_save_and_reload_roundtrip(tmp_path):
    sr = 16_000
    t = torch.arange(sr, dtype=torch.float32) / sr
    sine = 0.5 * torch.sin(2 * torch.pi * 440.0 * t)
    out = save_audio(tmp_path / "roundtrip.wav", sine, sample_rate=sr)
    back = load_audio(out)
    assert back.shape == sine.shape
    assert torch.allclose(back, sine, atol=1e-4)


def test_to_batch_layouts():
    t = torch.randn(16000)
    assert to_batch(t).shape == (1, 16000)
    assert to_batch(torch.randn(2, 16000)).shape == (2, 16000)
    assert to_batch(torch.randn(2, 1, 16000)).shape == (2, 16000)
    # stereo downmix
    assert to_batch(torch.randn(2, 2, 16000)).shape == (2, 16000)
