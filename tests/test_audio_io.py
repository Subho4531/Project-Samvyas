"""Unit tests for Audio utilities."""

import torch

from voicellm.audio.io import normalize_loudness, trim_silence
from voicellm.audio.processing import extract_mel_spectrogram


def test_audio_normalize_and_trim():
    sr = 24000
    t = torch.linspace(0, 1.0, sr)
    signal = torch.sin(2 * 3.14159 * 440.0 * t).unsqueeze(0)
    silence = torch.zeros(1, sr // 2)

    padded_signal = torch.cat([silence, signal, silence], dim=-1)

    trimmed = trim_silence(padded_signal, top_db=30.0)
    assert trimmed.shape[-1] < padded_signal.shape[-1]

    normed = normalize_loudness(trimmed, sample_rate=sr, target_lufs=-20.0)
    assert torch.max(torch.abs(normed)) <= 1.0


def test_extract_mel_spectrogram():
    sr = 24000
    waveform = torch.randn(1, sr)
    mel = extract_mel_spectrogram(waveform, sample_rate=sr, n_mels=80)
    assert mel.shape[1] == 80
