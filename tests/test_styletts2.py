"""Unit tests for Bengali G2P, Style Bank, and StyleTTS 2 engine."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from samvyas.audio.phonemize_bn import (
    BengaliPhonemeTokenizer,
    BengaliPhonemizer,
    BengaliTextNormalizer,
)
from samvyas.models.tts.style_bank import StyleBankManager
from samvyas.models.tts.styletts2 import StyleTTS2Config, StyleTTS2Engine


def test_bengali_text_normalizer():
    # Digits expansion
    raw = "আমি ১২৩৪ নম্বর কক্ষে আছি।"
    norm = BengaliTextNormalizer.normalize(raw)
    assert "এক" in norm
    assert "দুই" in norm
    assert "তিন" in norm
    assert "চার" in norm

    # Punctuation spacing
    raw_punct = "হ্যালো!কেমন আছো?ভালো।"
    norm_punct = BengaliTextNormalizer.normalize(raw_punct)
    assert "হ্যালো । কেমন আছো । ভালো" in norm_punct


def test_bengali_phonemizer_and_tokenizer():
    phonemizer = BengaliPhonemizer()
    tokenizer = BengaliPhonemeTokenizer()

    text = "নমস্কার"
    ipa = phonemizer.phonemize(text)
    assert len(ipa) > 0
    # Must contain phonetic segments
    assert any(c in ipa for c in ["n", "m", "ʃ", "k", "r"])

    # Token encoding and decoding
    tokens = tokenizer.encode(ipa)
    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)

    decoded = tokenizer.decode(tokens)
    assert len(decoded) > 0


def test_style_bank_manager():
    bank = StyleBankManager(style_dim=64)

    # Retrieval of standard emotions
    neutral = bank.get_style("neutral")
    excited = bank.get_style("excited")
    assert neutral.shape == (1, 64)
    assert excited.shape == (1, 64)
    assert pytest.approx(torch.norm(neutral).item(), rel=1e-3) == 1.0

    # Unrecognized emotion fallback
    unknown = bank.get_style("completely_unknown_emotion")
    assert unknown.shape == (1, 64)
    # Fallback matches neutral
    assert torch.allclose(unknown, neutral)

    # Interpolation
    blended = bank.interpolate("neutral", "excited", alpha=0.5)
    assert blended.shape == (1, 64)
    assert pytest.approx(torch.norm(blended).item(), rel=1e-3) == 1.0

    # Custom registration
    custom_vec = torch.randn(64)
    bank.register_style("sarcastic", custom_vec)
    retrieved = bank.get_style("sarcastic")
    assert retrieved.shape == (1, 64)

    # Serialization roundtrip
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test_bank.pt"
        bank.save(tmp_path)
        assert tmp_path.exists()

        loaded_bank = StyleBankManager.load(tmp_path)
        assert loaded_bank.style_dim == 64
        assert "sarcastic" in loaded_bank.styles


def test_styletts2_engine_synthesis():
    config = StyleTTS2Config(device="cpu")
    engine = StyleTTS2Engine(config=config)

    # Regular synthesis
    output = engine.synthesize("নমস্কার, আমি জিত।", emotion="warm", speed=1.0)
    assert output.sample_rate_24k == 24000
    assert output.sample_rate_16k == 16000
    assert output.audio_24k.ndim == 2
    assert output.audio_24k.shape[0] == 1
    assert output.audio_16k.ndim == 2
    assert output.audio_16k.shape[0] == 1
    assert output.duration_sec > 0.0
    assert output.latency_ms > 0.0

    # Bounds check
    assert torch.max(torch.abs(output.audio_24k)).item() <= 1.0
    assert torch.max(torch.abs(output.audio_16k)).item() <= 1.0

    # Empty text handling
    empty_output = engine.synthesize("", emotion="neutral")
    assert empty_output.audio_24k.shape[0] == 1
    assert empty_output.duration_sec > 0.0
