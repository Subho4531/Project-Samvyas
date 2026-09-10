"""Tests for DeepFilterNet v2 denoiser wrapper.

Fast tests avoid weight downloads (bypass + stubbed core). The real-model
test is gated behind SAMVYAS_RUN_HEAVY=1 and a short synthetic clip.
"""

import os

import pytest
import torch

from samvyas.audio.denoiser.deepfilter import DeepFilterDenoiser

HEAVY = os.getenv("SAMVYAS_RUN_HEAVY") == "1"


def _sine(secs: float = 1.0, sr: int = 16_000) -> torch.Tensor:
    t = torch.arange(int(secs * sr), dtype=torch.float32) / sr
    return 0.4 * torch.sin(2 * torch.pi * 440.0 * t)


def test_bypass_passthrough_all_layouts():
    den = DeepFilterDenoiser(enabled=False)
    assert den._model is None  # no weights touched
    a = _sine()
    assert torch.equal(den.enhance(a), a)
    b = torch.stack([a, a])
    assert den.enhance(b).shape == (2, a.shape[0])
    c = b.unsqueeze(1)
    assert den.enhance(c).shape == (2, 1, a.shape[0])


def test_wrong_sample_rate_rejected():
    den = DeepFilterDenoiser(enabled=False)
    # Bypass returns early; enabled path must validate instead.
    den_on = DeepFilterDenoiser(enabled=True)
    den_on._ensure_model = lambda: None  # avoid download; validation happens first
    with pytest.raises(ValueError, match="16000Hz"):
        den_on.enhance(_sine(), sample_rate=8000)


def test_chunked_path_uses_stub_core():
    den = DeepFilterDenoiser(enabled=True, chunk_secs=0.25)
    den._ensure_model = lambda: None  # stub out weight loading
    calls: list[int] = []

    def fake_utt(wav_1d: torch.Tensor) -> torch.Tensor:
        calls.append(wav_1d.shape[-1])
        return wav_1d

    den._enhance_utterance = fake_utt  # type: ignore[method-assign]
    out = den.enhance(_sine(secs=1.0))
    assert out.shape == (16_000,)  # 1-D in -> 1-D out
    assert len(calls) == 4  # 1s split into 4 x 0.25s chunks
    out_b = den.enhance(_sine(secs=1.0).unsqueeze(0))
    assert out_b.shape == (1, 16_000)


@pytest.mark.skipif(not HEAVY, reason="needs DeepFilter weights + GPU; set SAMVYAS_RUN_HEAVY=1")
def test_real_deepfilter_short_clip_shape():
    den = DeepFilterDenoiser(enabled=True)
    noisy = _sine(secs=2.0) + 0.02 * torch.randn(32_000)
    out = den.enhance(noisy.unsqueeze(0))
    assert out.shape == (1, 32_000)
    assert torch.isfinite(out).all()
