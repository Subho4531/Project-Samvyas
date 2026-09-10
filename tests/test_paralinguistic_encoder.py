"""Tests for the paralinguistic branch (acoustics + prosody).

Fast tests use stub HF objects — no downloads. The real-checkpoint test is
gated behind SAMVYAS_RUN_HEAVY=1 and skips cleanly until HuggingFace access
to ai4bharat/indicwav2vec_v1_bengali is granted (see configs/encoders/paralinguistic.yaml).
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import BatchFeature

from samvyas.models.encoders.paralinguistic.base import (
    ParalinguisticEncoderOutput,
    align_frames,
)
from samvyas.models.encoders.paralinguistic.indic_wav2vec import IndicWav2VecEncoder
from samvyas.models.encoders.paralinguistic.prosody import (
    PROSODY_DIM,
    ProsodicFeatureExtractor,
)

HEAVY = os.getenv("SAMVYAS_RUN_HEAVY") == "1"


def _sine(secs: float = 1.0, freq: float = 220.0, sr: int = 16_000) -> torch.Tensor:
    t = torch.arange(int(secs * sr), dtype=torch.float32) / sr
    return 0.4 * torch.sin(2 * torch.pi * freq * t)


# -- align_frames ------------------------------------------------------
def test_align_identity():
    x = torch.randn(2, 50, 16)
    assert torch.equal(align_frames(x, 50), x)


def test_align_upsample_downsample():
    x = torch.randn(1, 40, 8)
    assert align_frames(x, 80).shape == (1, 80, 8)
    assert align_frames(x, 20).shape == (1, 20, 8)


def test_align_rejects_bad_dim():
    with pytest.raises(ValueError, match=r"\(B, T, D\)"):
        align_frames(torch.randn(10, 8), 10)


# -- prosody -----------------------------------------------------------
def test_prosody_shape_and_ranges():
    ext = ProsodicFeatureExtractor()
    out = ext.extract(torch.stack([_sine(), _sine(freq=330.0)]))
    assert out.shape[0] == 2 and out.shape[2] == PROSODY_DIM
    f0, voiced, energy, zcr = out.unbind(dim=-1)
    assert ((f0 >= 0) & (f0 <= 1)).all()
    assert set(voiced.unique().tolist()) <= {0.0, 1.0}
    assert torch.isfinite(out).all()


def test_prosody_silence_no_crash():
    ext = ProsodicFeatureExtractor()
    out = ext.extract(torch.zeros(16_000))
    assert out.shape[2] == PROSODY_DIM
    assert torch.isfinite(out).all()


# -- encoder with stubs -------------------------------------------------
class _StubProcessor:
    def __call__(self, audios, sampling_rate=16_000, return_tensors="pt", padding=True):
        import numpy as np

        arr = np.stack([np.asarray(a, dtype=np.float32) for a in audios], axis=0)
        t = arr.shape[1]
        return BatchFeature(
            {"input_values": torch.from_numpy(arr), "attention_mask": torch.ones(1, t)}
        )


class _StubWav2Vec(nn.Module):
    def __init__(self, dim: int = 32, layers: int = 4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=dim)
        self.dim = dim
        self.layers = layers

    def forward(self, input_values, attention_mask=None, output_hidden_states=False):
        b, t = input_values.shape
        frames = max(1, t // 320)
        states = tuple(torch.randn(b, frames, self.dim) for _ in range(self.layers + 1))
        return SimpleNamespace(hidden_states=states if output_hidden_states else None)


def _encoder_with_stub(dim: int = 32, **kwargs) -> IndicWav2VecEncoder:
    enc = IndicWav2VecEncoder(repo_id="stub/test", device="cpu", **kwargs)
    enc._processor = _StubProcessor()
    enc._model = _StubWav2Vec(dim)
    enc._hidden_dim = dim
    return enc


def test_acoustic_output_contract():
    enc = _encoder_with_stub(32, include_prosody=False)
    out = enc.encode(_sine().unsqueeze(0))
    assert isinstance(out, ParalinguisticEncoderOutput)
    b, t, d = out.acoustic_embeddings.shape
    assert (b, d) == (1, 32) and t > 0
    assert out.prosody_latents is None
    assert enc.hidden_dim == 32


def test_prosody_aligned_to_acoustic():
    enc = _encoder_with_stub(32, include_prosody=True)
    out = enc.encode(torch.randn(1, 16_000))
    assert out.prosody_latents is not None
    assert out.prosody_latents.shape[:2] == out.acoustic_embeddings.shape[:2]
    assert out.prosody_latents.shape[2] == PROSODY_DIM


def test_mean_last4_same_shape():
    enc = _encoder_with_stub(32, layer="mean_last4", include_prosody=False)
    out = enc.encode(torch.randn(2, 16_000))
    assert out.acoustic_embeddings.shape == (2, out.acoustic_embeddings.shape[1], 32)


def test_align_to_semantic():
    enc = _encoder_with_stub(32)
    out = enc.encode(torch.randn(1, 16_000))
    fused = enc.align_to_semantic(out, semantic_frames=100)
    assert fused.acoustic_embeddings.shape == (1, 100, 32)
    assert fused.prosody_latents is not None
    assert fused.prosody_latents.shape == (1, 100, PROSODY_DIM)


def test_wrong_sr_and_layer_rejected():
    enc = _encoder_with_stub()
    with pytest.raises(ValueError, match="16000Hz"):
        enc.encode(torch.randn(1, 8000), sample_rate=8000)
    with pytest.raises(ValueError, match="layer"):
        IndicWav2VecEncoder(layer="middle")  # type: ignore[arg-type]


@pytest.mark.skipif(not HEAVY, reason="needs gated HF checkpoint + GPU; set SAMVYAS_RUN_HEAVY=1")
def test_real_indicwav2vec_short_clip():
    from pathlib import Path

    from samvyas.audio.io import load_audio

    wav = load_audio(Path(__file__).resolve().parents[1] / "assets" / "audio" / "vocaltest.m4a")
    clip = wav[: 5 * 16_000].unsqueeze(0)  # <=5s guard for 4GB VRAM
    enc = IndicWav2VecEncoder()
    try:
        out = enc.encode(clip)
    except RuntimeError as e:
        if "Gated" in str(e) or "gated" in str(e):
            pytest.skip(f"HF access pending: {e}")
        raise
    assert out.acoustic_embeddings.dim() == 3
    assert out.acoustic_embeddings.shape[-1] == enc.hidden_dim
    assert out.prosody_latents is not None
    assert out.prosody_latents.shape[:2] == out.acoustic_embeddings.shape[:2]
