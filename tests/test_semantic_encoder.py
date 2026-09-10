"""Tests for IndicASREncoder (Vdim=768 contract).

Fast tests inject a stub NeMo model — no 523MB download. The real-checkpoint
test is gated behind SAMVYAS_RUN_HEAVY=1 with a <=5s clip (4GB VRAM guard).
"""

import os

import pytest
import torch
import torch.nn as nn

from samvyas.models.encoders.semantic.base import SemanticEncoderOutput
from samvyas.models.encoders.semantic.indic_asr import HIDDEN_DIM, IndicASREncoder

HEAVY = os.getenv("SAMVYAS_RUN_HEAVY") == "1"


class _StubPreprocessor(nn.Module):
    def forward(self, input_signal, length):
        b, t = input_signal.shape
        frames = t // 160  # 10ms hop at 16kHz
        return torch.randn(b, 80, frames), torch.full((b,), frames)


class _StubEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, audio_signal, length):
        b, _, frames = audio_signal.shape
        t_frames = frames // 2
        return torch.randn(b, t_frames, self.dim), torch.full((b,), t_frames)


class _StubCTC(nn.Module):
    def __init__(self, dim: int, vocab: int = 64):
        super().__init__()
        self.lin = nn.Linear(dim, vocab)

    def forward(self, encoder_output):
        return self.lin(encoder_output)


class _StubNeMo(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.preprocessor = _StubPreprocessor()
        self.encoder = _StubEncoder(dim)
        self.ctc_decoder = _StubCTC(dim)

    def freeze(self):
        pass

    def change_decoding_strategy(self, decoder_type=None, verbose=False):
        pass


def _encoder_with_stub(enc_dim: int) -> IndicASREncoder:
    enc = IndicASREncoder(device="cpu")
    enc._model = _StubNeMo(enc_dim)
    return enc


def test_output_contract_native_768():
    enc = _encoder_with_stub(768)
    wav = torch.randn(1, 16_000)
    out = enc.encode(wav)
    assert isinstance(out, SemanticEncoderOutput)
    b, t, d = out.hidden_states.shape
    assert (b, d) == (1, HIDDEN_DIM)
    assert out.logits.shape[:2] == (b, t)
    assert out.tokens.shape == (b, t)
    assert out.hidden_states.dtype == torch.float32


def test_output_contract_adapted_dim():
    enc = _encoder_with_stub(512)  # non-768 backbone -> adapter path
    out = enc.encode(torch.randn(2, 16_000))
    assert out.hidden_states.shape == (2, out.hidden_states.shape[1], 768)


def test_wrong_sample_rate_rejected():
    enc = _encoder_with_stub(768)
    with pytest.raises(ValueError, match="16000Hz"):
        enc.encode(torch.randn(1, 8000), sample_rate=8000)
    with pytest.raises(ValueError, match="16000Hz"):
        enc.transcribe(torch.randn(1, 8000), sample_rate=8000)


def test_invalid_decoder_rejected():
    with pytest.raises(ValueError, match="decoder"):
        IndicASREncoder(decoder="beam")


@pytest.mark.skipif(not HEAVY, reason="needs .nemo checkpoint + GPU; set SAMVYAS_RUN_HEAVY=1")
def test_real_checkpoint_short_clip():
    from pathlib import Path

    from samvyas.audio.io import load_audio

    wav = load_audio(Path(__file__).resolve().parents[1] / "assets" / "audio" / "vocaltest.m4a")
    clip = wav[: 5 * 16_000].unsqueeze(0)  # <=5s guard for 4GB VRAM
    enc = IndicASREncoder(decoder="ctc")
    out = enc.encode_with_text(clip)
    assert out.hidden_states.shape[-1] == 768
    assert isinstance(out.text[0], str) and len(out.text[0]) > 0
