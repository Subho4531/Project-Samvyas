"""Tests for STTPipeline composition (denoiser + semantic encoder)."""

from pathlib import Path

import torch

from samvyas.models.encoders.semantic.base import SemanticEncoderOutput
from samvyas.pipelines.stt_pipeline import STTPipeline, STTPipelineOutput

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "audio"


class _FakeDenoiser:
    def enhance(self, audio, sample_rate=16_000):
        assert sample_rate == 16_000
        return audio.reshape(-1)


class _FakeEncoder:
    def encode_with_text(self, waveform, sample_rate=16_000):
        assert sample_rate == 16_000
        t_frames = max(1, waveform.shape[-1] // 320)
        return SemanticEncoderOutput(
            logits=torch.randn(1, t_frames, 16),
            hidden_states=torch.randn(1, t_frames, 768),
            tokens=torch.zeros(1, t_frames, dtype=torch.long),
            text=["আমি বাংলায় কথা বলছি"],
        )


def test_pipeline_tensor_input():
    pipe = STTPipeline(denoiser=_FakeDenoiser(), encoder=_FakeEncoder())  # type: ignore[arg-type]
    out = pipe.process(torch.randn(16_000))
    assert isinstance(out, STTPipelineOutput)
    assert out.cleaned_audio.shape == (16_000,)
    assert out.hidden_states.shape[-1] == 768
    assert "বাংলা" in out.transcription


def test_pipeline_file_input():
    pipe = STTPipeline(denoiser=_FakeDenoiser(), encoder=_FakeEncoder())  # type: ignore[arg-type]
    out = pipe.process(ASSETS / "audio_for_inference.wav")
    assert out.cleaned_audio.dim() == 1
    assert out.logits.dim() == 3


def test_pipeline_rejects_wrong_sr():
    pipe = STTPipeline(denoiser=_FakeDenoiser(), encoder=_FakeEncoder())  # type: ignore[arg-type]
    try:
        pipe.process(torch.randn(8000), sample_rate=8000)
    except ValueError as e:
        assert "16000Hz" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
