"""Phase-1 STT pipeline: audio in -> clean audio + Bengali transcript + 768-dim latents.

Diagram path covered:
    Mic 16kHz -> DeepFilterNet v2 -> IndicConformer (CTC decode, Vdim 768)

Paralinguistic branch + Audio Projector (d=4096) are deferred to later phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

from samvyas.audio.denoiser.deepfilter import DeepFilterDenoiser
from samvyas.audio.io import load_audio
from samvyas.models.encoders.semantic.indic_asr import IndicASREncoder


@dataclass
class STTPipelineOutput:
    cleaned_audio: torch.Tensor  # (T,) 16kHz enhanced
    transcription: str  # decoded text
    hidden_states: torch.Tensor  # (1, T_frames, 768) semantic latents
    logits: torch.Tensor  # (1, T_frames, vocab) CTC log-probs


class STTPipeline:
    """Denoiser + semantic ASR composed for 4GB-VRAM sequential inference."""

    def __init__(
        self,
        denoiser: Optional[DeepFilterDenoiser] = None,
        encoder: Optional[IndicASREncoder] = None,
        device: str = "auto",
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.denoiser = denoiser or DeepFilterDenoiser(device=str(self.device))
        self.encoder = encoder or IndicASREncoder(device=str(self.device))

    def process(
        self,
        audio: Union[str, Path, torch.Tensor],
        sample_rate: int = 16_000,
    ) -> STTPipelineOutput:
        """Run full STT pass. `audio` may be a file path or raw tensor."""
        if isinstance(audio, (str, Path)):
            wav = load_audio(audio, target_sr=16_000)  # (T,)
        else:
            wav = audio.float().reshape(-1).cpu()
            if sample_rate != 16_000:
                raise ValueError(f"Expected 16000Hz audio, got {sample_rate}Hz.")

        # Stage 1: denoise (lazy-loads DeepFilter weights on first call).
        clean = self.denoiser.enhance(wav, sample_rate=16_000)
        clean_1d = clean.reshape(-1).cpu()

        # Stage 2: encode + transcribe (lazy-loads .nemo on first call).
        with torch.no_grad():
            enc_out = self.encoder.encode_with_text(clean_1d, sample_rate=16_000)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        text = enc_out.text[0] if enc_out.text else ""
        return STTPipelineOutput(
            cleaned_audio=clean_1d,
            transcription=text,
            hidden_states=enc_out.hidden_states,
            logits=enc_out.logits,
        )
