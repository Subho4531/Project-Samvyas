"""Denoiser interfaces and DeepFilterNet v2 wrapper."""

from samvyas.audio.denoiser.base import BaseDenoiser
from samvyas.audio.denoiser.deepfilter import DeepFilterDenoiser

__all__ = ["BaseDenoiser", "DeepFilterDenoiser"]
