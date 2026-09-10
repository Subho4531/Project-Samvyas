"""Abstract denoiser interface (audio frontend contract)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseDenoiser(ABC):
    """Denoise 16kHz mono audio.

    Implementations accept (B, T) or (B, 1, T) and return the same layout
    with identical time length.
    """

    def __init__(self, sample_rate: int = 16_000, enabled: bool = True) -> None:
        self.sample_rate = sample_rate
        self.enabled = enabled

    @abstractmethod
    def enhance(self, audio: torch.Tensor, sample_rate: int = 16_000) -> torch.Tensor:
        """Enhance audio. Args: audio (B,T) or (B,1,T). Returns same shape."""
        raise NotImplementedError

    def __call__(self, audio: torch.Tensor, sample_rate: int = 16_000) -> torch.Tensor:
        return self.enhance(audio, sample_rate)
