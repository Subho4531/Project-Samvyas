"""Abstract base class for Neural Audio Codecs."""

from abc import ABC, abstractmethod
from typing import List, Tuple
import torch
import torch.nn as nn


class BaseAudioCodec(nn.Module, ABC):
    """Abstract interface for audio neural codecs (e.g. SNAC, DAC, EnCodec)."""

    def __init__(self, sample_rate: int = 24000, num_codebooks: int = 4, codebook_size: int = 4096):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

    @abstractmethod
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encodes audio waveform to discrete multi-codebook acoustic tokens.
        
        Args:
            waveform: Tensor of shape (B, 1, T)
            
        Returns:
            tokens: Tensor of shape (B, num_codebooks, seq_len)
        """
        pass

    @abstractmethod
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decodes multi-codebook acoustic tokens back to audio waveform.
        
        Args:
            tokens: Tensor of shape (B, num_codebooks, seq_len)
            
        Returns:
            waveform: Tensor of shape (B, 1, T)
        """
        pass
