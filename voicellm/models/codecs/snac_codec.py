"""SNAC (Multi-Scale Neural Audio Codec) Adapter."""

from typing import Optional
import torch

from voicellm.core.utils import setup_logger
from voicellm.models.codecs.base_codec import BaseAudioCodec

logger = setup_logger("SNACCodec")


class SNACCodec(BaseAudioCodec):
    """Wrapper for SNAC (Multi-Scale Neural Audio Codec) by Hubert Siuzdak.
    Supports 24kHz / 32kHz / 44.1kHz hierarchical multi-resolution codebooks.
    """

    def __init__(
        self,
        model_name: str = "hubertsiuzdak/snac_24khz",
        sample_rate: int = 24000,
        device: Optional[str] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_codebooks=4, codebook_size=4096)
        self.model_name = model_name
        self.codec_model = None
        self._load_model(device)

    def _load_model(self, device: Optional[str] = None):
        try:
            from snac import SNAC
            logger.info(f"Loading pretrained SNAC codec model: {self.model_name}")
            self.codec_model = SNAC.from_pretrained(self.model_name)
            if device:
                self.codec_model.to(device)
            self.codec_model.eval()
        except ImportError:
            logger.warning(
                "snac package is not installed. Codec operations will use synthetic fallback. "
                "Install with `pip install snac`."
            )
            self.codec_model = None

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encodes waveform to discrete tokens."""
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)  # (B, 1, T)

        if self.codec_model is not None:
            # SNAC returns a list of tensors for each hierarchy level
            codes = self.codec_model.encode(waveform)
            # Reconstruct into a uniform representation or flat delay pattern
            # For 4 codebooks in SNAC: codes[0] (B, T), codes[1] (B, 2T), codes[2] (B, 4T)
            # Standard uniform stacked token grid for base modeling:
            return codes
        else:
            # Synthetic mock token generator for offline / lightweight dry-runs
            B, _, T = waveform.shape
            seq_len = max(1, T // 320)
            return torch.randint(0, self.codebook_size, (B, self.num_codebooks, seq_len), device=waveform.device)

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decodes discrete tokens back to audio waveform."""
        if self.codec_model is not None:
            audio = self.codec_model.decode(tokens)
            return audio
        else:
            # Synthetic sine-wave mock for dry-runs
            B = tokens.shape[0]
            seq_len = tokens.shape[-1]
            samples = seq_len * 320
            t = torch.linspace(0, 1.0, samples, device=tokens.device)
            mock_audio = torch.sin(2 * 3.14159 * 440.0 * t).repeat(B, 1, 1)
            return mock_audio
