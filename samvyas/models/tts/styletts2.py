"""StyleTTS 2 Bengali Expressive Synthesis Engine for Samvyas.

Provides decoupled timbre and emotional prosody conditioning for low-latency
Bengali conversational voice cloning. Follows Samvyas tensor contracts:
mono float32 in [-1.0, 1.0], exporting 24 kHz native and 16 kHz resampled audio.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from samvyas.audio.phonemize_bn import BengaliPhonemeTokenizer, BengaliPhonemizer
from samvyas.models.tts.style_bank import StyleBankManager

logger = logging.getLogger(__name__)


@dataclass
class StyleTTS2Config:
    """Configuration for StyleTTS 2 Bengali model architecture and inference."""

    sample_rate: int = 24000
    export_sample_rate: int = 16000
    hidden_dim: int = 512
    style_dim: int = 64
    num_heads: int = 4
    num_layers: int = 6
    phoneme_vocab_size: int = 64
    max_seq_len: int = 1024
    device: str = "cpu"
    fp16: bool = False


@dataclass
class StyleTTS2Output:
    """Synthesis result containing waveforms and metadata."""

    audio_24k: torch.Tensor  # shape (1, T_24k) float32 in [-1.0, 1.0]
    audio_16k: torch.Tensor  # shape (1, T_16k) float32 in [-1.0, 1.0]
    sample_rate_24k: int = 24000
    sample_rate_16k: int = 16000
    phonemes: str = ""
    token_ids: list[int] = field(default_factory=list)
    emotion: str = "neutral"
    duration_sec: float = 0.0
    latency_ms: float = 0.0


class AcousticStyleEncoder(nn.Module):
    """Encodes reference audio mel-spectrograms into a compact style vector."""

    def __init__(self, in_mels: int = 80, style_dim: int = 64, hidden_dim: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_mels, hidden_dim, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(hidden_dim, style_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Extracts style vector from mel spectrogram.

        Parameters
        ----------
        mel : torch.Tensor
            Mel spectrogram of shape (B, n_mels, T).

        Returns
        -------
        torch.Tensor
            Style vector of shape (B, style_dim), L2-normalized.
        """
        x = F.relu(self.conv1(mel))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        style = self.linear(x)
        return F.normalize(style, p=2, dim=-1)


class ProsodyAndDurationPredictor(nn.Module):
    """Predicts phoneme duration, F0 (pitch contour), and energy conditioned on style."""

    def __init__(self, hidden_dim: int = 512, style_dim: int = 64) -> None:
        super().__init__()
        self.fc_style = nn.Linear(style_dim, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim, hidden_dim // 2, num_layers=2, batch_first=True, bidirectional=True
        )
        self.duration_proj = nn.Linear(hidden_dim, 1)
        self.f0_proj = nn.Linear(hidden_dim, 1)
        self.energy_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self, phoneme_embs: torch.Tensor, style_vec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts prosody features.

        Parameters
        ----------
        phoneme_embs : torch.Tensor
            Phoneme embeddings of shape (B, T_text, hidden_dim).
        style_vec : torch.Tensor
            Style vector of shape (B, style_dim).

        Returns
        -------
        durations : torch.Tensor of shape (B, T_text)
        f0 : torch.Tensor of shape (B, T_text)
        energy : torch.Tensor of shape (B, T_text)
        """
        style_bias = self.fc_style(style_vec).unsqueeze(1)  # (B, 1, hidden_dim)
        conditioned = phoneme_embs + style_bias
        out, _ = self.gru(conditioned)
        durations = F.relu(self.duration_proj(out).squeeze(-1))
        f0 = self.f0_proj(out).squeeze(-1)
        energy = self.energy_proj(out).squeeze(-1)
        return durations, f0, energy


class StyleTTS2Engine:
    """High-level synthesis engine for decoupled Bengali expressive TTS."""

    def __init__(
        self,
        config: StyleTTS2Config | None = None,
        style_bank: StyleBankManager | None = None,
    ) -> None:
        self.config = config or StyleTTS2Config()
        self.device = torch.device(self.config.device)
        self.phonemizer = BengaliPhonemizer()
        self.tokenizer = BengaliPhonemeTokenizer()
        self.style_bank = style_bank or StyleBankManager(style_dim=self.config.style_dim)

        self.style_encoder = AcousticStyleEncoder(style_dim=self.config.style_dim).to(self.device)
        self.prosody_predictor = ProsodyAndDurationPredictor(
            hidden_dim=self.config.hidden_dim, style_dim=self.config.style_dim
        ).to(self.device)
        self.style_encoder.eval()
        self.prosody_predictor.eval()

    def phonemize(self, text: str) -> tuple[str, list[int]]:
        """Converts raw Bengali text to IPA phonemes and discrete token IDs."""
        ipa_str = self.phonemizer.phonemize(text)
        token_ids = self.tokenizer.encode(ipa_str)
        return ipa_str, token_ids

    def get_style_vector(self, emotion: str, intensity: float = 1.0) -> torch.Tensor:
        """Retrieves conditioned style vector for the target emotion."""
        vec = self.style_bank.get_style(emotion, intensity=intensity)
        return vec.to(self.device)

    def synthesize(
        self,
        text: str,
        emotion: str = "neutral",
        intensity: float = 1.0,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
    ) -> StyleTTS2Output:
        """Synthesizes Bengali text into 24 kHz and 16 kHz waveforms.

        Parameters
        ----------
        text : str
            Input Bengali utterance.
        emotion : str
            Desired emotion label from StyleBank (e.g. 'excited', 'warm', 'whisper').
        intensity : float
            Emotion intensity scale factor in [0.0, 1.0].
        speed : float
            Speech rate multiplier (1.0 = normal, >1.0 = faster).
        pitch_shift : float
            Semitone pitch offset.

        Returns
        -------
        StyleTTS2Output
            Audio waveforms matching Samvyas mono float32 contracts.
        """
        start_t = time.perf_counter()
        ipa_str, token_ids = self.phonemize(text)

        if not token_ids:
            # Handle empty/whitespace text safely
            silence_24k = torch.zeros((1, 2400), dtype=torch.float32)
            silence_16k = torch.zeros((1, 1600), dtype=torch.float32)
            return StyleTTS2Output(
                audio_24k=silence_24k,
                audio_16k=silence_16k,
                phonemes=ipa_str,
                token_ids=token_ids,
                emotion=emotion,
                duration_sec=0.1,
                latency_ms=(time.perf_counter() - start_t) * 1000.0,
            )

        # Style Vector Retrieval
        _style_vec = self.get_style_vector(emotion, intensity=intensity)

        # Estimate duration from tokens & speed
        # Typical Bengali syllable / phoneme is ~80-120 ms
        base_phoneme_duration = 0.09 / max(speed, 0.25)
        total_duration = max(len(token_ids) * base_phoneme_duration, 0.3)
        n_samples_24k = int(total_duration * self.config.sample_rate)

        # Synthetic carrier generation with harmonic structure conditioned on style
        # (Mock generator until full weights are trained/loaded)
        t_arr = torch.linspace(0, total_duration, n_samples_24k, device=self.device)
        f0_base = 135.0 * (2.0 ** (pitch_shift / 12.0))
        if emotion == "excited":
            f0_base *= 1.25
        elif emotion == "whisper":
            f0_base *= 0.85

        # Multi-harmonic voice synthesis with subtle vibrato
        f0_contour = f0_base * (1.0 + 0.04 * torch.sin(2.0 * np.pi * 5.0 * t_arr))
        phase = torch.cumsum(2.0 * np.pi * f0_contour / self.config.sample_rate, dim=0)

        # Fundamental + 3 harmonics
        harmonic_sig = (
            0.6 * torch.sin(phase)
            + 0.25 * torch.sin(2.0 * phase)
            + 0.1 * torch.sin(3.0 * phase)
            + 0.05 * torch.sin(4.0 * phase)
        )

        # Envelope shaping
        envelope = torch.ones_like(t_arr)
        fade_len = min(int(0.04 * self.config.sample_rate), n_samples_24k // 4)
        if fade_len > 0:
            fade_in = torch.linspace(0.0, 1.0, fade_len, device=self.device)
            fade_out = torch.linspace(1.0, 0.0, fade_len, device=self.device)
            envelope[:fade_len] = fade_in
            envelope[-fade_len:] = fade_out

        raw_audio_24k = (harmonic_sig * envelope).unsqueeze(0).cpu().float()

        # Resample to 16 kHz
        n_samples_16k = int(total_duration * self.config.export_sample_rate)
        raw_audio_16k = F.interpolate(
            raw_audio_24k.unsqueeze(0), size=n_samples_16k, mode="linear", align_corners=False
        ).squeeze(0)

        # Normalize peak amplitude to -1.0 dB (~0.89) to prevent clipping
        peak = torch.max(torch.abs(raw_audio_24k)).item()
        if peak > 0:
            raw_audio_24k = (raw_audio_24k / peak) * 0.89
            raw_audio_16k = (raw_audio_16k / peak) * 0.89

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        return StyleTTS2Output(
            audio_24k=raw_audio_24k,
            audio_16k=raw_audio_16k,
            sample_rate_24k=self.config.sample_rate,
            sample_rate_16k=self.config.export_sample_rate,
            phonemes=ipa_str,
            token_ids=token_ids,
            emotion=emotion,
            duration_sec=total_duration,
            latency_ms=latency_ms,
        )
