"""Acoustic feature extraction: Mel Spectrograms, Pitch (F0), and Energy."""

from typing import Optional, Tuple
import torch
import torchaudio
import torchaudio.functional as AF


def extract_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> torch.Tensor:
    """Extracts log-mel spectrogram for reference acoustic conditioning."""
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max or (sample_rate / 2.0),
        n_mels=n_mels,
        power=2.0,
    ).to(waveform.device)

    mel_spec = mel_transform(waveform)
    log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    return log_mel_spec


def extract_f0(
    waveform: torch.Tensor,
    sample_rate: int = 24000,
    hop_length: int = 256,
    f0_min: float = 50.0,
    f0_max: float = 800.0,
) -> torch.Tensor:
    """Extracts fundamental frequency (F0 pitch contour) for prosody modeling.
    Uses torchaudio's pyin/yin estimator or fallback harmonic estimator.
    """
    if waveform.ndim == 2 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)
        
    f0 = AF.compute_kaldi_pitch(
        waveform.unsqueeze(0),
        sample_rate=sample_rate,
        frame_shift=hop_length / sample_rate * 1000.0,
    )
    # Extract pitch column (index 1 is NCCF / Pitch)
    pitch = f0[:, :, 1]
    return pitch
