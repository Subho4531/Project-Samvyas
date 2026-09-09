"""Audio I/O, format conversion, trimming, and loudness normalization."""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_audio(
    file_path: Union[str, Path],
    target_sample_rate: int = 24000,
    mono: bool = True,
    normalize: bool = True,
    target_lufs: float = -20.0,
    trim: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Loads an audio file, converts to mono, resamples, trims silence, and normalizes loudness.
    
    Returns:
        waveform: Tensor of shape (1, T) or (C, T) in range [-1.0, 1.0]
        sample_rate: int
    """
    path_str = str(file_path)
    waveform, sr = torchaudio.load(path_str)

    # Convert to mono
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sr = target_sample_rate

    # Trim leading and trailing silence
    if trim:
        waveform = trim_silence(waveform, top_db=40.0)

    # Loudness normalization
    if normalize:
        waveform = normalize_loudness(waveform, sr, target_lufs=target_lufs)

    return waveform, sr


def save_audio(
    file_path: Union[str, Path],
    waveform: torch.Tensor,
    sample_rate: int = 24000,
) -> None:
    """Saves a torch audio tensor to disk."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    
    # Detach & move to CPU
    waveform = waveform.detach().cpu()
    
    # Clip to prevent digital distortion
    waveform = torch.clamp(waveform, -1.0, 1.0)
    torchaudio.save(str(path), waveform, sample_rate=sample_rate)


def trim_silence(waveform: torch.Tensor, top_db: float = 40.0) -> torch.Tensor:
    """Trims leading and trailing silence from audio tensor using energy thresholding."""
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    
    # Compute frame energy
    frame_length = 1024
    hop_length = 256
    
    unfolded = waveform.unfold(-1, frame_length, hop_length)
    energy = unfolded.pow(2).mean(dim=-1).squeeze(0)  # (Num_frames,)
    
    energy_db = 10 * torch.log10(torch.clamp(energy, min=1e-8))
    max_db = torch.max(energy_db)
    threshold = max_db - top_db
    
    active_frames = torch.where(energy_db > threshold)[0]
    if len(active_frames) == 0:
        return waveform
    
    start_sample = active_frames[0].item() * hop_length
    end_sample = min((active_frames[-1].item() + 1) * hop_length + frame_length, waveform.shape[-1])
    
    return waveform[:, start_sample:end_sample]


def normalize_loudness(
    waveform: torch.Tensor,
    sample_rate: int,
    target_lufs: float = -20.0,
) -> torch.Tensor:
    """Simple RMS peak/energy normalization approximating target loudness."""
    rms = torch.sqrt(torch.mean(waveform**2) + 1e-8)
    # Approximate conversion from RMS to rough LUFS scaling
    target_rms = 10.0 ** (target_lufs / 20.0)
    scaling = target_rms / rms
    normalized = waveform * scaling
    
    # Peak clamp if clipping
    peak = torch.max(torch.abs(normalized))
    if peak > 0.99:
        normalized = normalized * (0.99 / peak)
        
    return normalized
