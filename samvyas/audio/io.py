"""Audio I/O helpers: load any common format as 16kHz mono float32.

Canonical format across Samvyas (AGENTS.md):
    - sample rate: 16_000 Hz
    - channels: 1 (mono)
    - dtype: float32 in [-1.0, 1.0]
    - shape: (T,) for a single utterance
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

TARGET_SR = 16_000


def _resample(waveform: torch.Tensor, orig_sr: int, target_sr: int = TARGET_SR) -> torch.Tensor:
    """Resample a (C, T) waveform with torchaudio (fallback: linear interp)."""
    if orig_sr == target_sr:
        return waveform
    try:
        import torchaudio.functional as F

        return F.resample(waveform, orig_freq=orig_sr, new_freq=target_sr)
    except Exception:
        # Last-resort linear interpolation (keeps no extra deps).
        ratio = target_sr / orig_sr
        new_len = int(waveform.shape[-1] * ratio)
        wav = waveform.unsqueeze(0)  # (1, C, T)
        wav = torch.nn.functional.interpolate(wav, size=new_len, mode="linear", align_corners=False)
        return wav.squeeze(0)


def load_audio(
    path_or_bytes: Union[str, Path, bytes],
    target_sr: int = TARGET_SR,
    mono: bool = True,
) -> torch.Tensor:
    """Load audio as mono float32 tensor of shape (T,) at `target_sr`.

    Args:
        path_or_bytes: file path or raw WAV bytes.
        target_sr: resampling target (default 16000).
        mono: downmix multi-channel to mono by averaging.

    Returns:
        1-D float32 tensor in [-1.0, 1.0].
    """
    import io as _io

    import soundfile as sf

    if isinstance(path_or_bytes, bytes):
        data, sr = sf.read(_io.BytesIO(path_or_bytes), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T.copy())  # (C, T)
    else:
        path = Path(path_or_bytes)
        if not path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {path}")
        waveform, sr = None, None
        try:
            # Fast path: WAV/FLAC/OGG via libsndfile.
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            waveform = torch.from_numpy(data.T.copy())  # (C, T)
        except Exception:
            pass
        if waveform is None:
            try:
                # Compressed formats (.m4a/.mp3) via ffmpeg-backed audioread.
                import librosa

                y, sr = librosa.load(str(path), sr=target_sr, mono=False)
                waveform = torch.from_numpy(y).float()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                sr = target_sr  # librosa already resampled
            except Exception:
                pass
        if waveform is None:
            # Last resort: torchaudio (backend-dependent).
            import torchaudio

            waveform, sr = torchaudio.load(str(path))  # (C, T)
            waveform = waveform.float()

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = _resample(waveform, orig_sr=sr, target_sr=target_sr)
    waveform = waveform.squeeze(0).float()  # (T,)

    # Peak-normalize guard: keep in [-1, 1] without amplifying quiet clips.
    peak = waveform.abs().max()
    if peak > 1.0:
        waveform = waveform / peak
    return waveform


def save_audio(path: Union[str, Path], waveform: torch.Tensor, sample_rate: int = TARGET_SR) -> Path:
    """Save a (T,) or (C, T) float tensor as WAV. Returns the path."""
    import soundfile as sf

    out = Path(path)
    wav = waveform.detach().cpu()
    if wav.dim() == 1:
        data = wav.numpy()
    else:
        data = wav.transpose(0, 1).numpy() if wav.dim() == 2 else wav.numpy()
    sf.write(str(out), data, sample_rate)
    return out


def to_batch(waveform: torch.Tensor) -> torch.Tensor:
    """Normalize (T,) or (B, T) or (B, 1, T) to (B, T)."""
    wav = waveform.float()
    if wav.dim() == 1:
        return wav.unsqueeze(0)
    if wav.dim() == 3:
        if wav.shape[1] == 1:
            return wav.squeeze(1)
        # Multi-channel batch: downmix.
        return wav.mean(dim=1)
    return wav
