"""Conservative pause cuts for mono 16 kHz recordings; no speech recognition."""

from __future__ import annotations

import math

import numpy as np

from samvyas.data.voice_dataset import SAMPLE_RATE, audio_quality


def pause_segments(
    waveform: np.ndarray,
    min_pause: float = 1.2,
    threshold: float = 0.008,
    padding: float = 0.2,
) -> list[tuple[int, int]]:
    """Return sample bounds, splitting only quiet gaps >= min_pause seconds.

    Inputs are finite float32 mono (T,) in [-1, 1] at 16 kHz. A 20 ms RMS
    detector supplies candidate boundaries, not word or transcript alignment.
    Leading/trailing silence is trimmed with padding; samples are unmodified.
    """
    wav = np.asarray(waveform, dtype=np.float32)
    audio_quality(wav)
    if (
        not all(math.isfinite(x) for x in (min_pause, threshold, padding))
        or min_pause <= 0
        or not 0 < threshold < 1
        or padding < 0
        or padding * 2 >= min_pause
    ):
        raise ValueError("Use a positive pause, threshold in (0, 1), and smaller padding.")
    hop = SAMPLE_RATE // 50
    frames = np.pad(wav, (0, (-wav.size) % hop)).reshape(-1, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    active = np.flatnonzero(rms >= threshold)
    if not active.size:
        raise ValueError("No speech-level audio detected; check the microphone/threshold.")
    boundaries = np.flatnonzero((np.diff(active) - 1) * hop >= min_pause * SAMPLE_RATE)
    groups = np.split(active, boundaries + 1)
    margin = round(padding * SAMPLE_RATE)
    return [
        (max(0, int(group[0]) * hop - margin), min(wav.size, (int(group[-1]) + 1) * hop + margin))
        for group in groups
    ]
