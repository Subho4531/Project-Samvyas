"""Hand-crafted acoustic/prosody features (imperfections, rhythm, intonation).

Complements the SSL acoustic embeddings with interpretable per-frame signals:
    - normalized F0 (pitch) + voicing flag  -> intonation, hesitations, pauses
    - log RMS energy                         -> stress, rhythm, breathing
    - zero-crossing rate                     -> frication, breath noise

Frame hop defaults to 320 samples (20ms @16kHz) to match the Wav2Vec2
convolutional front-end stride, so prosody frames line up 1:1 with
`IndicWav2VecEncoder` acoustic frames (residual drift fixed by `align_frames`).
"""

from __future__ import annotations

import numpy as np
import torch

HOP_LENGTH = 320  # 20ms @16kHz == wav2vec2 total stride
FRAME_LENGTH = 1024
F0_FMIN = 50.0
F0_FMAX = 500.0
PROSODY_DIM = 4  # [f0_norm, voicing, log_energy, zcr]


class ProsodicFeatureExtractor:
    """CPU-only prosody extractor (librosa). No trainable parameters."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        hop_length: int = HOP_LENGTH,
        frame_length: int = FRAME_LENGTH,
    ) -> None:
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_length = frame_length

    @staticmethod
    def _to_batch(waveform: torch.Tensor) -> torch.Tensor:
        wav = waveform.float().cpu()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        elif wav.dim() == 3:
            wav = wav.squeeze(1) if wav.shape[1] == 1 else wav.mean(dim=1)
        return wav

    def _frame_count(self, n_samples: int) -> int:
        return 1 + max(0, (n_samples - self.frame_length) // self.hop_length)

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract prosody latents of shape (B, T_frames, 4)."""
        import librosa

        wav = self._to_batch(waveform)
        feats: list[np.ndarray] = []
        for utt in wav.numpy():
            n_frames = self._frame_count(len(utt))
            # --- F0 + voicing (pyin) ---
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    utt,
                    fmin=F0_FMIN,
                    fmax=F0_FMAX,
                    sr=self.sample_rate,
                    frame_length=self.frame_length,
                    hop_length=self.hop_length,
                )
                f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
                voiced = voiced_flag.astype(np.float32)
            except Exception:
                f0 = np.zeros(n_frames, dtype=np.float32)
                voiced = np.zeros(n_frames, dtype=np.float32)
            # --- log energy (RMS) ---
            rms = librosa.feature.rms(
                y=utt, frame_length=self.frame_length, hop_length=self.hop_length
            )[0].astype(np.float32)
            log_energy = np.log1p(rms * 100.0)
            # --- zero-crossing rate ---
            zcr = librosa.feature.zero_crossing_rate(
                utt, frame_length=self.frame_length, hop_length=self.hop_length
            )[0].astype(np.float32)

            t = min(n_frames, len(f0), len(voiced), len(log_energy), len(zcr))
            f0_n = np.clip(f0[:t] / F0_FMAX, 0.0, 1.0)  # [0,1]
            frame = np.stack([f0_n, voiced[:t], log_energy[:t], zcr[:t]], axis=1)
            feats.append(frame)

        # Pad batch to common length (pauses/utterance-length differences).
        t_max = max(f.shape[0] for f in feats)
        padded = np.stack(
            [np.pad(f, ((0, t_max - f.shape[0]), (0, 0))) for f in feats], axis=0
        )
        return torch.from_numpy(padded).float()
