"""Paralinguistic analysis branch interface (diagram: acoustics/prosody/imperfections).

Contract (.agents/rules/stt_architecture.md):
    - acoustic_embeddings: (B, T_frames, D_acoustic) dense SSL features
    - prosody_latents: optional (B, T_frames, P) hand-crafted prosody
    - frame rate must align (or be alignable) with the semantic branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ParalinguisticEncoderOutput:
    acoustic_embeddings: torch.Tensor  # (B, T_frames, D_acoustic)
    prosody_latents: Optional[torch.Tensor] = None  # (B, T_frames, P)
    lengths: Optional[torch.Tensor] = None  # valid frame counts (B,)


def align_frames(features: torch.Tensor, target_len: int) -> torch.Tensor:
    """Temporally align (B, T, D) features to `target_len` frames.

    Uses linear interpolation when lengths differ (covers the semantic 50Hz
    vs acoustic backbone rate mismatch); identity when already aligned.
    """
    if features.dim() != 3:
        raise ValueError(f"Expected (B, T, D) features, got {tuple(features.shape)}")
    if features.shape[1] == target_len:
        return features
    x = features.transpose(1, 2).float()  # (B, D, T)
    y = torch.nn.functional.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return y.transpose(1, 2).contiguous()
