"""Semantic / STT branch interface (diagram: Bengali/Indic ASR block).

Contract (.agents/rules/stt_architecture.md):
    - hidden_states strictly (B, T_frames, 768)  [Vdim = 768]
    - logits (B, T_frames, vocab_size) for CTC/RNNT
    - tokens: optional decoded token IDs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SemanticEncoderOutput:
    logits: torch.Tensor  # (B, T_frames, vocab_size) CTC log-probs
    hidden_states: torch.Tensor  # (B, T_frames, 768) semantic latents
    tokens: Optional[torch.Tensor] = None  # greedy argmax token IDs (B, T_frames)
    lengths: Optional[torch.Tensor] = None  # valid frame counts (B,)
    text: Optional[list[str]] = None  # decoded transcripts (B,)
