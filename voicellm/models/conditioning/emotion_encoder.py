"""Emotion and prosody conditioning modules."""

from typing import Optional
import torch
import torch.nn as nn


class EmotionConditioner(nn.Module):
    """Conditioning module supporting both discrete emotion tokens and continuous valence/arousal vectors."""

    def __init__(
        self,
        num_emotions: int = 16,
        emotion_dim: int = 128,
        hidden_dim: int = 768,
        use_continuous_latents: bool = True,
    ):
        super().__init__()
        self.num_emotions = num_emotions
        self.use_continuous_latents = use_continuous_latents

        # Discrete emotion table (happy, sad, whisper, excited, angry, calm, etc.)
        self.discrete_emb = nn.Embedding(num_emotions, hidden_dim)

        # Continuous valence/arousal/dominance projection
        if use_continuous_latents:
            self.continuous_proj = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.continuous_proj = None

    def forward(
        self,
        emotion_id: Optional[torch.Tensor] = None,
        continuous_vad: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Args:
            emotion_id: Optional (B,) or (B, 1) discrete IDs
            continuous_vad: Optional (B, 3) valence-arousal-dominance scores [-1.0, 1.0]
        Returns:
            conditioned_emb: (B, 1, hidden_dim)
        """
        embs = []
        if emotion_id is not None:
            if emotion_id.ndim == 1:
                emotion_id = emotion_id.unsqueeze(1)
            embs.append(self.discrete_emb(emotion_id))

        if continuous_vad is not None and self.continuous_proj is not None:
            if continuous_vad.ndim == 2:
                continuous_vad = continuous_vad.unsqueeze(1)
            embs.append(self.continuous_proj(continuous_vad))

        if not embs:
            return None

        # Combine active emotion embeddings
        return torch.sum(torch.stack(embs, dim=0), dim=0)
