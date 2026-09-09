"""Speaker conditioning and zero-shot voice reference encoder."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerProjector(nn.Module):
    """Projects fixed-dimension speaker verification embeddings into the Transformer hidden dimension."""

    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 768):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, speaker_emb: torch.Tensor) -> torch.Tensor:
        """Args:
            speaker_emb: (B, embedding_dim) or (B, 1, embedding_dim)
        Returns:
            projected: (B, 1, hidden_dim)
        """
        if speaker_emb.ndim == 2:
            speaker_emb = speaker_emb.unsqueeze(1)
        # Normalize incoming speaker vector (cosine space)
        normed = F.normalize(speaker_emb, p=2, dim=-1)
        return self.net(normed)


class ReferenceAudioEncoder(nn.Module):
    """Encodes a short reference audio mel-spectrogram into variable-length acoustic prompt tokens."""

    def __init__(self, n_mels: int = 80, hidden_dim: int = 768, num_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = n_mels
        for _ in range(num_layers):
            layers.extend([
                nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
            ])
            in_dim = hidden_dim

        self.conv = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Args:
            mel_spec: (B, n_mels, T_mel)
        Returns:
            prompt_tokens: (B, T_mel, hidden_dim)
        """
        x = self.conv(mel_spec)  # (B, hidden_dim, T_mel)
        x = x.transpose(1, 2)    # (B, T_mel, hidden_dim)
        return self.proj(x)
