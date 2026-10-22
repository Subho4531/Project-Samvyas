"""Style Bank Manager for StyleTTS 2 in Samvyas.

Manages curated acoustic style vectors for emotional prosody control.
Provides storage, serialization, interpolation, and retrieval for
discrete emotion states (e.g. neutral, warm, excited, whisper, serious).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import torch

logger = logging.getLogger(__name__)


# Supported standard emotion tags
STANDARD_EMOTIONS: tuple[str, ...] = (
    "neutral",
    "warm",
    "excited",
    "whisper",
    "serious",
    "inquisitive",
)


@dataclass
class StyleBankManager:
    """Stores and retrieves emotion style vectors for speech synthesis conditioning."""

    DEFAULT_STYLE_DIM: ClassVar[int] = 64
    style_dim: int = DEFAULT_STYLE_DIM
    styles: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.styles:
            self._initialize_fallback_styles()

    def _initialize_fallback_styles(self) -> None:
        """Initializes deterministic normalized fallback style vectors for dry-runs."""
        generator = torch.Generator().manual_seed(42)
        for emotion in STANDARD_EMOTIONS:
            # Deterministic, unique initial vectors per emotion
            vec = torch.randn(self.style_dim, generator=generator)
            vec = torch.nn.functional.normalize(vec, p=2, dim=0)
            self.styles[emotion] = vec

    def register_style(self, emotion: str, vector: torch.Tensor, overwrite: bool = True) -> None:
        """Registers or updates a style vector for a given emotion tag.

        Parameters
        ----------
        emotion : str
            Emotion label (e.g., 'excited', 'warm').
        vector : torch.Tensor
            Acoustic style vector of shape (style_dim,) or (1, style_dim).
        overwrite : bool
            Whether to overwrite an existing emotion vector.
        """
        vec = vector.squeeze().detach().cpu().float()
        if vec.shape[-1] != self.style_dim:
            msg = f"Vector dimension {vec.shape[-1]} does not match bank dimension {self.style_dim}"
            raise ValueError(msg)
        if emotion in self.styles and not overwrite:
            logger.warning("Emotion '%s' already exists in style bank; skipping.", emotion)
            return

        self.styles[emotion] = torch.nn.functional.normalize(vec, p=2, dim=0)
        logger.info("Registered style vector for emotion '%s'.", emotion)

    def get_style(self, emotion: str, intensity: float = 1.0) -> torch.Tensor:
        """Retrieves a style vector, scaled or fallen back to neutral.

        Parameters
        ----------
        emotion : str
            Target emotion tag.
        intensity : float
            Scale factor for deviation from neutral style (1.0 = full target emotion).

        Returns
        -------
        torch.Tensor
            Normalized acoustic style vector of shape (1, style_dim).
        """
        emotion = emotion.strip().lower()
        if emotion not in self.styles:
            logger.warning(
                "Emotion '%s' not found in Style Bank. Falling back to 'neutral'.", emotion
            )
            target = self.styles.get("neutral", next(iter(self.styles.values())))
            return target.unsqueeze(0)

        if intensity == 1.0 or "neutral" not in self.styles:
            return self.styles[emotion].unsqueeze(0)

        # Blend with neutral according to intensity
        neutral = self.styles["neutral"]
        target = self.styles[emotion]
        blended = (1.0 - intensity) * neutral + intensity * target
        normalized = torch.nn.functional.normalize(blended, p=2, dim=0)
        return normalized.unsqueeze(0)

    def interpolate(self, emotion_a: str, emotion_b: str, alpha: float) -> torch.Tensor:
        """Interpolates between two emotion styles: (1 - alpha) * a + alpha * b.

        Parameters
        ----------
        emotion_a : str
            First emotion tag.
        emotion_b : str
            Second emotion tag.
        alpha : float
            Interpolation weight in [0.0, 1.0].
        """
        vec_a = self.get_style(emotion_a).squeeze(0)
        vec_b = self.get_style(emotion_b).squeeze(0)
        blended = (1.0 - alpha) * vec_a + alpha * vec_b
        return torch.nn.functional.normalize(blended, p=2, dim=0).unsqueeze(0)

    def save(self, output_path: str | Path) -> None:
        """Serializes the style bank to a PyTorch checkpoint."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "style_dim": self.style_dim,
                "styles": self.styles,
            },
            path,
        )
        logger.info("Saved Style Bank to %s (%d styles)", path, len(self.styles))

    @classmethod
    def load(cls, checkpoint_path: str | Path) -> StyleBankManager:
        """Loads a style bank from a saved checkpoint."""
        path = Path(checkpoint_path)
        if not path.exists():
            msg = f"Style bank checkpoint not found: {path}"
            raise FileNotFoundError(msg)

        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls(style_dim=data["style_dim"], styles=data["styles"])
