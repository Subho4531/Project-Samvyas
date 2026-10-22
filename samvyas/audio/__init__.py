"""Audio I/O and enhancement front-end (16kHz mono)."""

from typing import Any

__all__ = ["load_audio"]


def __getattr__(name: str) -> Any:
    """Keep microphone collection usable without loading PyTorch/STT packages."""
    if name == "load_audio":
        from samvyas.audio.io import load_audio

        return load_audio
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
