"""Core utility functions for VoiceLLM."""

import logging
import os
import random
import sys
from typing import Optional
import numpy as np
import torch


def setup_logger(name: str = "VoiceLLM", level: int = logging.INFO) -> logging.Logger:
    """Configures a clean, formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(preferred_device: Optional[str] = None) -> torch.device:
    """Returns available torch device (cuda, mps, cpu) with sensible fallbacks."""
    if preferred_device:
        return torch.device(preferred_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
