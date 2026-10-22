"""CLI tool to build and calibrate the Style Bank for StyleTTS 2.

Extracts and registers acoustic style vectors for emotion conditioning
(e.g., neutral, warm, excited, whisper, serious, inquisitive) and saves
the resulting checkpoint to configs/tts/style_bank.pt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from samvyas.models.tts.style_bank import STANDARD_EMOTIONS, StyleBankManager


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="Calibrate and save Samvyas Style Bank")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/tts/style_bank.pt"),
        help="Target output path for the style bank checkpoint",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=64,
        help="Latent style vector dimension (default: 64)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print registered emotions and summary norm",
    )
    args = parser.parse_args()

    print(f"[*] Initializing Style Bank with dimension {args.dimension}...")
    bank = StyleBankManager(style_dim=args.dimension)

    for emotion in STANDARD_EMOTIONS:
        vec = bank.get_style(emotion)
        norm = torch.norm(vec).item()
        print(f"  - [{emotion.upper():<12}] shape={list(vec.shape)}, L2-norm={norm:.4f}")

    # Test interpolation
    blended = bank.interpolate("warm", "excited", alpha=0.5)
    print(
        f"  - [WARM+EXCITED 50%] shape={list(blended.shape)}, L2-norm={torch.norm(blended).item():.4f}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bank.save(args.output)
    print(f"[+] Style Bank successfully saved to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
