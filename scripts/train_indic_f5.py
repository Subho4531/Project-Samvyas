"""Fine-tune final IndicF5 layers on a prepared speaker dataset (Kaggle GPU recommended)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from samvyas.models.tts.adaptation import train_speaker
from samvyas.models.tts.indic_f5 import load_indic_model


def main() -> None:
    """Load pinned base weights and write an adapted checkpoint without a vocoder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--last-blocks", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    report = train_speaker(
        load_indic_model(),
        args.data_root,
        args.output_dir,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        accumulation=args.accumulation,
        last_blocks=args.last_blocks,
        patience=args.patience,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
