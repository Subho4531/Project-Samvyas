"""Prepare a portable training dataset from one accepted speaker profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from samvyas.data.tts_adaptation import prepare_dataset


def main() -> None:
    """Write train/validation WAVs and metadata without editing source samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.dataset, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
