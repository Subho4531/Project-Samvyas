"""CLI script to bundle the repository into a Kaggle-ready notebook."""

import argparse
from pathlib import Path
from voicellm.kaggle.packager import KagglePackager


def main():
    parser = argparse.ArgumentParser(description="Package VoiceLLM for Kaggle training")
    parser.add_argument("--output", type=str, default="notebooks/kaggle_voicellm_train.ipynb", help="Path to save output notebook")
    parser.add_argument("--dataset_slug", type=str, default="user/voicellm-data", help="Kaggle dataset slug")
    args = parser.parse_args()

    packager = KagglePackager()
    out_path = Path(args.output)
    packager.generate_kaggle_notebook(out_path, dataset_slug=args.dataset_slug)
    print(f"✅ Successfully generated Kaggle notebook at: {out_path.resolve()}")


if __name__ == "__main__":
    main()
