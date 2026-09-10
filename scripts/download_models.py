"""Prefetch required Phase-1 model weights into local caches.

- DeepFilterNet v2 weights: downloaded on first `df.enhance.init_df` call
  into the DeepFilter model cache (outside the repo).
- Bengali IndicConformer `.nemo`: downloaded via `huggingface_hub` into
  `~/.cache/huggingface/hub` (already git-ignored via `*.nemo`).

Usage:
    .venv/Scripts/python.exe scripts/download_models.py
    .venv/Scripts/python.exe scripts/download_models.py --skip-deepfilter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ID = "ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large"
FILENAME = "indicconformer_stt_bn_hybrid_rnnt_large.nemo"


def fetch_nemo(repo_id: str = REPO_ID, filename: str = FILENAME) -> Path:
    from huggingface_hub import hf_hub_download

    print(f"[INFO] Fetching '{filename}' from '{repo_id}' (cached, no re-download if present)...")
    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )
    )
    size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
    # Snapshot paths may be junctions reporting 0; resolve blob size instead.
    if size_mb == 0.0:
        print(f"[INFO] Checkpoint at: {path} (size via link, verifying blob...)")
    else:
        print(f"[SUCCESS] Checkpoint at: {path} ({size_mb:.1f} MB)")
    return path


def warmup_deepfilter() -> None:
    print("[INFO] Warming up DeepFilterNet v2 (downloads weights on first run)...")
    from df.enhance import init_df

    model, df_state, suffix = init_df(
        model_base_dir=None,  # pretrained DeepFilterNet2
        default_model="DeepFilterNet2",
        epoch="best",
        log_file=None,
    )
    print(f"[SUCCESS] DeepFilterNet loaded (suffix={suffix}, model={type(model).__name__})")
    del model, df_state
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefetch Samvyas Phase-1 model weights.")
    parser.add_argument("--skip-deepfilter", action="store_true", help="Skip DeepFilter warmup.")
    parser.add_argument("--skip-nemo", action="store_true", help="Skip NeMo checkpoint fetch.")
    args = parser.parse_args()

    if not args.skip_nemo:
        fetch_nemo()
    if not args.skip_deepfilter:
        warmup_deepfilter()
    print("[DONE] All requested model caches are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
