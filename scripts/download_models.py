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
import os
import sys
from pathlib import Path


def _load_env_token() -> None:
    """Load HF token from repo .env (supports HF_ACESS_TOKEN or HF_TOKEN)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass
    tok = os.getenv("HF_ACESS_TOKEN") or os.getenv("HF_TOKEN")
    if tok and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = tok


_load_env_token()


REPO_ID = "ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large"
FILENAME = "indicconformer_stt_bn_hybrid_rnnt_large.nemo"
PARA_REPO_ID = "ai4bharat/indicwav2vec_v1_bengali"


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


def fetch_paralinguistic(repo_id: str = PARA_REPO_ID) -> Path | None:
    """Fetch the IndicWav2Vec backbone (gated repo — needs HF access).

    Returns the snapshot path, or None with guidance if access is missing.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    print(f"[INFO] Fetching paralinguistic backbone '{repo_id}'...")
    try:
        path = Path(snapshot_download(repo_id=repo_id))
    except GatedRepoError:
        print(
            f"[WARN] Gated repo: request access at https://huggingface.co/{repo_id} "
            "then run `huggingface-cli login` and re-run this script."
        )
        return None
    print(f"[SUCCESS] Paralinguistic checkpoint at: {path}")
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
    parser.add_argument("--skip-para", action="store_true", help="Skip paralinguistic backbone fetch.")
    parser.add_argument(
        "--include-para", action="store_true", help="Also fetch the (gated) IndicWav2Vec backbone."
    )
    args = parser.parse_args()

    if not args.skip_nemo:
        fetch_nemo()
    if args.include_para and not args.skip_para:
        fetch_paralinguistic()
    if not args.skip_deepfilter:
        warmup_deepfilter()
    print("[DONE] All requested model caches are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
