"""Repack the IndicConformer .nemo so NeMo >= 2 can restore it.

The checkpoint's config uses the NeMo 1.x convention `tokenizer.type:
multilingual`, which NeMo >= 2 rejects (`KeyError: 'dir'`). Its per-language
`langs` block matches the NeMo 2 aggregate-tokenizer schema, so rewriting the
type to `agg` is sufficient — weights are untouched.

  .\\.venv\\Scripts\\python.exe scripts/patch_nemo_checkpoint.py
  $env:SAMVYAS_NEMO_PATH = "$env:USERPROFILE\\.cache\\samvyas\\indicconformer_bn_agg.nemo"
"""

from __future__ import annotations

import argparse
import io
import shutil
import tarfile
import tempfile
from pathlib import Path

REPO_ID = "ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large"
FILENAME = "indicconformer_stt_bn_hybrid_rnnt_large.nemo"
DEFAULT_OUTPUT = (
    Path.home() / ".cache" / "samvyas" / "indicconformer_bn_agg.nemo"
)
OLD_LINE = "  type: multilingual"
NEW_LINE = "  type: agg"
# NeMo 1.x-only scalar keys under the inference-time decoder/joint blocks.
DROP_LINES = {"multisoftmax: true", "multilingual: true"}


def patch_config(text: str) -> str:
    """Rewrite the NeMo 1.x config idioms that NeMo >= 2 rejects at restore."""
    if text.count(OLD_LINE) != 1:
        raise ValueError(f"want exactly 1 {OLD_LINE!r} line")
    lines = text.replace(OLD_LINE, NEW_LINE).splitlines()
    kept: list[str] = []
    index = 0
    dropped = {"multisoftmax": 0, "multilingual": 0, "language_keys": 0}
    while index < len(lines):
        line = lines[index]
        if line.strip() in DROP_LINES:
            dropped[line.strip().rstrip(": true")] += 1
            index += 1
        elif line == "  language_keys:":
            dropped["language_keys"] += 1
            index += 1
            while index < len(lines) and lines[index].startswith("  - "):
                index += 1
        else:
            kept.append(line)
            index += 1
    if dropped["multisoftmax"] != 2 or dropped["multilingual"] != 1 or dropped["language_keys"] != 1:
        raise ValueError(f"unexpected drop counts: {dropped}")
    return "\n".join(kept) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--filename", default=FILENAME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download

    source = Path(
        hf_hub_download(repo_id=args.repo_id, filename=args.filename)
    )
    with tempfile.TemporaryDirectory(prefix="nemo_patch_") as work:
        stage = Path(work) / "stage"
        stage.mkdir()
        with tarfile.open(source) as original:
            original.extractall(stage)
            members = original.getmembers()
        config = stage / "model_config.yaml"
        text = config.read_text(encoding="utf-8")
        try:
            config.write_text(patch_config(text), encoding="utf-8")
        except ValueError as exc:
            print(f"Refusing to patch: {exc}")
            return 1
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        with tarfile.open(temporary, "w") as repacked:
            for member in members:
                path = stage / member.name
                if member.isdir():
                    continue
                if member.name == "./model_config.yaml":
                    data = config.read_bytes()
                    info = tarfile.TarInfo("./model_config.yaml")
                    info.size = len(data)
                    repacked.addfile(info, io.BytesIO(data))
                elif path.is_file():
                    repacked.addfile(member, path.open("rb"))
        shutil.move(str(temporary), args.output)
    print(f"Patched checkpoint: {args.output} ({args.output.stat().st_size / 1e6:.0f} MB)")
    print("Point the encoder at it: $env:SAMVYAS_NEMO_PATH = that path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
