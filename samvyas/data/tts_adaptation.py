"""Portable, deterministic speaker-adaptation data from accepted mono 16 kHz takes."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from samvyas.data.voice_dataset import VoiceDataset


def prepare_dataset(source: str | Path, output: str | Path, seed: int = 42) -> dict[str, Any]:
    """Copy verified takes into a new bundle; split identical transcripts together."""
    dataset = VoiceDataset(source)
    root = Path(output)
    if root.exists():
        raise FileExistsError("Choose a new output directory; existing bundles are preserved.")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in dataset.records:
        audio, rate = sf.read(dataset.root / row["audio_path"], dtype="float32")
        if rate != 16000 or audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError(f"Invalid source audio: {row['clip_id']}")
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if not 1 <= row["duration"] <= 15 or rms < 0.001 or np.max(np.abs(audio)) >= 0.999:
            raise ValueError(f"Review silent, clipped, or out-of-range take: {row['clip_id']}")
        key = " ".join(row["transcript"].split())
        groups.setdefault(key, []).append(row)
    if len(groups) < 10:
        raise ValueError(
            "At least ten distinct accepted transcripts are needed for this experiment."
        )
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    valid_keys = set(keys[: max(1, round(len(keys) * 0.1))])
    root.mkdir(parents=True)
    (root / "wavs").mkdir()
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for key in keys:
        split = "validation" if key in valid_keys else "train"
        for row in groups[key]:
            audio_path = f"wavs/{row['clip_id']}.wav"
            shutil.copyfile(dataset.root / row["audio_path"], root / audio_path)
            digest = hashlib.sha256((root / audio_path).read_bytes()).hexdigest()
            splits[split].append(
                {
                    "clip_id": row["clip_id"],
                    "audio_path": audio_path,
                    "text": row["transcript"],
                    "duration": row["duration"],
                    "sha256": digest,
                }
            )
    for split, rows in splits.items():
        (root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    report = {
        "seed": seed,
        "sample_rate": 16000,
        "counts": {k: len(v) for k, v in splits.items()},
        "seconds": {k: sum(r["duration"] for r in v) for k, v in splits.items()},
        "reference_clip_id": splits["train"][0]["clip_id"],
        "transcripts": "Accepted by user; independent listening review pending.",
    }
    (root / "dataset.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
