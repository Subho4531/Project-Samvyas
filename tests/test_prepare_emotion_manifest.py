"""End-to-end test for the manifest builder using tiny synthetic WAVs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from samvyas.data.emotion_dataset import EmotionPairDataset


def _write_source(root: Path) -> Path:
    """Create 3 speakers x 3 sentences x 4 emotions of 1-second clips."""
    source = root / "source"
    rows = []
    for speaker in ("s1", "s2", "s3"):
        for sentence in ("a", "b", "c"):
            for emotion in ("neutral", "happy", "sad", "angry"):
                name = f"{speaker}_{sentence}_{emotion}"
                path = source / "audio" / f"{name}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                samples = 16000
                sf.write(path, 0.1 * np.sin(np.arange(samples) * 0.02), 16000)
                rows.append(
                    {
                        "clip_id": name,
                        "audio_path": f"audio/{name}.wav",
                        "text": "abc",
                        "speaker_id": speaker,
                        "sentence_id": sentence,
                        "emotion": emotion,
                        "duration": 1.0,
                    }
                )
    (source / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return source


def test_prepare_manifest_end_to_end(tmp_path: Path) -> None:
    """The CLI writes portable pairs, a report, and a loadable dataset."""
    source = _write_source(tmp_path)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("a\nb\nc\n \n", encoding="utf-8")
    output = tmp_path / "bundle"
    command = [
        sys.executable,
        "scripts/prepare_emotion_manifest.py",
        "--subesco",
        str(source),
        "--output-dir",
        str(output),
        "--vocab",
        str(vocab),
        "--holdout-speakers",
        "1",
        "--holdout-sentences",
        "1",
        "--limit-neutral-hours",
        "0",
        "--neutral-prompt-ratio",
        "1.0",
    ]
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["clips"] == 36
    assert report["train_pairs"] > 0
    assert report["validation_pairs"] > 0
    assert (output / "wavs" / "subesco").is_dir()
    assert (output / "train.jsonl").is_file()
    assert (output / "validation.jsonl").is_file()
    dataset = EmotionPairDataset(
        output,
        "train",
        mel_fn=lambda waveform: torch.ones(1, 100, max(1, waveform.shape[-1] // 256)),
    )
    assert len(dataset) == report["train_pairs"]
    item = dataset[0]
    assert item["prompt_mel"].shape[-1] == 100
    assert item["tag_id"] in (1, 2, 3, 4)
    assert item["emotion"] in ("neutral", "happy", "sad", "angry")


def _write_jeet(root: Path) -> None:
    """Create a minimal Jeet-style fixed-split bundle."""
    dataset = root / "dataset"
    for split, count in (("train", 4), ("validation", 2)):
        rows = []
        for index in range(count):
            name = f"j_{split}_{index}"
            path = dataset / "wavs" / f"{name}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, 0.1 * np.sin(np.arange(16000) * 0.02), 16000)
            rows.append(
                {
                    "clip_id": name,
                    "audio_path": f"wavs/{name}.wav",
                    "text": "abc",
                    "duration": 1.0,
                }
            )
        (dataset / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )


def test_jeet_replay_duplicates_train_targets_only(tmp_path: Path) -> None:
    """Small-source replay creates unique target IDs and leaves validation alone."""
    root = tmp_path / "jeet"
    _write_jeet(root)
    output = tmp_path / "bundle"
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_emotion_manifest.py",
            "--jeet-root",
            str(root),
            "--output-dir",
            str(output),
            "--limit-neutral-hours",
            "0",
            "--holdout-speakers",
            "0",
            "--holdout-sentences",
            "0",
            "--jeet-replay",
            "2",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["clips"] == 6
    assert report["train_pairs"] == 4
    assert report["train_pairs_after_replay"] == 8
    assert report["jeet_replay"]["extra"] == 4
    train_ids = {
        json.loads(line)["clip_id"]
        for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert len(train_ids) == 8
    assert sum("_r01" in clip_id for clip_id in train_ids) == 4
    assert len((output / "validation.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    dataset = EmotionPairDataset(
        output,
        "train",
        mel_fn=lambda waveform: torch.ones(1, 100, max(1, waveform.shape[-1] // 256)),
    )
    assert len(dataset) == 8
