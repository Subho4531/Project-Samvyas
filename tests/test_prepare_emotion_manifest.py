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


def test_subesco_disambiguates_male_female_speaker_id_and_filters_labels(tmp_path: Path) -> None:
    """SUBESCO integer speaker_id values 1..10 collide across genders unless disambiguated."""
    source = tmp_path / "subesco_src"
    rows = []
    # Female speaker 1 (OISHI) and Male speaker 1 (BASHIR) plus Female speaker 2 (for holdout)
    speakers = [
        ("Female", 1, "OISHI"),
        ("Male", 1, "BASHIR"),
        ("Female", 2, "MITA"),
    ]
    kept_labels = {0: "angry", 3: "happy", 4: "neutral", 5: "sad"}
    dropped_labels = {1: "disgust", 2: "fear", 6: "surprise"}
    for gender, sp_id, sp_name in speakers:
        for sentence in ("S_1", "S_2"):
            for label_int, label_name in {**kept_labels, **dropped_labels}.items():
                fname = f"{gender[0]}_{sp_id:02d}_{sp_name}_{sentence}_{label_name.upper()}_1.wav"
                wav_path = source / "audio" / fname
                wav_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(wav_path, 0.1 * np.sin(np.arange(16000) * 0.02), 16000)
                rows.append(
                    {
                        "file name": fname,
                        "audio": str(wav_path),
                        "transcription": "abc def",
                        "speaker_id": sp_id,
                        "speaker_name": sp_name,
                        "speaker_gender": gender,
                        "sentence_no": sentence,
                        "label": label_int,
                    }
                )
    (source / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    output = tmp_path / "bundle"
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_emotion_manifest.py",
            "--subesco",
            str(source),
            "--output-dir",
            str(output),
            "--holdout-speakers",
            "1",
            "--holdout-sentences",
            "1",
            "--limit-neutral-hours",
            "0",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    # 3 speakers x 2 sentences x 4 kept emotions = 24 clips; 3x2x3 = 18 unsupported emotion skips
    assert report["clips"] == 24
    assert report["skipped"]["subesco"]["unsupported emotion"] == 18
    assert len(report["splits"]["sources"]["subesco"]["holdout_speakers"]) == 1
    clips = [
        json.loads(line)
        for line in (output / "clips.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    speaker_ids = {clip["speaker_id"] for clip in clips}
    assert speaker_ids == {
        "subesco:Female_1_OISHI",
        "subesco:Male_1_BASHIR",
        "subesco:Female_2_MITA",
    }
    for split in ("train.jsonl", "validation.jsonl"):
        for line in (output / split).read_text(encoding="utf-8").splitlines():
            pair = json.loads(line)
            assert pair["prompt_speaker_id"] == pair["speaker_id"]
    # With default --neutral-prompt-ratio 0.8, non-neutral prompts exist and must still load.
    assert report["train_pairs_report"]["emotional_prompts"] > 0
    for split in ("train", "validation"):
        ds = EmotionPairDataset(
            output,
            split,
            mel_fn=lambda waveform: torch.ones(1, 100, max(1, waveform.shape[-1] // 256)),
        )
        assert len(ds) > 0


def test_filter_pair_durations_drops_text_exceeding_mel_frames() -> None:
    """Pairs whose combined text exceeds estimated mel frames are dropped before training."""
    from scripts.prepare_emotion_manifest import _filter_pair_durations

    kept, dropped = _filter_pair_durations(
        [
            {
                "clip_id": "ok",
                "duration": 2.0,
                "prompt_duration": 2.0,
                "text": "abc",
                "prompt_text": "def",
            },
            {
                "clip_id": "too_long_text",
                "duration": 1.0,
                "prompt_duration": 1.0,
                "text": "x" * 200,
                "prompt_text": "y" * 200,
            },
        ],
        max_total_seconds=19.0,
    )
    assert [row["clip_id"] for row in kept] == ["ok"]
    assert dropped == 1
