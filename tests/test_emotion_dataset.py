"""CPU tests for emotion manifests, prompt pairing, holdouts, and frame batching."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from samvyas.data.emotion_dataset import (
    EmotionPairDataset,
    FrameBatchSampler,
    build_pairs,
    collate_emotion_batch,
    find_unknown_characters,
    split_records,
    tag_id_for,
    write_manifest,
)


def _wav(path: Path, seconds: float = 1.0) -> None:
    """Write a finite mono 16 kHz test clip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(seconds * 16000)
    sf.write(path, 0.1 * np.sin(np.arange(samples) * 0.02), 16000, subtype="PCM_16")


def _row(root: Path, name: str, emotion: str, speaker: str, sentence: str) -> dict:
    """Build a minimal pair row with a same-speaker neutral prompt."""
    target = f"wavs/{name}.wav"
    prompt = f"wavs/{speaker}_prompt.wav"
    _wav(root / target)
    _wav(root / prompt)
    return {
        "clip_id": name,
        "audio_path": target,
        "text": "abc def",
        "emotion": emotion,
        "speaker_id": speaker,
        "sentence_id": sentence,
        "source": "unit",
        "duration": 1.0,
        "prompt_clip_id": f"{speaker}_prompt",
        "prompt_audio_path": prompt,
        "prompt_text": "abc",
        "prompt_emotion": "neutral",
        "prompt_speaker_id": speaker,
        "prompt_sentence_id": "prompt",
        "prompt_duration": 1.0,
    }


def test_dataset_and_collator(tmp_path: Path) -> None:
    """Pairs load as (T, 100) mels and collate into a padded prompt|target batch."""
    rows = [
        _row(tmp_path, "one", "happy", "s1", "a"),
        _row(tmp_path, "two", "sad", "s1", "b"),
    ]
    write_manifest(tmp_path / "train.jsonl", rows)
    write_manifest(tmp_path / "validation.jsonl", rows)
    dataset = EmotionPairDataset(
        tmp_path,
        "train",
        mel_fn=lambda waveform: torch.ones(1, 100, max(1, waveform.shape[-1] // 256)),
    )
    assert len(dataset) == 2
    item = dataset[0]
    assert item["prompt_mel"].shape[-1] == 100
    assert item["target_mel"].shape[-1] == 100
    batch = collate_emotion_batch([dataset[0], dataset[1]])
    assert batch["mel"].shape == (2, max(batch["lens"]), 100)
    assert torch.equal(batch["lens"], batch["prompt_lengths"] + 93)  # 1s at 24k/256
    assert batch["tag_ids"].tolist() == [tag_id_for("happy"), tag_id_for("sad")]
    assert batch["text"][0].startswith("abc ")


def test_mel_cache_reuses_features(tmp_path: Path) -> None:
    """A second access loads cached mels instead of rerunning mel_fn."""
    row = _row(tmp_path, "one", "happy", "s1", "a")
    write_manifest(tmp_path / "train.jsonl", [row])
    calls = 0

    def fake_mel(waveform: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.ones(1, 100, max(1, waveform.shape[-1] // 256))

    dataset = EmotionPairDataset(
        tmp_path, "train", mel_fn=fake_mel, mel_cache_dir=tmp_path / "cache"
    )
    dataset[0]
    first = calls
    dataset[0]
    assert calls == first
    assert list((tmp_path / "cache").glob("*.pt"))


def test_manifest_rejects_self_prompt_and_wrong_speaker(tmp_path: Path) -> None:
    """Prompt leakage and cross-speaker references fail closed."""
    row = _row(tmp_path, "one", "happy", "s1", "a")
    row["prompt_clip_id"] = "one"
    write_manifest(tmp_path / "train.jsonl", [row])
    with pytest.raises(ValueError, match="own prompt"):
        EmotionPairDataset(tmp_path, "train", mel_fn=lambda w: torch.zeros(1, 100, 2))
    row["prompt_clip_id"] = "p"
    row["prompt_speaker_id"] = "s2"
    write_manifest(tmp_path / "train.jsonl", [row])
    with pytest.raises(ValueError, match="share a speaker"):
        EmotionPairDataset(tmp_path, "train", mel_fn=lambda w: torch.zeros(1, 100, 2))


def test_frame_batch_sampler_covers_and_respects_budget() -> None:
    """Every index is yielded once and no batch exceeds the frame budget."""
    sampler = FrameBatchSampler(
        [100, 200, 300, 400], max_frames=500, max_batch_size=2, shuffle=False
    )
    batches = list(sampler)
    assert sorted(index for batch in batches for index in batch) == [0, 1, 2, 3]
    assert all(sum([100, 200, 300, 400][index] for index in batch) <= 500 for batch in batches)
    assert all(len(batch) <= 2 for batch in batches)


def test_frame_batch_sampler_shards_without_overlap() -> None:
    """Accelerate's two-process shard sees each packed batch exactly once."""
    pytest.importorskip("accelerate")
    from accelerate.data_loader import BatchSamplerShard

    sampler = FrameBatchSampler(
        [100, 200, 300, 400, 500], max_frames=500, max_batch_size=2, shuffle=False
    )
    expected = [index for batch in sampler for index in batch]
    shards = [
        [
            index
            for batch in BatchSamplerShard(
                sampler, num_processes=2, process_index=rank, even_batches=True
            )
            for index in batch
        ]
        for rank in range(2)
    ]
    assert sorted(shards[0] + shards[1]) == sorted(expected)
    assert set(shards[0]).isdisjoint(shards[1])


def test_build_pairs_uses_same_speaker_and_prefers_neutral() -> None:
    """Pairing never self-references and records the tag for each target."""
    records = [
        {
            "clip_id": "n1",
            "audio_path": "wavs/n1.wav",
            "text": "n",
            "emotion": "neutral",
            "speaker_id": "s1",
            "sentence_id": "a",
            "duration": 1.0,
        },
        {
            "clip_id": "h1",
            "audio_path": "wavs/h1.wav",
            "text": "h",
            "emotion": "happy",
            "speaker_id": "s1",
            "sentence_id": "b",
            "duration": 1.0,
        },
        {
            "clip_id": "h2",
            "audio_path": "wavs/h2.wav",
            "text": "h2",
            "emotion": "happy",
            "speaker_id": "s2",
            "sentence_id": "c",
            "duration": 1.0,
        },
    ]
    pairs, report, skipped = build_pairs(records, seed=4, neutral_prompt_ratio=1.0)
    assert report.pairs == 2
    assert all(row["prompt_speaker_id"] == row["speaker_id"] for row in pairs)
    assert all(row["prompt_clip_id"] != row["clip_id"] for row in pairs)
    assert next(row for row in pairs if row["clip_id"] == "h1")["prompt_clip_id"] == "n1"
    # s2 has no other clip, so its only candidate is itself; it must be skipped.
    assert any(row["clip_id"] == "h2" for row in skipped)


def test_split_records_holds_out_speakers_and_sentences() -> None:
    """Validation contains held-out speakers and held-out sentences only."""
    records = []
    for speaker in ("s1", "s2", "s3"):
        for sentence in ("a", "b", "c"):
            records.append(
                {
                    "clip_id": f"{speaker}_{sentence}",
                    "source": "unit",
                    "speaker_id": speaker,
                    "sentence_id": sentence,
                    "emotion": "neutral",
                    "text": "abc",
                    "duration": 1.0,
                }
            )
    splits, report = split_records(records, holdout_speakers=1, holdout_sentences=1, seed=5)
    train_ids = {row["clip_id"] for row in splits["train"]}
    valid_ids = {row["clip_id"] for row in splits["validation"]}
    assert train_ids.isdisjoint(valid_ids)
    assert report["sources"]["unit"]["holdout_speakers"]
    held_speaker = report["sources"]["unit"]["holdout_speakers"][0]
    assert all(row["speaker_id"] != held_speaker for row in splits["train"])


def test_vocabulary_check_and_tag_mapping() -> None:
    """Unknown characters are reported and tag indices match the model order."""
    assert find_unknown_characters(["abc"], {"a": 0, "b": 1, "c": 2}) == []
    assert find_unknown_characters(["abz"], {"a": 0, "b": 1}) == ["z"]
    assert tag_id_for("neutral") == 1
    assert tag_id_for("angry") == 4


def test_manifest_rejects_absolute_paths(tmp_path: Path) -> None:
    """Portable manifests must not point outside the bundle."""
    row = _row(tmp_path, "one", "happy", "s1", "a")
    row["audio_path"] = str((tmp_path / row["audio_path"]).resolve())
    write_manifest(tmp_path / "train.jsonl", [row])
    with pytest.raises(ValueError, match="relative"):
        EmotionPairDataset(tmp_path, "train", mel_fn=lambda w: torch.zeros(1, 100, 2))


def test_jsonl_round_trip(tmp_path: Path) -> None:
    """Manifests preserve Bengali text and prompt fields."""
    row = _row(tmp_path, "one", "happy", "s1", "a")
    row["text"] = "বাংলা"
    path = tmp_path / "train.jsonl"
    write_manifest(path, [row])
    loaded = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert loaded["text"] == "বাংলা"
    assert loaded["prompt_emotion"] == "neutral"


def test_default_mel_fn_matches_f5_shape() -> None:
    """The real F5 MelSpec import and (1, 100, T) contract stay valid."""
    pytest.importorskip("torchaudio")
    pytest.importorskip("f5_tts")
    from samvyas.data.emotion_dataset import _default_mel_fn

    mel = _default_mel_fn()(torch.zeros(1, 2400))
    assert mel.ndim == 3
    assert mel.shape[0] == 1
    assert mel.shape[1] == 100
    assert mel.shape[2] > 0
