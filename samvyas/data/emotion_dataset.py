"""Prompt/target emotion manifests, datasets, batching, and leakage checks.

A pair row is a normal clip record plus a neutral reference clip from the same
speaker.  Audio is stored as portable mono 16 kHz WAV; the dataset resamples to
the 24 kHz model rate and returns mel frames shaped ``(T, 100)``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import random
import threading
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch import Tensor
from torch.utils.data import Dataset, Sampler

EMOTIONS: tuple[str, ...] = ("neutral", "happy", "sad", "angry")
EMOTION_ALIASES: dict[str, str] = {
    "neutral": "neutral",
    "calm": "neutral",
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "sad": "sad",
    "sadness": "sad",
    "angry": "angry",
    "anger": "angry",
}
MODEL_SAMPLE_RATE = 24_000
HOP_LENGTH = 256

_REQUIRED_FIELDS = (
    "clip_id",
    "audio_path",
    "text",
    "emotion",
    "speaker_id",
    "sentence_id",
    "source",
    "prompt_clip_id",
    "prompt_audio_path",
    "prompt_text",
    "prompt_emotion",
    "prompt_speaker_id",
)


def normalize_emotion(value: str) -> str:
    """Map corpus-specific labels to the four supported tag classes."""
    key = value.strip().lower().replace("-", "_")
    if key not in EMOTION_ALIASES:
        raise ValueError(f"Unsupported emotion label: {value!r}")
    return EMOTION_ALIASES[key]


def tag_id_for(emotion: str) -> int:
    """Return the model tag index; neutral/happy/sad/angry map to 1..4."""
    return EMOTIONS.index(normalize_emotion(emotion)) + 1


def find_unknown_characters(texts: Sequence[str], vocab: dict[str, int]) -> list[str]:
    """Return sorted characters absent from the pinned IndicF5 vocabulary."""
    return sorted({char for text in texts for char in text if char not in vocab})


def _sha256(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_relative(root: Path, value: str) -> Path:
    """Resolve a manifest audio path while forbidding absolute/parent escapes."""
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"Audio path must be relative: {value!r}")
    resolved = (root / relative).resolve()
    resolved.relative_to(root)
    if not resolved.is_file():
        raise ValueError(f"Missing audio: {value!r}")
    return resolved


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL manifest and require object rows."""
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"Manifest line {number} must be a JSON object.")
        rows.append(value)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def _frame_count(seconds: float) -> int:
    """Convert seconds to 24 kHz / 256-hop mel frames."""
    if seconds <= 0:
        raise ValueError("Duration must be positive.")
    return max(1, math.ceil(seconds * MODEL_SAMPLE_RATE / HOP_LENGTH))


def _load_audio_24k(path: Path) -> Tensor:
    """Read mono WAV and resample to the model rate as a (1, T) tensor."""
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {path}")
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1, keepdims=True)
    if rate != MODEL_SAMPLE_RATE:
        divisor = math.gcd(MODEL_SAMPLE_RATE, rate)
        audio = resample_poly(audio, MODEL_SAMPLE_RATE // divisor, rate // divisor, axis=0)
    waveform = torch.from_numpy(np.ascontiguousarray(audio.T))
    return waveform


def _default_mel_fn() -> Callable[[Tensor], Tensor]:
    """Build the same Vocos-compatible mel extractor used by IndicF5."""
    from f5_tts.model.modules import MelSpec

    mel = MelSpec(
        n_fft=1024,
        hop_length=HOP_LENGTH,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=MODEL_SAMPLE_RATE,
        mel_spec_type="vocos",
    )

    def extract(waveform: Tensor) -> Tensor:
        return mel(waveform)

    return extract


class EmotionPairDataset(Dataset[dict[str, Any]]):
    """Read target/prompt pairs and return model-ready mel frames.

    Args:
        root: Directory containing ``train.jsonl``/``validation.jsonl`` and audio.
        split: Manifest split name.
        mel_fn: Optional callable mapping (1, T) 24 kHz audio to (1, 100, T).
        verify_sha256: Hash every source file at construction (slow, opt-in).
        max_total_frames: Reject pairs longer than the model's 4096-frame limit.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        mel_fn: Callable[[Tensor], Tensor] | None = None,
        verify_sha256: bool = False,
        max_total_frames: int = 4096,
        mel_cache_dir: str | Path | None = None,
    ) -> None:
        if split not in ("train", "validation"):
            raise ValueError("Split must be train or validation.")
        self.root = Path(root).resolve()
        self.split = split
        self.rows = _read_jsonl(self.root / f"{split}.jsonl")
        self.mel_fn = mel_fn or _default_mel_fn()
        self.max_total_frames = max_total_frames
        self.mel_cache_dir = Path(mel_cache_dir).resolve() if mel_cache_dir else None
        self._validate(verify_sha256)
        seen: set[str] = set()
        for row in self.rows:
            if row["clip_id"] in seen:
                raise ValueError(f"Duplicate clip_id in {split}: {row['clip_id']}")
            seen.add(row["clip_id"])

    def _validate(self, verify_sha256: bool) -> None:
        """Reject malformed rows, path escapes, leakage, and self-prompts."""
        for index, row in enumerate(self.rows):
            missing = [field for field in _REQUIRED_FIELDS if field not in row]
            if missing:
                raise ValueError(f"Row {index} is missing fields: {missing}")
            for field in ("clip_id", "text", "speaker_id", "sentence_id", "source"):
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValueError(f"Row {index} has an empty {field}.")
            if not isinstance(row["prompt_text"], str) or not row["prompt_text"].strip():
                raise ValueError(f"Row {index} has an empty prompt_text.")
            row["emotion"] = normalize_emotion(row["emotion"])
            row["prompt_emotion"] = normalize_emotion(row["prompt_emotion"])
            if row["prompt_emotion"] != "neutral":
                raise ValueError("Prompt audio must be neutral.")
            if row["prompt_clip_id"] == row["clip_id"]:
                raise ValueError("A clip cannot be its own prompt.")
            if row["prompt_speaker_id"] != row["speaker_id"]:
                raise ValueError("Prompt and target must share a speaker.")
            expected_tag = tag_id_for(row["emotion"])
            if "tag_id" in row and int(row["tag_id"]) != expected_tag:
                raise ValueError("tag_id disagrees with emotion.")
            target_path = _resolve_relative(self.root, row["audio_path"])
            prompt_path = _resolve_relative(self.root, row["prompt_audio_path"])
            if row.get("duration") is not None and float(row["duration"]) <= 0:
                raise ValueError("Target duration must be positive.")
            if row.get("prompt_duration") is not None and float(row["prompt_duration"]) <= 0:
                raise ValueError("Prompt duration must be positive.")
            if verify_sha256:
                for key, path in (("sha256", target_path), ("prompt_sha256", prompt_path)):
                    if key in row and _sha256(path) != row[key]:
                        raise ValueError(f"{key} mismatch for {row['clip_id']}.")

    def __len__(self) -> int:
        """Number of target/prompt pairs."""
        return len(self.rows)

    @property
    def frame_lengths(self) -> list[int]:
        """Total prompt+target frames per item, using manifest durations when present."""
        lengths: list[int] = []
        for row in self.rows:
            if row.get("duration") is not None and row.get("prompt_duration") is not None:
                total = _frame_count(float(row["duration"])) + _frame_count(
                    float(row["prompt_duration"])
                )
            else:
                target = _resolve_relative(self.root, row["audio_path"])
                prompt = _resolve_relative(self.root, row["prompt_audio_path"])
                total = _frame_count(sf.info(target).duration) + _frame_count(
                    sf.info(prompt).duration
                )
            if total > self.max_total_frames:
                raise ValueError(
                    f"Pair {row['clip_id']} is {total} frames; limit is {self.max_total_frames}."
                )
            lengths.append(total)
        return lengths

    def _mel(self, path: Path) -> Tensor:
        """Load 24 kHz audio and return (T, 100) finite mel frames.

        When ``mel_cache_dir`` is set, cache by absolute path, size, and mtime so
        interrupted/resumed runs do not repeatedly extract the same mel.
        """
        cache: Path | None = None
        if self.mel_cache_dir is not None:
            stat = path.stat()
            key = hashlib.sha256(
                f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
            ).hexdigest()
            cache = self.mel_cache_dir / f"{key}.pt"
            if cache.is_file():
                try:
                    cached = torch.load(cache, map_location="cpu", weights_only=True)
                except (RuntimeError, EOFError, pickle.UnpicklingError, ValueError):
                    cached = None
                if isinstance(cached, Tensor) and cached.ndim == 2 and cached.shape[-1] == 100:
                    return cached
        waveform = _load_audio_24k(path)
        mel = self.mel_fn(waveform)
        if mel.ndim != 3 or mel.shape[0] != 1 or mel.shape[1] != 100:
            raise ValueError("mel_fn must return (1, 100, T).")
        mel = mel.squeeze(0).transpose(0, 1).contiguous()
        if not torch.isfinite(mel).all():
            raise ValueError(f"Non-finite mel features for {path}.")
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_name(f"{cache.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            try:
                torch.save(mel, temporary)
                temporary.replace(cache)
            finally:
                temporary.unlink(missing_ok=True)
        return mel

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return prompt/target mels, text, tag, and provenance for one pair."""
        row = self.rows[index]
        prompt_mel = self._mel(_resolve_relative(self.root, row["prompt_audio_path"]))
        target_mel = self._mel(_resolve_relative(self.root, row["audio_path"]))
        total = prompt_mel.shape[0] + target_mel.shape[0]
        if total > self.max_total_frames:
            raise ValueError(f"Pair exceeds {self.max_total_frames} frames.")
        return {
            "prompt_mel": prompt_mel,
            "target_mel": target_mel,
            "prompt_text": row["prompt_text"].strip(),
            "target_text": row["text"].strip(),
            "tag_id": tag_id_for(row["emotion"]),
            "clip_id": row["clip_id"],
            "prompt_clip_id": row["prompt_clip_id"],
            "speaker_id": row["speaker_id"],
            "emotion": row["emotion"],
            "source": row["source"],
        }


def collate_emotion_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pad and concatenate prompts/targets into ``[prompt | target]`` mels.

    Returns ``mel`` (B, T, 100), ``prompt_lengths``, ``lens``, ``tag_ids``,
    and the combined text.  The model masks the loss after ``prompt_lengths``.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    feature_dim = int(batch[0]["prompt_mel"].shape[-1])
    prompt_lengths = torch.tensor(
        [int(item["prompt_mel"].shape[0]) for item in batch], dtype=torch.long
    )
    target_lengths = torch.tensor(
        [int(item["target_mel"].shape[0]) for item in batch], dtype=torch.long
    )
    lens = prompt_lengths + target_lengths
    max_length = int(lens.max().item())
    mel = torch.zeros((len(batch), max_length, feature_dim), dtype=batch[0]["prompt_mel"].dtype)
    texts: list[str] = []
    for index, item in enumerate(batch):
        prompt = item["prompt_mel"]
        target = item["target_mel"]
        if prompt.shape[-1] != feature_dim or target.shape[-1] != feature_dim:
            raise ValueError("All batch items must share the mel feature width.")
        if prompt.shape[0] == 0 or target.shape[0] == 0:
            raise ValueError("Prompt and target must contain at least one frame.")
        mel[index, : prompt.shape[0]] = prompt
        mel[index, prompt.shape[0] : prompt.shape[0] + target.shape[0]] = target
        combined = item["prompt_text"].strip() + " " + item["target_text"].strip()
        if len(combined) > int(lens[index]):
            raise ValueError(
                f"Text length {len(combined)} exceeds mel frames {int(lens[index])} "
                f"for {item['clip_id']}; F5 would truncate the target text."
            )
        texts.append(combined)
    return {
        "mel": mel,
        "prompt_lengths": prompt_lengths,
        "lens": lens,
        "tag_ids": torch.tensor([int(item["tag_id"]) for item in batch], dtype=torch.long),
        "text": texts,
        "clip_ids": [item["clip_id"] for item in batch],
        "speaker_ids": [item["speaker_id"] for item in batch],
        "emotions": [item["emotion"] for item in batch],
        "sources": [item["source"] for item in batch],
    }


class FrameBatchSampler(Sampler[list[int]]):
    """Length-bucketed sampler respecting a per-batch frame budget.

    F5's memory cost is dominated by sequence length, not sample count.  Sorting
    by total frames and packing similar lengths keeps T4 batches near the VRAM
    target without padding every short clip to the longest clip in the shard.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        max_frames: int,
        max_batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if not lengths:
            raise ValueError("lengths cannot be empty.")
        if max_frames < max(lengths):
            raise ValueError("max_frames must fit the longest sample.")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive.")
        self.lengths = [int(value) for value in lengths]
        self.max_frames = int(max_frames)
        self.max_batch_size = int(max_batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self._batches = self._build()

    def _build(self) -> list[list[int]]:
        """Greedily pack sorted indices into frame-limited batches."""
        order = sorted(range(len(self.lengths)), key=lambda index: self.lengths[index])
        batches: list[list[int]] = []
        current: list[int] = []
        frames = 0
        for index in order:
            length = self.lengths[index]
            if current and (
                len(current) >= self.max_batch_size or frames + length > self.max_frames
            ):
                batches.append(current)
                current = []
                frames = 0
            current.append(index)
            frames += length
        if current:
            batches.append(current)
        if self.drop_last and len(batches[-1]) < self.max_batch_size:
            batches.pop()
        if not batches:
            raise ValueError("Frame budget produced no batches.")
        return batches

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle batch order deterministically for a new epoch."""
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield index batches, optionally in shuffled order."""
        batches = [list(batch) for batch in self._batches]
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        """Number of batches."""
        return len(self._batches)


@dataclass(frozen=True)
class PairReport:
    """Summary of prompt pairing for a manifest build."""

    targets: int
    pairs: int
    skipped: int
    neutral_prompts: int
    emotional_prompts: int
    same_sentence_prompts: int


def _stable_key(*parts: str) -> str:
    """Stable sort key independent of Python hash randomization."""
    return hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()


def build_pairs(
    targets: Sequence[dict[str, Any]],
    prompt_pool: Sequence[dict[str, Any]] | None = None,
    *,
    seed: int = 42,
    neutral_prompt_ratio: float = 0.8,
) -> tuple[list[dict[str, Any]], PairReport, list[dict[str, str]]]:
    """Attach a same-speaker reference prompt to each target clip.

    Neutral prompts are preferred.  For non-neutral targets, a configurable
    fraction uses another emotion from the same speaker to teach the model to
    follow the tag rather than the prompt; the default is 80% neutral.
    """
    if not 0 <= neutral_prompt_ratio <= 1:
        raise ValueError("neutral_prompt_ratio must be in [0, 1].")
    pool = list(prompt_pool if prompt_pool is not None else targets)
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        by_speaker[str(row["speaker_id"])].append(row)
    pairs: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    neutral_count = 0
    emotional_count = 0
    same_sentence = 0
    for target in targets:
        speaker = str(target["speaker_id"])
        candidates = [
            candidate
            for candidate in by_speaker.get(speaker, [])
            if candidate["clip_id"] != target["clip_id"]
        ]
        if not candidates:
            skipped.append({"clip_id": str(target["clip_id"]), "reason": "no same-speaker prompt"})
            continue
        neutral = [item for item in candidates if normalize_emotion(item["emotion"]) == "neutral"]
        rng = random.Random(f"{seed}:{target['clip_id']}")
        target_emotion = normalize_emotion(target["emotion"])
        use_neutral = target_emotion == "neutral" or rng.random() < neutral_prompt_ratio
        preferred = neutral if use_neutral and neutral else candidates
        different_sentence = [
            item for item in preferred if item["sentence_id"] != target["sentence_id"]
        ]
        choices = different_sentence or preferred
        prompt = rng.choice(choices)
        prompt_emotion = normalize_emotion(prompt["emotion"])
        if prompt_emotion == "neutral":
            neutral_count += 1
        else:
            emotional_count += 1
        if prompt["sentence_id"] == target["sentence_id"]:
            same_sentence += 1
        row = dict(target)
        row.update(
            {
                "prompt_clip_id": prompt["clip_id"],
                "prompt_audio_path": prompt["audio_path"],
                "prompt_text": prompt["text"],
                "prompt_emotion": prompt_emotion,
                "prompt_speaker_id": prompt["speaker_id"],
                "prompt_sentence_id": prompt["sentence_id"],
                "prompt_duration": prompt.get("duration"),
                "prompt_sha256": prompt.get("sha256"),
                "tag_id": tag_id_for(target_emotion),
            }
        )
        pairs.append(row)
    report = PairReport(
        targets=len(targets),
        pairs=len(pairs),
        skipped=len(skipped),
        neutral_prompts=neutral_count,
        emotional_prompts=emotional_count,
        same_sentence_prompts=same_sentence,
    )
    return pairs, report, skipped


def _per_source(value: int | dict[str, int], source: str) -> int:
    """Read a per-source holdout count."""
    if isinstance(value, dict):
        return int(value.get(source, 0))
    return int(value)


def split_records(
    records: Sequence[dict[str, Any]],
    *,
    holdout_speakers: int | dict[str, int] = 2,
    holdout_sentences: int | dict[str, int] = 1,
    seed: int = 42,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Create speaker- and sentence-disjoint train/validation splits.

    Rows with an explicit ``split`` field (for example the Jeet v2 bundle) keep
    that split.  Other rows are split per source by held-out speakers and, for
    the remaining speakers, held-out sentence IDs.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["source"])].append(row)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    report: dict[str, Any] = {"sources": {}}
    for source, rows in sorted(grouped.items()):
        fixed = [row for row in rows if row.get("split") in ("train", "validation")]
        flexible = [row for row in rows if row.get("split") not in ("train", "validation")]
        source_report: dict[str, Any] = {
            "total": len(rows),
            "fixed": len(fixed),
            "holdout_speakers": [],
            "holdout_sentences": {},
        }
        for row in fixed:
            (train if row["split"] == "train" else validation).append(row)
        if flexible:
            speakers = sorted(
                {str(row["speaker_id"]) for row in flexible},
                key=lambda speaker: _stable_key(str(seed), source, speaker),
            )
            speaker_count = min(_per_source(holdout_speakers, source), max(0, len(speakers) - 1))
            held_speakers = set(speakers[:speaker_count])
            source_report["holdout_speakers"] = sorted(held_speakers)
            train_speakers = [speaker for speaker in speakers if speaker not in held_speakers]
            held_sentences: dict[str, list[str]] = {}
            for speaker in train_speakers:
                sentence_ids = sorted(
                    {
                        str(row["sentence_id"])
                        for row in flexible
                        if str(row["speaker_id"]) == speaker
                    },
                    key=lambda sentence: _stable_key(str(seed), source, speaker, sentence),
                )
                count = min(_per_source(holdout_sentences, source), max(0, len(sentence_ids) - 1))
                held_sentences[speaker] = sentence_ids[:count]
            source_report["holdout_sentences"] = held_sentences
            for row in flexible:
                speaker = str(row["speaker_id"])
                is_validation = speaker in held_speakers or str(
                    row["sentence_id"]
                ) in held_sentences.get(speaker, [])
                (validation if is_validation else train).append(row)
        source_report["train"] = sum(1 for row in train if row["source"] == source)
        source_report["validation"] = sum(1 for row in validation if row["source"] == source)
        report["sources"][source] = source_report
    if not train or not validation:
        raise ValueError("Split produced an empty train or validation set.")
    train.sort(key=lambda row: str(row["clip_id"]))
    validation.sort(key=lambda row: str(row["clip_id"]))
    report["train"] = len(train)
    report["validation"] = len(validation)
    return {"train": train, "validation": validation}, report


def write_manifest(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write a UTF-8 JSONL manifest atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
