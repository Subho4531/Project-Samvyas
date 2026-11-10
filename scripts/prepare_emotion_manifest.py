"""Build a portable emotion prompt/target manifest from SUBESCO, neutral Bengali, and Jeet.

The output bundle is self-contained: every clip is rewritten as mono 16 kHz
PCM-16 WAV under ``wavs/<source>/`` and the JSONL manifests use relative paths.
Inputs may be JSONL, CSV, parquet (including Hugging Face audio columns), or a
directory containing those files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from samvyas.data.emotion_dataset import (
    EMOTIONS,
    build_pairs,
    find_unknown_characters,
    normalize_emotion,
    split_records,
    write_manifest,
)

SOURCE_RATE = 16_000
SUBESCO_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}


def _clean_text(value: Any) -> str:
    """Normalize Unicode and collapse whitespace without dropping Bengali marks."""
    text = unicodedata.normalize("NFC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def _safe_id(value: Any, fallback: str) -> str:
    """Create a filename-safe, bounded clip identifier."""
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")
    if not raw:
        raw = hashlib.sha256(fallback.encode("utf-8")).hexdigest()[:16]
    return raw[:80]


def _stable_key(*parts: str) -> str:
    """Stable sort key independent of Python hash randomization."""
    return hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()


def _read_audio_ref(reference: Any, base: Path) -> tuple[np.ndarray, int]:
    """Read audio from bytes, a Hugging Face audio dict, or a path."""
    if isinstance(reference, list):
        if not reference:
            raise ValueError("Audio list is empty.")
        reference = reference[0]
    if isinstance(reference, dict):
        if reference.get("array") is not None:
            array = np.asarray(reference["array"])
            if np.issubdtype(array.dtype, np.integer):
                array = array.astype(np.float32) / np.iinfo(array.dtype).max
            else:
                array = array.astype(np.float32)
            if array.ndim == 1:
                array = array[:, None]
            return array, int(reference.get("sampling_rate") or SOURCE_RATE)
        if reference.get("bytes"):
            return sf.read(io.BytesIO(reference["bytes"]), dtype="float32", always_2d=True)
        reference = reference.get("path")
    if isinstance(reference, (bytes, bytearray, memoryview)):
        return sf.read(io.BytesIO(bytes(reference)), dtype="float32", always_2d=True)
    if isinstance(reference, str) and reference:
        path = Path(reference)
        if not path.is_absolute():
            path = base / path
        return sf.read(path, dtype="float32", always_2d=True)
    raise ValueError("Audio reference is empty or unsupported.")


def _write_clip(
    reference: Any,
    base: Path,
    output_root: Path,
    source: str,
    clip_id: str,
) -> dict[str, Any]:
    """Normalize one clip to mono 16 kHz PCM-16 and return its manifest fields."""
    audio, rate = _read_audio_ref(reference, base)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio is empty or non-finite.")
    waveform = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    if rate != SOURCE_RATE:
        divisor = math.gcd(SOURCE_RATE, rate)
        waveform = resample_poly(waveform, SOURCE_RATE // divisor, rate // divisor)
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    relative = Path("wavs") / source / f"{clip_id}.wav"
    destination = output_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".wav.tmp")
    sf.write(temporary, waveform, SOURCE_RATE, subtype="PCM_16", format="WAV")
    temporary.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {
        "audio_path": relative.as_posix(),
        "duration": waveform.size / SOURCE_RATE,
        "sha256": digest,
    }


def _row_text(row: dict[str, Any]) -> str:
    """Pick the transcript field used by the supported corpora."""
    for key in ("text", "transcription", "sentence", "normalized_text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return _clean_text(value)
    return ""


def _row_speaker(row: dict[str, Any]) -> str:
    """Pick a stable speaker identifier."""
    for key in ("speaker_id", "speaker_name", "speaker"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def _row_sentence(row: dict[str, Any], speaker: str, index: int, text: str) -> str:
    """Use corpus sentence IDs or a stable transcript hash for holdout grouping."""
    for key in ("sentence_id", "sentence_no", "sentence"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    if text:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{speaker}_{index}"


def _row_emotion(row: dict[str, Any], default: str) -> str | None:
    """Map SUBESCO integer labels and string emotions to the four classes."""
    value = row.get("emotion")
    if value is None:
        label = row.get("label")
        if isinstance(label, (int, np.integer)):
            value = SUBESCO_LABELS.get(int(label))
        else:
            value = label
    if isinstance(value, (int, np.integer)) or (isinstance(value, str) and value.strip().isdigit()):
        value = SUBESCO_LABELS.get(int(value))
    if value is None or str(value).strip() == "":
        value = default
    try:
        return normalize_emotion(str(value))
    except ValueError:
        return None


def _row_reference(row: dict[str, Any]) -> Any:
    """Return the audio reference from common column names."""
    for key in ("audio", "audio_path", "path", "file"):
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _clip_from_row(
    row: dict[str, Any],
    *,
    source: str,
    index: int,
    base: Path,
    output_root: Path,
    default_emotion: str,
    only_neutral: bool,
    vocab: set[str],
    strict_vocab: bool,
    min_duration: float,
    max_duration: float,
) -> dict[str, Any]:
    """Normalize one input row into a clip record, writing its WAV."""
    emotion = _row_emotion(row, default_emotion)
    if emotion is None:
        raise ValueError("unsupported emotion")
    if only_neutral and emotion != "neutral":
        raise ValueError("non-neutral row in neutral source")
    text = _row_text(row)
    if not text:
        raise ValueError("empty transcript")
    reference = _row_reference(row)
    if reference is None:
        raise ValueError("missing audio reference")
    speaker = _row_speaker(row)
    raw_id = row.get("clip_id") or row.get("file name") or row.get("file_name") or f"{index}"
    raw_id = re.sub(r"\.(wav|flac|mp3|m4a)$", "", str(raw_id), flags=re.IGNORECASE)
    clip_id = _safe_id(f"{source}_{raw_id}", f"{source}:{index}:{speaker}")
    if vocab:
        unknown = find_unknown_characters([text], vocab)
        if unknown:
            raise ValueError(f"vocabulary misses {unknown}")
    materialized = _write_clip(reference, base, output_root, source, clip_id)
    duration = materialized["duration"]
    if not min_duration <= duration <= max_duration:
        (output_root / materialized["audio_path"]).unlink(missing_ok=True)
        raise ValueError(f"duration {duration:.2f}s outside {min_duration}-{max_duration}s")
    row_out: dict[str, Any] = {
        "clip_id": clip_id,
        "audio_path": materialized["audio_path"],
        "text": text,
        "emotion": emotion,
        "speaker_id": f"{source}:{speaker}",
        "sentence_id": _row_sentence(row, speaker, index, text),
        "source": source,
        "duration": duration,
        "sha256": materialized["sha256"],
    }
    if row.get("split") in ("train", "validation"):
        row_out["split"] = row["split"]
    return row_out


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield object rows from UTF-8 JSONL."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object in {path}.")
            yield value


def _iter_csv(path: Path) -> Iterable[dict[str, Any]]:
    """Yield rows from a UTF-8 CSV file."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _iter_parquet(path: Path) -> Iterable[dict[str, Any]]:
    """Stream parquet rows so large audio shards do not fill RAM."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=32):
        yield from batch.to_pylist()


def _iter_path(path: Path) -> Iterable[dict[str, Any]]:
    """Dispatch on a supported single-file suffix."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl(path)
    elif suffix == ".csv":
        yield from _iter_csv(path)
    elif suffix == ".parquet":
        yield from _iter_parquet(path)
    else:
        raise ValueError(f"Unsupported input file: {path}")


def _source_files(path: Path) -> list[Path]:
    """Find supported files in a directory, preferring manifests over shards."""
    if path.is_file():
        return [path]
    jsonl = sorted(path.rglob("*.jsonl"))
    if jsonl:
        return jsonl
    parquet = sorted(path.rglob("*.parquet"))
    if parquet:
        return parquet
    csv_files = sorted(path.rglob("*.csv"))
    if csv_files:
        return csv_files
    raise FileNotFoundError(f"No JSONL, CSV, or parquet files under {path}.")


def _load_source(
    path: Path,
    source: str,
    output_root: Path,
    *,
    default_emotion: str,
    only_neutral: bool,
    vocab: set[str],
    strict_vocab: bool,
    min_duration: float,
    max_duration: float,
) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, str]]]:
    """Normalize every row from one source and count skipped reasons."""
    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    examples: list[dict[str, str]] = []
    index = 0
    for file in _source_files(path):
        for row in _iter_path(file):
            if "split" not in row and file.stem in ("train", "validation"):
                row = dict(row)
                row["split"] = file.stem
            index += 1
            try:
                record = _clip_from_row(
                    row,
                    source=source,
                    index=index,
                    base=file.parent,
                    output_root=output_root,
                    default_emotion=default_emotion,
                    only_neutral=only_neutral,
                    vocab=vocab,
                    strict_vocab=strict_vocab,
                    min_duration=min_duration,
                    max_duration=max_duration,
                )
            except (ValueError, RuntimeError, OSError) as exc:
                if strict_vocab and "vocabulary misses" in str(exc):
                    raise
                reason = str(exc).split(":")[0][:80] or "invalid row"
                skipped[reason] += 1
                if len(examples) < 100:
                    examples.append({"source": source, "reason": reason, "row": str(index)})
                continue
            records.append(record)
    return records, skipped, examples


def _load_jeet(
    path: Path,
    output_root: Path,
    *,
    vocab: set[str],
    strict_vocab: bool,
    min_duration: float,
    max_duration: float,
) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, str]]]:
    """Load the existing Jeet v2 split and treat every take as neutral."""
    root = path / "dataset" if (path / "dataset").is_dir() else path
    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    examples: list[dict[str, str]] = []
    for split in ("train", "validation"):
        manifest = root / f"{split}.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        for index, row in enumerate(_iter_jsonl(manifest)):
            row = dict(row)
            row["emotion"] = "neutral"
            row["speaker_id"] = "jeet"
            row["sentence_id"] = row.get("clip_id", f"jeet_{index}")
            row["split"] = split
            reference = row.get("audio_path")
            if reference is not None:
                row["audio_path"] = str((root / reference).resolve())
            try:
                record = _clip_from_row(
                    row,
                    source="jeet",
                    index=index,
                    base=root,
                    output_root=output_root,
                    default_emotion="neutral",
                    only_neutral=True,
                    vocab=vocab,
                    strict_vocab=strict_vocab,
                    min_duration=min_duration,
                    max_duration=max_duration,
                )
            except (ValueError, RuntimeError, OSError) as exc:
                if strict_vocab and "vocabulary misses" in str(exc):
                    raise
                reason = str(exc).split(":")[0][:80] or "invalid row"
                skipped[reason] += 1
                if len(examples) < 100:
                    examples.append({"source": "jeet", "reason": reason, "row": str(index)})
                continue
            record["split"] = split
            records.append(record)
    return records, skipped, examples


def _limit_neutral_hours(
    records: list[dict[str, Any]],
    *,
    limit_hours: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Round-robin across speakers until the neutral budget is reached."""
    total = sum(float(record["duration"]) for record in records) / 3600
    if limit_hours <= 0 or total <= limit_hours:
        return records, {"original_hours": total, "selected_hours": total, "limited": False}
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_speaker[str(record["speaker_id"])].append(record)
    for speaker, queue in by_speaker.items():
        queue.sort(key=lambda row: _stable_key(str(seed), speaker, row["clip_id"]))
    selected: list[dict[str, Any]] = []
    seconds = 0.0
    speakers = sorted(by_speaker)
    while True:
        progressed = False
        for speaker in speakers:
            queue = by_speaker[speaker]
            if not queue:
                continue
            progressed = True
            record = queue.pop(0)
            selected.append(record)
            seconds += float(record["duration"])
            if seconds >= limit_hours * 3600:
                break
        if not progressed or seconds >= limit_hours * 3600:
            break
    return selected, {
        "original_hours": total,
        "selected_hours": seconds / 3600,
        "limited": True,
        "speakers": len({record["speaker_id"] for record in selected}),
    }


def _write_clips(path: Path, records: list[dict[str, Any]]) -> None:
    """Write clip provenance without prompt fields."""
    write_manifest(path, records)


def _filter_pair_durations(
    pairs: list[dict[str, Any]], max_total_seconds: float
) -> tuple[list[dict[str, Any]], int]:
    """Drop pairs whose prompt+target exceed the trainer frame budget."""
    kept: list[dict[str, Any]] = []
    dropped = 0
    for pair in pairs:
        target = float(pair.get("duration") or 0.0)
        prompt = float(pair.get("prompt_duration") or 0.0)
        if target + prompt <= max_total_seconds:
            kept.append(pair)
        else:
            dropped += 1
    return kept, dropped


def _replay_pairs(
    pairs: list[dict[str, Any]], source: str, copies: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Repeat a small source with unique target IDs to cap its sampling share."""
    if copies < 1:
        raise ValueError("Replay copies must be positive.")
    replayed: list[dict[str, Any]] = []
    extra = 0
    original = 0
    for pair in pairs:
        replayed.append(pair)
        if pair.get("source") != source:
            continue
        original += 1
        for copy in range(1, copies):
            clone = dict(pair)
            clone["clip_id"] = f"{pair['clip_id']}_r{copy:02d}"
            replayed.append(clone)
            extra += 1
    return replayed, {
        "copies": copies,
        "original": original,
        "extra": extra,
        "total": original * copies,
    }


def main() -> None:
    """CLI entry point for building a Kaggle-ready emotion dataset bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subesco", type=Path, default=None)
    parser.add_argument("--neutral", type=Path, default=None)
    parser.add_argument("--jeet-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, default=None)
    parser.add_argument("--holdout-speakers", type=int, default=2)
    parser.add_argument("--holdout-sentences", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neutral-prompt-ratio", type=float, default=0.8)
    parser.add_argument("--limit-neutral-hours", type=float, default=5.0)
    parser.add_argument("--jeet-replay", type=int, default=8)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--max-total-duration", type=float, default=19.0)
    parser.add_argument("--strict-vocab", action="store_true")
    args = parser.parse_args()
    if not any((args.subesco, args.neutral, args.jeet_root)):
        raise ValueError("At least one of --subesco, --neutral, or --jeet-root is required.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            "Choose a new or empty output directory; existing bundles are preserved."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocab: set[str] = set()
    if args.vocab is not None:
        vocab = {
            line.rstrip("\n")
            for line in args.vocab.read_text(encoding="utf-8").splitlines()
            if line
        }
    records: list[dict[str, Any]] = []
    skipped: dict[str, Counter[str]] = {}
    examples: list[dict[str, str]] = []
    for source, path, default, neutral_only in (
        ("subesco", args.subesco, "neutral", False),
        ("neutral", args.neutral, "neutral", True),
        ("jeet", args.jeet_root, "neutral", True),
    ):
        if path is None:
            continue
        if source == "jeet":
            loaded, source_skipped, source_examples = _load_jeet(
                path,
                args.output_dir,
                vocab=vocab,
                strict_vocab=args.strict_vocab,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
            )
        else:
            loaded, source_skipped, source_examples = _load_source(
                path,
                source,
                args.output_dir,
                default_emotion=default,
                only_neutral=neutral_only,
                vocab=vocab,
                strict_vocab=args.strict_vocab,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
            )
        records.extend(loaded)
        skipped[source] = source_skipped
        examples.extend(source_examples)
    if not records:
        raise ValueError("No usable clips were found.")
    clip_ids = [str(record["clip_id"]) for record in records]
    id_counts = Counter(clip_ids)
    duplicates = sorted(clip_id for clip_id, count in id_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate clip_id values would overwrite audio: {duplicates[:5]}")
    neutral, neutral_report = _limit_neutral_hours(
        [record for record in records if record["source"] == "neutral"],
        limit_hours=args.limit_neutral_hours,
        seed=args.seed,
    )
    if neutral:
        selected_ids = {record["clip_id"] for record in neutral}
        dropped_neutral = [
            record
            for record in records
            if record["source"] == "neutral" and record["clip_id"] not in selected_ids
        ]
        for record in dropped_neutral:
            (args.output_dir / record["audio_path"]).unlink(missing_ok=True)
        neutral_report["removed_clips"] = len(dropped_neutral)
        records = [
            record
            for record in records
            if record["source"] != "neutral" or record["clip_id"] in selected_ids
        ]
    splits, split_report = split_records(
        records,
        holdout_speakers={
            "subesco": args.holdout_speakers,
            "neutral": args.holdout_speakers,
            "jeet": 0,
        },
        holdout_sentences={
            "subesco": args.holdout_sentences,
            "neutral": args.holdout_sentences,
            "jeet": 0,
        },
        seed=args.seed,
    )
    all_records = splits["train"] + splits["validation"]
    train_pairs, train_pairs_report, train_skipped = build_pairs(
        splits["train"],
        prompt_pool=splits["train"],
        seed=args.seed,
        neutral_prompt_ratio=args.neutral_prompt_ratio,
    )
    valid_pairs, valid_pairs_report, valid_skipped = build_pairs(
        splits["validation"],
        prompt_pool=all_records,
        seed=args.seed,
        neutral_prompt_ratio=args.neutral_prompt_ratio,
    )
    train_pairs, dropped_train = _filter_pair_durations(train_pairs, args.max_total_duration)
    valid_pairs, dropped_valid = _filter_pair_durations(valid_pairs, args.max_total_duration)
    train_pairs, replay_report = _replay_pairs(train_pairs, "jeet", args.jeet_replay)
    if not train_pairs or not valid_pairs:
        raise ValueError("Pair building produced an empty split; check neutral prompts/speakers.")
    write_manifest(args.output_dir / "train.jsonl", train_pairs)
    write_manifest(args.output_dir / "validation.jsonl", valid_pairs)
    split_ids = {record["clip_id"]: "train" for record in splits["train"]}
    split_ids.update({record["clip_id"]: "validation" for record in splits["validation"]})
    _write_clips(
        args.output_dir / "clips.jsonl",
        [{**record, "split": split_ids.get(record["clip_id"], "train")} for record in all_records],
    )
    counts: dict[str, Any] = {}
    for record in all_records:
        key = f"{record['source']}/{record['emotion']}"
        bucket = counts.setdefault(key, {"clips": 0, "hours": 0.0})
        bucket["clips"] += 1
        bucket["hours"] += float(record["duration"])
    report = {
        "output_dir": str(args.output_dir),
        "vocab": str(args.vocab) if args.vocab else None,
        "vocab_size": len(vocab),
        "emotions": list(EMOTIONS),
        "clips": len(all_records),
        "train_pairs": len(train_pairs) - replay_report["extra"],
        "validation_pairs": len(valid_pairs),
        "counts": counts,
        "skipped": {source: dict(counter) for source, counter in skipped.items()},
        "skipped_examples": examples[:100],
        "neutral_limit": neutral_report,
        "splits": split_report,
        "train_pairs_report": asdict(train_pairs_report),
        "validation_pairs_report": asdict(valid_pairs_report),
        "train_pairs_after_replay": len(train_pairs),
        "jeet_replay": replay_report,
        "dropped_long_pairs": {"train": dropped_train, "validation": dropped_valid},
        "max_total_duration": args.max_total_duration,
        "train_pair_skips": train_skipped[:100],
        "validation_pair_skips": valid_skipped[:100],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
