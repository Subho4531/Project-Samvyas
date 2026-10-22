"""Build a fresh IndicF5 dataset from accepted samples and corrected sentence takes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from samvyas.audio.pause_segmentation import pause_segments
from samvyas.data.tts_adaptation import prepare_dataset
from samvyas.data.voice_dataset import SAMPLE_RATE, VoiceDataset, VoicePrompt, audio_quality


def decode_mono_16k(source: Path, destination: Path) -> None:
    """Decode a source recording losslessly to a temporary canonical WAV."""
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(destination),
        ],
        check=True,
    )


def prepare_recordings(
    accepted_root: Path,
    recordings_root: Path,
    sentences_path: Path,
    output_root: Path,
    min_pause: float = 1.2,
) -> dict[str, Any]:
    """Create an immutable new source copy and train/validation dataset.

    Sentence transcripts are assigned in file order only after both parts
    produce exactly the same total number of cuts as nonempty script lines.
    The long story remains a preserved source recording, outside this dataset.
    """
    if output_root.exists():
        raise FileExistsError(f"Choose a new output directory: {output_root}")
    texts = [
        line.strip()
        for line in sentences_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    old = VoiceDataset(accepted_root)
    provenance: list[dict[str, Any]] = []
    files = (
        recordings_root / "continuous_sentences_bn_pt1.m4a",
        recordings_root / "continuous_sentences_bn_pt2.m4a",
    )
    with tempfile.TemporaryDirectory(prefix="samvyas-v2-") as temporary:
        durations: list[dict[str, Any]] = []
        all_segments: list[np.ndarray] = []
        for source in files:
            canonical = Path(temporary) / f"{source.stem}.wav"
            decode_mono_16k(source, canonical)
            waveform, rate = sf.read(canonical, dtype="float32")
            if rate != SAMPLE_RATE or waveform.ndim != 1 or not np.isfinite(waveform).all():
                raise ValueError(f"Invalid decoded recording: {source.name}")
            bounds = pause_segments(waveform, min_pause=min_pause)
            durations.append({"source": source.name, "duration": waveform.size / rate,
                              "cuts": len(bounds)})
            for start, end in bounds:
                all_segments.append(waveform[start:end].copy())
                provenance.append({"source": source.name, "start": start / rate,
                                   "end": end / rate})
        if len(all_segments) != len(texts):
            raise ValueError(
                f"Pause detection found {len(all_segments)} clips for {len(texts)} lines; "
                "no transcript alignment or new source dataset was accepted."
            )
        root = output_root / "source"
        target = VoiceDataset(root)
        for row in old.records:
            waveform, rate = sf.read(old.root / row["audio_path"], dtype="float32")
            if rate != SAMPLE_RATE or waveform.ndim != 1:
                raise ValueError(f"Invalid accepted source clip: {row['clip_id']}")
            target.accept(
                VoicePrompt(row["prompt_id"], row["prompt_text"], row["delivery"]),
                row["transcript"],
                waveform,
            )
        for index, (transcript, waveform) in enumerate(zip(texts, all_segments, strict=True)):
            quality = audio_quality(waveform)
            if not 1 <= waveform.size / SAMPLE_RATE <= 15 or quality["clipped_fraction"]:
                raise ValueError(f"Clip {index + 1} needs quality review before training.")
            prompt_id = f"recording_v2_{index + 1:03d}"
            record = target.accept(
                VoicePrompt(prompt_id, transcript, "Natural continuous sentence"),
                transcript,
                waveform,
            )
            provenance[index].update(
                {"clip_id": record["clip_id"], "prompt_id": prompt_id,
                 "transcript": transcript, "quality": quality}
            )

    manifest = VoiceDataset(root)
    shutil.copy2(recordings_root / "continuous_story_bn.m4a", output_root / "story_pending_alignment.m4a")
    (root / "recording_provenance.json").write_text(
        json.dumps({"pause_threshold_seconds": min_pause, "recordings": durations,
                    "new_recordings": provenance,
                    "story_status": "Preserved separately; pending sentence alignment."},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    training = prepare_dataset(root, output_root / "dataset")
    report = {
        "accepted_clips": len(manifest.records),
        "accepted_seconds": manifest.total_seconds,
        "new_sentence_clips": len(all_segments),
        "sentence_seconds": sum(segment.size / SAMPLE_RATE for segment in all_segments),
        "story_seconds_pending_alignment": 532.387542,
        "recording_parts": durations,
        "training_split": training,
        "private_sources_preserved": True,
    }
    (output_root / "preparation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    """Prepare only new, isolated adaptation-v2 artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-root", type=Path, default=ROOT / "assets/audio/voice_users/jeet")
    parser.add_argument("--recordings-root", type=Path, default=ROOT / "configs/tts/recordings")
    parser.add_argument("--sentences", type=Path, default=ROOT / "configs/tts/continuous_sentences_bn.txt")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-pause", type=float, default=1.2)
    args = parser.parse_args()
    print(json.dumps(prepare_recordings(args.accepted_root, args.recordings_root,
                                        args.sentences, args.output_root, args.min_pause),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
