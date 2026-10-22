"""Resumable single-writer voice datasets; waveforms are (T,) mono at 16 kHz."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class VoicePrompt:
    """A sentence and suggested delivery; text must match what was recorded."""

    id: str
    text: str
    delivery: str


def load_prompts(path: str | Path) -> list[VoicePrompt]:
    """Read a UTF-8 JSON array of unique {id, text, delivery} objects."""
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Prompt file must contain a nonempty JSON array.")
    prompts = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every prompt must be an object.")  # noqa: TRY004
        fields = [row.get(key) for key in ("id", "text", "delivery")]
        if any(not isinstance(value, str) or not value.strip() for value in fields):
            raise ValueError("Prompt id, text, and delivery must be nonempty strings.")
        prompt = VoicePrompt(*(value.strip() for value in fields))
        if prompt.id in seen:
            raise ValueError(f"Duplicate prompt ID: {prompt.id}")
        seen.add(prompt.id)
        prompts.append(prompt)
    return prompts


def audio_quality(waveform: np.ndarray) -> dict[str, float | list[str]]:
    """Validate float audio and measure it without normalization or denoising."""
    wav = np.asarray(waveform, dtype=np.float32)
    if wav.ndim != 1 or wav.size == 0 or not np.isfinite(wav).all():
        raise ValueError("Audio must be a nonempty finite mono (T,) array.")
    peak = float(np.max(np.abs(wav)))
    if peak > 1.0:
        raise ValueError("Audio amplitude must be in [-1, 1].")
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    clipped_fraction = float(np.mean(np.abs(wav) >= 0.999))
    warnings = []
    if rms < 0.001:
        warnings.append("Near-silence: check your microphone and distance.")
    if clipped_fraction > 0:
        warnings.append("Possible clipping: lower microphone gain and retry.")
    duration = wav.size / SAMPLE_RATE
    if duration < 5 or duration > 12:
        warnings.append("Outside the suggested 5–12 second duration.")
    return {"peak": peak, "rms": rms, "clipped_fraction": clipped_fraction, "warnings": warnings}


class VoiceDataset:
    """Store accepted takes only. Use one recorder process per dataset folder."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.wav_dir = self.root / "wavs"
        self.manifest = self.root / "manifest.jsonl"
        self.wav_dir.mkdir(parents=True, exist_ok=True)
        self.records = self._read_records()

    def _read_records(self) -> list[dict[str, Any]]:
        if not self.manifest.exists():
            return []
        records = []
        seen = set()
        for number, line in enumerate(self.manifest.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
                relative = Path(row["audio_path"])
                audio = (self.root / relative).resolve()
                audio.relative_to(self.root.resolve())
                if relative.is_absolute() or not audio.is_file():
                    raise ValueError("Missing audio or invalid relative path")
                info = sf.info(audio)
                if info.format != "WAV" or info.samplerate != SAMPLE_RATE or info.channels != 1:
                    raise ValueError("Expected mono 16 kHz WAV")
                if abs(info.duration - row["duration"]) > 1 / SAMPLE_RATE:
                    raise ValueError("Duration disagrees with WAV")
                if row["sample_rate"] != SAMPLE_RATE or not row["transcript"].strip():
                    raise ValueError("Invalid sample rate or transcript")
                if row["prompt_id"] in seen:
                    raise ValueError("Duplicate accepted prompt")
                seen.add(row["prompt_id"])
            except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as exc:
                raise ValueError(f"Invalid manifest line {number}: {exc}") from exc
            records.append(row)
        return records

    @property
    def accepted_ids(self) -> set[str]:
        """Prompt IDs already accepted."""
        return {row["prompt_id"] for row in self.records}

    @property
    def total_seconds(self) -> float:
        """Total duration of accepted takes."""
        return sum(row["duration"] for row in self.records)

    def orphans(self) -> list[Path]:
        """Report unreferenced audio/temp files; never delete them automatically."""
        referenced = {(self.root / row["audio_path"]).resolve() for row in self.records}
        return sorted(
            path
            for path in self.wav_dir.iterdir()
            if path.is_file() and path.resolve() not in referenced
        )

    def accept(self, prompt: VoicePrompt, transcript: str, waveform: np.ndarray) -> dict[str, Any]:
        """Atomically publish one confirmed take; interruption may leave orphan audio."""
        if prompt.id in self.accepted_ids:
            raise ValueError(f"Prompt {prompt.id} already accepted.")
        if not transcript.strip():
            raise ValueError("Transcript cannot be empty.")
        wav = np.asarray(waveform, dtype=np.float32)
        quality = audio_quality(wav)
        clip_id = uuid.uuid4().hex
        audio = self.wav_dir / f"{clip_id}.wav"
        temporary = self.wav_dir / f"{clip_id}.wav.tmp"
        row = {
            "clip_id": clip_id,
            "audio_path": f"wavs/{clip_id}.wav",
            "transcript": transcript.strip(),
            "prompt_id": prompt.id,
            "prompt_text": prompt.text,
            "delivery": prompt.delivery,
            "duration": wav.size / SAMPLE_RATE,
            "sample_rate": SAMPLE_RATE,
            "quality": quality,
        }
        try:
            sf.write(temporary, wav, SAMPLE_RATE, subtype="PCM_16", format="WAV")
            # Exclusive creation prevents accidental overwrites even on ID collisions.
            with audio.open("xb") as destination, temporary.open("rb") as source:
                destination.write(source.read())
                destination.flush()
                os.fsync(destination.fileno())
            staged = self.root / f"manifest.{clip_id}.tmp"
            try:
                with staged.open("x", encoding="utf-8", newline="\n") as handle:
                    for record in [*self.records, row]:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(staged, self.manifest)
            finally:
                staged.unlink(missing_ok=True)
        finally:
            temporary.unlink(missing_ok=True)
        self.records.append(row)
        return row
