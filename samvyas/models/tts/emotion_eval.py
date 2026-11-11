"""Acoustic feature extraction and proxy emotion classification for TTS evaluation.

These helpers are deliberately dependency-light so they can be unit-tested
without loading the diffusion model.  They are proxies, not a trained emotion
classifier: human listening review remains the voice-quality gate.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression

DEFAULT_UNSEEN_BENGALI_SENTENCES: tuple[str, ...] = (
    "আজ বিকেলে আকাশে খুব সুন্দর রঙের খেলা দেখা যাচ্ছে।",
    "তুমি কি জানো কালকের ট্রেনটা ঠিক কটার সময় ছাড়বে?",
    "অনেকদিন পর পুরোনো বন্ধুর সঙ্গে দেখা হয়ে সত্যিই ভালো লাগল।",
    "একই কথা বারবার বলার পরও কেউ একটুও মন দিয়ে শুনল না।",
    "সবুজ মাঠের ওপর দিয়ে ঠান্ডা বাতাস বয়ে যাচ্ছে আর পাখিরা ডাকছে।",
)
JEET_CANONICAL_CLIP_IDS: tuple[str, ...] = (
    "76f1edc2b50a49f19901fc83baf5bf2f",
    "5bc3ac951d34427b834c670110c0fac6",
)
JEET_CANONICAL_TRANSCRIPT = (
    "আমরা বিকেলে ছাদে বসেছিলাম, দূরের গাছের মাথায় পাখিরা ফিরছিল আর চারদিকে হালকা বাতাস বইছিল।"
)


def extract_acoustic_features(
    waveform: np.ndarray, rate: int = 24000, text: str = ""
) -> dict[str, float]:
    """Extract fundamental frequency (F0), RMS energy, and speech rate.

    Args:
        waveform: 1-D float32 audio at ``rate`` Hz.
        rate: Sample rate in Hz.
        text: Target text used to compute words-per-second speech rate.

    Returns:
        Dict with ``f0_mean``, ``f0_std``, ``energy``, and ``rate`` keys.
    """
    if waveform.ndim != 1 or waveform.size == 0:
        return {"f0_mean": 0.0, "f0_std": 0.0, "energy": 0.0, "rate": 0.0}

    # F0 extraction via Yin
    try:
        f0 = librosa.yin(waveform.astype(np.float64), fmin=50, fmax=500, sr=rate)
        f0_valid = f0[np.isfinite(f0) & (f0 > 50) & (f0 < 500)]
        f0_mean = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0
        f0_std = float(np.std(f0_valid)) if len(f0_valid) > 0 else 0.0
    except (RuntimeError, ValueError, librosa.ParameterError):
        # Degenerate signal or unsupported parameters: report silence.
        f0_mean = 0.0
        f0_std = 0.0

    # RMS energy
    energy = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))

    # Speech rate (words per second)
    duration = len(waveform) / float(rate)
    words = len(text.strip().split()) if text.strip() else 0
    # If energy is effectively zero, treat as silence and report rate = 0
    if duration <= 0 or energy < 1e-10:
        speech_rate = 0.0
    else:
        speech_rate = float(words / duration)

    return {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "energy": energy,
        "rate": speech_rate,
    }


def train_ground_truth_classifier(
    audio_features: list[tuple[list[float], int]],
) -> tuple[LogisticRegression | None, dict[str, Any]]:
    """Train a simple logistic-regression proxy classifier on ground-truth audio.

    Args:
        audio_features: List of ``(feature_vector, emotion_tag_id)`` pairs.

    Returns:
        ``(classifier, info_dict)``.  The classifier is ``None`` when there is
        insufficient coverage (fewer than 8 samples or a single emotion class).
    """
    if len(audio_features) < 8:
        return None, {"trained": False, "sample_count": len(audio_features)}

    labels = {label for _, label in audio_features}
    if len(labels) < 2:
        return None, {"trained": False, "sample_count": len(audio_features)}

    X = np.array([features for features, _ in audio_features], dtype=np.float32)
    y = np.array([label for _, label in audio_features], dtype=np.int32)

    # Standardize features manually for reproducible LR fitting
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-6
    X_scaled = (X - mean) / std

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_scaled, y)
    train_acc = float(clf.score(X_scaled, y))

    return clf, {
        "trained": True,
        "sample_count": len(audio_features),
        "train_accuracy": train_acc,
        "feature_means": mean.squeeze(0).tolist(),
        "feature_stds": std.squeeze(0).tolist(),
    }


def predict_emotion(
    clf: LogisticRegression,
    features: dict[str, float],
    clf_info: dict[str, Any],
) -> int | None:
    """Predict an emotion tag id from extracted features using the proxy classifier."""
    if clf is None:
        return None
    X_sample = np.array(
        [[features["f0_mean"], features["f0_std"], features["energy"], features["rate"]]],
        dtype=np.float32,
    )
    X_scaled = (X_sample - np.array(clf_info["feature_means"])) / np.array(clf_info["feature_stds"])
    return int(clf.predict(X_scaled)[0])


def read_audio_mono(path) -> tuple[np.ndarray, int] | None:
    """Read a WAV as mono float32, returning ``(waveform, sample_rate)`` or ``None``."""
    try:
        audio, sr = sf.read(path, dtype="float32")
    except (RuntimeError, ValueError, sf.LibsndfileError):
        return None
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, int(sr)


def estimate_target_frames(
    prompt_frames: int,
    prompt_text: str,
    target_text: str,
    *,
    speed: float = 1.0,
    min_frames: int = 40,
    max_total_frames: int = 4096,
) -> int:
    """Estimate target mel frames via the F5 UTF-8 byte ratio, guarding text length."""
    if prompt_frames < 1:
        raise ValueError("prompt_frames must be positive.")
    if speed <= 0:
        raise ValueError("speed must be positive.")
    ref_bytes = max(1, len(prompt_text.strip().encode("utf-8")))
    gen_bytes = max(1, len(target_text.strip().encode("utf-8")))
    estimated = int(prompt_frames / ref_bytes * gen_bytes / speed)
    combined_len = len(f"{prompt_text.strip()} {target_text.strip()}")
    required_target = max(min_frames, combined_len + 1 - prompt_frames)
    available_budget = max(1, max_total_frames - prompt_frames)
    if required_target > available_budget:
        raise ValueError(
            f"Combined prompt+target cannot fit within max_total_frames={max_total_frames}."
        )
    return min(available_budget, max(required_target, estimated))


def _load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows without failing on blank lines."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def resolve_prompt_clip(
    *,
    data_root: Path | None = None,
    jeet_root: Path | None = None,
    prompt_audio: Path | None = None,
    prompt_text: str | None = None,
    preferred_clip_ids: Sequence[str] = JEET_CANONICAL_CLIP_IDS,
    preferred_transcript: str = JEET_CANONICAL_TRANSCRIPT,
) -> tuple[Path, str, str]:
    """Locate Jeet's canonical neutral prompt WAV, verified transcript, and clip ID."""
    if prompt_audio is not None or prompt_text is not None:
        if prompt_audio is None or not prompt_text or not prompt_text.strip():
            raise ValueError("Both --prompt-audio and --prompt-text must be provided together.")
        if not prompt_audio.is_file():
            raise FileNotFoundError(prompt_audio)
        return prompt_audio, prompt_text.strip(), prompt_audio.stem

    candidates: list[tuple[Path, dict[str, Any]]] = []
    if jeet_root is not None:
        base = jeet_root / "dataset" if (jeet_root / "dataset").is_dir() else jeet_root
        for split in ("train.jsonl", "validation.jsonl"):
            manifest = base / split
            if manifest.is_file():
                for row in _load_manifest_rows(manifest):
                    candidates.append((base, row))
    if data_root is not None:
        for name in ("clips.jsonl", "train.jsonl", "validation.jsonl"):
            manifest = data_root / name
            if manifest.is_file():
                for row in _load_manifest_rows(manifest):
                    if row.get("source") == "jeet":
                        candidates.append((data_root, row))
                break

    if not candidates:
        raise FileNotFoundError("Could not find Jeet manifests under --jeet-root or --data-root.")

    target_norm = unicodedata.normalize("NFC", preferred_transcript).strip()
    id_set = set(preferred_clip_ids) | {f"jeet_{cid}" for cid in preferred_clip_ids}

    for rank in (0, 1, 2):
        for base_dir, row in candidates:
            clip_id = str(row.get("clip_id", "")).strip()
            text = unicodedata.normalize("NFC", str(row.get("text", ""))).strip()
            rel = str(row.get("audio_path", "")).strip()
            if not text or not rel:
                continue
            wav = base_dir / rel
            if not wav.is_file():
                continue
            duration = float(row.get("duration") or 0.0)
            if rank == 0 and clip_id in id_set:
                return wav, text, clip_id
            if rank == 1 and target_norm and text == target_norm:
                return wav, text, clip_id or wav.stem
            if rank == 2 and 3.5 <= duration <= 9.0:
                return wav, text, clip_id or wav.stem

    # Final fallback: first valid row with an existing WAV
    for base_dir, row in candidates:
        rel = str(row.get("audio_path", "")).strip()
        text = str(row.get("text", "")).strip()
        if rel and text and (base_dir / rel).is_file():
            return base_dir / rel, text, str(row.get("clip_id") or (base_dir / rel).stem)
    raise FileNotFoundError("No readable Jeet prompt audio file found.")
