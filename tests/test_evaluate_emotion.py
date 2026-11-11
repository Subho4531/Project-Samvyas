"""CPU tests for acoustic emotion evaluation, proxy classifier, and prompt resolution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from samvyas.models.tts.emotion_eval import (
    DEFAULT_UNSEEN_BENGALI_SENTENCES,
    JEET_CANONICAL_TRANSCRIPT,
    estimate_target_frames,
    extract_acoustic_features,
    predict_emotion,
    resolve_prompt_clip,
    train_ground_truth_classifier,
)
from scripts.evaluate_emotion_lora import _aggregate_stats


def _sine_wave(freq: float = 220.0, seconds: float = 1.0, rate: int = 24000) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(rate * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_extract_features_finite_and_positive() -> None:
    wave = _sine_wave()
    features = extract_acoustic_features(wave, rate=24000, text="two words here")
    assert features["energy"] > 0.0
    assert features["rate"] == pytest.approx(3.0 / 1.0, rel=1e-3)
    assert features["f0_mean"] > 0.0


def test_extract_features_silent() -> None:
    wave = np.zeros(2400, dtype=np.float32)
    features = extract_acoustic_features(wave, rate=24000, text="a b c")
    assert features["energy"] == 0.0
    assert features["f0_mean"] == 0.0
    assert features["rate"] == 0.0


def test_extract_features_resample_16k() -> None:
    wave_24k = _sine_wave(seconds=0.5)
    wave_16k = resample_poly(wave_24k, 2, 3).astype(np.float32)
    features = extract_acoustic_features(wave_16k, rate=16000, text="hello world")
    assert features["energy"] > 0.0
    assert features["rate"] == pytest.approx(2.0 / 0.5, rel=1e-2)


def test_proxy_classifier_trains_and_predicts_separable_classes() -> None:
    samples = []
    for _ in range(5):
        samples.append(([110.0, 5.0, 0.03, 2.0], 3))  # sad-like low pitch/energy
        samples.append(([260.0, 35.0, 0.25, 4.5], 4))  # angry-like high pitch/energy
    clf, info = train_ground_truth_classifier(samples)
    assert clf is not None
    assert info["trained"] is True
    assert info["train_accuracy"] == pytest.approx(1.0)
    assert (
        predict_emotion(clf, {"f0_mean": 115.0, "f0_std": 6.0, "energy": 0.035, "rate": 2.1}, info)
        == 3
    )
    assert (
        predict_emotion(clf, {"f0_mean": 255.0, "f0_std": 32.0, "energy": 0.24, "rate": 4.3}, info)
        == 4
    )

    none_clf, small_info = train_ground_truth_classifier(samples[:4])
    assert none_clf is None
    assert small_info["trained"] is False


def test_estimate_target_frames_and_unseen_sentences() -> None:
    assert len(DEFAULT_UNSEEN_BENGALI_SENTENCES) == 5
    frames = estimate_target_frames(
        prompt_frames=737,
        prompt_text=JEET_CANONICAL_TRANSCRIPT,
        target_text=DEFAULT_UNSEEN_BENGALI_SENTENCES[0],
    )
    assert frames >= 40
    assert (
        737 + frames
        >= len(f"{JEET_CANONICAL_TRANSCRIPT} {DEFAULT_UNSEEN_BENGALI_SENTENCES[0]}") + 1
    )


def test_resolve_prompt_clip_finds_canonical_id_or_transcript(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    wavs = dataset / "wavs"
    wavs.mkdir(parents=True)
    wav_other = wavs / "other.wav"
    wav_canon = wavs / "canon.wav"
    sf.write(wav_other, _sine_wave(seconds=2.0, rate=16000), 16000)
    sf.write(wav_canon, _sine_wave(seconds=2.0, rate=16000), 16000)
    rows = [
        {"clip_id": "other", "audio_path": "wavs/other.wav", "text": "hello", "duration": 2.0},
        {
            "clip_id": "custom_uuid",
            "audio_path": "wavs/canon.wav",
            "text": JEET_CANONICAL_TRANSCRIPT,
            "duration": 7.86,
        },
    ]
    (dataset / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    (dataset / "validation.jsonl").write_text("", encoding="utf-8")
    resolved_wav, resolved_text, resolved_id = resolve_prompt_clip(jeet_root=tmp_path)
    assert resolved_wav == wav_canon
    assert resolved_text == JEET_CANONICAL_TRANSCRIPT
    assert resolved_id == "custom_uuid"


def test_aggregate_stats_groups_by_tag() -> None:
    records = [
        {
            "intended_tag": "<neutral>",
            "features": {"f0_mean": 120.0, "f0_std": 10.0, "energy": 0.05, "rate": 2.0},
        },
        {
            "intended_tag": "<neutral>",
            "features": {"f0_mean": 140.0, "f0_std": 20.0, "energy": 0.07, "rate": 3.0},
        },
    ]
    stats = _aggregate_stats(records, "intended_tag")
    assert stats["<neutral>"]["f0_mean_avg"] == pytest.approx(130.0)
    assert stats["<neutral>"]["count"] == 2
