"""Dataset invariants, transcripts, quality, and crash-safe publication."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from samvyas.data.voice_dataset import VoiceDataset, VoicePrompt, audio_quality, load_prompts


def tone(seconds=1):
    return (0.2 * np.sin(np.arange(int(seconds * 16000)) * 0.1)).astype(np.float32)


def test_default_prompts():
    prompts = load_prompts(Path(__file__).parents[1] / "configs/tts/bengali_prompts.json")
    assert len(prompts) == 50
    assert len({p.id for p in prompts}) == 50
    assert all(any("\u0980" <= char <= "\u09ff" for char in p.text) for p in prompts)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        {},
        [{"id": "x", "text": "", "delivery": "normal"}],
        [{"id": "x", "text": "a", "delivery": "n"}] * 2,
    ],
)
def test_invalid_prompts(tmp_path, rows):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError):
        load_prompts(path)


def test_accept_resume_and_audio_integrity(tmp_path):
    dataset = VoiceDataset(tmp_path)
    wav = tone(6)
    row = dataset.accept(VoicePrompt("one", "আমি ভালো আছি।", "স্বাভাবিক"), "আমি ভালো আছি।", wav)
    audio, rate = sf.read(tmp_path / row["audio_path"], dtype="float32")
    info = sf.info(tmp_path / row["audio_path"])
    assert rate == 16000 and info.channels == 1 and info.subtype == "PCM_16"
    assert audio.shape == wav.shape and np.max(np.abs(audio)) <= 1
    assert np.allclose(audio, wav, atol=1 / 32768)
    resumed = VoiceDataset(tmp_path)
    assert resumed.accepted_ids == {"one"}
    assert resumed.total_seconds == 6
    assert resumed.records[0]["transcript"] == "আমি ভালো আছি।"
    assert "আমি" in dataset.manifest.read_text(encoding="utf-8")
    assert resumed.orphans() == []
    original = (tmp_path / row["audio_path"]).read_bytes()
    with pytest.raises(ValueError, match="already accepted"):
        resumed.accept(VoicePrompt("one", "different", "normal"), "different", tone())
    assert (tmp_path / row["audio_path"]).read_bytes() == original


def test_manifest_failure_leaves_reported_orphan(tmp_path, monkeypatch):
    import samvyas.data.voice_dataset as module

    dataset = VoiceDataset(tmp_path)
    dataset.accept(VoicePrompt("one", "first", "normal"), "first", tone())
    before = dataset.manifest.read_bytes()

    def fail(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError):
        dataset.accept(VoicePrompt("two", "second", "normal"), "second", tone())
    assert dataset.manifest.read_bytes() == before
    resumed = VoiceDataset(tmp_path)
    assert resumed.accepted_ids == {"one"}
    assert len(resumed.orphans()) == 1


def test_resume_rejects_missing_audio(tmp_path):
    dataset = VoiceDataset(tmp_path)
    row = dataset.accept(VoicePrompt("one", "hello", "normal"), "hello", tone())
    (tmp_path / row["audio_path"]).unlink()
    with pytest.raises(ValueError, match="Missing audio"):
        VoiceDataset(tmp_path)


@pytest.mark.parametrize(
    "wav", [np.array([]), np.zeros((1, 3)), np.array([np.nan]), np.array([1.1]), np.array([np.inf])]
)
def test_invalid_audio(wav):
    with pytest.raises(ValueError):
        audio_quality(wav)


def test_quality_warnings_do_not_process_audio():
    silent = np.zeros(16000, dtype=np.float32)
    assert any("silence" in value for value in audio_quality(silent)["warnings"])
    clipped = np.ones(16000, dtype=np.float32)
    assert any("clipping" in value for value in audio_quality(clipped)["warnings"])
    assert np.all(clipped == 1)
