"""Synthetic checks for continuous takes and conservative pause segmentation."""

import io
import json
import threading
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from samvyas.audio.pause_segmentation import pause_segments
from samvyas.data.voice_dataset import VoiceDataset
from samvyas.web.voice_recorder import VoiceRecorderService, make_server


def speech(seconds: float) -> np.ndarray:
    """Deterministic non-clipping tone for boundary tests at 16 kHz."""
    return (0.2 * np.sin(np.arange(round(seconds * 16000)) * 2 * np.pi * 220 / 16000)).astype(
        np.float32
    )


def take() -> np.ndarray:
    """Two phrases, one with a short natural internal pause."""
    return np.concatenate(
        [
            np.zeros(8000),
            speech(1),
            np.zeros(6400),
            speech(1),
            np.zeros(32000),
            speech(2),
            np.zeros(8000),
        ]
    ).astype(np.float32)


def wav_bytes(audio: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, 16000, subtype="PCM_16", format="WAV")
    return buffer.getvalue()


def test_cuts_keep_short_pauses_and_padding() -> None:
    bounds = pause_segments(take())
    assert len(bounds) == 2
    assert bounds[0][0] == 4800
    assert bounds[0][1] == 49600
    assert bounds[1][0] == 75200
    assert bounds[0][1] < bounds[1][0]


@pytest.mark.parametrize("audio", [np.zeros(16000), np.array([np.nan]), np.array([1.1])])
def test_invalid_or_silent_audio(audio: np.ndarray) -> None:
    with pytest.raises(ValueError):
        pause_segments(audio)


@pytest.mark.parametrize(
    "pause,threshold", [(0, 0.008), (1.2, float("nan")), (float("inf"), 0.008)]
)
def test_invalid_settings(pause: float, threshold: float) -> None:
    with pytest.raises(ValueError):
        pause_segments(speech(1), pause, threshold)


def test_master_crops_and_explicit_acceptance(tmp_path: Path) -> None:
    service = VoiceRecorderService(tmp_path)
    service.create_or_open_user("Jeet")
    data = wav_bytes(take())
    result = service.save_continuous("jeet", "sentences", "প্রথম বাক্য।\nদ্বিতীয় বাক্য।", data)
    folder = Path(result["folder"])
    assert (folder / "original.wav").read_bytes() == data
    assert len(result["clips"]) == 2
    assert not VoiceDataset(tmp_path / "jeet").records
    for clip in result["clips"]:
        info = sf.info(folder / clip["file"])
        assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16"
        assert info.duration == pytest.approx(clip["end"] - clip["start"])
    with pytest.raises(ValueError, match="confirm"):
        service.accept_continuous_clip("jeet", result["session_id"], 0, "সম্পাদিত বাক্য।", False)
    record = service.accept_continuous_clip("jeet", result["session_id"], 0, "সম্পাদিত বাক্য।", True)
    assert record["transcript"] == "সম্পাদিত বাক্য।"
    with pytest.raises(ValueError, match="already accepted"):
        service.accept_continuous_clip("jeet", result["session_id"], 0, "সম্পাদিত বাক্য।", True)
    assert len(VoiceDataset(tmp_path / "jeet").records) == 1


def test_mismatch_does_not_guess_transcripts_and_story_stays_whole(tmp_path: Path) -> None:
    service = VoiceRecorderService(tmp_path)
    service.create_or_open_user("Jeet")
    result = service.save_continuous("jeet", "sentences", "Only one line", wav_bytes(take()))
    assert all(clip["transcript"] == "" for clip in result["clips"])
    assert "2 clips for 1 lines" in result["warning"]
    story = service.save_continuous("jeet", "story", "A whole story", wav_bytes(speech(25)))
    assert story["clips"] == [] and story["duration"] == 25
    assert story["session_id"] != result["session_id"]
    assert not VoiceDataset(tmp_path / "jeet").records


def test_scripts_and_path_safety(tmp_path: Path) -> None:
    service = VoiceRecorderService(tmp_path)
    assert len(service.scripts()["sentences"].splitlines()) == 30
    service.create_or_open_user("Jeet")
    with pytest.raises(ValueError):
        service.continuous_folder("jeet", "../outside")


def test_http_continuous_upload_playback_and_accept(tmp_path: Path) -> None:
    """Exercise the browser's new API paths against a real local HTTP server."""
    service = VoiceRecorderService(tmp_path)
    service.create_or_open_user("Jeet")
    server = make_server("127.0.0.1", 0, service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        query = urllib.parse.urlencode(
            {"user": "jeet", "mode": "sentences", "transcript": "প্রথম বাক্য।\nদ্বিতীয় বাক্য।"}
        )
        request = urllib.request.Request(
            base + "/api/continuous?" + query,
            data=wav_bytes(take()),
            headers={"Content-Type": "audio/wav"},
        )
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
            assert response.status == 201
        query = urllib.parse.urlencode(
            {"user": "jeet", "session_id": result["session_id"], "file": "clip_001.wav"}
        )
        with urllib.request.urlopen(base + "/api/continuous/audio?" + query) as response:
            audio, rate = sf.read(io.BytesIO(response.read()))
            assert rate == 16000 and audio.ndim == 1
        data = {
            "user": "jeet",
            "session_id": result["session_id"],
            "index": 0,
            "transcript": "প্রথম বাক্য।",
            "verified": True,
        }
        request = urllib.request.Request(
            base + "/api/continuous/accept",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["transcript"] == data["transcript"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
