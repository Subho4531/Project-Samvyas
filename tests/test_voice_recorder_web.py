"""Tests for the local browser voice-recorder service."""

from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from samvyas.web.voice_recorder import VoiceRecorderService, make_server, user_slug

PROMPTS = Path("configs/tts/bengali_prompts.json")


def wav_bytes(sample_rate: int = 48_000, duration: float = 1.0) -> bytes:
    """Create a browser-like mono PCM WAV at an arbitrary input sample rate."""
    frames = int(sample_rate * duration)
    timeline = np.arange(frames, dtype=np.float32) / sample_rate
    waveform = (0.12 * np.sin(2 * np.pi * 220 * timeline)).astype(np.float32)
    output = io.BytesIO()
    sf.write(output, waveform, sample_rate, subtype="PCM_16", format="WAV")
    return output.getvalue()


def test_user_slug_rejects_empty_and_path_like_values() -> None:
    """Profile names map safely into a local data-root child directory."""
    assert user_slug("  A Test_User  ") == "a-test_user"
    with pytest.raises(ValueError):
        user_slug(".")
    with pytest.raises(ValueError):
        user_slug("a")


def test_service_creates_user_and_canonical_wav(tmp_path: Path) -> None:
    """Browser input is persisted through the normal 16 kHz dataset writer."""
    service = VoiceRecorderService(tmp_path / "voice_users", PROMPTS)
    profile = service.create_or_open_user("Subhajeet Bose")

    assert profile == {"name": "Subhajeet Bose", "slug": "subhajeet-bose"}
    record = service.save_recording(profile["slug"], "bn001", "আমি আজ বাড়িতে কাজ করছি।", wav_bytes())
    saved = tmp_path / "voice_users" / profile["slug"] / record["audio_path"]
    info = sf.info(saved)

    assert info.samplerate == 16_000
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    assert record["sample_rate"] == 16_000
    assert record["duration"] == pytest.approx(1.0, abs=1 / 16_000)
    assert service.session(profile["slug"])["accepted_prompt_ids"] == ["bn001"]
    with pytest.raises(ValueError, match="already accepted"):
        service.save_recording(profile["slug"], "bn001", "আবার বলছি।", wav_bytes())


def request_json(
    url: str, data: bytes | None = None, headers: dict[str, str] | None = None
) -> dict:
    """Issue one local API request and decode its JSON response."""
    request = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def test_http_routes_create_user_and_reject_bad_audio(tmp_path: Path) -> None:
    """The browser endpoints expose prompts and reject malformed uploads."""
    server = make_server("127.0.0.1", 0, VoiceRecorderService(tmp_path / "users", PROMPTS))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        page = urllib.request.urlopen(f"{base_url}/").read().decode("utf-8")
        assert "Samvyas Voice Studio" in page
        assert len(request_json(f"{base_url}/api/prompts")["prompts"]) == 50
        profile = request_json(
            f"{base_url}/api/users",
            json.dumps({"name": "Riya"}).encode("utf-8"),
            {"Content-Type": "application/json"},
        )
        query = urllib.parse.urlencode(
            {"user": profile["slug"], "prompt_id": "bn001", "transcript": "আমি আজ বাড়িতে কাজ করছি।"}
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            request_json(
                f"{base_url}/api/recordings?{query}", b"not a wav", {"Content-Type": "audio/wav"}
            )
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_browser_ui_prefers_audio_worklet_with_safe_capture_fallback() -> None:
    """Modern PCM capture is preferred without making unsupported browsers unusable."""
    page = Path("samvyas/web/static/voice_recorder.html").read_text(encoding="utf-8")

    assert "AudioWorkletNode" in page
    assert "samvyas-capture" in page
    assert "makeCompatibilityCaptureNode" in page
    assert "createScriptProcessor" in page
    assert "window.AudioContext || window.webkitAudioContext" in page
    assert "sampleRate: 48000" in page
    assert "this.frameCount >= 4096" in page
    assert "state.captureNode.port.postMessage('flush')" in page
    assert "sampleRate: { ideal: 48000 }" in page
