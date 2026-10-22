"""Tests for the local voice-chat service (fakes stand in for Gemini/TTS)."""

from __future__ import annotations

import json
import io
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from samvyas.web.voice_chat import STTWorker, VoiceChatService, make_server, split_sentences

REPLY = "আমি ভালো আছি।"


def fake_gemini(history: list[dict], system: str, model: str, api_key: str) -> str:
    """Echo the last user turn so trimming is observable."""
    assert api_key == "test-key"
    return f"{REPLY} ({len(history)})"


def fake_tts(text: str, turn_dir: Path) -> Path:
    """Write a short 16 kHz wav shaped like test_indic_f5 output."""
    assert text.strip(), "TTS needs non-empty text"
    turn_dir.mkdir(parents=True, exist_ok=True)
    wav = turn_dir / "baseline_01_16k.wav"
    frames = np.zeros(16_000, dtype=np.float32)
    sf.write(wav, frames, 16_000, subtype="PCM_24", format="WAV")
    return wav


def make_service(tmp_path: Path, **kwargs) -> VoiceChatService:
    """Chat service with canned LLM/TTS and an isolated session directory."""
    options = {
        "api_key": "test-key",
        "tts_python": "python",
        "reference": "ref.wav",
        "reference_text": "রেফারেন্স কথা।",
        "output_root": tmp_path,
        "gemini_fn": fake_gemini,
        "tts_fn": fake_tts,
    }
    options.update(kwargs)
    return VoiceChatService(**options)


def test_chat_turn_saves_audio_and_session_log(tmp_path: Path) -> None:
    """One turn returns a reply, a servable wav, and a session record."""
    service = make_service(tmp_path)
    result = service.chat("তুমি কেমন আছ?")
    assert result["reply"].startswith(REPLY)
    assert result["audio"] == "turn_01/baseline_01_16k.wav"
    wav = service.audio_path(result["audio"])
    assert wav.is_file()
    rows = (service.session_dir / "session.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(rows[0])["user"] == "তুমি কেমন আছ?"
    with pytest.raises(ValueError, match="Invalid audio"):
        service.audio_path("../other.wav")


def test_history_window_trims_context(tmp_path: Path) -> None:
    """Only the most recent history_limit messages reach the model."""
    service = make_service(tmp_path, history_limit=4)
    service.chat("এক")
    service.chat("দুই")
    result = service.chat("তিন")
    assert result["reply"].endswith("(4)")


def test_split_sentences_buffers_partial_and_force_splits_long() -> None:
    """Streaming TTS waits for end punctuation but never unboundedly."""
    complete, rest = split_sentences("আমি ভালো")
    assert complete == [] and rest == "আমি ভালো"
    complete, rest = split_sentences("আমি ভালো। তুমি?")
    assert complete == ["আমি ভালো।", "তুমি?"] and rest == ""
    long = "ক" * 500
    complete, rest = split_sentences(long)
    assert complete and len(rest) < len(long)


def fake_stream(history: list[dict], system: str, model: str, api_key: str):
    """Yield one reply in small deltas across a sentence boundary."""
    assert api_key == "test-key"
    yield "আমি ভালো"
    yield " আছি। তুমি"
    yield " কেমন?"


def drain(service: VoiceChatService, turn: int) -> list[dict]:
    """Collect every event for a turn; the worker always ends done/error."""
    return list(service.stream_events(turn, timeout=30.0))


def test_stream_turn_voices_each_sentence(tmp_path: Path) -> None:
    """Tokens flow first; each finished sentence gets its own wav."""
    service = make_service(tmp_path, gemini_stream_fn=fake_stream)
    turn = service.start_stream("তুমি কেমন আছ?")["turn"]
    events = drain(service, turn)
    assert "".join(e.get("token", "") for e in events) == "আমি ভালো আছি। তুমি কেমন?"
    audios = [e["audio"] for e in events if "audio" in e]
    assert audios == [
        "turn_01/sent_01/baseline_01_16k.wav",
        "turn_01/sent_02/baseline_01_16k.wav",
    ]
    done = events[-1]["done"]
    assert done["reply"].startswith(REPLY)
    for name in audios:
        assert service.audio_path(name).is_file()
    rows = (service.session_dir / "session.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(rows[0])["user"] == "তুমি কেমন আছ?"


def test_stream_empty_text_rejected_and_failure_rolls_back(tmp_path: Path) -> None:
    """Bad input never starts a turn; a dead LLM leaves no orphan user turn."""

    def boom(*args, **kwargs):
        yield "half"
        raise RuntimeError("down")

    with pytest.raises(ValueError, match="required"):
        make_service(tmp_path).start_stream("   ")
    service = make_service(tmp_path, gemini_stream_fn=boom, enable_tts=False)
    turn = service.start_stream("হ্যালো")["turn"]
    events = drain(service, turn)
    assert events[0] == {"token": "half"}
    assert "error" in events[-1]
    assert service.history == []


def test_text_only_mode_returns_no_audio(tmp_path: Path) -> None:
    """--no-tts windows still chat; playback is simply absent."""
    service = make_service(tmp_path, enable_tts=False)
    assert service.chat("হ্যালো")["audio"] is None


def test_http_stream_relays_tokens_and_sentence_audio(tmp_path: Path) -> None:
    """The SSE route replays queued events until done."""
    service = make_service(tmp_path, gemini_stream_fn=fake_stream)
    server = make_server("127.0.0.1", 0, service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        turn = request_json(
            f"{base_url}/api/chat_stream", json.dumps({"text": "বলো"}).encode("utf-8")
        )["turn"]
        raw = urllib.request.urlopen(f"{base_url}/api/stream?turn={turn}").read().decode("utf-8")
        events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ")]
        assert "".join(e.get("token", "") for e in events).startswith(REPLY)
        assert len([e for e in events if "audio" in e]) == 2
        assert events[-1]["done"]["turn"] == turn
    finally:
        server.shutdown()
        thread.join()


def request_json(url: str, data: bytes | None = None) -> dict:
    """Issue one local API request and decode its JSON response."""
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def wav_16k_bytes(duration: float = 1.0) -> bytes:
    """One 16 kHz mono PCM WAV, as the chat page capture would send."""
    frames = int(16_000 * duration)
    timeline = np.arange(frames, dtype=np.float32) / 16_000
    waveform = (0.2 * np.sin(2 * np.pi * 220 * timeline)).astype(np.float32)
    output = io.BytesIO()
    sf.write(output, waveform, 16_000, subtype="PCM_16", format="WAV")
    return output.getvalue()


def fake_stt(waveform) -> str:
    """Stand in for the worker: any real audio becomes one Bengali sentence."""
    assert len(waveform) >= 16_000
    return "আমি বাংলায় কথা বলছি।"


def test_voice_turn_runs_full_stt_llm_loop(tmp_path: Path) -> None:
    """Browser wav -> STT text -> streamed turn, mic wav kept beside replies."""
    service = make_service(tmp_path, stt_fn=fake_stt, gemini_stream_fn=fake_stream)
    result = service.voice_turn(wav_16k_bytes())
    assert result["transcript"] == "আমি বাংলায় কথা বলছি।"
    events = drain(service, result["turn"])
    assert events[-1]["done"]["reply"].startswith(REPLY)
    mic = service.audio_path(f"turn_{result['turn']:02d}/mic.wav")
    assert mic.is_file()
    rows = (service.session_dir / "session.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(rows[0])
    assert record["user"] == "আমি বাংলায় কথা বলছি।"
    assert record["mic"].endswith("mic.wav")


def test_voice_turn_rejects_bad_input_and_missing_stt(tmp_path: Path) -> None:
    """Empty/garbage audio is a 400; no worker means a clear 502 message."""
    with pytest.raises(ValueError, match="smaller than 16 MB"):
        make_service(tmp_path, stt_fn=fake_stt).voice_turn(b"")
    with pytest.raises(ValueError, match="valid WAV"):
        make_service(tmp_path, stt_fn=fake_stt).voice_turn(b"not-a-wav")
    with pytest.raises(RuntimeError, match="Server STT is off"):
        make_service(tmp_path).voice_turn(wav_16k_bytes())


def test_stt_worker_protocol_against_stub(tmp_path: Path) -> None:
    """Client speaks the stdio JSON protocol a stub worker answers."""
    stub = tmp_path / "stub_worker.py"
    stub.write_text(
        "import json, sys\n"
        "print(json.dumps({'ready': True}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    req = json.loads(line)\n"
        "    print(json.dumps({'id': req['id'], 'transcript': 'stub heard'}), flush=True)\n",
        encoding="utf-8",
    )
    worker = STTWorker(sys.executable, script=stub)
    try:
        assert worker.transcribe(tmp_path / "x.wav") == "stub heard"
    finally:
        worker.close()
    """--no-tts windows still chat; playback is simply absent."""
    service = make_service(tmp_path, enable_tts=False)
    assert service.chat("হ্যালো")["audio"] is None


def test_http_chat_reset_and_audio_routes(tmp_path: Path) -> None:
    """The window endpoints chat, serve wavs, reset context, reject bad input."""
    server = make_server("127.0.0.1", 0, make_service(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        page = urllib.request.urlopen(f"{base_url}/").read().decode("utf-8")
        assert "Samvyas Voice Chat" in page
        assert request_json(f"{base_url}/api/health")["turns"] == 0
        turn = request_json(
            f"{base_url}/api/chat", json.dumps({"text": "শুনতে পাচ্ছ?"}).encode("utf-8")
        )
        assert turn["reply"].startswith(REPLY)
        audio_url = f"{base_url}/api/audio?file={urllib.parse.quote(turn['audio'])}"
        assert urllib.request.urlopen(audio_url).read(4) == b"RIFF"
        assert request_json(f"{base_url}/api/reset", b"{}") == {"turns": 1}
        try:
            request_json(f"{base_url}/api/chat", json.dumps({"text": "  "}).encode("utf-8"))
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
        else:
            raise AssertionError("empty chat text must be rejected")
    finally:
        server.shutdown()
        thread.join()


def test_http_voice_turn_transcribes_then_streams(tmp_path: Path) -> None:
    """POSTed mic audio becomes a streamed turn with a saved mic wav."""
    service = make_service(tmp_path, stt_fn=fake_stt, gemini_stream_fn=fake_stream)
    server = make_server("127.0.0.1", 0, service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert request_json(f"{base_url}/api/health")["stt"] is True
        request = urllib.request.Request(
            f"{base_url}/api/voice", data=wav_16k_bytes(), headers={"Content-Type": "audio/wav"}
        )
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read().decode("utf-8"))
        assert result["transcript"] == "আমি বাংলায় কথা বলছি।"
        raw = urllib.request.urlopen(f"{base_url}/api/stream?turn={result['turn']}").read()
        events = [json.loads(line[6:]) for line in raw.decode("utf-8").splitlines() if line.startswith("data: ")]
        assert events[-1]["done"]["reply"].startswith(REPLY)
        mic_url = f"{base_url}/api/audio?file=" + urllib.parse.quote(
            f"turn_{result['turn']:02d}/mic.wav"
        )
        assert urllib.request.urlopen(mic_url).read(4) == b"RIFF"
    finally:
        server.shutdown()
        thread.join()
