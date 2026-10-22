"""Local-only HTTP service for end-to-end voice chat.

Browser (mic/text) -> Gemini -> IndicF5 TTS -> browser playback. The server
runs in `.venv-recorder` (stdlib only); TTS shells out to `.venv-tts` so the
venv separation rule holds. Spoken input is transcribed in the browser with
the Web Speech API (bn-IN) because the local STT pipeline needs `.venv`
rebuilt; swap `transcribe_text` when it is back.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import urllib.request
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).with_name("static")
DEFAULT_MODEL = "gemini-3.5-flash"
AUDIO_PATTERN = re.compile(r"turn_[0-9]{2}/(baseline_01_16k\.wav|sent_[0-9]{2}/baseline_01_16k\.wav|mic\.wav)")
MAX_SENTENCE_CHARS = 400  # force-split runaway sentences to bound TTS latency
MAX_VOICE_BYTES = 16 * 1024 * 1024
MAX_VOICE_SECONDS = 30.0


class STTWorker:
    """Client for one persistent scripts/stt_worker.py process (its own venv).

    The worker holds GPU weights across turns; this side only passes wav
    paths over stdio JSON lines. Restarts the worker once if it dies.
    """

    def __init__(
        self,
        python: str | Path,
        device: str = "auto",
        no_denoise: bool = False,
        timeout: float = 300.0,
        script: str | Path | None = None,
        lang: str = "bn",
    ) -> None:
        self.cmd = [
            str(python),
            str(script or (ROOT / "scripts" / "stt_worker.py")),
            "--device",
            device,
            "--lang",
            lang,
        ]
        if no_denoise:
            self.cmd.append("--no-denoise")
        self.timeout = timeout
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.request_id = 0
        self._spawn()

    def _spawn(self) -> None:
        """Start the worker and wait for its ready line (model load)."""
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except (OSError, subprocess.SubprocessError):
                pass
        try:
            self.proc = subprocess.Popen(
                self.cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise RuntimeError(f"Could not start STT worker: {exc}") from exc
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("STT worker exited during startup (see its terminal log).")
        for _ in range(49):
            try:
                hello = json.loads(line)
                break
            except json.JSONDecodeError:
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError("STT worker exited during startup (see its terminal log).")
        else:
            raise RuntimeError("STT worker spoke no ready line.")
        if not hello.get("ready"):
            raise RuntimeError(f"STT worker failed startup: {line[:200]}")

    def _read_reply(self) -> dict:
        """Read one protocol line, skipping library chatter on stdout.

        DeepFilter/NeMo install loguru sinks lazily (even onto stdout); those
        lines are noise, not protocol. Cap the skipping so real breakage still
        surfaces with the offending line attached.
        """
        last = ""
        for _ in range(50):
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("STT worker died mid-turn.")
            last = line
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise RuntimeError(f"STT worker sent no protocol JSON: {last.strip()[:200]!r}")

    def transcribe(self, wav_path: str | Path) -> str:
        """Transcribe one wav file; restarts a dead worker once."""
        with self.lock:
            for attempt in (0, 1):
                try:
                    return self._ask(wav_path)
                except (RuntimeError, OSError, ValueError):
                    if attempt == 1 or self.proc is None or self.proc.poll() is None:
                        raise
                    self._spawn()
            raise RuntimeError("STT worker gave up.")  # unreachable

    def _ask(self, wav_path: str | Path) -> str:
        self.request_id += 1
        self.proc.stdin.write(json.dumps({"id": self.request_id, "audio": str(wav_path)}) + "\n")
        self.proc.stdin.flush()
        reply = self._read_reply()
        if reply.get("error"):
            raise RuntimeError(f"STT: {reply['error']}")
        if reply.get("id") != self.request_id or not (reply.get("transcript") or "").strip():
            raise RuntimeError("STT worker sent a mismatched or empty reply.")
        return reply["transcript"].strip()

    def close(self) -> None:
        """Stop the worker process."""
        with self.lock:
            if self.proc is not None:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=10)
                except (OSError, subprocess.SubprocessError):
                    pass
                self.proc = None


class TTSWorker:
    """Client for one persistent scripts/tts_worker.py process (running in .venv-tts).

    Strictly preloads and holds IndicF5 and Vocos model weights in GPU VRAM across
    the entire session. Eliminates per-sentence cold-boot and disk-loading delays.
    """

    def __init__(
        self,
        python: str | Path,
        checkpoint: str | Path | None,
        reference: str | Path,
        reference_text: str,
        device: str = "cuda",
        steps: int = 16,
        timeout: float = 300.0,
        script: str | Path | None = None,
    ) -> None:
        self.cmd = [
            str(python),
            str(script or (ROOT / "scripts" / "tts_worker.py")),
            "--device",
            device,
            "--reference",
            str(reference),
            "--reference-text",
            reference_text,
            "--steps",
            str(steps),
        ]
        if checkpoint:
            self.cmd.extend(["--checkpoint", str(checkpoint)])
        self.timeout = timeout
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.request_id = 0
        self._spawn()

    def _spawn(self) -> None:
        """Start the TTS worker and wait for its GPU-ready signal."""
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except (OSError, subprocess.SubprocessError):
                pass
        try:
            self.proc = subprocess.Popen(
                self.cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise RuntimeError(f"Could not start TTS worker: {exc}") from exc
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("TTS worker exited during startup (see its terminal log).")
        for _ in range(49):
            try:
                hello = json.loads(line)
                break
            except json.JSONDecodeError:
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError("TTS worker exited during startup (see its terminal log).")
        else:
            raise RuntimeError("TTS worker spoke no ready line.")
        if not hello.get("ready"):
            raise RuntimeError(f"TTS worker failed startup: {hello.get('error') or line[:200]}")

    def _read_reply(self) -> dict:
        for _ in range(50):
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("TTS worker died mid-turn.")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise RuntimeError("TTS worker sent no protocol JSON.")

    def synthesize(self, text: str, output_dir: str | Path, steps: int | None = None) -> Path:
        """Synthesize one sentence on GPU; restarts a dead worker once."""
        with self.lock:
            for attempt in (0, 1):
                try:
                    return self._ask(text, output_dir, steps)
                except (RuntimeError, OSError, ValueError):
                    if attempt == 1 or self.proc is None or self.proc.poll() is None:
                        raise
                    self._spawn()
            raise RuntimeError("TTS worker gave up.")

    def _ask(self, text: str, output_dir: str | Path, steps: int | None = None) -> Path:
        self.request_id += 1
        payload: dict[str, Any] = {
            "id": self.request_id,
            "text": text,
            "output_dir": str(output_dir),
        }
        if steps is not None:
            payload["steps"] = steps
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        reply = self._read_reply()
        if reply.get("error"):
            raise RuntimeError(f"TTS worker: {reply['error']}")
        if reply.get("id") != self.request_id:
            raise RuntimeError("TTS worker sent mismatched request id.")
        wav = Path(reply["audio_16k"])
        if not wav.is_file():
            raise RuntimeError(f"TTS worker finished without audio file at {wav}")
        return wav

    def close(self) -> None:
        with self.lock:
            if self.proc is not None:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=10)
                except (OSError, subprocess.SubprocessError):
                    pass
                self.proc = None


def load_env_file(path: str | Path = ROOT / ".env") -> None:
    """Load KEY=VALUE lines without needing python-dotenv in this venv."""
    env_file = Path(path)
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, _, value = line.partition("=")
            if name.strip() not in os.environ:
                os.environ[name.strip()] = value.strip().strip("\"'")


def resolve_api_key(explicit: str | None = None) -> str:
    """API key from flag, shell env, or repo .env — never logged."""
    if explicit:
        return explicit
    load_env_file()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    if not key:
        raise ValueError("Set GEMINI_API_KEY (shell or repo .env) before chatting.")
    return key


def ask_gemini(
    history: list[dict[str, str]],
    system: str,
    model: str,
    api_key: str,
    timeout: float = 90.0,
) -> str:
    """POST one chat turn with stdlib only; raises RuntimeError downstream."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [
            {"role": msg["role"], "parts": [{"text": msg["text"]}]} for msg in history
        ],
        "generationConfig": {"maxOutputTokens": 512},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"Gemini HTTP {exc.code}: {detail}") from exc
    candidates = payload.get("candidates") or []
    parts = (candidates[0].get("content") or {}).get("parts") if candidates else None
    text = "".join(part.get("text", "") for part in parts or []).strip()
    if not text:
        raise RuntimeError("Gemini returned an empty reply.")
    return text


def split_sentences(text: str) -> tuple[list[str], str]:
    """Split complete Bengali sentences off a streaming buffer.

    Returns (complete, remainder). A trailing fragment without end
    punctuation stays buffered; overlong buffers force-split at a space so
    one TTS call never waits on an unbounded sentence.
    """
    parts = re.split(r"(?<=[।?!\n])\s+", text)
    if re.search(r"[।?!\n]\s*$", text):
        return [part for part in parts if part.strip()], ""
    *complete, rest = parts
    if len(rest) > MAX_SENTENCE_CHARS:
        cut = rest.rfind(" ", 0, MAX_SENTENCE_CHARS)
        cut = cut if cut > 0 else MAX_SENTENCE_CHARS
        return [part for part in complete if part.strip()] + [rest[:cut]], rest[cut:].lstrip()
    return [part for part in complete if part.strip()], rest


def stream_gemini(
    history: list[dict[str, str]],
    system: str,
    model: str,
    api_key: str,
    timeout: float = 300.0,
):
    """Yield reply text deltas via streamGenerateContent (server-sent events)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [
            {"role": msg["role"], "parts": [{"text": msg["text"]}]} for msg in history
        ],
        "generationConfig": {"maxOutputTokens": 512},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"Gemini HTTP {exc.code}: {detail}") from exc
    with response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            for candidate in payload.get("candidates") or []:
                for part in ((candidate.get("content") or {}).get("parts") or []):
                    if part.get("text"):
                        yield part["text"]


class VoiceChatService:
    """In-memory Bengali voice-chat session: Gemini turn + TTS wav per turn."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str,
        history_limit: int = 10,
        system: str | None = None,
        tts_python: str | Path,
        tts_device: str = "auto",
        cpu_fallback: bool = False,
        steps: int = 32,
        checkpoint: str | Path | None = None,
        reference: str | Path | None = None,
        reference_text: str | None = None,
        output_root: str | Path = ROOT / "outputs" / "tts",
        enable_tts: bool = True,
        gemini_fn: Callable[..., str] = ask_gemini,
        gemini_stream_fn: Callable[..., Any] = stream_gemini,
        stt_fn: Callable[[Any], str] | None = None,
        tts_fn: Callable[[str, Path], Path] | None = None,
        tts_worker: TTSWorker | None = None,
    ) -> None:
        if history_limit < 1:
            raise ValueError("History limit must be positive.")
        if enable_tts and (not reference or not (reference_text or "").strip()):
            raise ValueError("TTS needs a reference WAV and its verified transcript.")
        self.model = model
        self.api_key = api_key
        self.history_limit = history_limit
        self.system = system or (
            "You are a friendly voice assistant. Reply in Bengali, in short "
            "spoken style (1-3 sentences), since replies are read aloud."
        )
        self.tts_python = str(tts_python)
        self.tts_device = tts_device
        self.cpu_fallback = cpu_fallback
        self.steps = steps
        self.checkpoint = str(checkpoint) if checkpoint else None
        self.reference = str(reference) if reference else None
        self.reference_text = (reference_text or "").strip()
        self.enable_tts = enable_tts
        self.gemini_fn = gemini_fn
        self.gemini_stream_fn = gemini_stream_fn
        self.stt_fn = stt_fn
        self.tts_worker = tts_worker
        self.tts_fn = tts_fn or self._run_tts
        self.lock = threading.Lock()
        self.streams: dict[int, queue.Queue] = {}
        self.history: list[dict[str, str]] = []
        self._mic_files: dict[int, str] = {}
        self.turn = 0
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        unique = uuid.uuid4().hex[:8]
        self.session_dir = Path(output_root) / f"chat_web_{stamp}_{unique}"
        self.session_dir.mkdir(parents=True, exist_ok=False)

    def _run_tts(self, text: str, turn_dir: Path) -> Path:
        """Synthesize one reply; delegates to persistent GPU TTSWorker if present."""
        if self.tts_worker is not None:
            return self.tts_worker.synthesize(text, turn_dir, steps=self.steps)
        cmd = [
            self.tts_python,
            str(ROOT / "scripts" / "test_indic_f5.py"),
            "--text",
            text,
            "--reference",
            self.reference,
            "--reference-text",
            self.reference_text,
            "--output-dir",
            str(turn_dir),
            "--steps",
            str(self.steps),
            "--device",
            self.tts_device,
        ]
        if self.cpu_fallback:
            cmd.append("--cpu-fallback")
        if self.checkpoint:
            cmd += ["--checkpoint", self.checkpoint]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", check=False
        )
        if proc.returncode != 0:
            raise RuntimeError(f"TTS failed: {(proc.stderr or proc.stdout)[-500:]}")
        wav = turn_dir / "baseline_01_16k.wav"
        if not wav.is_file():
            raise RuntimeError("TTS finished without the expected 16 kHz file.")
        return wav

    def chat(self, user_text: str) -> dict[str, Any]:
        """Run one turn; returns reply text plus the servable audio path or None."""
        text = (user_text or "").strip()
        if not text:
            raise ValueError("Message text is required.")
        with self.lock:
            self.history.append({"role": "user", "text": text})
            try:
                reply = self.gemini_fn(
                    self.history[-self.history_limit :],
                    self.system,
                    self.model,
                    self.api_key,
                )
            except Exception:
                self.history.pop()
                raise
            self.history.append({"role": "model", "text": reply})
            self.turn += 1
            turn = self.turn
            audio: str | None = None
            if self.enable_tts:
                turn_dir = self.session_dir / f"turn_{turn:02d}"
                turn_dir.mkdir(parents=True, exist_ok=False)
                wav = self.tts_fn(reply, turn_dir)
                audio = f"turn_{turn:02d}/{wav.name}"
            with (self.session_dir / "session.jsonl").open("a", encoding="utf-8") as log:
                log.write(
                    json.dumps(
                        {"turn": turn, "user": text, "reply": reply, "audio": audio},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            return {"reply": reply, "audio": audio, "turn": turn}

    def voice_turn(self, wav_bytes: bytes) -> dict[str, Any]:
        """Full voice turn: browser wav -> our STT -> streamed LLM/TTS turn.

        Returns {"turn", "transcript"}; the browser then opens /api/stream.
        """
        if self.stt_fn is None:
            raise RuntimeError("Server STT is off; type or use browser recognition.")
        if not wav_bytes or len(wav_bytes) > MAX_VOICE_BYTES:
            raise ValueError("Voice audio must be a WAV smaller than 16 MB.")
        import io
        import math

        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly

        try:
            audio, rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
        except RuntimeError as exc:
            raise ValueError("Voice audio must be a valid WAV file.") from exc
        waveform = audio.mean(axis=1, dtype=np.float32)
        if rate != 16_000:
            if rate < 8000:
                raise ValueError("Voice audio sample rate is too low.")
            divisor = math.gcd(int(rate), 16_000)
            waveform = resample_poly(waveform, 16_000 // divisor, rate // divisor).astype(
                np.float32
            )
        duration = waveform.size / 16_000
        if not 0 < duration <= MAX_VOICE_SECONDS:
            raise ValueError(f"Voice audio must be 0-{MAX_VOICE_SECONDS:g} seconds.")
        transcript = self.stt_fn(waveform)
        if not transcript.strip():
            raise RuntimeError("STT heard no speech; speak closer or retry.")
        turn = self.start_stream(transcript)["turn"]
        with self.lock:
            mic_path = self.session_dir / f"turn_{turn:02d}" / "mic.wav"
            sf.write(mic_path, waveform, 16_000, subtype="PCM_16", format="WAV")
            self._mic_files[turn] = "mic.wav"
        return {"turn": turn, "transcript": transcript.strip()}

    def reset(self) -> dict[str, int]:
        """Clear context; saved wavs stay on disk for review."""
        with self.lock:
            self.history.clear()
            return {"turns": self.turn}

    def start_stream(self, user_text: str) -> dict[str, int]:
        """Begin a streamed turn; events flow from stream_events()."""
        text = (user_text or "").strip()
        if not text:
            raise ValueError("Message text is required.")
        with self.lock:
            self.history.append({"role": "user", "text": text})
            snapshot = self.history[-self.history_limit :]
            self.turn += 1
            turn = self.turn
            turn_dir = self.session_dir / f"turn_{turn:02d}"
            turn_dir.mkdir(parents=True, exist_ok=False)
            event_queue: queue.Queue = queue.Queue()
            self.streams[turn] = event_queue
        worker = threading.Thread(
            target=self._stream_worker, args=(turn, text, snapshot, turn_dir), daemon=True
        )
        worker.start()
        return {"turn": turn}

    def stream_events(self, turn: int, timeout: float = 300.0):
        """Yield queued events for a turn until done/error (SSE source)."""
        event_queue = self.streams.get(turn)
        if event_queue is None:
            raise ValueError("Unknown turn.")
        try:
            while True:
                try:
                    event = event_queue.get(timeout=timeout)
                except queue.Empty as exc:
                    raise RuntimeError("Stream timed out.") from exc
                yield event
                if "done" in event or "error" in event:
                    return
        finally:
            self.streams.pop(turn, None)

    def _synth_sentence(self, sentence: str, sent_dir: Path) -> Path:
        """Synthesize one streamed sentence; test_indic_f5 owns the files."""
        sent_dir.mkdir(parents=True, exist_ok=False)
        return self.tts_fn(sentence, sent_dir)

    def _stream_worker(
        self,
        turn: int,
        user_text: str,
        snapshot: list[dict[str, str]],
        turn_dir: Path,
    ) -> None:
        """Stream tokens, voice each finished sentence, then log the turn."""
        event_queue = self.streams[turn]
        deltas: list[str] = []
        pending = ""
        audios: list[str] = []
        sentence_index = 0

        def voice(text: str) -> None:
            nonlocal sentence_index
            sentence_index += 1
            sent_dir = turn_dir / f"sent_{sentence_index:02d}"
            wav = self._synth_sentence(text, sent_dir)
            name = f"turn_{turn:02d}/sent_{sentence_index:02d}/{wav.name}"
            audios.append(name)
            event_queue.put({"audio": name, "sentence": text})

        try:
            for delta in self.gemini_stream_fn(snapshot, self.system, self.model, self.api_key):
                deltas.append(delta)
                pending += delta
                event_queue.put({"token": delta})
                complete, pending = split_sentences(pending)
                if self.enable_tts:
                    for sentence in complete:
                        voice(sentence)
            if pending.strip() and self.enable_tts:
                voice(pending.strip())
            reply = "".join(deltas).strip()
            if not reply:
                raise RuntimeError("Gemini returned an empty reply.")
            with self.lock:
                self.history.append({"role": "model", "text": reply})
                mic = self._mic_files.pop(turn, None)
                with (self.session_dir / "session.jsonl").open("a", encoding="utf-8") as log:
                    log.write(
                        json.dumps(
                            {
                                "turn": turn,
                                "user": user_text,
                                "reply": reply,
                                "audio": audios,
                                "mic": f"turn_{turn:02d}/{mic}" if mic else None,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            event_queue.put({"done": {"turn": turn, "reply": reply, "audios": audios}})
        except Exception as exc:  # noqa: BLE001 — surfaced to the browser as an event
            with self.lock:
                if self.history and self.history[-1].get("role") == "user":
                    self.history.pop()
            event_queue.put({"error": str(exc)[:500]})

    def audio_path(self, name: str) -> Path:
        """Resolve a servable wav name, confined to this session directory."""
        if not AUDIO_PATTERN.fullmatch(name or ""):
            raise ValueError("Invalid audio file.")
        path = (self.session_dir / name).resolve()
        try:
            path.relative_to(self.session_dir.resolve())
        except ValueError as exc:
            raise ValueError("Invalid audio file.") from exc
        if not path.is_file():
            raise ValueError("Audio file does not exist.")
        return path


def worker_transcriber(worker: STTWorker, tmp_dir: str | Path) -> Callable[[Any], str]:
    """Adapt file-protocol STTWorker to the service's waveform -> text contract."""
    import itertools

    scratch = Path(tmp_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    counter = itertools.count()

    def transcribe(waveform: Any) -> str:
        import soundfile as sf

        path = scratch / f"stt_{next(counter):04d}.wav"
        sf.write(path, waveform, 16_000, subtype="PCM_16", format="WAV")
        try:
            return worker.transcribe(path)
        finally:
            path.unlink(missing_ok=True)

    return transcribe


class VoiceChatRequestHandler(BaseHTTPRequestHandler):
    """Chat page, JSON API, and per-turn wav routes for one chat service."""

    service: VoiceChatService

    def log_message(self, format: str, *args: object) -> None:
        """Keep polling out of the terminal."""

    def _send_json(self, body: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status)

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > 64 * 1024:
            raise ValueError("Request body is missing or too large.")
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Request body must be JSON.") from exc
        if not isinstance(payload, dict):
            raise TypeError("Request body must be an object.")
        return payload

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                self._send_json(
                    {
                        "model": self.service.model,
                        "history_limit": self.service.history_limit,
                        "turns": self.service.turn,
                        "tts": self.service.enable_tts,
                        "stt": self.service.stt_fn is not None,
                    }
                )
            elif parsed.path == "/api/audio":
                path = self.service.audio_path(parse_qs(parsed.query).get("file", [""])[0])
                payload = path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif parsed.path == "/api/stream":
                try:
                    turn = int(parse_qs(parsed.query).get("turn", [""])[0])
                except (TypeError, ValueError) as exc:
                    raise ValueError("Turn id is required.") from exc
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                # No Content-Length on SSE: the client reads to EOF, so this
                # route must close (keep-alive here hangs read-to-EOF clients).
                self.close_connection = True
                self.end_headers()
                try:
                    for event in self.service.stream_events(turn):
                        self.wfile.write(
                            ("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode(
                                "utf-8"
                            )
                        )
                        self.wfile.flush()
                except (ValueError, RuntimeError) as exc:
                    self.wfile.write(
                        ("data: " + json.dumps({"error": str(exc)[:500]}) + "\n\n").encode(
                            "utf-8"
                        )
                    )
                    self.wfile.flush()
            elif parsed.path in ("/", "/chat.html"):
                html = (STATIC_DIR / "voice_chat.html").read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
            else:
                self._send_error_json(HTTPStatus.NOT_FOUND, "Not found.")
        except (TypeError, ValueError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/chat":
                result = self.service.chat(self._read_json().get("text", ""))
                self._send_json(result)
            elif parsed.path == "/api/chat_stream":
                result = self.service.start_stream(self._read_json().get("text", ""))
                self._send_json(result)
            elif parsed.path == "/api/voice":
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_VOICE_BYTES:
                    raise ValueError("Voice audio must be a WAV smaller than 16 MB.")
                result = self.service.voice_turn(self.rfile.read(length))
                self._send_json(result)
            elif parsed.path == "/api/reset":
                self._send_json(self.service.reset())
            else:
                self._send_error_json(HTTPStatus.NOT_FOUND, "Not found.")
        except (TypeError, ValueError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except RuntimeError as exc:
            self._send_error_json(HTTPStatus.BAD_GATEWAY, str(exc))
        except Exception as exc:  # noqa: BLE001 — never drop a client connection
            self._send_error_json(HTTPStatus.BAD_GATEWAY, f"voice turn failed: {exc}"[:500])


def make_server(
    host: str,
    port: int,
    service: VoiceChatService,
) -> ThreadingHTTPServer:
    """Create the local voice-chat HTTP server."""
    handler = type(
        "BoundVoiceChatRequestHandler",
        (VoiceChatRequestHandler,),
        {"service": service},
    )
    return ThreadingHTTPServer((host, port), handler)
