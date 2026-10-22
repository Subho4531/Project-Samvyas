"""Local-only HTTP service for collecting Bengali voice samples in a browser."""

from __future__ import annotations

import io
import json
import math
import re
import threading
import uuid
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from samvyas.audio.pause_segmentation import pause_segments
from samvyas.data.voice_dataset import (
    SAMPLE_RATE,
    VoiceDataset,
    VoicePrompt,
    audio_quality,
    load_prompts,
)

DEFAULT_DATA_ROOT = Path("assets/audio/voice_users")
DEFAULT_PROMPTS_PATH = Path("configs/tts/bengali_prompts.json")
STATIC_DIR = Path(__file__).with_name("static")
MAX_RECORDING_BYTES = 16 * 1024 * 1024
MAX_CONTINUOUS_BYTES = 64 * 1024 * 1024
SCRIPT_DIR = Path(__file__).resolve().parents[2] / "configs/tts"
USER_NAME_PATTERN = re.compile(r"[^\w-]+", flags=re.UNICODE)


def user_slug(name: str) -> str:
    """Make a safe, readable profile folder name from a user-provided name."""
    normalized = USER_NAME_PATTERN.sub("-", name.strip().casefold()).strip("-._")
    if not 2 <= len(normalized) <= 48:
        raise ValueError(
            "User name must contain 2–48 letters, numbers, spaces, hyphens, or underscores."
        )
    return normalized


class VoiceRecorderService:
    """User profiles, prompt progress, and browser WAV persistence."""

    def __init__(
        self,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        prompts_path: str | Path = DEFAULT_PROMPTS_PATH,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.prompts = load_prompts(prompts_path)
        self.prompts_by_id = {prompt.id: prompt for prompt in self.prompts}
        self.write_lock = threading.Lock()

    def scripts(self) -> dict[str, str]:
        """Return original UTF-8 scripts, also available as plain text files."""
        return {
            mode: (SCRIPT_DIR / f"continuous_{mode}_bn.txt").read_text(encoding="utf-8")
            for mode in ("sentences", "story")
        }

    def continuous_folder(self, slug: str, session_id: str) -> Path:
        """Resolve a session only under an existing safe user profile."""
        self._profile(slug)
        if not re.fullmatch(r"[0-9a-f]{32}", session_id):
            raise ValueError("Invalid continuous session ID.")
        return self._profile_dir(slug) / "sessions" / session_id

    def save_continuous(
        self,
        slug: str,
        mode: str,
        transcript: str,
        wav_data: bytes,
        min_pause: float = 1.2,
        threshold: float = 0.008,
    ) -> dict[str, Any]:
        """Preserve a master and propose crops; never auto-accept transcripts."""
        self._profile(slug)
        if mode not in ("sentences", "story") or not transcript.strip():
            raise ValueError("Choose sentences/story and provide the spoken script.")
        if not wav_data or len(wav_data) > MAX_CONTINUOUS_BYTES:
            raise ValueError("Continuous WAV must be smaller than 64 MB.")
        try:
            audio, rate = sf.read(io.BytesIO(wav_data), dtype="float32", always_2d=True)
        except RuntimeError as exc:
            raise ValueError("Invalid continuous WAV.") from exc
        if rate < 8000 or not 0 < audio.shape[0] / rate <= 600:
            raise ValueError("Record up to ten minutes at 8 kHz or higher.")
        wav = audio.mean(axis=1, dtype=np.float32)
        if rate != SAMPLE_RATE:
            divisor = math.gcd(rate, SAMPLE_RATE)
            wav = resample_poly(wav, SAMPLE_RATE // divisor, rate // divisor).astype(np.float32)
        quality = audio_quality(wav)
        bounds = []
        warning = "Story saved whole; transcript alignment is required before training."
        lines = [line.strip() for line in transcript.splitlines() if line.strip()]
        if mode == "sentences":
            bounds = pause_segments(wav, min_pause, threshold)
            warning = (
                "Play every crop and confirm the exact words before accepting."
                if len(bounds) == len(lines)
                else f"Found {len(bounds)} clips for {len(lines)} lines. Transcripts left blank; "
                "listen and enter the exact words, or retry with longer pauses."
            )
        session_id = uuid.uuid4().hex
        folder = self.continuous_folder(slug, session_id)
        folder.mkdir(parents=True, exist_ok=False)
        # Keep the exact input bytes as well as canonical audio; neither is a dataset entry.
        (folder / "original.wav").write_bytes(wav_data)
        sf.write(folder / "master_16k.wav", wav, SAMPLE_RATE, subtype="PCM_16")
        (folder / "script.txt").write_text(transcript.strip() + "\n", encoding="utf-8")
        clips = []
        for index, (start, end) in enumerate(bounds):
            filename = f"clip_{index + 1:03d}.wav"
            sf.write(folder / filename, wav[start:end], SAMPLE_RATE, subtype="PCM_16")
            clips.append(
                {
                    "index": index,
                    "file": filename,
                    "start": start / SAMPLE_RATE,
                    "end": end / SAMPLE_RATE,
                    "transcript": lines[index] if len(bounds) == len(lines) else "",
                    "quality": audio_quality(wav[start:end]),
                }
            )
        result = {
            "session_id": session_id,
            "mode": mode,
            "folder": str(folder),
            "duration": wav.size / SAMPLE_RATE,
            "quality": quality,
            "warning": warning,
            "clips": clips,
        }
        (folder / "review.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    def accept_continuous_clip(
        self,
        slug: str,
        session_id: str,
        index: int,
        transcript: str,
        verified: bool,
    ) -> dict[str, Any]:
        """Accept one explicitly verified crop through the existing dataset writer."""
        if verified is not True or not isinstance(transcript, str) or not transcript.strip():
            raise ValueError("Listen and confirm the exact transcript first.")
        folder = self.continuous_folder(slug, session_id)
        if not (folder / "review.json").is_file():
            raise ValueError("Continuous session does not exist.")
        review = json.loads((folder / "review.json").read_text(encoding="utf-8"))
        if type(index) is not int or not 0 <= index < len(review["clips"]):
            raise ValueError("Invalid clip index.")
        clip = review["clips"][index]
        wav, rate = sf.read(folder / clip["file"], dtype="float32")
        if rate != SAMPLE_RATE:
            raise ValueError("Candidate sample rate is invalid.")
        prompt = VoicePrompt(f"continuous_{session_id}_{index:03d}", transcript.strip(), "Natural")
        with self.write_lock:
            return VoiceDataset(self._profile_dir(slug)).accept(prompt, transcript, wav)

    def _profile_dir(self, slug: str) -> Path:
        if user_slug(slug) != slug:
            raise ValueError("Invalid user profile.")
        profile = (self.data_root / slug).resolve()
        try:
            profile.relative_to(self.data_root)
        except ValueError as exc:
            raise ValueError("Invalid user profile.") from exc
        return profile

    def _profile(self, slug: str) -> dict[str, str]:
        path = self._profile_dir(slug) / "profile.json"
        if not path.is_file():
            raise ValueError("User profile does not exist.")
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
            if profile["slug"] != slug or not profile["name"].strip():
                raise ValueError("Invalid profile.")
            return {"name": profile["name"], "slug": profile["slug"]}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid profile.") from exc

    def create_or_open_user(self, name: str) -> dict[str, str]:
        """Create/open a dataset profile without overwriting its display name."""
        name = name.strip()
        slug = user_slug(name)
        root = self._profile_dir(slug)
        root.mkdir(parents=True, exist_ok=True)
        profile_file = root / "profile.json"
        if profile_file.is_file():
            profile = self._profile(slug)
        else:
            profile = {"name": name, "slug": slug}
            temporary = root / "profile.tmp"
            temporary.write_text(
                json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(profile_file)
        VoiceDataset(root)
        return profile

    def users(self) -> list[dict[str, str]]:
        """Return known profiles alphabetically by display name."""
        profiles = []
        for folder in self.data_root.iterdir():
            if folder.is_dir() and (folder / "profile.json").is_file():
                try:
                    profiles.append(self._profile(folder.name))
                except ValueError:
                    continue
        return sorted(profiles, key=lambda profile: profile["name"].casefold())

    def session(self, slug: str) -> dict[str, Any]:
        """Return profile, completed prompts, and total accepted duration."""
        profile = self._profile(slug)
        dataset = VoiceDataset(self._profile_dir(slug))
        return {
            "profile": profile,
            "accepted_prompt_ids": sorted(dataset.accepted_ids),
            "accepted_count": len(dataset.records),
            "total_seconds": dataset.total_seconds,
            "orphan_count": len(dataset.orphans()),
        }

    def prompt_rows(self) -> list[dict[str, str]]:
        """Serialize the configured prompts for the browser."""
        return [asdict(prompt) for prompt in self.prompts]

    def save_recording(
        self,
        slug: str,
        prompt_id: str,
        transcript: str,
        wav_data: bytes,
    ) -> dict[str, Any]:
        """Decode browser WAV, resample it to canonical audio, and accept it once."""
        if not wav_data or len(wav_data) > MAX_RECORDING_BYTES:
            raise ValueError("Recording must be a WAV file smaller than 16 MB.")
        prompt = self.prompts_by_id.get(prompt_id)
        if prompt is None:
            raise ValueError("Unknown prompt.")
        try:
            audio, sample_rate = sf.read(io.BytesIO(wav_data), dtype="float32", always_2d=True)
        except RuntimeError as exc:
            raise ValueError("The browser did not send a valid WAV recording.") from exc
        waveform = audio.mean(axis=1, dtype=np.float32)
        if sample_rate != SAMPLE_RATE:
            divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
            waveform = resample_poly(
                waveform, SAMPLE_RATE // divisor, sample_rate // divisor
            ).astype(np.float32)
        if waveform.size > 20 * SAMPLE_RATE:
            raise ValueError("Recording is longer than the 20-second limit.")
        dataset = VoiceDataset(self._profile_dir(slug))
        with self.write_lock:
            dataset = VoiceDataset(self._profile_dir(slug))
            record = dataset.accept(prompt, transcript, waveform)
        return record


class VoiceRecorderRequestHandler(BaseHTTPRequestHandler):
    """JSON API and single-page application routes for one recorder service."""

    service: VoiceRecorderService

    def log_message(self, format: str, *args: object) -> None:
        """Keep normal browser polling out of the terminal."""

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
            if parsed.path == "/api/users":
                self._send_json({"users": self.service.users()})
            elif parsed.path == "/api/prompts":
                self._send_json({"prompts": self.service.prompt_rows()})
            elif parsed.path == "/api/scripts":
                self._send_json(self.service.scripts())
            elif parsed.path == "/api/continuous/audio":
                query = parse_qs(parsed.query)
                folder = self.service.continuous_folder(
                    query.get("user", [""])[0], query.get("session_id", [""])[0]
                )
                filename = query.get("file", [""])[0]
                if not re.fullmatch(r"clip_[0-9]{3}\.wav|master_16k\.wav", filename):
                    raise ValueError("Invalid audio file.")
                path = folder / filename
                if not path.is_file():
                    raise ValueError("Audio file does not exist.")
                payload = path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif parsed.path == "/api/session":
                slug = parse_qs(parsed.query).get("user", [""])[0]
                self._send_json(self.service.session(slug))
            elif parsed.path in ("/", "/index.html"):
                html = (STATIC_DIR / "voice_recorder.html").read_bytes()
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
            if parsed.path == "/api/users":
                payload = self._read_json()
                name = payload.get("name")
                if not isinstance(name, str):
                    raise ValueError("User name is required.")
                self._send_json(self.service.create_or_open_user(name), HTTPStatus.CREATED)
                return
            if parsed.path == "/api/continuous/accept":
                data = self._read_json()
                record = self.service.accept_continuous_clip(
                    data.get("user", ""),
                    data.get("session_id", ""),
                    data.get("index"),
                    data.get("transcript", ""),
                    data.get("verified", False),
                )
                self._send_json(record, HTTPStatus.CREATED)
                return
            if parsed.path == "/api/continuous":
                query = parse_qs(parsed.query)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_CONTINUOUS_BYTES:
                    raise ValueError("Continuous WAV must be smaller than 64 MB.")
                result = self.service.save_continuous(
                    query.get("user", [""])[0],
                    query.get("mode", [""])[0],
                    query.get("transcript", [""])[0],
                    self.rfile.read(length),
                    float(query.get("pause", ["1.2"])[0]),
                    float(query.get("threshold", ["0.008"])[0]),
                )
                self._send_json(result, HTTPStatus.CREATED)
                return
            if parsed.path == "/api/recordings":
                query = parse_qs(parsed.query)
                slug = query.get("user", [""])[0]
                prompt_id = query.get("prompt_id", [""])[0]
                transcript = query.get("transcript", [""])[0]
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > MAX_RECORDING_BYTES:
                    raise ValueError("Recording must be a WAV file smaller than 16 MB.")
                record = self.service.save_recording(
                    slug, prompt_id, transcript, self.rfile.read(content_length)
                )
                self._send_json(record, HTTPStatus.CREATED)
                return
            self._send_error_json(HTTPStatus.NOT_FOUND, "Not found.")
        except (TypeError, ValueError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))


def make_server(
    host: str,
    port: int,
    service: VoiceRecorderService,
) -> ThreadingHTTPServer:
    """Create the local browser recorder HTTP server."""
    handler = type(
        "BoundVoiceRecorderRequestHandler",
        (VoiceRecorderRequestHandler,),
        {"service": service},
    )
    return ThreadingHTTPServer((host, port), handler)
