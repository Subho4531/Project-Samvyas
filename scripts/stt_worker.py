"""Persistent STT worker: load the pipeline once, transcribe many turns.

Runs in `.venv` (torch + DeepFilterNet + NeMo). The chat server spawns one
instance and speaks JSON lines over stdio so VRAM model weights stay loaded:

  request:  {"id": 1, "audio": "/abs/path/to/turn.wav"}
  response: {"id": 1, "transcript": "..."} | {"id": 1, "error": "..."}

One-shot mode transcribes a single file (smoke test without the server):

  .\\.venv\\Scripts\\python.exe scripts/stt_worker.py --file sample.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import os

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    _tok = os.getenv("HF_TOKEN") or os.getenv("HF_ACESS_TOKEN")
    if _tok and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = _tok
except (ImportError, OSError):
    pass

MAX_SECONDS = 30.0


def _keep_stdout_protocol_clean() -> None:
    """Force third-party loggers off stdout; the stdio protocol owns it.

    DeepFilter installs a loguru sink on sys.stdout at first use, which would
    corrupt the JSON-lines protocol. Re-assert stderr-only before inference.
    """
    try:
        import sys as _sys

        from loguru import logger

        logger.remove()
        logger.add(_sys.stderr)
    except Exception:
        pass


def transcribe_file(path: str, pipe, max_seconds: float = MAX_SECONDS) -> str:
    """Transcribe one wav file; raises ValueError/RuntimeError with reasons."""
    import soundfile as sf
    import torch

    _keep_stdout_protocol_clean()

    info = sf.info(path)
    if info.channels != 1:
        raise ValueError("STT worker needs mono WAV input.")
    if info.samplerate != 16_000:
        raise ValueError(f"STT worker needs 16000 Hz, got {info.samplerate} Hz.")
    if not 0 < info.duration <= max_seconds:
        raise ValueError(f"Audio must be 0-{max_seconds:g} seconds, got {info.duration:.1f}.")
    audio, _ = sf.read(path, dtype="float32")
    with torch.no_grad():
        result = pipe.process(torch.from_numpy(audio), 16_000)
    text = (result.transcription or "").strip()
    if not text:
        raise RuntimeError("STT returned an empty transcript; speak closer or retry.")
    return text


def serve(pipe, max_seconds: float) -> int:
    """Serve transcribe requests on stdin until EOF; never exit on bad input."""
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            text = transcribe_file(request["audio"], pipe, max_seconds)
            print(json.dumps({"id": request.get("id"), "transcript": text}), flush=True)
        except (ValueError, RuntimeError, KeyError, OSError) as exc:
            try:
                request_id = json.loads(line).get("id")
            except (ValueError, AttributeError):
                request_id = None
            print(json.dumps({"id": request_id, "error": str(exc)[:500]}), flush=True)
        except Exception as exc:  # noqa: BLE001 — worker must stay alive
            print(json.dumps({"id": None, "error": f"worker: {exc}"[:500]}), flush=True)
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--lang", default="bn", help="Aggregate-tokenizer language for decode.")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--file", default=None, help="One-shot transcribe, then exit.")
    args = parser.parse_args()
    try:
        from samvyas.models.encoders.semantic.indic_asr import IndicASREncoder
        from samvyas.pipelines.stt_pipeline import STTPipeline
    except ImportError as exc:
        print(f"STT worker needs the STT venv (torch/DeepFilterNet/NeMo): {exc}")
        return 2
    try:
        pipe = STTPipeline(
            encoder=IndicASREncoder(device=args.device, lang=args.lang),
            device=args.device,
        )
        if args.no_denoise:
            pipe.denoiser.enabled = False
        # Strict immediate GPU preloading at worker startup
        if hasattr(pipe.encoder, "load"):
            pipe.encoder.load()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"STT worker failed to load: {exc}")
        return 1
    try:
        if args.file:
            print(transcribe_file(args.file, pipe, args.max_seconds))
            return 0
        return serve(pipe, args.max_seconds)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"STT worker failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
