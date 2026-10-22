"""Persistent TTS worker: load IndicF5 + Vocos once on CUDA, synthesize many sentences.

Runs in `.venv-tts` (PyTorch 2.5.1+cu121). The chat server spawns one instance
and communicates via JSON lines over stdio so weights stay resident in GPU VRAM:

  request:  {"id": 1, "text": "...", "output_dir": "/path/to/sent_01", "steps": 16}
  response: {"id": 1, "audio_16k": "/path/to/baseline_01_16k.wav", "audio_24k": "..."}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    _tok = os.getenv("HF_TOKEN") or os.getenv("HF_ACESS_TOKEN")
    if _tok and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = _tok
except (ImportError, OSError):
    pass


def _silence_stdout_noise() -> None:
    """Redirect third-party stdout loggers so stdio protocol stays clean."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    _silence_stdout_noise()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()

    # Strict GPU enforcement
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        err = {"ready": False, "error": "CUDA is strictly required for fast GPU TTS inference, but no CUDA GPU was detected."}
        print(json.dumps(err), flush=True)
        return 1

    # Resolve reference text
    ref_text = (
        args.reference_text_file.read_text(encoding="utf-8-sig")
        if args.reference_text_file
        else args.reference_text
    )
    if not ref_text or not ref_text.strip():
        err = {"ready": False, "error": "Verified reference transcript is required."}
        print(json.dumps(err), flush=True)
        return 1
    ref_text = ref_text.strip()
    ref_path = Path(args.reference).resolve()
    if not ref_path.is_file():
        err = {"ready": False, "error": f"Reference audio not found: {ref_path}"}
        print(json.dumps(err), flush=True)
        return 1

    # Strict GPU Preloading
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly

        from samvyas.models.tts.indic_f5 import MODEL_REVISION, IndicF5Backend

        backend = IndicF5Backend(device=args.device, checkpoint=args.checkpoint)
        vram_mb = round(torch.cuda.memory_allocated() / (1024**2), 1) if args.device == "cuda" else 0.0
    except Exception as exc:  # noqa: BLE001
        err = {"ready": False, "error": f"Failed to preload TTS model on {args.device}: {exc}"}
        print(json.dumps(err), flush=True)
        return 1

    # Ready signal to parent process
    ready_msg = {
        "ready": True,
        "device": args.device,
        "vram_mb": vram_mb,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "default_steps": args.steps,
    }
    print(json.dumps(ready_msg), flush=True)

    # Process stdin loop
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = None
        try:
            request = json.loads(line)
            req_id = request.get("id")
            text = (request.get("text") or "").strip()
            if not text:
                raise ValueError("Synthesis text cannot be empty.")

            output_dir = Path(request.get("output_dir", ROOT / "outputs" / "tts" / "tmp"))
            output_dir.mkdir(parents=True, exist_ok=True)
            steps = int(request.get("steps", args.steps))
            target_ref = Path(request.get("reference", ref_path))
            target_ref_text = request.get("reference_text", ref_text)

            t0 = time.perf_counter()
            native = backend.generate(text, target_ref, target_ref_text, steps=steps)
            native = np.asarray(native)
            if native.dtype == np.int16:
                native = native.astype(np.float32) / 32768.0
            elif not np.issubdtype(native.dtype, np.floating):
                raise ValueError("Backend must return floating point or int16 audio.")
            native = native.astype(np.float32)

            # Resample to 16 kHz
            canonical = resample_poly(native, 2, 3).astype(np.float32)
            scale = max(1.0, float(np.max(np.abs(native))), float(np.max(np.abs(canonical))))
            scaled_native = native / scale
            scaled_canonical = canonical / scale

            # Save WAV files
            path_24k = output_dir / "baseline_01_24k.wav"
            path_16k = output_dir / "baseline_01_16k.wav"
            sf.write(str(path_24k), scaled_native, 24_000, subtype="PCM_24", format="WAV")
            sf.write(str(path_16k), scaled_canonical, 16_000, subtype="PCM_24", format="WAV")

            dt = round(time.perf_counter() - t0, 3)

            # Write run.json
            report = {
                "model": "ai4bharat/IndicF5",
                "revision": MODEL_REVISION,
                "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
                "reference": str(target_ref),
                "reference_transcript": target_ref_text,
                "steps": steps,
                "results": [
                    {
                        "text": text,
                        "device": args.device,
                        "audio_24k": path_24k.name,
                        "audio_16k": path_16k.name,
                        "peak_scale": scale,
                        "duration_s": round(len(native) / 24000.0, 3),
                        "inference_s": dt,
                    }
                ],
            }
            (output_dir / "run.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            reply = {
                "id": req_id,
                "audio_16k": str(path_16k.resolve()),
                "audio_24k": str(path_24k.resolve()),
                "device": args.device,
                "duration_s": round(len(native) / 24000.0, 3),
                "inference_s": dt,
            }
            print(json.dumps(reply), flush=True)

        except Exception as exc:  # noqa: BLE001
            err_reply = {"id": req_id, "error": f"TTS synthesis error: {exc}"}
            print(json.dumps(err_reply), flush=True)

    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
