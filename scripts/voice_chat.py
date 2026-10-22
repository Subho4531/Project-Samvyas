"""Push-to-talk voice chat: mic -> STT -> Gemini -> IndicF5 TTS.

Runs in `.venv-recorder` (mic + playback, stdlib-only Gemini REST). STT is
attempted lazily and degrades to typed input until `.venv` is rebuilt; TTS
runs out-of-process in `.venv-tts` via scripts/test_indic_f5.py so the venv
separation rule is preserved.

  $env:GEMINI_API_KEY = "<key>"   # never pass --api-key on a shared screen
  .\\.venv-recorder\\Scripts\\python.exe scripts/voice_chat.py --text-mode
  .\\.venv-recorder\\Scripts\\python.exe scripts/voice_chat.py ^
      --clip-id YOUR_CLIP_ID --dataset assets/audio/voice_users/jeet

Controls per turn: Enter = talk (or type in --text-mode), t = type once,
q = quit. The last --history messages (default 10) are sent as context.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from samvyas.audio.recording import (  # noqa: E402
    MicrophoneRecorder,
    configure_unicode_terminal,
    report_take,
    terminal_write,
)

DEFAULT_MODEL = "gemini-2.5-flash"  # user asked for "3.5 flash"; override with --model
TTS_SCRIPT = ROOT / "scripts" / "test_indic_f5.py"
CHAT_ROOT = ROOT / "outputs" / "tts"


def trim_history(history: list[dict], limit: int) -> list[dict]:
    """Keep the most recent `limit` messages for the context window."""
    return history[-limit:] if limit > 0 else []


def build_payload(history: list[dict], system: str) -> dict:
    """Build a generateContent body from role/text history."""
    contents = [
        {"role": msg["role"], "parts": [{"text": msg["text"]}]} for msg in history
    ]
    return {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 512},
    }


def extract_text(body: dict) -> str:
    """Pull the reply text out of a generateContent response."""
    candidates = body.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {json.dumps(body)[:300]}")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        raise RuntimeError("Gemini returned an empty reply.")
    return text


def ask_gemini(
    history: list[dict], system: str, model: str, api_key: str, timeout: float = 90.0
) -> str:
    """POST one chat turn; stdlib only so no new dependency is needed."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    request = urllib.request.Request(
        url,
        data=json.dumps(build_payload(history, system)).encode("utf-8"),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return extract_text(json.loads(response.read().decode("utf-8")))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"Gemini HTTP {exc.code}: {detail}") from exc


def resolve_mic(value: str | None):
    """Resolve --mic: index, name substring, or None (Realtek array, else default).

    System default here is the WO Mic phone-app device, which is often not
    live — so an unset --mic prefers a Realtek array input instead.
    """
    import sounddevice as sd

    if value is not None and value.strip() != "":
        text = value.strip()
        try:
            return int(text)
        except ValueError:
            tokens = text.lower().split()
            inputs = [
                i
                for i, dev in enumerate(sd.query_devices())
                if dev["max_input_channels"] > 0
                and all(tok in dev["name"].lower() for tok in tokens)
            ]
            if not inputs:
                raise SystemExit(f"No input device matches {value!r} (see --list-devices).")
            return inputs[0]
    order = ("wasapi", "directsound", "mme", "wdm-ks")

    def rank(index: int) -> tuple[int, str]:
        dev = sd.query_devices(index)
        api = sd.query_hostapis(dev["hostapi"])["name"].lower()
        return (
            next((n for n, key in enumerate(order) if key in api), len(order)),
            dev["name"],
        )

    arrays = [
        i
        for i, dev in enumerate(sd.query_devices())
        if dev["max_input_channels"] > 0 and "realtek" in dev["name"].lower()
    ]
    if arrays:
        return sorted(arrays, key=rank)[0]
    return None


def describe_mic(device) -> str:
    """Human-readable name of the resolved capture device."""
    if device is None:
        return "system default input"
    import sounddevice as sd

    try:
        return f"[{device}] {sd.query_devices(device)['name']}"
    except Exception:
        return str(device)


def transcribe(waveform, sample_rate: int, no_denoise: bool) -> str:
    """Run the STT pipeline; raises with a rebuild hint when `.venv` is down."""
    try:
        import torch

        from samvyas.pipelines.stt_pipeline import STTPipeline
    except ImportError as exc:
        raise RuntimeError(
            "STT needs the rebuilt `.venv` (torch/DeepFilterNet/NeMo); "
            "use --text-mode until then."
        ) from exc
    pipe = STTPipeline()
    if no_denoise:
        pipe.denoiser.enabled = False
    with torch.no_grad():
        result = pipe.process(torch.from_numpy(waveform), sample_rate)
    return result.transcription.strip()


def synthesize_reply(
    text: str,
    args: argparse.Namespace,
    tts_python: str,
    turn: int,
) -> Path | None:
    """Shell out to .venv-tts; return the 16 kHz wav path or None on failure."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out_dir = CHAT_ROOT / f"chat_{stamp}_t{turn:02d}"
    cmd = [
        tts_python,
        str(TTS_SCRIPT),
        "--text",
        text,
        "--output-dir",
        str(out_dir),
        "--steps",
        str(args.steps),
        "--device",
        args.tts_device,
    ]
    if args.cpu_fallback:
        cmd.append("--cpu-fallback")
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    if args.clip_id:
        cmd += ["--clip-id", args.clip_id, "--dataset", args.dataset]
    else:
        cmd += ["--reference", args.reference]
        if args.reference_text_file:
            cmd += ["--reference-text-file", args.reference_text_file]
        else:
            cmd += ["--reference-text", args.reference_text]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        terminal_write(f"TTS failed:\n{(proc.stderr or proc.stdout)[-800:]}")
        return None
    wav = out_dir / "baseline_01_16k.wav"
    return wav if wav.is_file() else None


def play_wav(path: Path, recorder: MicrophoneRecorder) -> None:
    """Play a 16 kHz wav through the default output device."""
    import sounddevice as sd
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32")
    if rate != 16_000:
        terminal_write(f"Unexpected playback rate {rate}; skipping audio.")
        return
    sd.play(audio, rate)
    sd.wait()


def resolve_api_key(args: argparse.Namespace) -> str:
    """Key from flag, env, or repo .env — never printed or stored."""
    import os

    if args.api_key:
        return args.api_key
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        # .venv-recorder has no python-dotenv; parse KEY=VALUE lines manually.
        env_file = ROOT / ".env"
        if env_file.is_file():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    name, _, value = line.partition("=")
                    if name.strip() not in os.environ:
                        os.environ[name.strip()] = value.strip().strip("\"'")
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    if not key:
        raise SystemExit(
            "Set $env:GEMINI_API_KEY (or GOOGLE_API_KEY in .env) before chatting."
        )
    return key


def main() -> int:
    configure_unicode_terminal()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None, help="Prefer $env:GEMINI_API_KEY.")
    parser.add_argument("--history", type=int, default=10, help="Context window (msgs).")
    parser.add_argument("--system", default=None, help="Override system instruction.")
    parser.add_argument("--text-mode", action="store_true", help="Type instead of mic.")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=15.0)
    parser.add_argument(
        "--mic",
        default=None,
        help="Mic index or name substring; unset prefers the Realtek array.",
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--tts-python", default=str(ROOT / ".venv-tts/Scripts/python.exe"))
    parser.add_argument("--tts-device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--cpu-fallback", action="store_true")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--dataset", default=str(ROOT / "assets/audio/voice_dataset"))
    parser.add_argument("--reference", default=None)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-text-file", default=None)
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd

        terminal_write(str(sd.query_devices()))
        return 0
    if not args.no_tts and not args.clip_id and not args.reference:
        parser.error("TTS needs --clip-id or --reference (+ transcript), or --no-tts.")
    if args.reference and not (args.reference_text or args.reference_text_file):
        parser.error("--reference needs --reference-text or --reference-text-file.")

    api_key = resolve_api_key(args)
    system = args.system or (
        "You are a friendly voice assistant. Reply in Bengali, in short "
        "spoken style (1-3 sentences), since replies are read aloud."
    )
    recorder = MicrophoneRecorder(device=resolve_mic(args.mic))
    terminal_write(f"Mic: {describe_mic(recorder.device)}")
    history: list[dict] = []
    terminal_write(f"Voice chat [{args.model}], history={args.history}. q=quit.")
    turn = 0
    while True:
        try:
            if args.text_mode:
                user_text = input("you> ").strip()
                if user_text.lower() == "q":
                    break
            else:
                choice = input("Enter=talk, t=type, q=quit: ").strip().lower()
                if choice == "q":
                    break
                if choice == "t":
                    user_text = input("you> ").strip()
                else:
                    terminal_write("Speak now — Enter to stop.")
                    try:
                        take = recorder.record(args.max_seconds)
                    except (RuntimeError, ValueError, OSError) as exc:
                        terminal_write(f"Recording failed: {exc}")
                        continue
                    report_take(take, terminal_write)
                    try:
                        user_text = transcribe(take, 16_000, args.no_denoise)
                    except (RuntimeError, OSError, ValueError) as exc:
                        terminal_write(f"{exc} Type this turn instead (t) or --text-mode.")
                        continue
                    terminal_write(f"heard> {user_text}")
            if not user_text:
                continue
            history.append({"role": "user", "text": user_text})
            try:
                reply = ask_gemini(trim_history(history, args.history), system, args.model, api_key)
            except (RuntimeError, OSError) as exc:
                history.pop()
                terminal_write(f"LLM failed: {exc}")
                continue
            history.append({"role": "model", "text": reply})
            turn += 1
            terminal_write(f"gemini> {reply}")
            if not args.no_tts:
                wav = synthesize_reply(reply, args, args.tts_python, turn)
                if wav is None:
                    continue
                try:
                    play_wav(wav, recorder)
                except (RuntimeError, OSError) as exc:
                    terminal_write(f"Playback failed ({wav}): {exc}")
        except (KeyboardInterrupt, EOFError):
            terminal_write("\nBye.")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
