"""Run the local Samvyas end-to-end voice-chat window."""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from samvyas.web.voice_chat import (
    DEFAULT_MODEL,
    STTWorker,
    TTSWorker,
    VoiceChatService,
    make_server,
    resolve_api_key,
    worker_transcriber,
)

# Assistant's default voice: Jeet's conversational preset (verified 7.9 s take).
# Push-to-talk input needs no clip; this only sets whose voice replies use.
DEFAULT_VOICE_DATASET = ROOT / "assets/audio/voice_users/jeet"
DEFAULT_VOICE_CLIP = "76f1edc2b50a49f19901fc83baf5bf2f"


def resolve_reference(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Return an absolute reference WAV + verified transcript, or (None, None)."""
    from samvyas.data.voice_dataset import VoiceDataset

    def from_clip(dataset: str | Path, clip_id: str) -> tuple[str, str]:
        records = VoiceDataset(dataset).records
        record = next((row for row in records if row["clip_id"] == clip_id), None)
        if record is None:
            raise ValueError(f"Clip ID {clip_id} not found under {dataset}.")
        wav = (Path(dataset) / record["audio_path"]).resolve()
        return str(wav), record["transcript"]

    if args.no_tts:
        return None, None
    if args.clip_id:
        return from_clip(args.dataset, args.clip_id)
    if not args.reference:
        try:
            wav, transcript = from_clip(DEFAULT_VOICE_DATASET, DEFAULT_VOICE_CLIP)
        except ValueError:
            raise ValueError(
                "Default voice preset is gone; pass --clip-id, --reference, or --no-tts."
            )
        print(f"Assistant voice: default preset {DEFAULT_VOICE_CLIP} (override: --clip-id).")
        return wav, transcript
    if not (args.reference_text or args.reference_text_file):
        raise ValueError("--reference needs --reference-text or --reference-text-file.")
    text = (
        Path(args.reference_text_file).read_text(encoding="utf-8-sig")
        if args.reference_text_file
        else args.reference_text
    )
    if not text.strip():
        raise ValueError("Reference transcript is empty.")
    return str(Path(args.reference).resolve()), text.strip()


def main() -> int:
    """Serve the local-only voice chat window until Ctrl+C."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Local bind address.")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None, help="Prefer GEMINI_API_KEY in shell/.env.")
    parser.add_argument("--history", type=int, default=10, help="Context window (msgs).")
    parser.add_argument("--system", default=None)
    parser.add_argument("--no-tts", action="store_true", help="Text-only window.")
    parser.add_argument("--tts-python", default=str(ROOT / ".venv-tts/Scripts/python.exe"))
    parser.add_argument("--tts-device", default="cuda", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--cpu-fallback", action="store_true")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--dataset", default=str(ROOT / "assets/audio/voice_dataset"))
    parser.add_argument("--reference", default=None)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-text-file", default=None)
    parser.add_argument("--no-stt", action="store_true", help="No server STT worker.")
    parser.add_argument("--stt-python", default=str(ROOT / ".venv/Scripts/python.exe"))
    parser.add_argument("--stt-device", default="cuda", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--stt-lang", default="bn")
    parser.add_argument("--stt-no-denoise", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        parser.error("For privacy, voice chat only binds to a loopback address.")
    stt_worker = None
    tts_worker = None
    try:
        reference, reference_text = resolve_reference(args)
        if not args.no_stt:
            if not os.getenv("SAMVYAS_NEMO_PATH"):
                default_patched = (
                    Path.home() / ".cache" / "samvyas" / "indicconformer_bn_agg.nemo"
                )
                if default_patched.is_file():
                    os.environ["SAMVYAS_NEMO_PATH"] = str(default_patched)
                    print(f"Using patched NeMo checkpoint: {default_patched}")
            try:
                print(f"Preloading STT worker onto {args.stt_device}...")
                stt_worker = STTWorker(
                    args.stt_python,
                    device=args.stt_device,
                    no_denoise=args.stt_no_denoise,
                    lang=args.stt_lang,
                )
                print("Server STT worker ready (preloaded on GPU).")
            except RuntimeError as exc:
                print(f"No server STT ({exc}); Talk falls back to browser recognition.")

        if not args.no_tts and reference:
            try:
                print(f"Preloading IndicF5 TTS model onto {args.tts_device} (strict GPU preloading)...")
                tts_worker = TTSWorker(
                    args.tts_python,
                    checkpoint=args.checkpoint,
                    reference=reference,
                    reference_text=reference_text,
                    device=args.tts_device,
                    steps=args.steps,
                )
                print("Server TTS worker ready (preloaded on GPU, zero per-sentence disk loading).")
            except RuntimeError as exc:
                print(f"TTS preloading error: {exc}. Falling back to one-shot.")

        service = VoiceChatService(
            model=args.model,
            api_key=resolve_api_key(args.api_key),
            history_limit=args.history,
            system=args.system,
            tts_python=args.tts_python,
            tts_device=args.tts_device,
            cpu_fallback=args.cpu_fallback,
            steps=args.steps,
            checkpoint=args.checkpoint,
            reference=reference,
            reference_text=reference_text,
            enable_tts=not args.no_tts,
            tts_worker=tts_worker,
        )
        if stt_worker is not None:
            service.stt_fn = worker_transcriber(stt_worker, service.session_dir / "stt_tmp")
        server = make_server(args.host, args.port, service)
    except (OSError, ValueError) as exc:
        print(f"Could not start voice chat: {exc}", file=sys.stderr)
        return 1
    address = "localhost" if args.host == "localhost" else args.host
    url = f"http://{address}:{server.server_port}"
    print(f"Samvyas Voice Chat: {url}")
    print("Talk uses our STT pipeline when ready, else browser recognition; typing always works.")
    print(f"Replies are saved under: {service.session_dir}")
    print("Press Ctrl+C to stop the local server.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nVoice chat stopped.")
    finally:
        server.server_close()
        if stt_worker is not None:
            stt_worker.close()
        if tts_worker is not None:
            tts_worker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
