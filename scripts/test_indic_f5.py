"""Generate Bengali reference-conditioned speech; no fine-tuning is performed."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from samvyas.data.voice_dataset import VoiceDataset
from samvyas.models.tts.indic_f5 import MODEL_REVISION, IndicF5Backend, IndicF5Synthesizer

BASELINE_TEXTS = [
    "আজ নতুন একটি গল্প নিয়ে তোমার সঙ্গে কথা বলতে চাই।",
    "আগামী রবিবার আমরা কি নদীর ধারে একটু বেড়াতে যাব?",
    "কী আনন্দের খবর! তোমার সাফল্যে আমি খুব খুশি হয়েছি।",
]


def main() -> int:
    """Select an accepted clip explicitly or provide a WAV and verified transcript."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--clip-id", help="Explicit accepted manifest clip_id")
    selection.add_argument("--reference", type=Path, help="Mono 16 kHz reference WAV")
    parser.add_argument("--dataset", type=Path, default=ROOT / "assets/audio/voice_dataset")
    transcript = parser.add_mutually_exclusive_group()
    transcript.add_argument("--reference-text", help="Exact, verified reference words")
    transcript.add_argument("--reference-text-file", type=Path, help="UTF-8 verified transcript")
    parser.add_argument("--text", help="Target text; omitted runs the three-sentence baseline")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-fallback", action="store_true", help="Retry CUDA OOM on CPU")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, help="Adapted model.safetensors from fine-tuning"
    )
    args = parser.parse_args()
    synthesizer = None
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        if args.clip_id:
            if args.reference_text is not None or args.reference_text_file is not None:
                parser.error("--clip-id uses its confirmed manifest transcript.")
            if not (args.dataset / "manifest.jsonl").is_file():
                parser.error("Dataset manifest not found; record and accept a clip first.")
            dataset = VoiceDataset(args.dataset)
            record = next((row for row in dataset.records if row["clip_id"] == args.clip_id), None)
            if record is None:
                parser.error("Clip ID not found in accepted recordings.")
            reference = args.dataset / record["audio_path"]
            ref_text = record["transcript"]
        else:
            reference = args.reference
            ref_text = (
                args.reference_text_file.read_text(encoding="utf-8-sig")
                if args.reference_text_file
                else args.reference_text
            )
            if not ref_text or not ref_text.strip():
                parser.error(
                    "Provide --reference-text or --reference-text-file with verified words."
                )
        output_dir = args.output_dir or ROOT / "outputs/tts" / datetime.now(timezone.utc).strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        if output_dir.exists() and any(output_dir.iterdir()):
            parser.error(
                "Output directory must be new or empty; existing results will not be overwritten."
            )
        texts = [args.text] if args.text is not None else BASELINE_TEXTS
        factory = lambda device: IndicF5Backend(device, args.checkpoint)
        synthesizer = IndicF5Synthesizer(args.device, args.cpu_fallback, backend_factory=factory)
        results = []
        for index, text in enumerate(texts, 1):
            print(f"Generating {index}/{len(texts)}: {text}")
            result = synthesizer.synthesize(text, reference, ref_text, args.steps)
            paths = result.save(output_dir, f"baseline_{index:02d}")
            results.append(
                {
                    "text": text,
                    "device": result.device,
                    "audio_24k": paths[0].name,
                    "audio_16k": paths[1].name,
                    "peak_scale": result.peak_scale,
                }
            )
        report = {
            "model": "ai4bharat/IndicF5",
            "revision": MODEL_REVISION,
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "reference": str(reference.resolve()),
            "reference_transcript": ref_text,
            "steps": args.steps,
            "results": results,
            "listening_review": "pending: intelligibility, identity, delivery, artifacts",
        }
        (output_dir / "run.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved to {output_dir}. Listen to both rates; voice similarity is not yet verified.")
        return 0
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        print(f"TTS test failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Check the output folder for any completed sentences.")
        return 130
    finally:
        if synthesizer is not None:
            synthesizer.close()


if __name__ == "__main__":
    raise SystemExit(main())
