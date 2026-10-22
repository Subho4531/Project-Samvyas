"""Guided Bengali voice collection. Run --help for options."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from samvyas.audio.recording import (
    MicrophoneRecorder,
    configure_unicode_terminal,
    guided_session,
    microphone_check,
    terminal_write,
)
from samvyas.data.voice_dataset import VoiceDataset, load_prompts


def main() -> int:
    """Run a resumable recording session; models and GPU are not required."""
    configure_unicode_terminal()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/audio/voice_dataset")
    parser.add_argument("--prompts", type=Path, default=ROOT / "configs/tts/bengali_prompts.json")
    parser.add_argument("--device", help="Input device index or sounddevice name substring")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
    try:
        if args.list_devices:
            import sounddevice as sd

            terminal_write(str(sd.query_devices()))
            return 0
        if not sys.stdin.isatty():
            parser.error("An interactive terminal is required for recording.")
        device = int(args.device) if args.device and args.device.isdecimal() else args.device
        prompts = load_prompts(args.prompts)
        dataset = VoiceDataset(args.output_dir)
        if all(prompt.id in dataset.accepted_ids for prompt in prompts):
            terminal_write(f"All prompts accepted: {dataset.total_seconds / 60:.2f} minutes.")
            for orphan in dataset.orphans():
                terminal_write(f"Orphan file (inspect manually): {orphan}")
            return 0
        terminal_write(
            "Use a quiet room and consistent microphone distance. Read naturally; "
            "aim for 5–12 seconds. Controls use only r, y, n, p, e, s, and q."
        )
        recorder = MicrophoneRecorder(device)
        if microphone_check(recorder, write=terminal_write):
            guided_session(dataset, prompts, recorder, write=terminal_write)
        return 0
    except (KeyboardInterrupt, EOFError):
        terminal_write("\nStopped. Previously accepted recordings are preserved.")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Recorder error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
