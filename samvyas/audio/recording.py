"""Bounded microphone capture and a guided terminal recording session."""

from __future__ import annotations

import os
import select
import sys
import time
from collections.abc import Callable
from ctypes import byref, c_uint32, c_void_p

import numpy as np

from samvyas.data.voice_dataset import SAMPLE_RATE, VoiceDataset, VoicePrompt, audio_quality

_STD_OUTPUT_HANDLE = -11
_INVALID_HANDLE_VALUE = c_void_p(-1).value


def configure_unicode_terminal() -> None:
    """Configure a Windows terminal for Bengali text input and output.

    Windows uses UTF-16 internally. Setting both console code pages to UTF-8
    and writing prompts with :func:`terminal_write` avoids legacy code-page
    corruption in PowerShell and Command Prompt. Non-console output keeps the
    normal Python UTF-8 stream behavior.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except (AttributeError, OSError):
        # A redirected stream or non-Windows Python build needs no special setup.
        return


def terminal_write(message: str) -> None:
    """Print Unicode directly to a Windows console, preserving Bengali glyphs."""
    text = str(message)
    if os.name == "nt" and sys.stdout.isatty():
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(_STD_OUTPUT_HANDLE)
            mode = c_uint32()
            if handle not in (None, 0, _INVALID_HANDLE_VALUE) and kernel32.GetConsoleMode(
                handle, byref(mode)
            ):
                written = c_uint32()
                if kernel32.WriteConsoleW(handle, f"{text}\n", len(text) + 1, byref(written), None):
                    return
        except (AttributeError, OSError):
            pass
    print(text, flush=True)


def wait_for_enter(timeout: float) -> None:
    """Wait for Enter with a deadline, without leaving a blocked input thread."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.name == "nt":
            import msvcrt

            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key == "\x03":
                    raise KeyboardInterrupt
                if key in ("\r", "\n"):
                    return
        elif select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.readline() == "":
                raise EOFError
            return
        time.sleep(0.02)


class MicrophoneRecorder:
    """Capture (T,) float32 at 16 kHz, capped independently of terminal input."""

    def __init__(self, device: int | str | None = None) -> None:
        self.device = device

    def record(
        self,
        max_seconds: float = 20.0,
        wait: Callable[[float], None] = wait_for_enter,
    ) -> np.ndarray:
        """Start capture immediately; `wait` returns on Enter or the time limit."""
        import sounddevice as sd

        try:
            sd.check_input_settings(
                device=self.device, channels=1, dtype="float32", samplerate=SAMPLE_RATE
            )
        except sd.PortAudioError as exc:
            raise RuntimeError(f"Microphone does not support mono 16 kHz capture: {exc}") from exc
        maximum = int(max_seconds * SAMPLE_RATE)
        if maximum <= 0:
            raise ValueError("Recording duration must be positive.")
        chunks: list[np.ndarray] = []
        failures: list[str] = []
        captured = 0

        def callback(indata: np.ndarray, frames: int, timing: object, status: object) -> None:
            nonlocal captured
            if status:
                failures.append(str(status))
                raise sd.CallbackAbort
            count = min(frames, maximum - captured)
            chunks.append(indata[:count, 0].copy())
            captured += count
            if captured >= maximum:
                raise sd.CallbackStop

        try:
            with sd.InputStream(
                device=self.device,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=callback,
            ):
                wait(max_seconds)
        except sd.PortAudioError as exc:
            raise RuntimeError(f"Microphone capture failed: {exc}") from exc
        if failures:
            raise RuntimeError(f"Microphone stream failed: {'; '.join(failures)}")
        if not chunks:
            raise RuntimeError("No microphone samples captured; retry or select another device.")
        return np.concatenate(chunks).astype(np.float32, copy=False)

    def play(self, waveform: np.ndarray) -> None:
        """Play a take through the default output device without processing."""
        import sounddevice as sd

        try:
            sd.play(waveform, SAMPLE_RATE)
            sd.wait()
        except sd.PortAudioError as exc:
            raise RuntimeError(f"Audio playback failed: {exc}") from exc
        finally:
            sd.stop()


def countdown(
    write: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """A three-second preparation countdown."""
    for second in (3, 2, 1):
        write(str(second))
        sleep(1)


def report_take(waveform: np.ndarray, write: Callable[[str], None] = print) -> None:
    """Display measurements and actionable quality warnings."""
    quality = audio_quality(waveform)
    write(
        f"Duration: {len(waveform) / SAMPLE_RATE:.2f}s | "
        f"Peak: {quality['peak']:.3f} | RMS: {quality['rms']:.4f}"
    )
    for warning in quality["warnings"]:
        write(f"Warning: {warning}")


def guided_session(
    dataset: VoiceDataset,
    prompts: list[VoicePrompt],
    recorder: MicrophoneRecorder,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    prepare: Callable[[], None] = countdown,
) -> None:
    """Record/review/confirm takes; unsaved takes are discarded on quit/interruption."""
    for orphan in dataset.orphans():
        write(f"Orphan file (not counted; inspect manually): {orphan}")
    pending = [prompt for prompt in prompts if prompt.id not in dataset.accepted_ids]
    write(
        f"Accepted: {len(dataset.records)} clips, {dataset.total_seconds / 60:.2f} minutes. "
        f"Remaining: {len(pending)} prompts."
    )
    try:
        for index, prompt in enumerate(pending, 1):
            transcript = prompt.text
            take = None
            write(f"\n[{index}/{len(pending)}] {prompt.id} | Delivery: {prompt.delivery}")
            write(prompt.text)
            while True:
                if take is None:
                    start = read("r=record, s=skip, q=quit: ").strip().lower()
                    if start == "q":
                        return
                    if start == "s":
                        break
                    if start != "r":
                        continue
                    prepare()
                    write("Recording — speak now. Enter to stop (maximum 20 seconds).")
                    try:
                        take = recorder.record()
                        report_take(take, write)
                    except (RuntimeError, ValueError, OSError) as exc:
                        write(f"Recording failed: {exc}")
                        take = None
                        continue
                action = read("y=save, n=record again, p=play, e=edit text, s=skip, q=quit: ")
                action = action.strip().lower()
                if action == "q":
                    return
                if action == "s":
                    break
                if action == "n":
                    take = None
                elif action == "p":
                    try:
                        recorder.play(take)
                    except (RuntimeError, OSError) as exc:
                        write(f"Playback failed: {exc}")
                elif action == "e":
                    replacement = read("Exact words spoken (Bengali): ").strip()
                    if replacement:
                        transcript = replacement
                    else:
                        write("Transcript cannot be empty.")
                elif action == "y":
                    write(f"Transcript: {transcript}")
                    record = dataset.accept(prompt, transcript, take)
                    write(
                        f"Saved clip {record['clip_id']} to {dataset.root / record['audio_path']}."
                    )
                    write(
                        f"Accepted: {len(dataset.records)} clips, "
                        f"{dataset.total_seconds / 60:.2f} minutes."
                    )
                    break
    except (KeyboardInterrupt, EOFError):
        write("\nSession stopped. Accepted takes are saved; the current take was discarded.")


def microphone_check(
    recorder: MicrophoneRecorder,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> bool:
    """Record/play a three-second unsaved check before collecting prompts."""
    while True:
        start = read("Microphone check: r=record, q=quit: ").strip().lower()
        if start == "q":
            return False
        if start != "r":
            continue
        countdown(write)
        write("Speak now...")
        audio = recorder.record(3, wait=time.sleep)
        report_take(audio, write)
        recorder.play(audio)
        answer = read("Audio clear? y=continue, n=repeat, q=quit: ").strip().lower()
        if answer == "y":
            return True
        if answer == "q":
            return False
