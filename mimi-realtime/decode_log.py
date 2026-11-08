#!/usr/bin/env python3
"""Decode a Mimi token log (the text printed by `--show tokens`) back into audio.

    python decode_log.py                     # reads mimi_log.txt -> decoded_stream.wav
    python decode_log.py --log other.txt --out other.wav
    python decode_log.py --stride 1          # no time stretching
    python decode_log.py --print-every 8     # stride is inferred from the timestamps

Two things the naive version of this script gets wrong:
  1. `re.findall(r"(?<!\\.)\\b\\d+\\b(?!s)")` also matches the WHOLE-SECOND part of the
     timestamp ("0.00s" -> 0, "10.24s" -> 10, because the look-behind only rejects the
     part after the dot), so every line yields 33 numbers instead of 32 and
     `len(tokens) == 32` never matches -> zero frames.
  2. Printed lines are throttled: with `--print-every 8` the log holds every 8th frame
     (0.64 s apart), so decoding them back-to-back plays 8x too fast. Each frame is
     repeated `stride` times here to restore the real duration (content of the
     skipped frames is unrecoverable).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch

FRAME_RATE = 12.5
SAMPLE_RATE = 24000
FRAME_SAMPLES = int(SAMPLE_RATE / FRAME_RATE)  # 1920
HERE = Path(__file__).resolve().parent

# "  10.24s sem 1439  2029 1149 ..."  (1 or 2 spaces after the timestamp, both seen)
LINE_RE = re.compile(r"^\s*(?P<t>\d+(?:\.\d+)?)s\s+sem\s+(?P<sem>\d+)\s+(?P<rest>\d[\d\s]*)$")


def parse(path: Path) -> tuple[list[list[int]], list[float]]:
    frames: list[list[int]] = []
    times: list[float] = []
    bad = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        m = LINE_RE.match(raw)
        if m is None:
            bad += 1
            continue
        acoustic = [int(x) for x in m["rest"].split()]
        frames.append([int(m["sem"])] + acoustic)
        times.append(float(m["t"]))
    if bad:
        print(f"  skipped {bad} unparsable line(s)")
    return frames, times


def parse_jsonl(path: Path) -> tuple[list[list[int]], list[float]]:
    """Every frame, written by `realtime_mimi.py --dump frames.jsonl`."""
    import json
    frames: list[list[int]] = []
    times: list[float] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            rec = json.loads(raw)
            frames.append([int(c) for c in rec["codes"]])
            times.append(float(rec["t"]))
    return frames, times


def main() -> int:
    ap = argparse.ArgumentParser(description="Decode logged Mimi tokens to a WAV file.")
    ap.add_argument("--log", default=str(HERE / "mimi_log.txt"))
    ap.add_argument("--out", default=str(HERE / "decoded_stream.wav"))
    ap.add_argument("--model-dir", default=str(HERE / "mimi"))
    ap.add_argument("--stride", default="auto",
                    help="hold each logged frame N times to restore timing "
                         "(auto = infer from timestamps, 1 = decode back-to-back)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"no such log: {log_path}")
        return 1

    frames, times = parse_jsonl(log_path) if log_path.suffix == ".jsonl" else parse(log_path)
    if not frames:
        print("no frames parsed - is this really `--show tokens` output?")
        return 1

    widths = {len(f) for f in frames}
    if len(widths) != 1:
        # keep the most common width, drop the rest (usually a copy/paste slip)
        from collections import Counter
        common = Counter(len(f) for f in frames).most_common(1)[0][0]
        dropped = sum(1 for f in frames if len(f) != common)
        print(f"warning: mixed token counts {sorted(widths)}; "
              f"keeping {common}-wide frames and dropping {dropped} line(s)")
        keep = [i for i, f in enumerate(frames) if len(f) == common]
        frames = [frames[i] for i in keep]
        times = [times[i] for i in keep]
        widths = {common}
    n_q = widths.pop()
    print(f"parsed {len(frames)} frames x {n_q} tokens from {log_path.name}")

    # --- timing -----------------------------------------------------------------
    stride = 1
    if args.stride == "auto":
        if len(times) > 2:
            dt = float(np.median(np.diff(times)))
            stride = max(1, int(round(dt * FRAME_RATE)))
            print(f"  logged every {dt:.2f} s -> repeat each frame {stride}x")
        else:
            print("  too few timestamps, no time stretching")
    else:
        stride = max(1, int(args.stride))

    codes = torch.tensor(frames, dtype=torch.long).T.unsqueeze(0)   # (1, K, T_logged)
    if stride > 1:
        codes = codes.repeat_interleave(stride, dim=-1)              # (1, K, T_logged*stride)
    print(f"  codes: {tuple(codes.shape)}  ({codes.shape[-1] / FRAME_RATE:.2f} s at 12.5 Hz)")

    # --- decode -----------------------------------------------------------------
    from transformers import MimiModel

    model = MimiModel.from_pretrained(args.model_dir, dtype=torch.float32).to(args.device).eval()
    if codes.shape[1] != model.config.num_quantizers:
        print(f"model expects {model.config.num_quantizers} codebooks, log has {codes.shape[1]}")
        return 1

    with torch.inference_mode():
        audio = model.decode(codes.to(args.device), return_dict=True).audio_values

    wav = audio[0, 0].detach().cpu().numpy()
    peak = float(np.abs(wav).max())
    print(f"  decoded {len(wav)} samples ({len(wav) / SAMPLE_RATE:.2f} s), peak {peak:.3f}"
          f"  ({len(wav) // codes.shape[-1]} samples per codec frame)")

    from scipy.io import wavfile
    if peak > 0:
        wav = wav / peak * 0.95          # model output can exceed full scale
    wavfile.write(args.out, SAMPLE_RATE, (np.clip(wav, -1, 1) * 32767).astype(np.int16))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
