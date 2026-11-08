#!/usr/bin/env python3
"""
realtime_mimi.py - run Kyutai Mimi (neural audio codec) in real time.

    mic  ->  [ streaming encoder | quantization | streaming decoder ]  ->  speakers

Modes
-----
    python realtime_mimi.py                      # live mic -> speakers
    python realtime_mimi.py --wav in.wav --out out.wav   # stream a file through
    python realtime_mimi.py --bench              # measure per-block latency
    python realtime_mimi.py --list-devices

How the streaming works
-----------------------
* Encoder: Mimi is causal, so a block can be encoded on its own as long as the
  causal-convolution padding and the transformer KV cache are carried over.
  That is exactly what `use_streaming=True` + `padding_cache` +
  `encoder_past_key_values` do in transformers.  Verified token-identical to a
  one-shot encode.
* Decoder: transformers only forwards the transformer KV cache, so the conv
  stack is driven here with full state as well - `MimiConv1d` via its
  `MimiConv1dPaddingCache`, the transposed/upsampling convs via a small
  left-context wrapper.  The streamed output matches an offline decode to
  ~117 dB SNR (float32 noise floor), so no boundary artifacts.
* `--decode-mode overlap` is an older, chunked fallback (context frames +
  trim) in case a future transformers version moves these internals.

Audio never leaves the machine and nothing is stored unless --out is given.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from mimi_decoder import MimiDecoderStream

MODEL_SR = 24000
FRAME_RATE = 12.5
FRAME_SAMPLES = int(round(MODEL_SR / FRAME_RATE))  # 1920 samples == 80 ms


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_model(model_dir: str, device: str, num_quantizers: int | None):
    import torch
    from transformers import MimiModel

    model_dir = str(Path(model_dir).expanduser().resolve())
    if not Path(model_dir, "config.json").exists():
        raise SystemExit(f"No config.json in {model_dir!r} - is that the Mimi folder?")

    try:
        model = MimiModel.from_pretrained(model_dir, dtype=torch.float32)
    except TypeError:  # older transformers
        model = MimiModel.from_pretrained(model_dir, torch_dtype=torch.float32)

    model = model.to(device).eval()
    torch.set_grad_enabled(False)

    cfg = model.config
    if num_quantizers is None:
        num_quantizers = cfg.num_quantizers
    if not 1 <= num_quantizers <= cfg.num_quantizers:
        raise SystemExit(f"--num-quantizers must be 1..{cfg.num_quantizers}")

    n = sum(p.numel() for p in model.parameters())
    print(f"loaded {model_dir}")
    print(f"  device={device}  params={n/1e6:.1f}M  quantizers={num_quantizers}/{cfg.num_quantizers}"
          f"  bitrate={num_quantizers * 10 * FRAME_RATE / 1000:.2f} kbit/s")
    return model, num_quantizers


class StreamingMimi:
    """Stateful block-by-block Mimi encode/decode."""

    def __init__(self, model, num_quantizers: int, ctx_frames: int = 6,
                 device: str = "cpu", decode_mode: str = "state"):
        import torch
        self.torch = torch
        self.model = model
        self.num_quantizers = num_quantizers
        self.ctx_frames = ctx_frames
        self.device = device
        self.decode_mode = decode_mode

        self._padding_cache = None      # MimiConv1dPaddingCache (causal conv state)
        self._enc_past = None           # encoder transformer KV cache
        self._code_history = None       # (1, K, ctx) tail of emitted codes
        self.frames_seen = 0

        self.last_codes = None          # (K, n) int32 tokens from the last block
        self.last_latents = None        # (hidden, n) float latents from the last block

        self.decoder = None
        if decode_mode == "state":
            try:
                self.decoder = MimiDecoderStream(model, device=device, drop_encoder=False)
            except Exception as exc:                      # pragma: no cover
                print(f"  ! stateful decoder unavailable ({exc}); falling back to overlap decode")
                self.decode_mode = "overlap"
        if self.decode_mode == "overlap":
            print(f"  decode mode: overlap (ctx={ctx_frames} frames)")
        else:
            print("  decode mode: stateful (conv cache + KV cache, bit-exact)")

    def reset(self):
        self._padding_cache = None
        self._enc_past = None
        self._code_history = None
        self.last_codes = None
        self.last_latents = None
        self.frames_seen = 0
        if self.decoder is not None:
            self.decoder.reset()

    def step(self, block: np.ndarray) -> np.ndarray:
        with self.torch.inference_mode():
            return self._step(block)

    def _step(self, block: np.ndarray) -> np.ndarray:
        """block: float32 mono @24 kHz, length a multiple of 80 ms. Returns the
        decoded audio for exactly that block."""
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32))
        x = x.view(1, 1, -1).to(self.device)

        enc = self.model.encode(
            x,
            num_quantizers=self.num_quantizers,
            use_streaming=True,
            padding_cache=self._padding_cache,
            encoder_past_key_values=self._enc_past,
            return_dict=True,
        )
        self._padding_cache = enc.padding_cache
        self._enc_past = enc.encoder_past_key_values

        codes = enc.audio_codes            # (1, K, frames)
        n_new = codes.shape[-1]
        if n_new == 0:
            return np.zeros(0, dtype=np.float32)

        want = n_new * FRAME_SAMPLES

        if self.decoder is not None:
            # stateful decoder: codes -> 24 kHz audio, one block at a time
            out = self.decoder.decode_codes(codes)
            latents = self.decoder.last_latents
        else:
            # fallback: prepend context frames and keep the tail
            if self._code_history is not None:
                codes_in = torch.cat([self._code_history, codes], dim=-1)
            else:
                codes_in = codes
            audio = self.model.decode(codes_in, return_dict=True).audio_values[0, 0]
            self._code_history = codes_in[..., -self.ctx_frames:] if self.ctx_frames > 0 else None
            out = audio[-want:].detach().cpu().numpy().astype(np.float32, copy=False)
            latents = None

        if out.shape[-1] < want:
            out = np.pad(out, (want - out.shape[-1], 0))

        self.last_codes = codes[0].detach().cpu().numpy().astype(np.int32)
        self.last_latents = latents
        self.frames_seen += n_new
        return out.astype(np.float32, copy=False)


class FrameView:
    """Live view of what the codec is actually emitting, frame by frame.

    `codes`      - `(K, n)` discrete tokens, one row per codebook
                   (row 0 = semantic RVQ, rows 1.. = acoustic RVQ, 2048 entries each)
    `latents`    - `(hidden, n)` continuous 512-d latents before quantization
    """

    BARS = " .:-=+*#%@"

    def __init__(self, mode: str = "tokens", every: int = 4, dump: str | None = None):
        self.mode = mode          # none | tokens | latents | both
        self.every = max(1, every)
        self.n = 0
        self._dump = open(dump, "w", encoding="utf-8") if dump else None
        if self._dump:
            print(f"  writing per-frame tokens to {dump}")

    def close(self):
        if self._dump:
            self._dump.close()
            self._dump = None

    def __call__(self, mimi):                                    # (K,n) int32 / (D,n) float32
        codes = mimi.last_codes
        if codes is None or codes.shape[1] == 0:
            return
        lat = mimi.last_latents

        for j in range(codes.shape[1]):
            t = self.n / FRAME_RATE
            if self._dump:
                rec = {"frame": self.n, "t": round(t, 3),
                       "codes": [int(c) for c in codes[:, j]]}
                if lat is not None:
                    rec["latent_norm"] = round(float(np.linalg.norm(lat[:, j])), 4)
                self._dump.write(json.dumps(rec) + "\n")

            if self.n % self.every == 0:
                if self.mode in ("tokens", "both"):
                    print(self._tokens_line(t, codes[:, j]), flush=True)
                if self.mode in ("latents", "both"):
                    print(self._latent_line(t, None if lat is None else lat[:, j]), flush=True)
            self.n += 1

        if self._dump:
            self._dump.flush()

    # ------------------------------------------------------------------ format
    @staticmethod
    def _cols() -> int:
        import shutil
        return shutil.get_terminal_size((100, 24)).columns

    def _tokens_line(self, t: float, col: np.ndarray) -> str:
        """All codebooks for one frame, wrapped to the terminal width."""
        sem = int(col[0])
        prefix = f"{t:7.2f}s  sem {sem:4d}  "
        rest = [f"{int(c):04d}" for c in col[1:]]
        per_row = max(4, (self._cols() - len(prefix)) // 5)
        rows = [" ".join(rest[i: i + per_row]) for i in range(0, len(rest), per_row)] or [""]
        return prefix + ("\n" + " " * len(prefix)).join(rows)

    def _latent_line(self, t: float, vec: np.ndarray | None) -> str:
        if vec is None:
            return f"{t:7.2f}s  latents unavailable (overlap decode mode)"
        bins = 32
        step = max(1, len(vec) // bins)
        v = vec[: (len(vec) // step) * step].reshape(-1, step).mean(axis=1)
        peak = float(np.abs(v).max()) + 1e-9
        spark = "".join(self.BARS[min(len(self.BARS) - 1, int(abs(x) / peak * (len(self.BARS) - 1)))]
                        for x in v)
        return (f"{t:7.2f}s  latents dim={len(vec)} |x|={np.linalg.norm(vec):6.2f}"
                f"  [{spark}]")


# --------------------------------------------------------------------------- #
# live audio
# --------------------------------------------------------------------------- #
class MicIn:
    """Microphone -> fixed-size float32 blocks in a queue (oldest dropped)."""

    def __init__(self, block_samples: int, samplerate: int = MODEL_SR, device=None):
        import sounddevice as sd
        self.sd = sd
        self.block_samples = block_samples
        self.samplerate = samplerate
        self.ratio = samplerate // MODEL_SR  # 1 or 2
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
        self._buf = np.zeros(0, dtype=np.float32)
        self.dropped = 0
        self.level = 0.0

        # ask PortAudio for ~1 block of audio per callback, a bit less so we refill often
        blocksize = max(256, block_samples * self.ratio // 2)
        self.stream = sd.InputStream(
            samplerate=samplerate, channels=1, dtype="float32",
            blocksize=blocksize, device=device, callback=self._cb,
        )

    def _cb(self, indata, frames, time_info, status):
        x = indata[:, 0].copy()
        if self.ratio > 1:  # naive boxcar decimation, fine for speech
            x = x[: len(x) - len(x) % self.ratio].reshape(-1, self.ratio).mean(axis=1)
        self._buf = np.concatenate([self._buf, x]) if self._buf.size else x
        while self._buf.size >= self.block_samples:
            block, self._buf = self._buf[: self.block_samples], self._buf[self.block_samples:]
            try:
                self.queue.put_nowait(block)
            except queue.Full:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
                self.queue.put_nowait(block)
                self.dropped += 1
        if self._buf.size:
            self.level = 0.8 * self.level + 0.2 * float(np.sqrt(np.mean(self._buf ** 2)))

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()


class Speaker:
    """float32 blocks -> output device, with bounded latency."""

    def __init__(self, samplerate: int = MODEL_SR, device=None):
        import sounddevice as sd
        self.sd = sd
        self.samplerate = samplerate
        self.ratio = samplerate // MODEL_SR
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
        self.level = 0.0
        self._tail = np.zeros(0, dtype=np.float32)

        self.stream = sd.OutputStream(
            samplerate=samplerate, channels=1, dtype="float32",
            blocksize=0, device=device, callback=self._cb,
        )

    def _cb(self, outdata, frames, time_info, status):
        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        while filled < frames:
            if self._tail.size == 0:
                try:
                    self._tail = self.queue.get_nowait()
                except queue.Empty:
                    break
            take = min(frames - filled, self._tail.size)
            out[filled: filled + take] = self._tail[:take]
            self._tail = self._tail[take:]
            filled += take
        if self.ratio > 1:  # linear upsample 24k -> 48k
            out = np.interp(
                np.arange(frames * self.ratio) / self.ratio,
                np.arange(frames), out,
            ).astype(np.float32)
        outdata[:, 0] = out
        self.level = 0.8 * self.level + 0.2 * float(np.sqrt(np.mean(out ** 2)))

    def push(self, audio: np.ndarray):
        if audio.size == 0:
            return
        while True:
            try:
                self.queue.put_nowait(audio)
                return
            except queue.Full:      # keep latency bounded: drop the oldest audio
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    return

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()


def _dbfs(x: float) -> str:
    if x <= 1e-8:
        return " -inf"
    return f"{20 * np.log10(x):5.1f}"


def run_live(mimi: StreamingMimi, args):
    import sounddevice as sd

    block_frames = max(1, int(round(args.block_ms / (1000 / FRAME_RATE))))
    block_samples = block_frames * FRAME_SAMPLES

    in_rate, out_rate = MODEL_SR, MODEL_SR
    try:
        mic = MicIn(block_samples, MODEL_SR, args.input_device)
    except sd.PortAudioError:
        in_rate = 2 * MODEL_SR
        print(f"24 kHz capture refused, falling back to {in_rate} Hz (decimating 2:1)")
        mic = MicIn(block_samples, in_rate, args.input_device)
    try:
        spk = Speaker(MODEL_SR, args.output_device)
    except sd.PortAudioError:
        out_rate = 2 * MODEL_SR
        print(f"24 kHz playback refused, falling back to {out_rate} Hz (interpolating 1:2)")
        spk = Speaker(out_rate, args.output_device)

    print(f"\nblock = {block_frames} frame(s) = {block_samples} samples"
          f" = {1000*block_samples/MODEL_SR:.0f} ms | ctx = {mimi.ctx_frames} frame(s)")
    print(f"latency floor ~ {1000*block_samples/MODEL_SR:.0f} ms + device buffers")
    if args.show in ("tokens", "both"):
        print(f"token stream: {mimi.num_quantizers} codebooks x {FRAME_RATE} Hz, ids 0..2047"
              f" (sem = semantic RVQ, then acoustic RVQ)")
        print(f"printing 1 frame in every {max(1, args.print_every)}; Ctrl+C to stop.\n")
    else:
        print("Ctrl+C to stop.\n")

    view = None
    if args.show != "none" or args.dump:
        view = FrameView(args.show, args.print_every, args.dump)

    mic.start()
    spk.start()

    total_ms, n_blocks = 0.0, 0
    wall_start = time.perf_counter()
    try:
        while True:
            try:
                block = mic.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            audio = mimi.step(block)
            dt = (time.perf_counter() - t0) * 1000
            spk.push(audio)
            if view is not None:
                view(mimi)
            total_ms += dt
            n_blocks += 1
            if n_blocks % 4 == 0:
                rt = dt / (1000 * block_samples / MODEL_SR)
                line = (f"  {mimi.frames_seen:6d} frames | mic {_dbfs(mic.level)} dBFS"
                        f" | out {_dbfs(spk.level)} dBFS"
                        f" | {dt:5.1f} ms/block ({rt:4.2f}x realtime)"
                        f" | drop {mic.dropped:3d}")
                if view is None:
                    print("\r" + line, end="", flush=True)
                elif n_blocks % 8 == 0:
                    print(line, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
        spk.stop()
        if view is not None:
            view.close()
        wall = time.perf_counter() - wall_start
        print()
        if n_blocks:
            print(f"\nstopped: {n_blocks} blocks in {wall:.1f} s"
                  f" | mean {total_ms/n_blocks:.1f} ms/block"
                  f" | {100*(total_ms/1000)/wall:.0f}% of one core"
                  f" | {mimi.frames_seen} codec frames ({mimi.frames_seen/FRAME_RATE:.1f} s of audio)")


# --------------------------------------------------------------------------- #
# file mode / benchmark
# --------------------------------------------------------------------------- #
def _read_wav(path: str) -> np.ndarray:
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    if data.dtype.kind == "i":
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    elif data.dtype.kind == "u":
        data = (data.astype(np.float32) - 128) / 128.0
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != MODEL_SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), MODEL_SR)
        data = resample_poly(data, MODEL_SR // g, int(sr) // g).astype(np.float32)
        print(f"resampled {path}: {sr} Hz -> {MODEL_SR} Hz")
    return data.astype(np.float32)


def run_wav(mimi: StreamingMimi, args):
    from scipy.io import wavfile

    audio = _read_wav(args.wav)
    block_frames = max(1, int(round(args.block_ms / (1000 / FRAME_RATE))))
    block_samples = block_frames * FRAME_SAMPLES
    n_blocks = int(np.ceil(len(audio) / block_samples))
    audio = np.pad(audio, (0, n_blocks * block_samples - len(audio)))

    out = np.zeros_like(audio)
    times = []
    print(f"\n{len(audio)/MODEL_SR:.2f} s of audio, {n_blocks} blocks of {block_ms_label(block_samples)}")
    view = FrameView(args.show, args.print_every, args.dump) if args.show != "none" or args.dump else None
    for i in range(n_blocks):
        block = audio[i * block_samples:(i + 1) * block_samples]
        t0 = time.perf_counter()
        dec = mimi.step(block)
        times.append((time.perf_counter() - t0) * 1000)
        if view is not None:
            view(mimi)
        n = min(len(dec), len(block))
        out[i * block_samples:i * block_samples + n] = dec[:n]
        if view is None and ((i + 1) % 10 == 0 or i + 1 == n_blocks):
            print(f"\r  {i+1}/{n_blocks} blocks | {np.mean(times):5.1f} ms/block"
                  f" ({np.mean(times)/(1000*block_samples/MODEL_SR):4.2f}x realtime)", end="", flush=True)
    if view is not None:
        view.close()
    print()

    if args.out:
        wavfile.write(args.out, MODEL_SR, np.clip(out, -1.0, 1.0))
        print(f"wrote {args.out}")

    if args.check:
        # same audio, one shot, no chunking
        import torch
        x = torch.from_numpy(audio).view(1, 1, -1).to(mimi.device)
        ref = mimi.model(x).audio_values[0, 0].cpu().numpy()
        n = min(len(ref), len(out))
        ref, y = ref[:n], out[:n]
        noise = ref - y
        snr = 10 * np.log10(max(np.sum(ref ** 2), 1e-12) / max(np.sum(noise ** 2), 1e-12))
        print(f"streamed vs offline decode: SNR = {snr:.1f} dB"
              f" (peak ref {np.max(np.abs(ref)):.3f}, peak stream {np.max(np.abs(y)):.3f})")
        if args.out:
            wavfile.write(str(args.out).replace(".wav", "_offline.wav"), MODEL_SR,
                          np.clip(ref, -1, 1))


def run_bench(mimi: StreamingMimi, args):
    block_frames = max(1, int(round(args.block_ms / (1000 / FRAME_RATE))))
    block_samples = block_frames * FRAME_SAMPLES
    x = (0.05 * np.random.randn(block_samples)).astype(np.float32)
    warm = 5
    n = 30
    print(f"\nbenchmark: {n} blocks of {block_ms_label(block_samples)}, warmup {warm}")
    for _ in range(warm):
        mimi.step(x)
    mimi.reset()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        mimi.step(x)
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    block_ms = 1000 * block_samples / MODEL_SR
    print(f"  mean  {t.mean():6.2f} ms   p50 {np.percentile(t,50):6.2f}   "
          f"p95 {np.percentile(t,95):6.2f}   max {t.max():6.2f}")
    print(f"  realtime factor: {t.mean()/block_ms:.2f}x "
          f"({'OK' if t.mean() < block_ms else 'TOO SLOW - use a GPU or bigger --block-ms'})")


def block_ms_label(samples: int) -> str:
    return f"{1000*samples/MODEL_SR:.0f} ms"


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Run Kyutai Mimi as a realtime audio codec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", default=str(Path(__file__).parent / "mimi"),
                   help="folder with config.json + model.safetensors")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--block-ms", type=float, default=160,
                   help="audio per streaming step, rounded to 80 ms frames")
    p.add_argument("--ctx", type=int, default=6,
                   help="with --decode-mode overlap: context frames prepended to each block")
    p.add_argument("--decode-mode", default="state", choices=["state", "overlap"],
                   help="stateful = exact streaming decoder, overlap = chunked fallback")
    p.add_argument("--num-quantizers", type=int, default=None,
                   help="use fewer codebooks (lower bitrate, lower quality)")
    p.add_argument("--threads", type=int, default=0,
                   help="torch CPU threads (0 = torch default)")
    p.add_argument("--wav", help="stream this wav file instead of the microphone")
    p.add_argument("--out", help="write decoded audio to this wav file (with --wav)")
    p.add_argument("--check", action="store_true",
                   help="with --wav: compare streaming output against an offline decode")
    p.add_argument("--bench", action="store_true", help="benchmark only, no audio I/O")
    p.add_argument("--show", default="tokens", choices=["none", "tokens", "latents", "both"],
                   help="what to print per frame: discrete tokens, latent vectors, or neither")
    p.add_argument("--print-every", type=int, default=8,
                   help="with --show: print 1 frame in every N frames")
    p.add_argument("--dump", help="write per-frame tokens (+ latent norms) to a JSONL file")
    p.add_argument("--input-device", default=None, help="index or name (see --list-devices)")
    p.add_argument("--output-device", default=None, help="index or name (see --list-devices)")
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")

    model, nq = load_model(args.model_dir, args.device, args.num_quantizers)
    mimi = StreamingMimi(model, nq, ctx_frames=args.ctx, device=args.device,
                         decode_mode=args.decode_mode)

    if args.bench:
        run_bench(mimi, args)
    elif args.wav:
        run_wav(mimi, args)
    else:
        run_live(mimi, args)


if __name__ == "__main__":
    sys.exit(main())
