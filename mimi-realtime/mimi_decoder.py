#!/usr/bin/env python3
"""Mimi decoder only - feed it your own codes (or latents), get 24 kHz audio.

The `kyutai/mimi` checkpoint ships three parts:

    encoder  : waveform (24 kHz)  -> latent (512-d, 12.5 Hz)   ~ half the params
    quantizer: latent             -> 32 discrete codes (2048 entries each)
    decoder  : latent (512-d)     -> waveform (24 kHz)

If your own converter already produces tokens/latents, you only need the last
two - the encoder can be dropped entirely (`drop_encoder=True`, the default).

Two entry points:

    decode_codes(codes)     # (K, T) int, K = 1..32, T = number of 80 ms frames
    decode_latents(latents) # (512, T) float - skips the quantizer

Both are *streaming*: state (causal conv padding + transformer KV cache) is kept
between calls, so calling them every 80 ms yields gapless audio at any block
size. Output is exactly `T * 1920` samples @ 24 kHz, and the cold start matches
a one-shot `MimiModel.decode` to float32 precision.

Why a custom encoder usually needs work: the decoder was trained against Mimi's
*own* latent space. Tokens from a foreign tokenizer (or latents from an untrained
projection) decode into noise, not speech. What works:

    * your converter emits Mimi-compatible codes (Mimi encoder, or one distilled
      against it)                -> use decode_codes() as is
    * your converter emits a different latent space -> train a linear/MLP
      projection onto Mimi's 512-d latents, or fine-tune the decoder on
      (your_latent, audio) pairs. Untrained projections do not work.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

FRAME_RATE = 12.5
SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920      # 24000 / 12.5 : samples per codec frame (80 ms)


class _ConvTStream:
    """Stateful wrapper for a causal `MimiConvTranspose1d`.

    The module itself is stateless: full convolution, then trim padding. Since
    it only ever looks at past inputs, feeding `kernel` frames of history and
    keeping the last `n * stride` output samples is an exact continuation.
    """

    def __init__(self, module):
        self.m = module
        self.kernel = int(module.conv.kernel_size[0])
        self.stride = int(module.conv.stride[0])
        self.pl, self.pr = int(module.padding_left), int(module.padding_right)
        self.state: torch.Tensor | None = None

    def reset(self) -> None:
        self.state = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, C', T * stride)"""
        n = x.shape[-1]
        if self.state is None:
            self.state = x.new_zeros(x.shape[0], x.shape[1], self.kernel)
        xin = torch.cat([self.state, x], dim=-1)
        y = self.m.conv(xin)
        y = y[..., self.pl: y.shape[-1] - self.pr]
        self.state = xin[..., -self.kernel:].contiguous()
        return y[..., -n * self.stride:]


class MimiDecoderStream:
    """Decode Mimi codes/latents to audio, block by block, with full state."""

    def __init__(self, model_or_dir: str | Path | torch.nn.Module, device: str = "cpu",
                 drop_encoder: bool = True):
        from transformers import MimiModel

        if isinstance(model_or_dir, (str, Path)):
            model = MimiModel.from_pretrained(str(model_or_dir), dtype=torch.float32)
        else:
            model = model_or_dir
        model = model.to(device).eval()

        self.model = model
        self.device = device
        self.quantizer = model.quantizer
        self.decoder = model.decoder

        cfg = model.config
        self.sample_rate = int(cfg.sampling_rate)
        self.frame_rate = float(cfg.frame_rate)
        self.frame_samples = int(round(self.sample_rate / self.frame_rate))
        self.num_quantizers = int(cfg.num_quantizers)
        self.latent_dim = int(cfg.hidden_size)

        self._init_state()
        if drop_encoder:
            model.encoder = None                     # frees ~half the parameters
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.last_latents: np.ndarray | None = None  # (D, T) from the last call

    # ------------------------------------------------------------------ state
    def _init_state(self) -> None:
        from transformers.models.mimi.modeling_mimi import (
            MimiConv1d, MimiConv1dPaddingCache, MimiConvTranspose1d, MimiResnetBlock,
        )
        self._conv1d_t = MimiConv1d
        self._resnet_t = MimiResnetBlock

        seanet = self.decoder.decoder_seanet
        self._transformer = self.decoder.decoder_transformer
        self._upsample = (_ConvTStream(self.decoder.upsample)
                          if self.decoder.upsample is not None else None)
        self._layers = [_ConvTStream(layer) if isinstance(layer, MimiConvTranspose1d) else layer
                        for layer in seanet.layers]

        conv1ds = [m for m in seanet.modules() if isinstance(m, MimiConv1d)]
        for i, m in enumerate(conv1ds):
            m.layer_idx = i
        self._cache = MimiConv1dPaddingCache(
            num_layers=len(conv1ds),
            per_layer_padding=[int(m.padding_total) for m in conv1ds],
            per_layer_padding_mode=[m.pad_mode for m in conv1ds],
            per_layer_in_channels=[int(m.in_channels) for m in conv1ds],
        )
        self._past = None

    def reset(self) -> None:
        """Forget all history - call this when you switch audio sources."""
        self._cache.padding_cache = [None] * len(self._cache.per_layer_padding)
        self._past = None
        if self._upsample is not None:
            self._upsample.reset()
        for layer in self._layers:
            if isinstance(layer, _ConvTStream):
                layer.reset()
        self.last_latents = None

    # ----------------------------------------------------------------- decode
    @torch.inference_mode()
    def decode_codes(self, codes) -> np.ndarray:
        """codes: (K, T) or (1, K, T) int -> (T * frame_samples,) float32 @ 24 kHz"""
        t = self._as_tensor(codes, dtype=torch.long)
        if t.shape[1] > self.num_quantizers:
            raise ValueError(f"got {t.shape[1]} codebooks, model has {self.num_quantizers}")
        latents = self.quantizer.decode(t)                    # (1, D, T)
        self.last_latents = latents[0].detach().cpu().numpy()
        return self._run(latents)

    @torch.inference_mode()
    def decode_latents(self, latents) -> np.ndarray:
        """latents: (D, T) or (1, D, T) float -> (T * frame_samples,) float32 @ 24 kHz

        `D` must be `self.latent_dim` (512) and in Mimi's latent space - see the
        module docstring about foreign encoders.
        """
        t = self._as_tensor(latents, dtype=torch.float32)
        if t.shape[1] != self.latent_dim:
            raise ValueError(f"expected {self.latent_dim}-d latents, got {t.shape[1]}")
        self.last_latents = t[0].detach().cpu().numpy()
        return self._run(t)

    @torch.inference_mode()
    def decode_offline(self, codes) -> np.ndarray:
        """One shot, no streaming state - handy for sanity checks / training."""
        t = self._as_tensor(codes, dtype=torch.long)
        audio = self.decoder(self.quantizer.decode(t), return_dict=True).audio_values
        return audio[0, 0].detach().cpu().numpy()

    def _as_tensor(self, x, dtype) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(np.asarray(x))
        x = x.to(self.device, dtype=dtype)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f"expected (K, T) or (1, K, T), got shape {tuple(x.shape)}")
        return x

    def _run(self, latents: torch.Tensor) -> np.ndarray:
        """latents: (1, D, T) -> (T * frame_samples,) float32"""
        h = self._upsample(latents) if self._upsample is not None else latents
        out = self._transformer(h.transpose(1, 2), past_key_values=self._past, use_cache=True)
        self._past = out.past_key_values
        x = out.last_hidden_state.transpose(1, 2)              # (1, hidden, T')
        for layer in self._layers:
            if isinstance(layer, (self._conv1d_t, self._resnet_t)):
                x = layer(x, padding_cache=self._cache)
            else:
                x = layer(x)
        audio = x[0, 0].detach().cpu().numpy().astype(np.float32)

        want = latents.shape[-1] * self.frame_samples
        if audio.shape[-1] > want:                              # trailing conv tail
            audio = audio[-want:]
        elif audio.shape[-1] < want:
            audio = np.pad(audio, (want - audio.shape[-1], 0))
        return audio


# --------------------------------------------------------------------------- CLI
def _read_codes(path: Path) -> np.ndarray:
    """(T, K) from `realtime_mimi.py --dump frames.jsonl`, or console text."""
    import json
    import re
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            rows.append([int(c) for c in json.loads(line)["codes"]])
            continue
        m = re.match(r"^(?:\d+(?:\.\d+)?s\s+)?sem\s+(\d+)\s+(.*)$", line)
        if m:
            rows.append([int(m.group(1))] + [int(v) for v in m.group(2).split()])
    if not rows:
        raise SystemExit(f"no tokens found in {path}")
    if len({len(r) for r in rows}) != 1:
        raise SystemExit("inconsistent token counts per frame")
    return np.asarray(rows, dtype=np.int64).T                      # (K, T)


def main() -> int:
    ap = argparse.ArgumentParser(description="Decode Mimi codes with the decoder only.")
    ap.add_argument("--model-dir", default=str(Path(__file__).resolve().parent / "mimi"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log", help="tokens from --dump frames.jsonl or the console log")
    ap.add_argument("--codes", help='single frame, e.g. "1049 1700 1626 ..." (32 ids)')
    ap.add_argument("--random", type=int, metavar="T", help="synthesize T frames of random codes")
    ap.add_argument("--offline", action="store_true", help="bypass the streaming state")
    ap.add_argument("--out", default="decoder_out.wav")
    args = ap.parse_args()

    dec = MimiDecoderStream(args.model_dir, device=args.device)
    print(f"decoder: {dec.latent_dim}-d latents, {dec.num_quantizers} codebooks, "
          f"{dec.frame_rate} Hz frames, {dec.frame_samples} samples/frame @ {dec.sample_rate} Hz")

    if args.log:
        codes = _read_codes(Path(args.log))
        print(f"codes: {codes.shape} -> {codes.shape[-1] / dec.frame_rate:.2f} s")
    elif args.codes:
        ids = [int(v) for v in args.codes.replace(",", " ").split()]
        codes = np.asarray(ids, dtype=np.int64)[:, None]            # (K, 1)
        print(f"codes: {codes.shape} -> {1 / dec.frame_rate:.2f} s")
    elif args.random:
        codes = np.random.randint(0, 2048, size=(dec.num_quantizers, args.random), dtype=np.int64)
        print(f"random codes: {codes.shape} -> {args.random / dec.frame_rate:.2f} s "
              f"(noise by construction, useful only to test plumbing)")
    else:
        ap.print_help()
        return 1

    if args.offline:
        audio = dec.decode_offline(codes)
    else:
        chunks = []
        for i in range(codes.shape[-1]):                           # one frame at a time
            chunks.append(dec.decode_codes(codes[:, i: i + 1]))
        audio = np.concatenate(chunks)

    from scipy.io import wavfile
    wavfile.write(args.out, dec.sample_rate, np.clip(audio, -1.0, 1.0))
    print(f"wrote {args.out}  ({len(audio)} samples, {len(audio) / dec.sample_rate:.2f} s, "
          f"peak {np.abs(audio).max():.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
