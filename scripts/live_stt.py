"""Live microphone STT demo: default mic -> transcript + 768-dim vector + PCA.

Flow (diagram path: Mic 16kHz -> DeepFilterNet v2 -> IndicConformer CTC):
    record fixed clip from default input -> STTPipeline.process()
    -> print transcript, hidden-state stats, sklearn PCA-2D of frame latents.

Usage:
    .venv/Scripts/python.exe scripts/live_stt.py                 # 5s clips, loop till Ctrl+C
    .venv/Scripts/python.exe scripts/live_stt.py --once          # single clip then exit
    .venv/Scripts/python.exe scripts/live_stt.py --secs 4        # custom clip length
    .venv/Scripts/python.exe scripts/live_stt.py --no-denoise    # skip DeepFilter
    .venv/Scripts/python.exe scripts/live_stt.py --file assets/audio/vocaltest.m4a  # mic-less dry run

VRAM guard (RTX 2050 4GB): short clips (default 5s), sequential inference,
torch.cuda.empty_cache() after every turn.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from samvyas.audio.denoiser.deepfilter import DeepFilterDenoiser
from samvyas.audio.io import TARGET_SR, load_audio
from samvyas.models.encoders.semantic.indic_asr import IndicASREncoder
from samvyas.pipelines.stt_pipeline import STTPipeline


def record_clip(secs: float, device=None) -> torch.Tensor:
    """Record `secs` seconds of mono 16kHz float32 audio from the default mic."""
    import sounddevice as sd

    print(f"[MIC] Recording {secs:.1f}s @ {TARGET_SR}Hz — speak now...")
    frames = int(secs * TARGET_SR)
    audio = sd.rec(frames, samplerate=TARGET_SR, channels=1, dtype="float32", device=device)
    sd.wait()
    wav = torch.from_numpy(audio.reshape(-1).copy())
    peak = wav.abs().max().item()
    print(f"[MIC] Captured {wav.shape[0] / TARGET_SR:.2f}s (peak {peak:.3f})")
    if peak < 1e-4:
        print("[WARN] Near-silence recorded — check the default input device.")
    return wav


def report_vector(hidden: torch.Tensor) -> None:
    """Print 768-dim latent stats + sklearn StandardScaler/PCA-2D transform."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    assert hidden.dim() == 3 and hidden.shape[0] == 1, f"expected (1,T,768), got {tuple(hidden.shape)}"
    frames = hidden.squeeze(0).double().numpy()  # (T, 768) float64 for stable PCA
    t, d = frames.shape
    pooled = frames.mean(axis=0)  # time-averaged utterance vector (768,)
    print(f"[VEC] hidden_states: (1, {t}, {d}), T_frames={t}")
    print(f"[VEC] pooled[0:8] = {np.array2string(pooled[:8], precision=4, separator=', ')}")
    print(f"[VEC] pooled L2={np.linalg.norm(pooled):.4f} mean={pooled.mean():+.4f} std={pooled.std():.4f}")

    n_comp = min(2, t, d)
    if t < 2:
        print("[VEC] Too few frames for PCA — skipping transform.")
        return
    z = StandardScaler().fit_transform(frames)
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(z)
    print(f"[VEC] PCA-2D explained variance: {np.array2string(pca.explained_variance_ratio_, precision=4)}")
    head = min(5, t)
    print(f"[VEC] PCA coords first {head} frames (T_frames x 2):")
    for i in range(head):
        print(f"      f{i:04d}: ({coords[i, 0]:+.4f}, {coords[i, 1]:+.4f})")


def run_turn(pipe: STTPipeline, wav: torch.Tensor) -> None:
    out = pipe.process(wav, sample_rate=TARGET_SR)
    print(f"[TXT] {out.transcription.strip() or '(empty transcript)'}")
    report_vector(out.hidden_states)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    # Windows consoles default to cp1252, which cannot print Bengali text.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Live mic STT + vector demo (Samvyas Phase 1).")
    parser.add_argument("--secs", type=float, default=5.0, help="Seconds per clip (default 5).")
    parser.add_argument("--once", action="store_true", help="Single clip then exit.")
    parser.add_argument("--no-denoise", action="store_true", help="Bypass DeepFilterNet.")
    parser.add_argument("--file", type=str, default=None, help="Dry-run on audio file instead of mic.")
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu.")
    parser.add_argument("--mic", type=int, default=None, help="sounddevice input device index.")
    args = parser.parse_args()

    if args.secs <= 0 or args.secs > 25:
        print("[ERR] --secs must be in (0, 25] (NeMo guidance + 4GB VRAM).")
        return 2

    denoiser = DeepFilterDenoiser(enabled=not args.no_denoise, device=args.device)
    encoder = IndicASREncoder(device=args.device)
    pipe = STTPipeline(denoiser=denoiser, encoder=encoder, device=args.device)

    if args.file:
        print(f"[FILE] Dry run on {args.file}")
        run_turn(pipe, load_audio(Path(args.file), target_sr=TARGET_SR))
        return 0

    try:
        import sounddevice as sd

        print(f"[MIC] Default input: {sd.query_devices(kind='input')['name']}")
    except Exception as e:
        print(f"[ERR] No input device: {e}")
        return 2

    turn = 0
    print("[INFO] Fixed-clip loop — press Ctrl+C to stop.")
    try:
        while True:
            turn += 1
            print(f"\n===== Turn {turn} =====")
            run_turn(pipe, record_clip(args.secs, device=args.mic))
            if args.once:
                break
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
