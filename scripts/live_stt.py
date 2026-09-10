"""Live dual-stream demo: default mic -> STT + paralinguistic vectors.

Flow (diagram: Mic 16kHz -> DeepFilterNet v2 -> STT branch + Para branch):
    record fixed clip -> STTPipeline (transcript + 768-dim latents)
                      -> IndicWav2VecEncoder (1024-dim acoustics + prosody)
    -> print transcript, semantic vector + PCA, acoustic/prosody parameters.

Usage:
    .venv/Scripts/python.exe scripts/live_stt.py                 # 5s clips, loop till Ctrl+C
    .venv/Scripts/python.exe scripts/live_stt.py --once          # single clip then exit
    .venv/Scripts/python.exe scripts/live_stt.py --no-para       # STT branch only
    .venv/Scripts/python.exe scripts/live_stt.py --para-device cuda  # para on GPU (OOM risk)
    .venv/Scripts/python.exe scripts/live_stt.py --file assets/audio/vocaltest.m4a  # dry run

VRAM guard (RTX 2050 4GB): short clips (default 5s), sequential inference,
paralinguistic encoder defaults to CPU (NeMo + DeepFilter already hold the
GPU), torch.cuda.empty_cache() after every turn.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from samvyas.audio.denoiser.deepfilter import DeepFilterDenoiser
from samvyas.audio.io import TARGET_SR, load_audio
from samvyas.models.encoders.paralinguistic import IndicWav2VecEncoder
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


def report_vector(hidden: torch.Tensor) -> int:
    """Print 768-dim latent stats + sklearn StandardScaler/PCA-2D transform.

    Returns the semantic frame count (alignment target for the para branch).
    """
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
        return t
    z = StandardScaler().fit_transform(frames)
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(z)
    print(f"[VEC] PCA-2D explained variance: {np.array2string(pca.explained_variance_ratio_, precision=4)}")
    head = min(5, t)
    print(f"[VEC] PCA coords first {head} frames (T_frames x 2):")
    for i in range(head):
        print(f"      f{i:04d}: ({coords[i, 0]:+.4f}, {coords[i, 1]:+.4f})")
    return t


def report_paralinguistic(
    para_enc: IndicWav2VecEncoder, clean: torch.Tensor, semantic_frames: int
) -> None:
    """Run the para branch on cleaned audio and print all other parameters."""
    out = para_enc.encode(clean, sample_rate=TARGET_SR)
    a = out.acoustic_embeddings  # (1, T_para, D)
    b, t_para, d_para = (a.shape[0], a.shape[1], a.shape[2])
    a_np = a.squeeze(0).double().numpy()
    pooled = a_np.mean(axis=0)
    print(f"[PARA] backbone={para_enc.repo_id} layer={para_enc.layer} device={para_enc.device}")
    print(f"[PARA] acoustic: ({b}, {t_para}, {d_para}), D_acoustic={d_para}")
    print(f"[PARA] pooled[0:8] = {np.array2string(pooled[:8], precision=4, separator=', ')}")
    print(f"[PARA] pooled L2={np.linalg.norm(pooled):.4f} mean={pooled.mean():+.4f} std={pooled.std():.4f}")

    if out.prosody_latents is not None:
        p = out.prosody_latents.squeeze(0).numpy()  # (T, 4): f0_norm, voiced, logE, zcr
        f0_hz = p[:, 0] * 500.0
        voiced = p[:, 1] > 0.5
        print(
            f"[PARA] prosody: frames={p.shape[0]} voiced_ratio={voiced.mean():.3f} "
            f"mean_F0={f0_hz[voiced].mean() if voiced.any() else 0.0:.1f}Hz "
            f"mean_logE={p[:, 2].mean():.4f} mean_ZCR={p[:, 3].mean():.4f}"
        )

    fused = para_enc.align_to_semantic(out, semantic_frames)
    msg = (
        f"[PARA] aligned para_T={t_para} -> semantic_T={semantic_frames}: "
        f"{tuple(fused.acoustic_embeddings.shape)}"
    )
    if fused.prosody_latents is not None:
        msg += f" + prosody {tuple(fused.prosody_latents.shape)}"
    print(msg)


def run_turn(pipe: STTPipeline, para_enc: IndicWav2VecEncoder | None, wav: torch.Tensor) -> None:
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    out = pipe.process(wav, sample_rate=TARGET_SR)
    timings["stt"] = time.perf_counter() - t0
    print(f"[TXT] {out.transcription.strip() or '(empty transcript)'}")
    sem_frames = report_vector(out.hidden_states)

    if para_enc is not None:
        t0 = time.perf_counter()
        report_paralinguistic(para_enc, out.cleaned_audio, sem_frames)
        timings["para"] = time.perf_counter() - t0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[TIME] " + " ".join(f"{k}={v:.1f}s" for k, v in timings.items()))


def main() -> int:
    # Windows consoles default to cp1252, which cannot print Bengali text.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Live dual-stream demo (STT + paralinguistic).")
    parser.add_argument("--secs", type=float, default=5.0, help="Seconds per clip (default 5).")
    parser.add_argument("--once", action="store_true", help="Single clip then exit.")
    parser.add_argument("--no-denoise", action="store_true", help="Bypass DeepFilterNet.")
    parser.add_argument("--no-para", action="store_true", help="STT branch only (skip paralinguistic).")
    parser.add_argument(
        "--para-device", type=str, default="cpu", help="cpu (safe, default) | cuda (OOM risk on 4GB)."
    )
    parser.add_argument("--para-layer", type=str, default="last", help="last | mean_last4.")
    parser.add_argument("--file", type=str, default=None, help="Dry-run on audio file instead of mic.")
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu (STT branch).")
    parser.add_argument("--mic", type=int, default=None, help="sounddevice input device index.")
    args = parser.parse_args()

    if args.secs <= 0 or args.secs > 25:
        print("[ERR] --secs must be in (0, 25] (NeMo guidance + 4GB VRAM).")
        return 2

    denoiser = DeepFilterDenoiser(enabled=not args.no_denoise, device=args.device)
    encoder = IndicASREncoder(device=args.device)
    pipe = STTPipeline(denoiser=denoiser, encoder=encoder, device=args.device)
    para_enc = (
        None
        if args.no_para
        else IndicWav2VecEncoder(layer=args.para_layer, device=args.para_device)  # type: ignore[arg-type]
    )

    if args.file:
        print(f"[FILE] Dry run on {args.file}")
        wav = load_audio(Path(args.file), target_sr=TARGET_SR)
        wav = wav[: int(args.secs * TARGET_SR)]  # same fixed-clip guard as mic mode
        run_turn(pipe, para_enc, wav)
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
            run_turn(pipe, para_enc, record_clip(args.secs, device=args.mic))
            if args.once:
                break
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
