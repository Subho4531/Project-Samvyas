"""Evaluate emotion-conditioned IndicF5 LoRA: fixed unseen grid, validation pairs, acoustics, and listening manifest."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from samvyas.data.emotion_dataset import (
    EMOTIONS,
    EmotionPairDataset,
    _load_audio_24k,
    find_unknown_characters,
    tag_id_for,
)
from samvyas.models.tts.emotion_eval import (
    DEFAULT_UNSEEN_BENGALI_SENTENCES,
    estimate_target_frames,
    extract_acoustic_features,
    predict_emotion,
    read_audio_mono,
    resolve_prompt_clip,
    train_ground_truth_classifier,
)
from samvyas.models.tts.emotion_lora import EmotionCFM, load_adapter, load_emotion_model
from samvyas.models.tts.indic_f5 import VOCODER_REVISION, _FullPrecisionVocoder

TAG_SPECS: tuple[tuple[str, int], ...] = tuple((f"<{name}>", tag_id_for(name)) for name in EMOTIONS)
TAG_NAME_BY_ID: dict[int, str] = {tag_id_for(name): name for name in EMOTIONS}


def fit_dataset_classifier(
    dataset: EmotionPairDataset,
) -> tuple[Any | None, dict[str, Any]]:
    """Train the acoustic proxy classifier on ground-truth training clips."""
    samples: list[tuple[list[float], int]] = []
    for row in dataset.rows:
        target_path = dataset.root / row["audio_path"]
        if not target_path.is_file():
            continue
        loaded = read_audio_mono(target_path)
        if loaded is None:
            continue
        audio, sr = loaded
        feat = extract_acoustic_features(audio, rate=sr, text=row["text"])
        samples.append(
            (
                [feat["f0_mean"], feat["f0_std"], feat["energy"], feat["rate"]],
                tag_id_for(row["emotion"]),
            )
        )
    return train_ground_truth_classifier(samples)


def _synthesize_sample(
    model: EmotionCFM,
    vocoder: _FullPrecisionVocoder,
    *,
    prompt_mel: torch.Tensor,
    combined_text: str,
    total_frames: int,
    tag_id: int,
    steps: int,
    cfg_strength: float,
    sway_coef: float | None,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate target audio under CUDA bf16 autocast while decoding in float32."""
    prompt_frames = int(prompt_mel.shape[1])
    tag_tensor = torch.tensor([tag_id], device=device, dtype=torch.long)
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    with (
        torch.inference_mode(),
        torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp),
    ):
        sampled_mel, _ = model.sample(
            cond=prompt_mel,
            text=[combined_text],
            duration=total_frames,
            tag_ids=tag_tensor,
            steps=steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_coef,
            seed=seed,
        )
    target_sampled_mel = sampled_mel[:, prompt_frames:total_frames, :]
    target_mel_perm = target_sampled_mel.permute(0, 2, 1)
    audio_24k = vocoder.decode(target_mel_perm).squeeze(0).detach().cpu().numpy().astype(np.float32)
    if audio_24k.size == 0 or not np.isfinite(audio_24k).all():
        raise RuntimeError("Generated audio is empty or non-finite.")
    audio_16k = resample_poly(audio_24k, 2, 3).astype(np.float32)
    scale = max(1.0, float(np.max(np.abs(audio_24k))), float(np.max(np.abs(audio_16k))))
    return audio_24k / scale, audio_16k / scale, scale


def _aggregate_stats(records: Sequence[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Compute mean acoustic stats grouped by ``key`` (e.g. ``intended_tag``)."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(str(record[key]), []).append(record)
    summary: dict[str, dict[str, Any]] = {}
    for bucket_key, items in buckets.items():
        f0_means = [r["features"]["f0_mean"] for r in items if r["features"]["f0_mean"] > 0]
        f0_stds = [r["features"]["f0_std"] for r in items]
        energies = [r["features"]["energy"] for r in items]
        rates = [r["features"]["rate"] for r in items]
        summary[bucket_key] = {
            "f0_mean_avg": float(np.mean(f0_means)) if f0_means else 0.0,
            "f0_std_avg": float(np.mean(f0_stds)) if f0_stds else 0.0,
            "energy_avg": float(np.mean(energies)) if energies else 0.0,
            "rate_avg": float(np.mean(rates)) if rates else 0.0,
            "count": len(items),
        }
    return summary


def evaluate_emotion_lora(
    adapter_path: Path,
    output_dir: Path,
    *,
    mode: str = "unseen",
    data_root: Path | None = None,
    jeet_root: Path | None = None,
    prompt_audio: Path | None = None,
    prompt_text: str | None = None,
    unseen_sentences: Sequence[str] | None = None,
    base_checkpoint: Path | None = None,
    steps_list: tuple[int, ...] = (32, 16, 8),
    cfg_strength: float = 2.0,
    sway_coef: float = -1.0,
    speed: float = 1.0,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_samples: int = 0,
) -> dict[str, Any]:
    """Evaluate an emotion LoRA adapter on unseen Jeet texts and/or held-out validation pairs."""
    if mode not in ("unseen", "validation", "all"):
        raise ValueError("mode must be unseen, validation, or all.")
    if mode in ("validation", "all") and data_root is None:
        raise ValueError("--data-root is required when evaluating validation pairs.")
    output_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT / ".env")

    print(
        f"Loading emotion model from {base_checkpoint or 'pinned base'} on {device}...",
        flush=True,
    )
    model, adapter_meta = load_emotion_model(checkpoint=base_checkpoint, device=device)
    loaded_meta = load_adapter(model, adapter_path)
    model.eval()
    vocab = model.base.vocab_char_map

    from f5_tts.infer.utils_infer import load_vocoder
    from huggingface_hub import hf_hub_download

    vocoder_config = hf_hub_download(
        repo_id="charactr/vocos-mel-24khz",
        filename="config.yaml",
        revision=VOCODER_REVISION,
        token=False,
    )
    hf_hub_download(
        repo_id="charactr/vocos-mel-24khz",
        filename="pytorch_model.bin",
        revision=VOCODER_REVISION,
        token=False,
    )
    vocoder_raw = load_vocoder(
        vocoder_name="vocos",
        is_local=True,
        local_path=str(Path(vocoder_config).parent),
        device=device,
    )
    vocoder = _FullPrecisionVocoder(vocoder_raw)

    clf = None
    clf_info: dict[str, Any] = {"trained": False, "sample_count": 0}
    if data_root is not None and (data_root / "train.jsonl").is_file():
        print("Training ground-truth acoustic proxy classifier on training targets...", flush=True)
        train_ds = EmotionPairDataset(data_root, split="train", max_total_frames=4096)
        clf, clf_info = fit_dataset_classifier(train_ds)

    eval_audio_dir = output_dir / "audio"
    eval_audio_dir.mkdir(parents=True, exist_ok=True)

    synth_results: list[dict[str, Any]] = []
    listening_entries: list[dict[str, Any]] = []
    start_time = time.time()

    resolved_prompt_info: dict[str, Any] | None = None
    sentences = list(unseen_sentences or DEFAULT_UNSEEN_BENGALI_SENTENCES)

    if mode in ("unseen", "all"):
        unknown = find_unknown_characters(sentences, vocab)
        if unknown:
            raise ValueError(
                f"Unseen evaluation sentences contain unsupported characters: {unknown}"
            )
        p_wav, p_text, p_clip_id = resolve_prompt_clip(
            data_root=data_root,
            jeet_root=jeet_root,
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
        )
        p_wav_24k = _load_audio_24k(p_wav).to(device=device)
        with torch.no_grad():
            p_mel = (
                model.base.mel_spec(p_wav_24k).permute(0, 2, 1).to(device=device, dtype=model.dtype)
            )
        p_frames = int(p_mel.shape[1])
        resolved_prompt_info = {
            "clip_id": p_clip_id,
            "audio_path": str(p_wav),
            "text": p_text,
            "prompt_frames": p_frames,
        }
        print(
            f"Synthesizing unseen grid: {len(sentences)} texts x {len(TAG_SPECS)} tags x "
            f"{len(steps_list)} steps using Jeet prompt {p_clip_id} ({p_frames} frames)...",
            flush=True,
        )
        for sent_idx, target_text in enumerate(sentences):
            combined_text = f"{p_text.strip()} {target_text.strip()}"
            target_frames = estimate_target_frames(
                p_frames,
                p_text,
                target_text,
                speed=speed,
            )
            total_frames = p_frames + target_frames
            for tag_str, tag_id in TAG_SPECS:
                for steps in steps_list:
                    sample_seed = seed + sent_idx * 1000 + steps
                    audio_24k, audio_16k, scale = _synthesize_sample(
                        model,
                        vocoder,
                        prompt_mel=p_mel,
                        combined_text=combined_text,
                        total_frames=total_frames,
                        tag_id=tag_id,
                        steps=steps,
                        cfg_strength=cfg_strength,
                        sway_coef=sway_coef,
                        seed=sample_seed,
                        device=device,
                    )
                    tag_slug = tag_str.strip("<>")
                    stem = f"unseen_s{sent_idx + 1:02d}_{tag_slug}_step{steps}"
                    out_24k = eval_audio_dir / f"{stem}_24k.wav"
                    out_16k = eval_audio_dir / f"{stem}_16k.wav"
                    sf.write(out_24k, audio_24k, 24000, subtype="PCM_24")
                    sf.write(out_16k, audio_16k, 16000, subtype="PCM_24")
                    feat = extract_acoustic_features(audio_24k, rate=24000, text=target_text)
                    pred_id = predict_emotion(clf, feat, clf_info) if clf is not None else None
                    record = {
                        "suite": "unseen",
                        "item_id": f"unseen_s{sent_idx + 1:02d}",
                        "speaker_id": "jeet",
                        "target_text": target_text,
                        "intended_tag": tag_str,
                        "intended_tag_id": tag_id,
                        "tag_step_key": f"{tag_str}@step{steps}",
                        "steps": steps,
                        "seed": sample_seed,
                        "audio_24k": out_24k.relative_to(output_dir).as_posix(),
                        "audio_16k": out_16k.relative_to(output_dir).as_posix(),
                        "peak_scale": scale,
                        "duration_seconds": round(len(audio_24k) / 24000.0, 3),
                        "features": feat,
                        "predicted_tag_id": pred_id,
                        "predicted_tag": TAG_NAME_BY_ID.get(pred_id)
                        if pred_id is not None
                        else None,
                    }
                    synth_results.append(record)
                    listening_entries.append(
                        {
                            "suite": "unseen",
                            "item_id": record["item_id"],
                            "speaker_id": "jeet",
                            "intended_tag": tag_str,
                            "steps": steps,
                            "audio_24k": record["audio_24k"],
                            "audio_16k": record["audio_16k"],
                            "prompt_clip_id": p_clip_id,
                            "prompt_text": p_text,
                            "target_text": target_text,
                            "status": "pending_review",
                            "notes": "Compare <neutral>/<happy>/<sad>/<angry> and Jeet voice similarity.",
                        }
                    )
            print(f" Synthesized unseen sentence {sent_idx + 1}/{len(sentences)}", flush=True)

    valid_count = 0
    if mode in ("validation", "all"):
        assert data_root is not None
        valid_ds = EmotionPairDataset(data_root, split="validation", max_total_frames=4096)
        valid_rows = valid_ds.rows[:max_samples] if max_samples > 0 else valid_ds.rows
        valid_count = len(valid_rows)
        print(
            f"Synthesizing {valid_count} validation pairs across {len(TAG_SPECS)} tags and "
            f"{len(steps_list)} step settings...",
            flush=True,
        )
        for item_idx, row in enumerate(valid_rows):
            clip_id = row["clip_id"]
            item = valid_ds[item_idx]
            prompt_mel = item["prompt_mel"].unsqueeze(0).to(device=device, dtype=model.dtype)
            prompt_frames = int(prompt_mel.shape[1])
            target_frames = int(item["target_mel"].shape[0])
            total_frames = prompt_frames + target_frames
            combined_text = f"{item['prompt_text'].strip()} {item['target_text'].strip()}"

            gt_feat = None
            gt_path = valid_ds.root / row["audio_path"]
            if gt_path.is_file():
                loaded_gt = read_audio_mono(gt_path)
                if loaded_gt is not None:
                    gt_audio, gt_sr = loaded_gt
                    gt_feat = extract_acoustic_features(
                        gt_audio, rate=gt_sr, text=item["target_text"]
                    )

            for tag_str, tag_id in TAG_SPECS:
                for steps in steps_list:
                    sample_seed = seed + 50_000 + item_idx * 1000 + steps
                    audio_24k, audio_16k, scale = _synthesize_sample(
                        model,
                        vocoder,
                        prompt_mel=prompt_mel,
                        combined_text=combined_text,
                        total_frames=total_frames,
                        tag_id=tag_id,
                        steps=steps,
                        cfg_strength=cfg_strength,
                        sway_coef=sway_coef,
                        seed=sample_seed,
                        device=device,
                    )
                    tag_slug = tag_str.strip("<>")
                    stem = f"val_{clip_id}_{tag_slug}_step{steps}"
                    out_24k = eval_audio_dir / f"{stem}_24k.wav"
                    out_16k = eval_audio_dir / f"{stem}_16k.wav"
                    sf.write(out_24k, audio_24k, 24000, subtype="PCM_24")
                    sf.write(out_16k, audio_16k, 16000, subtype="PCM_24")
                    feat = extract_acoustic_features(
                        audio_24k, rate=24000, text=item["target_text"]
                    )
                    pred_id = predict_emotion(clf, feat, clf_info) if clf is not None else None
                    record = {
                        "suite": "validation",
                        "item_id": clip_id,
                        "speaker_id": row["speaker_id"],
                        "target_text": item["target_text"],
                        "ground_truth_emotion": row["emotion"],
                        "intended_tag": tag_str,
                        "intended_tag_id": tag_id,
                        "tag_step_key": f"{tag_str}@step{steps}",
                        "steps": steps,
                        "seed": sample_seed,
                        "audio_24k": out_24k.relative_to(output_dir).as_posix(),
                        "audio_16k": out_16k.relative_to(output_dir).as_posix(),
                        "peak_scale": scale,
                        "duration_seconds": round(len(audio_24k) / 24000.0, 3),
                        "features": feat,
                        "predicted_tag_id": pred_id,
                        "predicted_tag": TAG_NAME_BY_ID.get(pred_id)
                        if pred_id is not None
                        else None,
                        "gt_features": gt_feat,
                    }
                    synth_results.append(record)
                    listening_entries.append(
                        {
                            "suite": "validation",
                            "item_id": clip_id,
                            "speaker_id": row["speaker_id"],
                            "intended_tag": tag_str,
                            "steps": steps,
                            "audio_24k": record["audio_24k"],
                            "audio_16k": record["audio_16k"],
                            "prompt_text": item["prompt_text"],
                            "target_text": item["target_text"],
                            "status": "pending_review",
                            "notes": "Verify emotional expressiveness, clarity, and speaker consistency.",
                        }
                    )
            print(f" Evaluated validation clip {item_idx + 1}/{valid_count}", flush=True)

    elapsed = round(time.time() - start_time, 1)
    unseen_records = [r for r in synth_results if r["suite"] == "unseen"]
    tag_stats = _aggregate_stats(unseen_records or synth_results, "intended_tag")
    tag_step_stats = _aggregate_stats(unseen_records or synth_results, "tag_step_key")

    clf_synth_acc = None
    if clf is not None and synth_results:
        correct = sum(1 for r in synth_results if r["predicted_tag_id"] == r["intended_tag_id"])
        clf_synth_acc = float(correct / len(synth_results))

    manifest_path = output_dir / "listening_review.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in listening_entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    results_path = output_dir / "synthesized_samples.jsonl"
    with results_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in synth_results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "method": "IndicF5 emotion LoRA evaluation",
        "mode": mode,
        "adapter_path": str(adapter_path),
        "adapter_metadata": loaded_meta or bool(adapter_meta),
        "jeet_prompt": resolved_prompt_info,
        "unseen_sentences": sentences if mode in ("unseen", "all") else [],
        "validation_clips": valid_count,
        "total_synthesized": len(synth_results),
        "steps_evaluated": list(steps_list),
        "cfg_strength": cfg_strength,
        "sway_sampling_coef": sway_coef,
        "elapsed_seconds": elapsed,
        "classifier_ground_truth_info": clf_info,
        "classifier_synth_accuracy": clf_synth_acc,
        "tag_acoustic_statistics": tag_stats,
        "tag_step_acoustic_statistics": tag_step_stats,
        "stt_cer": "pending local STT verification in .venv",
        "synthesized_samples_manifest": results_path.relative_to(output_dir).as_posix(),
        "listening_review_manifest": manifest_path.relative_to(output_dir).as_posix(),
        "listening_review_status": "pending; human listening check required",
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evaluation complete in {elapsed}s. Saved to {output_dir}.", flush=True)
    return report


def main() -> None:
    """CLI entry point for end-of-run emotion LoRA evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter", type=Path, required=True, help="Path to adapter safetensors file."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for evaluated audio and report."
    )
    parser.add_argument(
        "--mode",
        choices=("unseen", "validation", "all"),
        default="unseen",
        help="Evaluation suite to synthesize.",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Prepared dataset directory.")
    parser.add_argument("--jeet-root", type=Path, default=None, help="Raw Jeet dataset directory.")
    parser.add_argument("--prompt-audio", type=Path, default=None, help="Explicit prompt WAV.")
    parser.add_argument("--prompt-text", type=str, default=None, help="Explicit prompt transcript.")
    parser.add_argument(
        "--texts-file",
        type=Path,
        default=None,
        help="Optional UTF-8 newline file overriding the 5 fixed unseen Bengali sentences.",
    )
    parser.add_argument(
        "--base-checkpoint", type=Path, default=None, help="Local base checkpoint directory."
    )
    parser.add_argument(
        "--steps", type=str, default="32,16,8", help="Comma-separated ODE step counts."
    )
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-coef", type=float, default=-1.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples", type=int, default=0, help="Max validation samples to evaluate."
    )
    args = parser.parse_args()

    steps_tuple = tuple(int(s.strip()) for s in args.steps.split(",") if s.strip())
    if not steps_tuple:
        raise ValueError("At least one ODE step count must be provided.")
    texts = None
    if args.texts_file is not None:
        texts = [
            line.strip()
            for line in args.texts_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    report = evaluate_emotion_lora(
        adapter_path=args.adapter,
        output_dir=args.output_dir,
        mode=args.mode,
        data_root=args.data_root,
        jeet_root=args.jeet_root,
        prompt_audio=args.prompt_audio,
        prompt_text=args.prompt_text,
        unseen_sentences=texts,
        base_checkpoint=args.base_checkpoint,
        steps_list=steps_tuple,
        cfg_strength=args.cfg_strength,
        sway_coef=args.sway_coef,
        speed=args.speed,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
