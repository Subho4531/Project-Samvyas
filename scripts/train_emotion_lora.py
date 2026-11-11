"""Train the IndicF5 emotion LoRA adapter on a prepared prompt/target manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from samvyas.models.tts.emotion_trainer import EmotionTrainingConfig, run_training


def main() -> None:
    """Parse CLI options, run Accelerate/DDP training, and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--mel-cache", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--tag-dropout", type=float, default=0.1)
    parser.add_argument("--max-frames", type=int, default=2400)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--attention-mask", action="store_true")
    parser.add_argument(
        "--mixed-precision",
        choices=("no", "fp16", "bf16"),
        default="fp16",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--find-unused-parameters", action="store_true")
    parser.add_argument(
        "--metadata",
        type=str,
        default="{}",
        help="Extra JSON merged into the adapter sidecar and run report.",
    )
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    metadata = json.loads(args.metadata)
    if not isinstance(metadata, dict):
        raise TypeError("--metadata must be a JSON object.")
    config = EmotionTrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        base_checkpoint=args.base_checkpoint,
        mel_cache_dir=args.mel_cache,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        accumulation=args.accumulation,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        tag_dropout=args.tag_dropout,
        max_frames=args.max_frames,
        max_batch_size=args.max_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        use_attention_mask=args.attention_mask,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        resume=args.resume,
        patience=args.patience,
        max_steps=args.max_steps,
        max_seconds=args.max_seconds,
        log_every=args.log_every,
        save_every=args.save_every,
        grad_clip=args.grad_clip,
        find_unused_parameters=args.find_unused_parameters,
        metadata=metadata,
    )
    print(json.dumps(run_training(config), indent=2))


if __name__ == "__main__":
    main()
