"""Accelerate/DDP trainer for the IndicF5 emotion LoRA adapter.

The training objective is deliberately different from upstream F5: a neutral
prompt is concatenated before the emotional target, the prompt frames remain
visible as conditioning, and the flow-matching loss is evaluated only on the
target suffix.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader

from samvyas.data.emotion_dataset import (
    EmotionPairDataset,
    FrameBatchSampler,
    collate_emotion_batch,
)
from samvyas.models.tts.emotion_lora import (
    ADAPTER_FORMAT,
    EMOTION_TAGS,
    LORA_TARGETS,
    EmotionCFM,
    LoRAConfig,
    load_adapter,
    load_emotion_model,
    save_adapter,
)


@dataclass
class EmotionTrainingConfig:
    """All knobs for a reproducible single- or multi-GPU emotion run."""

    data_root: Path
    output_dir: Path
    base_checkpoint: Path | None = None
    mel_cache_dir: Path | None = None
    epochs: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    accumulation: int = 4
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    tag_dropout: float = 0.1
    max_frames: int = 2400
    max_batch_size: int = 8
    eval_batch_size: int = 4
    num_workers: int = 2
    gradient_checkpointing: bool = False
    use_attention_mask: bool = False
    mixed_precision: str = "fp16"
    seed: int = 42
    resume: Path | None = None
    patience: int = 5
    max_steps: int = 0
    log_every: int = 10
    save_every: int = 100
    grad_clip: float = 1.0
    find_unused_parameters: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize paths and reject impossible schedules."""
        self.data_root = Path(self.data_root)
        self.output_dir = Path(self.output_dir)
        self.base_checkpoint = Path(self.base_checkpoint) if self.base_checkpoint else None
        self.mel_cache_dir = Path(self.mel_cache_dir) if self.mel_cache_dir else None
        self.resume = Path(self.resume) if self.resume else None
        if not self.data_root.is_dir():
            raise FileNotFoundError(self.data_root)
        if self.epochs < 1 and not self.max_steps:
            raise ValueError("epochs must be positive unless max_steps is set.")
        if self.learning_rate <= 0 or self.accumulation < 1:
            raise ValueError("Learning rate and accumulation must be positive.")
        if self.max_frames < 1 or self.max_batch_size < 1:
            raise ValueError("Frame and batch limits must be positive.")
        if self.mixed_precision not in ("no", "fp16", "bf16"):
            raise ValueError("mixed_precision must be no, fp16, or bf16.")
        if not 0 <= self.lora_dropout < 1 or not 0 <= self.tag_dropout < 1:
            raise ValueError("Dropout values must be in [0, 1).")
        if self.save_every < 0 or self.log_every < 1:
            raise ValueError("save_every must be nonnegative and log_every positive.")


def _atomic_torch_save(value: Any, path: Path) -> None:
    """Write a checkpoint atomically so interruption cannot leave a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _resume_paths(directory: Path) -> tuple[Path, Path]:
    """Return adapter and trainer-state paths for a resumable run directory."""
    return directory / "adapter.safetensors", directory / "trainer_state.pt"


def _load_trainer_state(path: Path) -> dict[str, Any]:
    """Load a trainer state across PyTorch versions."""
    return torch.load(path, map_location="cpu", weights_only=False)


def _save_run(
    model: EmotionCFM,
    accelerator: Accelerator,
    directory: Path,
    *,
    config: EmotionTrainingConfig,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best: float,
    initial: float,
    history: list[dict[str, Any]],
) -> None:
    """Save adapter and optimizer state on rank zero, then synchronize ranks."""
    if not accelerator.is_main_process:
        return
    directory.mkdir(parents=True, exist_ok=True)
    save_adapter(
        accelerator.unwrap_model(model),
        directory / "adapter.safetensors",
        config=LoRAConfig(
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            targets=LORA_TARGETS,
        ),
        metadata={
            "base_checkpoint": str(config.base_checkpoint or "pinned-ai4bharat-indicf5"),
            "tag_count": len(EMOTION_TAGS),
            "tag_names": list(EMOTION_TAGS),
            "global_step": global_step,
            "epoch": epoch,
            **config.metadata,
        },
    )
    _atomic_torch_save(
        {
            "format": ADAPTER_FORMAT,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_validation_loss": best,
            "initial_validation_loss": initial,
            "history": history,
        },
        directory / "trainer_state.pt",
    )


def _prune_steps(root: Path, keep: int = 2) -> None:
    """Keep only the newest numbered checkpoints while preserving best/last."""
    numbered = sorted(
        (path for path in root.glob("step_*") if path.is_dir()),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    for path in numbered[:-keep]:
        for child in path.iterdir():
            child.unlink(missing_ok=True)
        path.rmdir()


def enable_gradient_checkpointing(transformer: torch.nn.Module) -> None:
    """Checkpoint every DiT block to trade compute for activation memory."""
    from torch.utils.checkpoint import checkpoint

    if getattr(transformer, "_samvyas_checkpointed", False):
        return
    for block in transformer.transformer_blocks:
        original = block.forward

        def checkpointed(*args: Any, _original: Any = original, **kwargs: Any) -> torch.Tensor:
            return checkpoint(_original, *args, use_reentrant=False, **kwargs)

        block.forward = checkpointed
    transformer._samvyas_checkpointed = True


def _validate(
    model: torch.nn.Module,
    loader: DataLoader,
    accelerator: Accelerator,
) -> float:
    """Return a globally averaged loss over all validation shards."""
    model.eval()
    total = torch.zeros(2, device=accelerator.device, dtype=torch.float64)
    with torch.no_grad():
        for batch in loader:
            loss = model(
                mel=batch["mel"],
                text=batch["text"],
                tag_ids=batch["tag_ids"],
                prompt_lengths=batch["prompt_lengths"],
                lens=batch["lens"],
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation loss.")
            total[0] += loss.double() * batch["mel"].shape[0]
            total[1] += batch["mel"].shape[0]
    gathered = accelerator.gather(total).view(-1, 2)
    value = (gathered[:, 0].sum() / gathered[:, 1].sum()).item()
    model.train()
    return float(value)


def train_emotion(
    model: EmotionCFM,
    train_dataset: EmotionPairDataset,
    valid_dataset: EmotionPairDataset,
    config: EmotionTrainingConfig,
    *,
    accelerator: Accelerator | None = None,
) -> dict[str, Any]:
    """Run LoRA training and return the final report.

    The model must already have LoRA layers and a tag embedding.  The base CFM
    remains frozen; only adapter matrices and tag rows receive gradients.
    """
    if accelerator is None:
        precision = config.mixed_precision
        if not torch.cuda.is_available():
            precision = "no"
        accelerator = Accelerator(
            cpu=not torch.cuda.is_available(),
            gradient_accumulation_steps=config.accumulation,
            mixed_precision=precision,
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=config.find_unused_parameters)
            ],
        )
    if accelerator.is_main_process:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if any(config.output_dir.iterdir()) and config.resume is None:
            raise FileExistsError("Output directory must be new or empty without --resume.")
    accelerator.wait_for_everyone()

    parameters = model.configure_trainable()
    if not parameters:
        raise RuntimeError("No trainable parameters; LoRA/tag setup failed.")
    if config.gradient_checkpointing:
        enable_gradient_checkpointing(model.base.transformer)

    train_lengths = train_dataset.frame_lengths
    valid_lengths = valid_dataset.frame_lengths
    if max(train_lengths) > config.max_frames:
        raise ValueError(
            f"max_frames={config.max_frames} is smaller than the longest pair "
            f"({max(train_lengths)} frames)."
        )
    if max(valid_lengths) > config.max_frames:
        raise ValueError(
            f"max_frames={config.max_frames} is smaller than the longest validation "
            f"pair ({max(valid_lengths)} frames)."
        )
    train_sampler = FrameBatchSampler(
        train_lengths,
        max_frames=config.max_frames,
        max_batch_size=config.max_batch_size,
        shuffle=True,
        seed=config.seed,
    )
    valid_sampler = FrameBatchSampler(
        valid_lengths,
        max_frames=config.max_frames,
        max_batch_size=config.eval_batch_size,
        shuffle=False,
        seed=config.seed,
    )
    loader_kwargs = {
        "collate_fn": collate_emotion_batch,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, **loader_kwargs)

    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    warmup = min(100, max(1, len(train_loader) // 20))
    total_steps = max(1, len(train_loader) * max(1, config.epochs))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        valid_loader,
        scheduler,
    )

    start_epoch = 0
    global_step = 0
    best = float("inf")
    initial = float("inf")
    history: list[dict[str, Any]] = []
    if config.resume is not None:
        adapter_path, state_path = _resume_paths(config.resume)
        if not adapter_path.is_file() or not state_path.is_file():
            raise FileNotFoundError(f"Resume directory needs adapter and state: {config.resume}")
        load_adapter(accelerator.unwrap_model(model), adapter_path)
        state = _load_trainer_state(state_path)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        best = float(state.get("best_validation_loss", best))
        initial = float(state.get("initial_validation_loss", initial))
        history = list(state.get("history", []))

    if initial == float("inf"):
        initial = _validate(model, valid_loader, accelerator)
        best = min(best, initial)
        if accelerator.is_main_process:
            print(json.dumps({"initial_validation_loss": initial}), flush=True)

    model.train()
    stale = 0
    started = time.time()
    stop = False
    for epoch in range(start_epoch, config.epochs):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_count = 0
        for batch in train_loader:
            with accelerator.accumulate(model):
                loss = model(
                    mel=batch["mel"],
                    text=batch["text"],
                    tag_ids=batch["tag_ids"],
                    prompt_lengths=batch["prompt_lengths"],
                    lens=batch["lens"],
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss; no checkpoint was written.")
                batch_size = batch["mel"].shape[0]
                epoch_loss += float(loss.detach().item()) * batch_size
                epoch_count += batch_size
                accelerator.backward(loss / config.accumulation)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if (
                        config.log_every
                        and global_step % config.log_every == 0
                        and accelerator.is_main_process
                    ):
                        print(
                            json.dumps(
                                {
                                    "step": global_step,
                                    "epoch": epoch + 1,
                                    "loss": float(loss.detach().item()),
                                    "lr": scheduler.get_last_lr()[0],
                                }
                            ),
                            flush=True,
                        )
                    if config.save_every and global_step % config.save_every == 0:
                        _save_run(
                            model,
                            accelerator,
                            config.output_dir / f"step_{global_step:07d}",
                            config=config,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            global_step=global_step,
                            best=best,
                            initial=initial,
                            history=history,
                        )
                        if accelerator.is_main_process:
                            _prune_steps(config.output_dir, keep=2)
                    if config.max_steps and global_step >= config.max_steps:
                        stop = True
                        break
            if stop:
                break
        value = _validate(model, valid_loader, accelerator)
        improved = value < best
        if improved:
            best = value
            stale = 0
        else:
            stale += 1
        entry = {
            "epoch": epoch + 1,
            "steps": global_step,
            "train_loss": epoch_loss / max(1, epoch_count),
            "validation_loss": value,
            "improved": improved,
            "elapsed_seconds": round(time.time() - started, 1),
        }
        history.append(entry)
        if accelerator.is_main_process:
            print(json.dumps(entry), flush=True)
            (config.output_dir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
        _save_run(
            model,
            accelerator,
            config.output_dir / "last",
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            global_step=global_step,
            best=best,
            initial=initial,
            history=history,
        )
        if improved:
            _save_run(
                model,
                accelerator,
                config.output_dir / "best",
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                global_step=global_step,
                best=best,
                initial=initial,
                history=history,
            )
        if config.max_steps and global_step >= config.max_steps:
            break
        if config.patience > 0 and stale >= config.patience:
            if accelerator.is_main_process:
                print(json.dumps({"early_stop": True, "epoch": epoch + 1}), flush=True)
            break

    _save_run(
        model,
        accelerator,
        config.output_dir / "last",
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=min(config.epochs, start_epoch + len(history)),
        global_step=global_step,
        best=best,
        initial=initial,
        history=history,
    )
    accelerator.wait_for_everyone()
    report = {
        "method": "IndicF5 emotion LoRA (prompt/target suffix mask)",
        "base_checkpoint": str(config.base_checkpoint or "pinned-ai4bharat-indicf5"),
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "lora_targets": list(LORA_TARGETS),
        "tag_names": list(EMOTION_TAGS),
        "tag_dropout": config.tag_dropout,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "train_pairs": len(train_dataset),
        "validation_pairs": len(valid_dataset),
        "epochs": config.epochs,
        "steps": global_step,
        "initial_validation_loss": initial,
        "best_validation_loss": best,
        "final_validation_loss": history[-1]["validation_loss"] if history else None,
        "world_size": accelerator.num_processes,
        "mixed_precision": accelerator.mixed_precision,
        "max_frames": config.max_frames,
        "max_batch_size": config.max_batch_size,
        "accumulation": config.accumulation,
        "gradient_checkpointing": config.gradient_checkpointing,
        "seed": config.seed,
        "listening_review": "pending; loss does not measure emotional or voice quality",
        **config.metadata,
    }
    if accelerator.is_main_process:
        (config.output_dir / "training_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    accelerator.end_training()
    return report


def run_training(config: EmotionTrainingConfig) -> dict[str, Any]:
    """Load the pinned base, build manifests, and train an emotion adapter."""
    train_dataset = EmotionPairDataset(
        config.data_root, "train", mel_cache_dir=config.mel_cache_dir
    )
    valid_dataset = EmotionPairDataset(
        config.data_root, "validation", mel_cache_dir=config.mel_cache_dir
    )
    seed = config.seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, _ = load_emotion_model(
        config.base_checkpoint,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        tag_dropout=config.tag_dropout,
        device="cpu",
        use_attention_mask=config.use_attention_mask,
    )
    return train_emotion(model, train_dataset, valid_dataset, config)
