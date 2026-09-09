"""Kaggle-Optimized VoiceLLM Trainer."""

import math
import os
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from voicellm.core.config import TrainingConfig
from voicellm.core.utils import setup_logger
from voicellm.training.losses import MultiCodebookLoss

logger = setup_logger("VoiceLLMTrainer")


class VoiceLLMTrainer:
    """Trainer built for efficient single/multi-GPU training with gradient accumulation,
    mixed precision (FP16/BF16), cosine scheduling, and Kaggle-friendly checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or TrainingConfig()
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss function
        self.criterion = MultiCodebookLoss(num_codebooks=self.model.num_codebooks)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Mixed precision GradScaler
        self.use_amp = self.config.mixed_precision in ["fp16", "bf16"] and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.config.mixed_precision == "fp16"))

        # Checkpoint directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0

    def get_lr(self, step: int) -> float:
        """Linear warmup followed by cosine decay."""
        if step < self.config.warmup_steps:
            return self.config.learning_rate * (step + 1) / self.config.warmup_steps
        if step > self.config.max_steps:
            return self.config.min_learning_rate
        progress = (step - self.config.warmup_steps) / max(1, self.config.max_steps - self.config.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_learning_rate + (self.config.learning_rate - self.config.min_learning_rate) * cosine

    def update_lr(self):
        lr = self.get_lr(self.global_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def train_epoch(self, epoch: int) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            # Move batch to device
            text_tokens = batch["text_tokens"].to(self.device)
            acoustic_tokens = batch["acoustic_tokens"].to(self.device)
            speaker_emb = batch["speaker_emb"].to(self.device)
            emotion_id = batch["emotion_id"].to(self.device)

            # Target tokens (shift acoustic token prediction)
            targets = acoustic_tokens + self.model.special_token_offset

            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(
                    text_tokens=text_tokens,
                    acoustic_tokens=acoustic_tokens,
                    speaker_emb=speaker_emb,
                    emotion_id=emotion_id,
                )
                loss, metrics = self.criterion(outputs["logits"], targets)
                scaled_loss = loss / self.config.grad_accum_steps

            # Backward pass with scaler
            if self.scaler.is_enabled():
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            running_loss += loss.item()

            # Optimizer Step on accumulation boundary
            if (step + 1) % self.config.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                lr = self.update_lr()
                self.global_step += 1

                if self.global_step % self.config.log_every_steps == 0:
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "cb0_acc": f"{metrics.get('acc_cb_0', 0):.3f}",
                        "lr": f"{lr:.2e}",
                    })

                if self.global_step % self.config.save_every_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}.pt")

                if self.global_step >= self.config.max_steps:
                    logger.info(f"Reached max steps: {self.config.max_steps}. Halting training.")
                    return

    def save_checkpoint(self, filename: str) -> None:
        save_path = self.output_dir / filename
        logger.info(f"Saving model checkpoint to: {save_path}")
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.model.config,
            "global_step": self.global_step,
        }
        torch.save(state, str(save_path))
