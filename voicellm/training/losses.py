"""Loss functions for Multi-Codebook Voice LLM Training."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiCodebookLoss(nn.Module):
    """Computes weighted Cross-Entropy loss across all RVQ acoustic codebooks.
    
    Assigns higher weighting to primary / coarse codebooks (e.g. K=0) 
    which dictate phonetic identity and pitch, while fine codebooks model residual acoustic texture.
    """

    def __init__(
        self,
        num_codebooks: int = 4,
        codebook_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.05,
        ignore_index: int = 0,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        if codebook_weights is None:
            # Default: gentle decay across residual levels [1.0, 0.8, 0.6, 0.5]
            self.codebook_weights = [1.0 / (1.0 + 0.2 * i) for i in range(num_codebooks)]
        else:
            self.codebook_weights = codebook_weights

        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Args:
            logits: (B, num_codebooks, T_audio, vocab_size)
            targets: (B, num_codebooks, T_audio)
            
        Returns:
            total_loss: Scalar torch.Tensor
            metrics: Dict with per-codebook loss and accuracy
        """
        B, K, T, V = logits.shape
        total_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        metrics = {}

        for k in range(K):
            k_logits = logits[:, k, :, :].reshape(-1, V)
            k_targets = targets[:, k, :].reshape(-1)

            loss_k = F.cross_entropy(
                k_logits,
                k_targets,
                label_smoothing=self.label_smoothing,
                ignore_index=self.ignore_index,
            )
            weight = self.codebook_weights[k] if k < len(self.codebook_weights) else 1.0
            total_loss = total_loss + weight * loss_k

            # Compute accuracy on active tokens
            with torch.no_grad():
                valid_mask = k_targets != self.ignore_index
                if valid_mask.sum() > 0:
                    preds = torch.argmax(k_logits[valid_mask], dim=-1)
                    acc = (preds == k_targets[valid_mask]).float().mean().item()
                else:
                    acc = 0.0

            metrics[f"loss_cb_{k}"] = loss_k.item()
            metrics[f"acc_cb_{k}"] = acc

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics
