"""Evaluation metrics: Speaker Similarity (SIM-O), Audio Quality, and Intelligibility."""

from typing import Optional
import torch
import torch.nn.functional as F


def compute_speaker_similarity(
    ref_embedding: torch.Tensor,
    gen_embedding: torch.Tensor,
) -> float:
    """Computes cosine similarity between reference speaker embedding and synthesized speaker embedding.
    Range: [-1.0, 1.0]. SOTA systems typically achieve >= 0.75 on zero-shot benchmarks.
    """
    ref_norm = F.normalize(ref_embedding.view(1, -1), p=2, dim=-1)
    gen_norm = F.normalize(gen_embedding.view(1, -1), p=2, dim=-1)
    cos_sim = torch.sum(ref_norm * gen_norm).item()
    return cos_sim


def compute_wer(reference_text: str, hypothesis_text: str) -> float:
    """Calculates Word Error Rate (WER) using Levenshtein distance."""
    ref_words = reference_text.lower().strip().split()
    hyp_words = hypothesis_text.lower().strip().split()

    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[len(ref_words)][len(hyp_words)] / max(1, len(ref_words))
