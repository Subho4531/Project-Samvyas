"""Unit tests for VoiceLLM model components and forward/generate passes."""

import pytest
import torch

from voicellm.core.config import ModelConfig
from voicellm.models.backbones.rope import RotaryEmbedding
from voicellm.models.backbones.transformer import RMSNorm, SwiGLU, TransformerDecoder
from voicellm.models.voice_llm import VoiceLLM
from voicellm.training.losses import MultiCodebookLoss


def test_rmsnorm():
    norm = RMSNorm(dim=64)
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == (2, 10, 64)


def test_swiglu():
    ffn = SwiGLU(dim=64, hidden_dim=128)
    x = torch.randn(2, 10, 64)
    out = ffn(x)
    assert out.shape == (2, 10, 64)


def test_transformer_decoder():
    decoder = TransformerDecoder(
        hidden_dim=128,
        intermediate_dim=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        max_seq_len=512,
    )
    x = torch.randn(2, 16, 128)
    out, kv = decoder(x)
    assert out.shape == (2, 16, 128)
    assert len(kv) == 2


def test_voicellm_forward_and_loss():
    config = ModelConfig(
        vocab_size_text=1000,
        num_codebooks=4,
        codebook_vocab_size=512,
        hidden_dim=128,
        intermediate_dim=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        max_seq_len=512,
    )
    model = VoiceLLM(config)

    B, T_text, T_audio = 2, 8, 16
    text_tokens = torch.randint(0, 1000, (B, T_text))
    acoustic_tokens = torch.randint(0, 512, (B, 4, T_audio))
    speaker_emb = torch.randn(B, 256)
    emotion_id = torch.tensor([0, 1])

    outputs = model(
        text_tokens=text_tokens,
        acoustic_tokens=acoustic_tokens,
        speaker_emb=speaker_emb,
        emotion_id=emotion_id,
    )

    logits = outputs["logits"]
    assert logits.shape == (B, 4, T_audio, model.total_acoustic_vocab)

    criterion = MultiCodebookLoss(num_codebooks=4)
    targets = acoustic_tokens + model.special_token_offset
    loss, metrics = criterion(logits, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0.0
    assert "loss_cb_0" in metrics


def test_voicellm_generate():
    config = ModelConfig(
        vocab_size_text=1000,
        num_codebooks=4,
        codebook_vocab_size=512,
        hidden_dim=128,
        intermediate_dim=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        max_seq_len=512,
    )
    model = VoiceLLM(config)

    text_tokens = torch.randint(0, 1000, (1, 6))
    speaker_emb = torch.randn(1, 256)

    generated = model.generate(
        text_tokens=text_tokens,
        speaker_emb=speaker_emb,
        max_new_tokens=10,
        temperature=0.8,
    )

    assert generated.shape == (1, 4, 10)
