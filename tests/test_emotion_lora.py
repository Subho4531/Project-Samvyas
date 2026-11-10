"""CPU tests for the emotion LoRA wrapper, masked loss, and adapter export."""

from __future__ import annotations

import random

import pytest
import torch
from torch import nn

pytest.importorskip("f5_tts")

from f5_tts.model import CFM, DiT

from samvyas.models.tts.emotion_lora import (
    EMOTION_TAGS,
    EmotionCFM,
    LoRAConfig,
    apply_lora_to_dit,
    load_adapter,
    save_adapter,
)


class DummyMel(nn.Module):
    """Tiny mel module so tests do not allocate the real Vocos frontend."""

    n_mel_channels = 100

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        length = max(1, waveform.shape[-1] // 256)
        return torch.zeros(waveform.shape[0], 100, length)


def make_tiny_model() -> tuple[EmotionCFM, list]:
    """Build a small CFM with the same nested module names as IndicF5."""
    dit = DiT(
        dim=32,
        depth=2,
        heads=4,
        dim_head=8,
        ff_mult=2,
        mel_dim=100,
        text_num_embeds=64,
        text_dim=16,
        conv_layers=0,
    )
    vocab = {char: index for index, char in enumerate("abcdefghijklmnopqrstuvwxyz ")}
    cfm = CFM(
        dit,
        mel_spec_module=DummyMel(),
        num_channels=100,
        vocab_char_map=vocab,
    )
    cfm.requires_grad_(False)
    lora = apply_lora_to_dit(cfm.transformer, rank=4, alpha=8.0)
    model = EmotionCFM(cfm, lora, num_tags=len(EMOTION_TAGS), tag_dropout=0.1)
    return model, lora


def test_lora_targets_and_initial_identity() -> None:
    """Every expected projection is adapted and B=0 preserves the base output."""
    model, lora = make_tiny_model()
    assert len(lora) == 12  # two blocks x six projections
    names = [name for name, _ in model.base.transformer.named_modules()]
    assert any(name.endswith("to_out.0") for name in names)
    assert any(name.endswith("ff.ff.0.0") for name in names)
    assert any(name.endswith("ff.ff.2") for name in names)
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        model.base.transformer.eval()
        base_input = x.clone()
        # The adapter starts at zero, so the projection output is unchanged.
        for name, module in model.base.transformer.named_modules():
            if name.endswith("attn.to_q"):
                assert torch.allclose(module(base_input), module.base(base_input))


def test_masked_loss_gradients_and_tag_effect() -> None:
    """Only target frames contribute; LoRA and tag rows receive gradients."""
    model, _ = make_tiny_model()
    params = model.configure_trainable()
    assert params
    model.train()
    random.seed(7)
    torch.manual_seed(7)
    mel = torch.randn(2, 24, 100)
    text = ["abc def", "ghi jkl"]
    tags = torch.tensor([1, 3])
    prompt = torch.tensor([7, 9])
    lens = torch.tensor([24, 20])
    loss, cond, pred = model.forward_emotion(mel, text, tags, prompt, lens)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert pred.shape == mel.shape
    assert torch.equal(cond[:, :7], mel[:, :7])
    assert torch.count_nonzero(cond[0, 7:24]) == 0
    loss.backward()
    assert model.tag_embed.embed.weight.grad is not None
    assert any(module.lora_a.grad is not None for module in model.lora_modules)
    assert all(module.base.weight.grad is None for module in model.lora_modules)

    model.eval()
    with torch.no_grad():
        torch.manual_seed(11)
        pred_a = model._transformer_forward(
            x=mel,
            cond=cond,
            text=text,
            time=torch.full((2,), 0.5),
            tag_ids=torch.tensor([1, 3]),
            mask=None,
            drop_audio_cond=False,
            drop_text=False,
        )
        torch.manual_seed(11)
        pred_b = model._transformer_forward(
            x=mel,
            cond=cond,
            text=text,
            time=torch.full((2,), 0.5),
            tag_ids=torch.tensor([2, 4]),
            mask=None,
            drop_audio_cond=False,
            drop_text=False,
        )
    assert not torch.allclose(pred_a, pred_b)


def test_save_and_load_adapter(tmp_path) -> None:
    """Adapter round-trip preserves LoRA/tag tensors but not base weights."""
    model, _ = make_tiny_model()
    model.configure_trainable()
    path = tmp_path / "adapter.safetensors"
    metadata = save_adapter(
        model,
        path,
        config=LoRAConfig(rank=4, alpha=8.0),
        metadata={"run": "unit-test"},
    )
    assert metadata["tag_names"] == list(EMOTION_TAGS)
    assert path.is_file() and path.with_suffix(".json").is_file()

    restored, _ = make_tiny_model()
    loaded = load_adapter(restored, path)
    assert loaded["run"] == "unit-test"
    assert torch.equal(model.tag_embed.embed.weight, restored.tag_embed.embed.weight)
    for left, right in zip(model.lora_modules, restored.lora_modules):
        assert torch.equal(left.lora_a, right.lora_a)
        assert torch.equal(left.lora_b, right.lora_b)


def test_sample_returns_prompt_plus_target() -> None:
    """The sampling wrapper keeps the reference frames and generates the target."""
    model, _ = make_tiny_model()
    model.eval()
    out, trajectory = model.sample(
        cond=torch.zeros(1, 1200),
        text=["abc"],
        duration=10,
        tag_ids=torch.tensor([2]),
        steps=1,
        cfg_strength=0.0,
        seed=3,
    )
    assert out.shape == (1, 10, 100)
    assert trajectory.shape[0] == 2
    assert torch.equal(out[:, :4], torch.zeros(1, 4, 100))
