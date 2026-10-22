"""End-to-end verification and latency benchmark script for StyleTTS 2 Bengali pipeline."""

from __future__ import annotations

import sys
import time

import torch

from samvyas.audio.phonemize_bn import (
    BengaliPhonemeTokenizer,
    BengaliPhonemizer,
    BengaliTextNormalizer,
)
from samvyas.models.tts.style_bank import StyleBankManager
from samvyas.models.tts.styletts2 import StyleTTS2Config, StyleTTS2Engine


def run_benchmark() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    print("=" * 65)
    print(" SAMVYAS 2.0: StyleTTS 2 Bengali Pipeline Verification")
    print("=" * 65)

    test_sentences = [
        "নমস্কার! আপনি কেমন আছেন?",
        "আজকের দিনটি সত্যিই অসাধারণ, তাই না?",
        "১২৩৪ নম্বর কক্ষে সভাটি অনুষ্ঠিত হবে।",
        "বিজ্ঞান এবং প্রযুক্তির অগ্রগতি আমাদের জীবনকে সহজ করেছে।",
    ]

    print("\n--- 1. Testing Bengali Text Normalizer ---")
    for text in test_sentences:
        norm = BengaliTextNormalizer.normalize(text)
        print(f"  [RAW ] {text}")
        print(f"  [NORM] {norm}\n")

    print("--- 2. Testing G2P Phonemization & Tokenization ---")
    phonemizer = BengaliPhonemizer()
    tokenizer = BengaliPhonemeTokenizer()
    for text in test_sentences[:2]:
        ipa = phonemizer.phonemize(text)
        tokens = tokenizer.encode(ipa)
        decoded = tokenizer.decode(tokens)
        print(f"  Text: {text}")
        print(f"  IPA : {ipa}")
        print(f"  Tokens ({len(tokens)}): {tokens[:10]}...")
        print(f"  Decoded: {decoded[:40]}...\n")

    print("--- 3. Testing Style Bank & Emotions ---")
    bank = StyleBankManager(style_dim=64)
    emotions = ["neutral", "warm", "excited", "whisper", "serious", "inquisitive"]
    for emo in emotions:
        vec = bank.get_style(emo)
        print(f"  Emotion '{emo:<12}': shape={tuple(vec.shape)}, norm={torch.norm(vec).item():.3f}")

    print("--- 4. Testing End-to-End StyleTTS 2 Engine Synthesis ---")
    config = StyleTTS2Config(device="cpu")
    engine = StyleTTS2Engine(config=config, style_bank=bank)

    for i, emo in enumerate(["warm", "excited", "whisper", "serious"]):
        sentence = test_sentences[i % len(test_sentences)]
        t0 = time.perf_counter()
        output = engine.synthesize(sentence, emotion=emo, speed=1.0)
        dur = time.perf_counter() - t0

        rtf = dur / max(output.duration_sec, 0.001)
        print(f"  Sentence: '{sentence}'")
        print(f"  Emotion : {output.emotion}")
        print(f"  Audio 24k: shape={tuple(output.audio_24k.shape)}, dtype={output.audio_24k.dtype}")
        print(f"  Audio 16k: shape={tuple(output.audio_16k.shape)}, dtype={output.audio_16k.dtype}")
        print(
            f"  Duration : {output.duration_sec:.2f}s | Latency: {output.latency_ms:.1f}ms | RTF: {rtf:.3f}"
        )
        print(f"  Peak 24k : {torch.max(torch.abs(output.audio_24k)).item():.3f}\n")

        # Verify contracts
        assert output.audio_24k.ndim == 2 and output.audio_24k.shape[0] == 1, "Must be (1, T)"
        assert output.audio_16k.ndim == 2 and output.audio_16k.shape[0] == 1, "Must be (1, T)"
        assert torch.max(torch.abs(output.audio_24k)).item() <= 1.0, (
            "Audio out of [-1.0, 1.0] range"
        )

    print("=" * 65)
    print(" ALL CHECKS PASSED: StyleTTS 2 Bengali Pipeline is operational!")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(run_benchmark())
