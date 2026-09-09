"""CLI script for running VoiceLLM synthesis and zero-shot voice cloning."""

import argparse
from pathlib import Path
import torch

from voicellm.core.config import ModelConfig
from voicellm.inference.engine import VoiceLLMInferenceEngine
from voicellm.models.voice_llm import VoiceLLM


def main():
    parser = argparse.ArgumentParser(description="Generate speech using VoiceLLM")
    parser.add_argument("--text", type=str, required=True, help="Text prompt to synthesize")
    parser.add_argument("--ref_audio", type=str, default=None, help="Reference audio file for voice mimicry")
    parser.add_argument("--emotion_id", type=int, default=0, help="Emotion ID (0=neutral, 1=happy, 2=sad, etc.)")
    parser.add_argument("--output_path", type=str, default="outputs/generated_sample.wav", help="Output WAV path")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    config = ModelConfig()
    model = VoiceLLM(config)
    engine = VoiceLLMInferenceEngine(model=model, device=device)

    print(f"Synthesizing text: '{args.text}'")
    engine.synthesize(
        text=args.text,
        reference_audio_path=args.ref_audio,
        emotion_id=args.emotion_id,
        temperature=args.temperature,
        output_wav_path=args.output_path,
    )
    print(f"✅ Audio generated and saved to: {args.output_path}")


if __name__ == "__main__":
    main()
