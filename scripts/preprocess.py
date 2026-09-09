"""Preprocessing script to normalize and pre-tokenize audio datasets."""

import argparse
from pathlib import Path
from tqdm import tqdm
import torch

from voicellm.audio.io import load_audio
from voicellm.core.utils import setup_logger
from voicellm.models.codecs.snac_codec import SNACCodec

logger = setup_logger("Preprocess")


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio data for VoiceLLM")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to raw audio folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save preprocessed tokens")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Target sample rate")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    codec = SNACCodec(sample_rate=args.sample_rate)
    audio_files = list(input_path.glob("**/*.wav")) + list(input_path.glob("**/*.mp3")) + list(input_path.glob("**/*.flac"))

    logger.info(f"Found {len(audio_files)} audio files in {input_path}")

    for audio_file in tqdm(audio_files, desc="Pre-tokenizing audio"):
        try:
            waveform, sr = load_audio(
                audio_file,
                target_sample_rate=args.sample_rate,
                mono=True,
                normalize=True,
                trim=True,
            )
            tokens = codec.encode(waveform.unsqueeze(0))
            rel_path = audio_file.relative_to(input_path).with_suffix(".pt")
            save_dest = output_path / rel_path
            save_dest.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tokens.cpu(), str(save_dest))
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")

    logger.info(f"Finished preprocessing. Tokens saved to {output_path}")


if __name__ == "__main__":
    main()
