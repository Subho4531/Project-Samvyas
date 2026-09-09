"""Inference and Zero-Shot Voice Generation Pipeline."""

from pathlib import Path
from typing import Optional, Union
import torch
import torchaudio

from voicellm.audio.io import load_audio, save_audio
from voicellm.data.tokenizer import VoiceTextTokenizer
from voicellm.models.codecs.base_codec import BaseAudioCodec
from voicellm.models.codecs.snac_codec import SNACCodec
from voicellm.models.voice_llm import VoiceLLM


class VoiceLLMInferenceEngine:
    """High-level inference engine for zero-shot voice cloning and emotional generation."""

    def __init__(
        self,
        model: VoiceLLM,
        codec: Optional[BaseAudioCodec] = None,
        tokenizer: Optional[VoiceTextTokenizer] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.codec = codec or SNACCodec(device=str(self.device))
        self.tokenizer = tokenizer or VoiceTextTokenizer()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
    ) -> "VoiceLLMInferenceEngine":
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        config = ckpt["config"]
        model = VoiceLLM(config)
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model=model, device=device)

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        reference_audio_path: Optional[Union[str, Path]] = None,
        speaker_emb: Optional[torch.Tensor] = None,
        emotion_id: Optional[int] = None,
        continuous_vad: Optional[torch.Tensor] = None,
        max_new_tokens: int = 500,
        temperature: float = 0.8,
        top_p: float = 0.95,
        output_wav_path: Optional[Union[str, Path]] = None,
    ) -> torch.Tensor:
        """Synthesizes speech conditioned on prompt text and voice reference."""
        # Encode text
        text_tokens = self.tokenizer.encode(text).to(self.device)

        # Extract speaker embedding from reference audio if provided
        if speaker_emb is None and reference_audio_path is not None:
            # Placeholder for speaker encoder extraction
            speaker_emb = torch.randn(1, 256, device=self.device)
        elif speaker_emb is not None:
            speaker_emb = speaker_emb.to(self.device)

        emo_tensor = torch.tensor([emotion_id], device=self.device) if emotion_id is not None else None
        if continuous_vad is not None:
            continuous_vad = continuous_vad.to(self.device)

        # Generate acoustic tokens
        acoustic_tokens = self.model.generate(
            text_tokens=text_tokens,
            speaker_emb=speaker_emb,
            emotion_id=emo_tensor,
            continuous_vad=continuous_vad,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Decode tokens to audio waveform
        waveform = self.codec.decode(acoustic_tokens)

        # Optionally save to disk
        if output_wav_path is not None:
            save_audio(output_wav_path, waveform.squeeze(0), sample_rate=self.codec.sample_rate)

        return waveform
