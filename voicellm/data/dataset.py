"""Audio-Text paired dataset and collation pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset

from voicellm.audio.io import load_audio
from voicellm.data.tokenizer import VoiceTextTokenizer
from voicellm.models.codecs.base_codec import BaseAudioCodec


@dataclass
class VoiceSample:
    text: str
    audio_path: str
    speaker_id: Optional[str] = None
    speaker_emb: Optional[torch.Tensor] = None
    emotion_id: Optional[int] = None


class VoiceLLMDataset(Dataset):
    """Dataset for training VoiceLLM with paired Audio, Transcripts, and Speaker Embeddings."""

    def __init__(
        self,
        samples: List[VoiceSample],
        codec: Optional[BaseAudioCodec] = None,
        tokenizer: Optional[VoiceTextTokenizer] = None,
        sample_rate: int = 24000,
        max_audio_duration_sec: float = 15.0,
    ):
        self.samples = samples
        self.codec = codec
        self.tokenizer = tokenizer or VoiceTextTokenizer()
        self.sample_rate = sample_rate
        self.max_audio_samples = int(max_audio_duration_sec * sample_rate)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load & normalize audio
        waveform, _ = load_audio(
            sample.audio_path,
            target_sample_rate=self.sample_rate,
            mono=True,
            normalize=True,
            trim=True,
        )

        # Truncate if exceeds max length
        if waveform.shape[-1] > self.max_audio_samples:
            waveform = waveform[:, : self.max_audio_samples]

        # Tokenize text
        text_tokens = self.tokenizer.encode(sample.text).squeeze(0)

        # Acoustic tokens (via codec)
        if self.codec is not None:
            acoustic_tokens = self.codec.encode(waveform).squeeze(0)  # (K, T_audio)
        else:
            # Fallback mock tokens
            T_audio = max(1, waveform.shape[-1] // 320)
            acoustic_tokens = torch.randint(0, 4096, (4, T_audio), dtype=torch.long)

        # Speaker embedding
        spk_emb = sample.speaker_emb
        if spk_emb is None:
            spk_emb = torch.randn(256)

        return {
            "text_tokens": text_tokens,
            "acoustic_tokens": acoustic_tokens,
            "speaker_emb": spk_emb,
            "emotion_id": torch.tensor(sample.emotion_id if sample.emotion_id is not None else 0, dtype=torch.long),
        }


def voice_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collates variable-length text and acoustic sequences with dynamic padding."""
    text_list = [item["text_tokens"] for item in batch]
    acoustic_list = [item["acoustic_tokens"] for item in batch]
    spk_list = [item["speaker_emb"] for item in batch]
    emo_list = [item["emotion_id"] for item in batch]

    # Pad text sequences (PAD=0)
    padded_text = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=0)

    # Pad acoustic sequences (PAD=0 across time dimension)
    # Each acoustic tensor is (num_codebooks, T_i)
    max_acoustic_len = max(ac.shape[-1] for ac in acoustic_list)
    num_codebooks = acoustic_list[0].shape[0]
    B = len(batch)

    padded_acoustic = torch.zeros((B, num_codebooks, max_acoustic_len), dtype=torch.long)
    for i, ac in enumerate(acoustic_list):
        t_len = ac.shape[-1]
        padded_acoustic[i, :, :t_len] = ac

    speaker_embs = torch.stack(spk_list, dim=0)
    emotion_ids = torch.stack(emo_list, dim=0)

    return {
        "text_tokens": padded_text,
        "acoustic_tokens": padded_acoustic,
        "speaker_emb": speaker_embs,
        "emotion_id": emotion_ids,
    }
