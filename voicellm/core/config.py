"""Configuration schemas for VoiceLLM."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class AudioConfig:
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    target_lufs: float = -20.0
    normalize_loudness: bool = True
    trim_silence: bool = True


@dataclass
class CodecConfig:
    codec_type: str = "snac"  # "snac", "dac", "encodec"
    model_name: str = "hubertsiuzdak/snac_24khz"
    num_codebooks: int = 4
    codebook_size: int = 4096
    sample_rate: int = 24000
    token_rate_hz: int = 12  # frame rate of coarse tokens


@dataclass
class SpeakerEncoderConfig:
    encoder_type: str = "ecapa_tdnn"  # "ecapa_tdnn", "campplus", "linear_ref"
    embedding_dim: int = 256
    pretrained_path: Optional[str] = None
    freeze: bool = True


@dataclass
class EmotionConfig:
    use_emotion_tokens: bool = True
    num_emotions: int = 16  # e.g., happy, sad, angry, neutral, whispered, excited, etc.
    emotion_dim: int = 128
    prosody_conditioning: bool = True


@dataclass
class ModelConfig:
    # Backbone architecture
    vocab_size_text: int = 32000
    num_codebooks: int = 4
    codebook_vocab_size: int = 4096
    hidden_dim: int = 768
    intermediate_dim: int = 2048
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: Optional[int] = 4  # Grouped Query Attention
    head_dim: int = 64
    max_seq_len: int = 4096
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0
    
    # Sub-modules
    speaker_encoder: SpeakerEncoderConfig = field(default_factory=SpeakerEncoderConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    delay_pattern_step: int = 1  # Interleaving delay per codebook


@dataclass
class TrainingConfig:
    experiment_name: str = "voicellm_base"
    batch_size: int = 4
    grad_accum_steps: int = 8
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 1000
    max_steps: int = 50000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"  # "fp16", "bf16", "no"
    seed: int = 42
    log_every_steps: int = 20
    eval_every_steps: int = 500
    save_every_steps: int = 1000
    output_dir: str = "./outputs"
    wandb_project: Optional[str] = "voicellm-research"
    hf_repo_id: Optional[str] = None


@dataclass
class VoiceLLMConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    codec: CodecConfig = field(default_factory=CodecConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VoiceLLMConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**data.get("model", {})),
            audio=AudioConfig(**data.get("audio", {})),
            codec=CodecConfig(**data.get("codec", {})),
            training=TrainingConfig(**data.get("training", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, default_flow_style=False)
