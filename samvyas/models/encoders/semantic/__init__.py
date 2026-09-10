"""Semantic / STT branch (Bengali/Indic ASR, Vdim=768)."""

from samvyas.models.encoders.semantic.base import SemanticEncoderOutput
from samvyas.models.encoders.semantic.indic_asr import IndicASREncoder

__all__ = ["SemanticEncoderOutput", "IndicASREncoder"]
