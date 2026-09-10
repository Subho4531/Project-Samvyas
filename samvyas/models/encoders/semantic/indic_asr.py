"""Fine-tuned IndicConformer STT encoder (Bengali-first).

Wraps `nemo.collections.asr.models.EncDecHybridRNNTCTCBPEModel` restored
from the `.nemo` checkpoint fetched via `scripts/download_models.py`.

Pipeline per utterance batch (all under `torch.no_grad()`):
    waveform (B, T) 16kHz
      -> preprocessor -> mel features (B, feat_in, frames)
      -> conformer encoder -> hidden (B, T_frames, d_model)
      -> 768-dim adapter (identity if d_model == 768) -> hidden_states
      -> ctc_decoder -> CTC log-probs (B, T_frames, vocab)
      -> transcribe (numpy path, no temp files) -> text

VRAM (RTX 2050 4GB): model is frozen + eval; forward runs in fp16
autocast on CUDA with sequential batch-of-1 by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from samvyas.models.encoders.semantic.base import SemanticEncoderOutput

DEFAULT_REPO_ID = "ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large"
DEFAULT_FILENAME = "indicconformer_stt_bn_hybrid_rnnt_large.nemo"
HIDDEN_DIM = 768  # Vdim contract — do not change.


class IndicASREncoder(nn.Module):
    """IndicConformer semantic encoder with strict 768-dim output."""

    def __init__(
        self,
        nemo_path: Optional[Union[str, Path]] = None,
        repo_id: str = DEFAULT_REPO_ID,
        filename: str = DEFAULT_FILENAME,
        decoder: str = "ctc",
        freeze_encoder: bool = True,
        device: str = "auto",
    ) -> None:
        super().__init__()
        if decoder not in ("ctc", "rnnt"):
            raise ValueError(f"decoder must be 'ctc' or 'rnnt', got {decoder!r}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.repo_id = repo_id
        self.filename = filename
        self.decoder = decoder
        self.freeze_encoder = freeze_encoder
        self._nemo_path = Path(nemo_path) if nemo_path else None
        self._model = None
        self._adapter: Optional[nn.Module] = None  # lazy d_model -> 768
        self._enc_dim: Optional[int] = None
        self._text_cache: dict[int, list[str]] = {}

    # -- loading -------------------------------------------------------
    def _resolve_checkpoint(self) -> Path:
        if self._nemo_path is not None:
            if not self._nemo_path.exists():
                raise FileNotFoundError(f"NeMo checkpoint not found: {self._nemo_path}")
            return self._nemo_path
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=self.repo_id, filename=self.filename))

    def load(self) -> "IndicASREncoder":
        """Restore the NeMo model (downloads on first call via HF cache)."""
        if self._model is not None:
            return self
        import nemo.collections.asr as nemo_asr

        ckpt = self._resolve_checkpoint()
        self._model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
            restore_path=str(ckpt),
            map_location=self.device,
            strict=False,
        )
        if self.freeze_encoder:
            self._model.freeze()
        self._model.eval()
        self._model.change_decoding_strategy(decoder_type=self.decoder, verbose=False)
        # Adapter lives on the same device as the encoder output.
        return self

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    def _ensure_adapter(self, enc_dim: int) -> nn.Module:
        if self._adapter is None or self._enc_dim != enc_dim:
            self._enc_dim = enc_dim
            if enc_dim == HIDDEN_DIM:
                self._adapter = nn.Identity()
            else:
                self._adapter = nn.Sequential(
                    nn.LayerNorm(enc_dim),
                    nn.Linear(enc_dim, HIDDEN_DIM),
                )
            self._adapter.to(self.device).eval()
            for p in self._adapter.parameters():
                p.requires_grad_(False)
        return self._adapter

    # -- inference -----------------------------------------------------
    @staticmethod
    def _to_batch(waveform: torch.Tensor) -> torch.Tensor:
        wav = waveform.float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        elif wav.dim() == 3:
            wav = wav.squeeze(1) if wav.shape[1] == 1 else wav.mean(dim=1)
        return wav.contiguous()

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, sample_rate: int = 16_000) -> SemanticEncoderOutput:
        """Run encoder + CTC head. Returns `SemanticEncoderOutput` with (B,T,768)."""
        if sample_rate != 16_000:
            raise ValueError(f"Expected 16000Hz audio, got {sample_rate}Hz.")
        model = self.model
        wav = self._to_batch(waveform).to(self.device)
        lengths = torch.full((wav.shape[0],), wav.shape[1], dtype=torch.long, device=self.device)

        use_amp = self.device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            features, feat_len = model.preprocessor(input_signal=wav, length=lengths)
            encoded, enc_len = model.encoder(audio_signal=features, length=feat_len)
            # encoded: (B, T_frames, d_model)
            adapter = self._ensure_adapter(encoded.shape[-1])
            hidden = adapter(encoded)
            ctc_decoder = getattr(model, "ctc_decoder", None)
            if ctc_decoder is None:
                raise RuntimeError("NeMo model has no ctc_decoder; cannot produce CTC logits.")
            # CTC head was trained on raw encoder dim — always decode from `encoded`.
            logits = ctc_decoder(encoder_output=encoded)
            tokens = logits.argmax(dim=-1)

        if hidden.shape[-1] != HIDDEN_DIM:  # pragma: no cover - contract guard
            raise RuntimeError(f"Vdim violated: hidden dim {hidden.shape[-1]} != {HIDDEN_DIM}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return SemanticEncoderOutput(
            logits=logits.float().cpu(),
            hidden_states=hidden.float().cpu(),
            tokens=tokens.cpu(),
            lengths=enc_len.cpu() if torch.is_tensor(enc_len) else None,
        )

    def forward(self, waveform: torch.Tensor, sample_rate: int = 16_000) -> SemanticEncoderOutput:
        return self.encode(waveform, sample_rate)

    @torch.no_grad()
    def transcribe(self, waveform: torch.Tensor, sample_rate: int = 16_000) -> list[str]:
        """Greedy-decode waveforms to text via NeMo `transcribe` (numpy path)."""
        if sample_rate != 16_000:
            raise ValueError(f"Expected 16000Hz audio, got {sample_rate}Hz.")
        model = self.model
        wav = self._to_batch(waveform).cpu().numpy()
        audios = [utt.astype("float32") for utt in wav]
        out = model.transcribe(audios, batch_size=1, verbose=False)
        texts: list[str] = []
        for hyp in out:
            texts.append(hyp.text if hasattr(hyp, "text") else str(hyp))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return texts

    def encode_with_text(
        self, waveform: torch.Tensor, sample_rate: int = 16_000
    ) -> SemanticEncoderOutput:
        """Single call returning hidden states (B,T,768) plus decoded text."""
        output = self.encode(waveform, sample_rate)
        output.text = self.transcribe(waveform, sample_rate)
        return output
