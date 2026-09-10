"""DeepFilterNet v2 wrapper (diagram: noise-removal block).

Wraps the installed `df` package (`DeepFilterNet==0.5.6`,
`from df.enhance import enhance, init_df`).

Notes:
    - DeepFilterNet natively runs at 48kHz (`df_state.sr()`); this wrapper
      resamples 16kHz -> model SR -> 16kHz so the rest of Samvyas stays
      canonical 16kHz mono.
    - Model is lazily initialized on first `enhance` call so that
      `enabled=False` (clean/synthetic audio) never triggers a download.
    - `model_base_dir=None` loads the pretrained DeepFilterNet2 weights
      (matches public/stt.png); weights live in the DeepFilter cache
      outside the repo (git-ignored).
"""

from __future__ import annotations

import torch

from samvyas.audio.denoiser.base import BaseDenoiser


class DeepFilterDenoiser(BaseDenoiser):
    """DeepFilterNet v2 speech enhancement.

    Args:
        sample_rate: expected input SR (default 16000).
        enabled: if False, `enhance` is a pass-through (no model load).
        chunk_secs: process in chunks of this many seconds (0 = whole
            utterance). Useful for long files on 4GB VRAM.
        post_filter: extra post-filter noise reduction.
        atten_lim_db: optional attenuation limit in dB.
        device: 'auto' | 'cuda' | 'cpu'.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        enabled: bool = True,
        chunk_secs: float = 0,
        post_filter: bool = False,
        atten_lim_db: float | None = None,
        device: str = "auto",
    ) -> None:
        super().__init__(sample_rate=sample_rate, enabled=enabled)
        self.chunk_secs = chunk_secs
        self.post_filter = post_filter
        self.atten_lim_db = atten_lim_db
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._model = None
        self._df_state = None
        self._model_sr: int | None = None

    # -- lazy init -----------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from df.enhance import init_df

        self._model, self._df_state, _suffix = init_df(
            model_base_dir=None,  # pretrained DeepFilterNet2
            default_model="DeepFilterNet2",
            post_filter=self.post_filter,
            epoch="best",
            log_file=None,
        )
        self._model.eval()
        try:
            self._model_sr = int(self._df_state.sr())
        except Exception:
            self._model_sr = 48_000
        # init_df moves the model to GPU when available; keep our device
        # choice aligned with where the model actually lives.
        try:
            param = next(self._model.parameters())
            self.device = param.device
        except Exception:
            pass

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _normalize_layout(audio: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Return (B, T) float tensor plus flag whether input had channel dim."""
        wav = audio.float()
        had_channel = False
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        elif wav.dim() == 3:
            if wav.shape[1] == 1:
                had_channel = True
                wav = wav.squeeze(1)
            else:  # downmix multi-channel
                wav = wav.mean(dim=1)
        return wav.contiguous(), had_channel

    def _resample(self, wav: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
        if orig_sr == new_sr:
            return wav
        try:
            import torchaudio.functional as F

            return F.resample(wav, orig_freq=orig_sr, new_freq=new_sr)
        except Exception:
            ratio = new_sr / orig_sr
            new_len = max(1, int(wav.shape[-1] * ratio))
            out = torch.nn.functional.interpolate(
                wav.unsqueeze(1), size=new_len, mode="linear", align_corners=False
            )
            return out.squeeze(1)

    @torch.no_grad()
    def _enhance_utterance(self, wav_1d: torch.Tensor) -> torch.Tensor:
        """Enhance a single (T,) tensor at input sample_rate. Returns (T,)."""
        assert self._model is not None and self._df_state is not None
        from df.enhance import enhance

        model_sr = self._model_sr or 48_000
        wav_up = self._resample(wav_1d.unsqueeze(0), self.sample_rate, model_sr)

        # df.enhance runs STFT via numpy -> input must be a CPU tensor even
        # though the model itself lives on CUDA; device transfer is internal.
        df_in = wav_up.cpu()
        if not (df_in.dim() == 2 and df_in.shape[0] == 1):
            df_in = df_in.reshape(1, -1)
        enhanced = enhance(self._model, self._df_state, df_in, atten_lim_db=self.atten_lim_db)
        if torch.is_tensor(enhanced):
            enh = enhanced.detach().float()
        else:  # pragma: no cover - defensive for API drift
            enh = torch.as_tensor(enhanced, dtype=torch.float32)
        if enh.dim() == 2:
            enh = enh.mean(dim=0) if enh.shape[0] > 1 else enh.squeeze(0)
        else:
            enh = enh.reshape(-1)
        enh = self._resample(enh.unsqueeze(0).cpu(), model_sr, self.sample_rate).squeeze(0)

        # Length guard: resampling round-trips can drift by a sample.
        target_len = wav_1d.shape[-1]
        if enh.shape[-1] > target_len:
            enh = enh[:target_len]
        elif enh.shape[-1] < target_len:
            enh = torch.nn.functional.pad(enh, (0, target_len - enh.shape[-1]))
        return enh

    # -- public API ----------------------------------------------------
    @torch.no_grad()
    def enhance(self, audio: torch.Tensor, sample_rate: int = 16_000) -> torch.Tensor:
        if not self.enabled:
            return audio
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Expected {self.sample_rate}Hz audio, got {sample_rate}Hz. "
                "Resample with samvyas.audio.io.load_audio first."
            )
        self._ensure_model()
        wav, had_channel = self._normalize_layout(audio)

        out_items: list[torch.Tensor] = []
        chunk_len = int(self.chunk_secs * self.sample_rate) if self.chunk_secs else 0
        was_1d = audio.dim() == 1
        for utt in wav:
            if chunk_len and utt.shape[-1] > chunk_len:
                chunks = [
                    self._enhance_utterance(c)
                    for c in utt.split(chunk_len)
                ]
                out_items.append(torch.cat(chunks, dim=-1))
            else:
                out_items.append(self._enhance_utterance(utt))
        out = torch.stack(out_items, dim=0)  # (B, T)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if was_1d:
            return out.squeeze(0)
        if had_channel:
            return out.unsqueeze(1)
        return out
