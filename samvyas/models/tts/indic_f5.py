"""Frozen Bengali reference-conditioned IndicF5 synthesis.

Explicit checkpoint loading avoids remote-code/device placement surprises.
The F5 architecture and loading convention follow AI4Bharat's repository.
Supports base weights and locally adapted checkpoints with the same vocabulary.
"""

from __future__ import annotations

import gc
import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

if TYPE_CHECKING:
    from torch import Tensor, nn

MODEL_REVISION = "fb652a96e508091045890dae5bb99fe511facbcf"
VOCODER_REVISION = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
NATIVE_SAMPLE_RATE = 24_000


def load_indic_model(checkpoint: str | Path | None = None) -> nn.Module:
    """Load the pinned CFM on CPU without a vocoder; preserve training interfaces."""
    from f5_tts.infer.utils_infer import load_model
    from f5_tts.model import DiT
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    token = os.getenv("HF_TOKEN") or os.getenv("HF_ACESS_TOKEN")
    common = {"repo_id": "ai4bharat/IndicF5", "revision": MODEL_REVISION, "token": token}
    local_base = os.getenv("SAMVYAS_INDICF5_DIR")
    if local_base:
        vocab = str(Path(local_base) / "vocab.txt")
        weights = str(checkpoint) if checkpoint else str(Path(local_base) / "model.safetensors")
        if not Path(vocab).is_file() or not Path(weights).is_file():
            raise FileNotFoundError("Local IndicF5 dataset needs vocab.txt and model.safetensors.")
    else:
        vocab = hf_hub_download(filename="checkpoints/vocab.txt", **common)
        weights = (
            str(checkpoint)
            if checkpoint
            else hf_hub_download(filename="model.safetensors", **common)
        )
    model = load_model(
        DiT,
        {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "conv_layers": 4},
        mel_spec_type="vocos",
        vocab_file=vocab,
        device="cpu",
    )
    state = load_file(weights, device="cpu")
    ema = {
        key.removeprefix("ema_model.").removeprefix("_orig_mod."): value
        for key, value in state.items()
        if key.startswith("ema_model.")
    }
    if not ema:
        raise RuntimeError("IndicF5 checkpoint has no expected ema_model weights.")
    model.load_state_dict(ema, strict=True)
    return model


class _FullPrecisionVocoder:
    """Keep Vocos convolutions/exponentiation/FFT outside decoder autocast."""

    def __init__(self, vocoder: nn.Module) -> None:
        self.vocoder = vocoder

    def decode(self, mel: Tensor) -> Tensor:
        """Decode (B, 100, T) mel features as float32 into (B, T_audio)."""
        import torch

        if not torch.isfinite(mel).all():
            raise RuntimeError("IndicF5 decoder produced non-finite mel features.")
        with torch.amp.autocast(mel.device.type, enabled=False):
            return self.vocoder.decode(mel.float())


class SynthesisBackend(Protocol):
    """An independently loadable single-utterance synthesis backend."""

    def generate(self, text: str, reference: Path, transcript: str, steps: int) -> np.ndarray:
        """Return mono (T,) speech at 24 kHz."""
        ...

    def close(self) -> None:
        """Release model resources before switching devices."""
        ...


class IndicF5Backend:
    """Frozen diffusion transformer and Vocos, loaded on an explicit device."""

    def __init__(self, device: str, checkpoint: str | Path | None = None) -> None:
        from f5_tts.infer.utils_infer import load_vocoder
        from huggingface_hub import hf_hub_download

        self.device = device
        self.model = None
        self.vocoder = None
        try:
            self.model = load_indic_model(checkpoint)
        except Exception as exc:
            if "403" in str(exc) or "401" in str(exc) or "gated" in str(exc).lower():
                raise RuntimeError(
                    "IndicF5 access is required: accept conditions at "
                    "https://huggingface.co/ai4bharat/IndicF5 and configure HF_TOKEN."
                ) from exc
            raise
        # Public vocoder downloads must not inherit a stale cached OAuth token.
        vocoder_config = hf_hub_download(
            repo_id="charactr/vocos-mel-24khz",
            filename="config.yaml",
            revision=VOCODER_REVISION,
            token=False,
        )
        hf_hub_download(
            repo_id="charactr/vocos-mel-24khz",
            filename="pytorch_model.bin",
            revision=VOCODER_REVISION,
            token=False,
        )
        self.vocoder = load_vocoder(
            vocoder_name="vocos",
            is_local=True,
            local_path=str(Path(vocoder_config).parent),
            device="cpu",
        )
        for module in (self.model, self.vocoder):
            module.eval()
            module.requires_grad_(False)
            module.to(device)

    def generate(self, text: str, reference: Path, transcript: str, steps: int) -> np.ndarray:
        """Synthesize one text batch, preserving the complete reference/transcript."""
        import torch
        from f5_tts.infer.utils_infer import infer_batch_process

        audio, sample_rate = sf.read(reference, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio.T.copy())
        # BF16 preserves the exponent range needed by the flow decoder. FP16
        # overflow can produce invalid mel features even with a float32 vocoder.
        use_amp = self.device == "cuda" and torch.cuda.is_bf16_supported()
        with (
            torch.inference_mode(),
            torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=use_amp,
            ),
        ):
            result, rate, _ = infer_batch_process(
                (waveform, sample_rate),
                transcript + " ",
                [text],
                self.model,
                _FullPrecisionVocoder(self.vocoder),
                device=self.device,
                nfe_step=steps,
            )
        if rate != NATIVE_SAMPLE_RATE:
            raise RuntimeError(f"Expected 24000 Hz backend output, got {rate}.")
        return result

    def close(self) -> None:
        """Drop both model references before device fallback."""
        self.model = None
        self.vocoder = None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@dataclass(frozen=True)
class TTSOutput:
    """Native and canonical mono (T,) float32 arrays in [-1, 1]."""

    native_audio: np.ndarray
    audio_16k: np.ndarray
    device: str
    peak_scale: float

    def save(self, output_dir: str | Path, name: str) -> tuple[Path, Path]:
        """Write both rates without overwriting existing outputs."""
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("Output name must be a plain filename stem.")
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        paths = (root / f"{name}_24k.wav", root / f"{name}_16k.wav")
        if any(path.exists() for path in paths):
            raise FileExistsError("Output already exists; choose a new output directory/name.")
        for path, audio, rate in zip(paths, (self.native_audio, self.audio_16k), (24_000, 16_000)):
            with path.open("xb") as handle:
                sf.write(handle, audio, rate, subtype="PCM_24", format="WAV")
        return paths


class IndicF5Synthesizer:
    """Lazy batch-one synthesis with opt-in CUDA OOM fallback to CPU."""

    def __init__(
        self,
        device: str = "auto",
        cpu_fallback: bool = False,
        backend_factory: Callable[[str], SynthesisBackend] = IndicF5Backend,
    ) -> None:
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("Device must be auto, cpu, or cuda.")
        self.device = device
        self.cpu_fallback = cpu_fallback
        self._factory = backend_factory
        self._backend: SynthesisBackend | None = None

    def close(self) -> None:
        """Release the loaded backend."""
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    def synthesize(
        self,
        text: str,
        reference: str | Path,
        transcript: str,
        steps: int = 32,
    ) -> TTSOutput:
        """Use verified reference words to generate different target words.

        The reference must be a mono 16 kHz WAV, 1–15 seconds long. It is not
        clipped automatically, because that would invalidate the transcript.
        """
        if not text.strip() or not transcript.strip():
            raise ValueError("Target text and verified reference transcript are required.")
        if steps < 1 or steps > 100:
            raise ValueError("Steps must be between 1 and 100.")
        reference = Path(reference)
        info = sf.info(reference)
        if info.format != "WAV" or info.samplerate != 16_000 or info.channels != 1:
            raise ValueError("Reference must be a mono 16000 Hz WAV from the recorder.")
        if not 1 <= info.duration <= 15:
            raise ValueError("Select a complete 1–15 second reference and matching transcript.")
        ref_audio, _ = sf.read(reference, dtype="float32")
        if not np.isfinite(ref_audio).all() or np.max(np.abs(ref_audio)) > 1:
            raise ValueError("Reference must be finite audio in [-1, 1].")
        if float(np.sqrt(np.mean(ref_audio.astype(np.float64) ** 2))) < 0.001:
            raise ValueError("Reference is near-silent; select another accepted take.")
        if self.device == "auto":
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        retry_cpu = False
        try:
            if self._backend is None:
                self._backend = self._factory(self.device)
            native = self._backend.generate(text.strip(), reference, transcript.strip(), steps)
        except RuntimeError as exc:
            oom = "out of memory" in str(exc).lower()
            if not (oom and self.device == "cuda" and self.cpu_fallback):
                if oom and self.device == "cuda":
                    raise RuntimeError(
                        "CUDA out of memory. Retry with --cpu-fallback or --device cpu."
                    ) from exc
                raise
            retry_cpu = True
        # Retry outside the except block so the failed traceback releases tensors.
        if retry_cpu:
            warnings.warn("CUDA out of memory; retrying on CPU (may be slow).", RuntimeWarning)
            self.close()
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.device = "cpu"
            self._backend = self._factory("cpu")
            native = self._backend.generate(text.strip(), reference, transcript.strip(), steps)
        native = np.asarray(native)
        if native.dtype == np.int16:
            native = native.astype(np.float32) / 32768.0
        elif not np.issubdtype(native.dtype, np.floating):
            raise ValueError("Backend must return floating point or int16 audio.")
        native = native.astype(np.float32)
        if native.ndim != 1 or native.size == 0 or not np.isfinite(native).all():
            raise ValueError("Backend returned invalid mono (T,) audio.")
        # One shared attenuation preserves the native/canonical level relationship.
        canonical = resample_poly(native, 2, 3).astype(np.float32)
        scale = max(1.0, float(np.max(np.abs(native))), float(np.max(np.abs(canonical))))
        return TTSOutput(native / scale, canonical / scale, self.device, scale)
