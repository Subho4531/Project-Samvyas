"""Validate reference conditioning, exports, and OOM fallback without weights."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from samvyas.models.tts.indic_f5 import IndicF5Synthesizer


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "ref.wav"
    sf.write(path, 0.2 * np.sin(np.arange(32000) * 0.1), 16000)
    return path


class Backend:
    def __init__(self, device):
        self.device = device
        self.closed = False
        self.args = None

    def generate(self, *args):
        self.args = args
        return (0.2 * np.sin(np.arange(24000) * 0.1)).astype(np.float32)

    def close(self):
        self.closed = True


def test_local_base_weights_do_not_download_gated_files(tmp_path, monkeypatch):
    import torch

    from samvyas.models.tts.indic_f5 import load_indic_model

    (tmp_path / "vocab.txt").write_text(" \na\n", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"mock weights")
    monkeypatch.setenv("SAMVYAS_INDICF5_DIR", str(tmp_path))
    monkeypatch.setitem(
        sys.modules,
        "f5_tts.infer.utils_infer",
        SimpleNamespace(load_model=lambda *args, **kwargs: torch.nn.Linear(2, 2)),
    )
    monkeypatch.setitem(sys.modules, "f5_tts.model", SimpleNamespace(DiT=object))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            hf_hub_download=lambda **kwargs: pytest.fail(
                "Local weights must bypass gated downloads"
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "safetensors.torch",
        SimpleNamespace(
            load_file=lambda *args, **kwargs: {
                "ema_model.weight": torch.eye(2),
                "ema_model.bias": torch.zeros(2),
            }
        ),
    )
    model = load_indic_model()
    assert torch.equal(model.weight, torch.eye(2))
    (tmp_path / "vocab.txt").unlink()
    with pytest.raises(FileNotFoundError, match="Local IndicF5"):
        load_indic_model()


def test_synthesis_and_exports(reference, tmp_path):
    backend = Backend("cpu")
    model = IndicF5Synthesizer("cpu", backend_factory=lambda _: backend)
    out = model.synthesize("নতুন কথা।", reference, "পুরোনো কথা।", steps=16)
    assert backend.args == ("নতুন কথা।", reference, "পুরোনো কথা।", 16)
    assert out.native_audio.shape == (24000,) and out.audio_16k.shape == (16000,)
    assert out.native_audio.dtype == out.audio_16k.dtype == np.float32
    assert np.max(np.abs(out.audio_16k)) <= 1
    paths = out.save(tmp_path / "output", "test")
    assert [sf.info(p).samplerate for p in paths] == [24000, 16000]
    assert all(sf.info(p).channels == 1 for p in paths)
    before = paths[0].read_bytes()
    with pytest.raises(FileExistsError):
        out.save(tmp_path / "output", "test")
    assert paths[0].read_bytes() == before
    model.close()
    assert backend.closed


@pytest.mark.parametrize(
    "text,transcript,steps",
    [("", "ref", 32), ("target", " ", 32), ("target", "ref", 0), ("target", "ref", 101)],
)
def test_bad_inputs_do_not_load(reference, text, transcript, steps):
    model = IndicF5Synthesizer("cpu", backend_factory=lambda _: pytest.fail("must not load"))
    with pytest.raises(ValueError):
        model.synthesize(text, reference, transcript, steps)


@pytest.mark.parametrize(
    "rate,channels,seconds,level",
    [(24000, 1, 2, 0.1), (16000, 2, 2, 0.1), (16000, 1, 16, 0.1), (16000, 1, 2, 0)],
)
def test_bad_reference(tmp_path, rate, channels, seconds, level):
    path = tmp_path / "ref.wav"
    sf.write(path, np.full((rate * seconds, channels), level, dtype=np.float32), rate)
    with pytest.raises(ValueError):
        IndicF5Synthesizer("cpu").synthesize("text", path, "reference")


def test_explicit_cpu_fallback(reference, monkeypatch):
    instances = []

    class OOMBackend(Backend):
        def generate(self, *args):
            if self.device == "cuda":
                raise RuntimeError("CUDA out of memory")
            return super().generate(*args)

    def factory(device):
        instance = OOMBackend(device)
        instances.append(instance)
        return instance

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    model = IndicF5Synthesizer("cuda", cpu_fallback=True, backend_factory=factory)
    with pytest.warns(RuntimeWarning, match="retrying on CPU"):
        output = model.synthesize("new", reference, "old")
    assert output.device == "cpu"
    assert instances[0].closed and [b.device for b in instances] == ["cuda", "cpu"]


def test_oom_without_fallback(reference):
    class Failed(Backend):
        def generate(self, *args):
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="--cpu-fallback"):
        IndicF5Synthesizer("cuda", backend_factory=Failed).synthesize("new", reference, "old")


@pytest.mark.parametrize("native", [np.array([np.nan]), np.zeros((1, 24000)), np.array([])])
def test_reject_bad_backend_output(reference, native):
    backend = Backend("cpu")
    backend.generate = lambda *args: native
    with pytest.raises(ValueError):
        IndicF5Synthesizer("cpu", backend_factory=lambda _: backend).synthesize(
            "new", reference, "old"
        )


def test_int16_output(reference):
    backend = Backend("cpu")
    backend.generate = lambda *args: np.full(24000, 16384, dtype=np.int16)
    out = IndicF5Synthesizer("cpu", backend_factory=lambda _: backend).synthesize(
        "new", reference, "old"
    )
    assert np.allclose(out.native_audio, 0.5)


def test_cuda_load_failure_can_fall_back(reference, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )

    def factory(device):
        if device == "cuda":
            raise RuntimeError("CUDA out of memory during load")
        return Backend(device)

    with pytest.warns(RuntimeWarning):
        result = IndicF5Synthesizer("cuda", True, factory).synthesize("new", reference, "old")
    assert result.device == "cpu"


def test_cli_three_sentence_baseline(tmp_path, monkeypatch):
    from samvyas.data.voice_dataset import VoiceDataset, VoicePrompt
    from scripts import test_indic_f5 as cli

    dataset = VoiceDataset(tmp_path / "dataset")
    row = dataset.accept(
        VoicePrompt("one", "আমি ভালো আছি।", "normal"),
        "আমি ভালো আছি।",
        np.full(96000, 0.1, dtype=np.float32),
    )
    monkeypatch.setattr(
        cli,
        "IndicF5Synthesizer",
        lambda *args, **kwargs: IndicF5Synthesizer("cpu", backend_factory=Backend),
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_indic_f5.py",
            "--dataset",
            str(dataset.root),
            "--clip-id",
            row["clip_id"],
            "--output-dir",
            str(output),
        ],
    )
    assert cli.main() == 0
    import json

    report = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert len(report["results"]) == 3
    assert report["reference_transcript"] == "আমি ভালো আছি।"
    assert len(list(output.glob("*.wav"))) == 6
    assert report["listening_review"].startswith("pending")


def test_backend_loads_ema_and_freezes_modules(monkeypatch):
    import torch

    from samvyas.models.tts.indic_f5 import IndicF5Backend

    modules = []
    downloads = []

    def download(**kwargs):
        downloads.append(kwargs)
        return kwargs["filename"]

    def make_module(*args, **kwargs):
        module = torch.nn.Linear(2, 2)
        modules.append(module)
        return module

    monkeypatch.setitem(sys.modules, "f5_tts", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "f5_tts.model", SimpleNamespace(DiT=object))
    monkeypatch.setitem(sys.modules, "f5_tts.infer", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "f5_tts.infer.utils_infer",
        SimpleNamespace(load_model=make_module, load_vocoder=make_module),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=download),
    )
    monkeypatch.setitem(
        sys.modules,
        "safetensors.torch",
        SimpleNamespace(
            load_file=lambda *args, **kwargs: {
                "ema_model._orig_mod.weight": torch.eye(2),
                "ema_model._orig_mod.bias": torch.zeros(2),
                "vocoder.unrelated": torch.zeros(1),
            }
        ),
    )
    backend = IndicF5Backend("cpu")
    assert torch.equal(backend.model.weight, torch.eye(2))
    assert all(not module.training for module in modules)
    assert all(not param.requires_grad for module in modules for param in module.parameters())
    public_downloads = [call for call in downloads if call["repo_id"] == "charactr/vocos-mel-24khz"]
    assert len(public_downloads) == 2
    assert all(call["token"] is False and call["revision"] for call in public_downloads)
    backend.close()
    assert backend.model is None and backend.vocoder is None


def test_vocoder_decodes_in_full_precision_and_preserves_input_gradients():
    import torch

    from samvyas.models.tts.indic_f5 import _FullPrecisionVocoder

    class Vocoder(torch.nn.Module):
        def decode(self, mel):
            assert mel.dtype == torch.float32
            assert not torch.is_autocast_enabled("cpu")
            return mel.mean(dim=1)

    mel = torch.ones(1, 100, 8, dtype=torch.float16, requires_grad=True)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        waveform = _FullPrecisionVocoder(Vocoder()).decode(mel)
    assert waveform.shape == (1, 8) and waveform.dtype == torch.float32
    assert torch.isfinite(waveform).all()
    waveform.sum().backward()
    assert mel.grad is not None and torch.isfinite(mel.grad).all()


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_vocoder_rejects_invalid_mel_before_decoding(value):
    import torch

    from samvyas.models.tts.indic_f5 import _FullPrecisionVocoder

    with pytest.raises(RuntimeError, match="non-finite mel"):
        _FullPrecisionVocoder(torch.nn.Identity()).decode(torch.full((1, 100, 8), value))


@pytest.mark.skipif(
    os.getenv("SAMVYAS_RUN_TTS_REAL") != "1",
    reason="requires IndicF5 access and an explicitly verified voice reference",
)
def test_real_indic_f5_reference():
    """Opt-in model smoke test; file generation cannot establish voice similarity."""
    from dotenv import load_dotenv

    load_dotenv()
    reference = os.getenv("SAMVYAS_TTS_REFERENCE")
    transcript = os.getenv("SAMVYAS_TTS_REFERENCE_TEXT")
    if not reference or not transcript:
        pytest.fail(
            "Set SAMVYAS_TTS_REFERENCE and SAMVYAS_TTS_REFERENCE_TEXT to your verified pair."
        )
    model = IndicF5Synthesizer(os.getenv("SAMVYAS_TTS_DEVICE", "cpu"))
    try:
        output = model.synthesize("আজ তোমার সঙ্গে একটি নতুন গল্প বলতে চাই।", reference, transcript)
        assert output.native_audio.size > 0
        assert np.isfinite(output.audio_16k).all()
        assert np.max(np.abs(output.audio_16k)) <= 1
    finally:
        model.close()
