# Agent Execution Procedures & Runbook

This document details the step-by-step procedures for autonomous coding agents implementing components in **Samvyas**.

---

## 1. Implementing an Audio / ML Module

Follow this 5-step checklist whenever building a new pipeline component:

1. **Design Abstract Base Interface**:
   - Define expected inputs, outputs, and tensor shapes.
   - For example: `BaseAudioEncoder` with `extract_features(waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`.

2. **Implement Core Logic**:
   - Keep classes modular and decoupled.
   - Use PyTorch native operations or established libraries (`torchaudio`, `deepfilternet`, `transformers`).
   - Add explicit shape validation:
     ```python
     assert waveform.ndim == 2, f"Expected (B, T) audio, got {waveform.shape}"
     ```

3. **Handle Device Placement & Precision**:
   - Support `.to(device)` cleanly.
   - Ensure buffers and submodules properly move with the model.

4. **Write Unit Tests First / Concurrently**:
   - Add a test file in `tests/test_<module_name>.py`.
   - Test with synthetic/dummy tensors (e.g. `torch.randn(2, 16000)`).
   - Test with real sample audio from `assets/audio/` when applicable.

5. **Expose Clean Exports**:
   - Export primary classes through the subpackage's `__init__.py`.

---

## 2. Audio Pipeline Constraints

- **Input Sample Rate:** Must always be standardized to 16,000 Hz.
- **Microphone Stream Processing:** Chunk size should match real-time hop length (typically 10ms - 20ms frames, or 1-3 second streaming chunks).
- **Latency Budget:** Denoising + Dual Encoding + Projection should target < 150ms latency on modern GPU / < 300ms on CPU.

---

## 3. Git & Checkpoint Policy

- Checkpoints downloaded via `huggingface_hub` must NEVER be copied into Git-tracked directories.
- Always use `.cache/huggingface` or a dedicated `.checkpoints/` folder (enforced in `.gitignore`).
