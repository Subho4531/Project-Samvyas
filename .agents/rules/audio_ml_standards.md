---
description: Coding and architectural guidelines for Audio & Deep Learning modules in Samvyas
globs: samvyas/**/*.py, tests/**/*.py
always_on: true
---

# Audio & ML Engineering Standards

1. **PyTorch Tensor Conventions**:
   - Audio Waveform tensors: `(B, T_samples)` where values are normalized in range `[-1.0, 1.0]`.
   - Hidden representation tensors: `(B, T_frames, D)`.
   - Always document tensor shape transformations in module comments.

2. **Sample Rate Guarantees**:
   - Audio preprocessing must explicitly verify `sample_rate == 16000` or invoke resamplers.

3. **External Model Weight Loading**:
   - Model loaders must accept a local cache directory or HuggingFace repo ID.
   - Do not bundle multi-megabyte weights in repository code.

4. **Testing Rigor**:
   - Every neural module must have a unit test verifying forward pass and backward pass (gradient flow).
