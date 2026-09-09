---
description: Technical rules and interface contracts for Samvyas STT and Dual-Stream Audio Perception
globs: samvyas/**/*.py, configs/**/*.yaml, tests/**/*.py
---

# STT & Dual-Stream Perception Architecture Rules

When implementing, modifying, or testing components of the STT pipeline in `samvyas/`, adhere strictly to these interface contracts and constraints:

## 1. Audio Frontend Contract (`samvyas/audio/`)
- Input audio must always be validated or resampled to **16,000 Hz**, mono channel, float32 normalized in `[-1.0, 1.0]`.
- The DeepFilterNet v2 wrapper (`DeepFilterDenoiser`) must support:
  - `enhance(audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor`
  - A bypass flag (`enabled: bool = True`) to allow skipping enhancement when dealing with clean synthetic or pre-enhanced audio.
  - Streaming chunk processing buffer option for real-time microphone input.

## 2. Semantic / STT Branch (`samvyas/models/encoders/semantic/`)
- Encapsulates Indic ASR (supporting Bengali, Hindi, English).
- Forward signature must return a dataclass or structured dictionary:
  ```python
  @dataclass
  class SemanticEncoderOutput:
      logits: torch.Tensor             # (B, T_frames, vocab_size) for CTC/RNNT
      hidden_states: torch.Tensor      # (B, T_frames, 768) intermediate semantic representations
      tokens: Optional[torch.Tensor]   # Argmax / decoded token IDs
  ```
- The semantic representation dimension must be strictly **768** ($V_{\text{dim}} = 768$).
- Must support freezing parameters (`freeze_encoder: bool = True`) during projector and LLM training.

## 3. Paralinguistic Analysis Branch (`samvyas/models/encoders/paralinguistic/`)
- Encapsulates acoustic, prosodic, emotional, and vocal imperfection extraction from raw audio using AI4Bharat IndicWav2Vec backbone.
- Forward signature:
  ```python
  @dataclass
  class ParalinguisticEncoderOutput:
      acoustic_embeddings: torch.Tensor # (B, T_frames, D_acoustic)
      prosody_latents: Optional[torch.Tensor] = None
  ```
- Must align frame rate or temporal resolution with the semantic branch, or provide interpolation/pooling to match sequence lengths.

## 4. Audio Projector & Downsampler (`samvyas/models/projectors/`)
- **Input:** Fused representation of Semantic ($d=768$) and Paralinguistic ($d=D_{\text{para}}$) features.
- **Downsampling:** Must implement a temporal downsampler (e.g. strided 1D convolution or pooling) with configurable downsampling factors ($2\times$, $4\times$, or $8\times$) to keep sequence lengths compact for the LLM.
- **Output Projection:** Multi-Layer Perceptron (MLP) with GELU/SwiGLU activation projecting downsampled features to **$d = 4096$** (the LLM hidden dimension).
- Output shape: `(B, T_downsampled, 4096)`.

## 5. Device and Memory Constraints
- RTX 2050 (4GB VRAM): Encoders must run in `torch.no_grad()` and evaluation mode during local inference and testing.
- Batch testing scripts should run with batch size 1 or 2 with short audio segments ($\le 5$ seconds) to avoid CUDA Out-of-Memory (OOM).
