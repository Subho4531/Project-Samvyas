# Samvyas: STT & Dual-Stream Perception Implementation Plan

> **Scope**: Implementation plan for the Speech-to-Text (STT) and Dual-Stream Audio Perception frontend mapped to [`public/stt.png`](./public/stt.png).  
> **Status**: Phase 1 (STT-only: denoiser + Bengali semantic branch + pipeline) implemented & verified — 18/18 tests pass on RTX 2050 (see MEMORY.md).

---

## 1. Architectural Objective

Build the complete audio input, enhancement, dual-stream feature extraction, and projection pipeline:

```
[16kHz Audio Input]
        │
        ▼
[DeepFilterNet v2] (Noise Removal & Speech Enhancement)
        │
   ┌────┴───────────────────────────┐
   ▼                                ▼
[Branch 1: Semantic / STT]     [Branch 2: Paralinguistic]
Fine-tuned IndicWav2Vec/       AI4Bharat IndicWav2Vec
IndicConformer + CTC Head      Acoustics, Prosody & Imperfections
   │ (Vdim = 768)                   │ (D_para)
   └───────────────┬────────────────┘
                   ▼
       [Audio Projector Module]
       - Temporal Downsampler (Conv1d, stride 2/4)
       - MLP Projection Layer
                   │
                   ▼
     [Projected Embeddings (d = 4096)]
      (Ready for Voice LLM Backbone)
```

---

## 2. Phased Implementation Roadmap

### Phase 1: Environment & Dependency Foundation
- **Goal**: Validate external packages and Hugging Face model access.
- **Tasks**:
  1. Verify installed dependencies: `deepfilternet`, `torch`, `torchaudio`, `transformers`, `huggingface_hub`.
  2. Verify local HuggingFace cache access for AI4Bharat IndicConformer / IndicWav2Vec models (Bengali checkpoint already present in cache: `ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large`).
  3. Prepare default configuration files in `configs/`:
     - `configs/frontend/deepfilternet.yaml`
     - `configs/encoders/semantic_asr.yaml`
     - `configs/encoders/paralinguistic.yaml`
     - `configs/projectors/audio_projector.yaml`

---

### Phase 2: Audio Preprocessing & Denoising (`samvyas/audio/`)
- **Goal**: Robust audio ingestion, 16kHz enforcement, and DeepFilterNet v2 noise removal.
- **Components to Implement**:
  1. `samvyas/audio/io.py`:
     - `load_audio(path_or_bytes, target_sr=16000, mono=True) -> torch.Tensor`
     - Resampling (via `torchaudio.transforms.Resample` or `librosa`), peak/loudness normalization.
  2. `samvyas/audio/denoiser/base.py`:
     - Abstract `BaseDenoiser` base class with standard `enhance(audio, sr)` method.
  3. `samvyas/audio/denoiser/deepfilter.py`:
     - `DeepFilterDenoiser` wrapping DeepFilterNet v2 (`df.enhance`).
     - Includes:
       - Configurable bypass toggle (`enabled: bool = True`).
       - Batch processing support `(B, T)` and `(B, 1, T)`.
       - Frame/chunk buffer support for real-time streaming input.
  4. **Verification**:
     - `tests/test_audio_denoiser.py`: Test on synthetic sine waves and [assets/audio/audio_for_inference.wav](./assets/audio/audio_for_inference.wav).

---

### Phase 3: Semantic / STT Branch (`samvyas/models/encoders/semantic/`)
- **Goal**: Extract phonetic/linguistic tokens and the intermediate **$V_{\text{dim}} = 768$** hidden representations.
- **Components to Implement**:
  1. `samvyas/models/encoders/semantic/base.py`:
     - Abstract interface returning `SemanticEncoderOutput(logits, hidden_states_768, tokens)`.
  2. `samvyas/models/encoders/semantic/indic_asr.py`:
     - Supports loading fine-tuned IndicWav2Vec or IndicConformer models.
     - Extracts:
       - CTC logits & decoded text tokens (supporting Bengali, Hindi, English).
       - Intermediate encoder representations strictly constrained to dimension **768**.
     - Parameter freezing toggle (`freeze_encoder: bool = True`) to preserve VRAM when training projector.
  3. **Verification**:
     - `tests/test_semantic_encoder.py`: Validate forward pass with 16kHz audio, verifying output shapes `(B, T_frames, 768)`.

---

### Phase 4: Paralinguistic Analysis Branch (`samvyas/models/encoders/paralinguistic/`)
- **Goal**: Capture non-linguistic acoustic nuances: prosody, emotional tone, pitch variations, rhythm, and imperfections (breathing, pauses, hesitations).
- **Components to Implement**:
  1. `samvyas/models/encoders/paralinguistic/base.py`:
     - Abstract interface returning `ParalinguisticEncoderOutput(acoustic_embeddings, prosody_latents)`.
  2. `samvyas/models/encoders/paralinguistic/indic_wav2vec.py`:
     - AI4Bharat IndicWav2Vec backbone feature extractor.
     - Extract multi-layer acoustic representations (e.g. weighted sum of middle/upper transformer layers or projection).
     - Temporal alignment utility to ensure sequence length compatibility with the semantic branch.
  3. **Verification**:
     - `tests/test_paralinguistic_encoder.py`: Verify acoustic embedding shapes and temporal alignment.

---

### Phase 5: Audio Projector & Downsampler (`samvyas/models/projectors/`)
- **Goal**: Fuse dual-stream representations, compress temporal length, and project to $d = 4096$.
- **Components to Implement**:
  1. `samvyas/models/projectors/downsampler.py`:
     - 1D Convolutional downsampling module (kernel size 3/5, stride 2, 4, or 8) with GELU activation and LayerNorm/RMSNorm.
     - Compresses frame rate (e.g. 50Hz -> 12.5Hz) so long audio inputs fit into LLM context window.
  2. `samvyas/models/projectors/mlp.py`:
     - Two-layer MLP projector with SwiGLU/GELU activation projecting from fused/downsampled dimension to **$d = 4096$**.
  3. `samvyas/models/projectors/audio_projector.py`:
     - Unified `AudioProjector` class:
       - Accepts: Semantic features `(B, T_audio, 768)` + Paralinguistic features `(B, T_audio, D_para)`.
       - Fuses along feature dimension or via gated cross-attention.
       - Downsamples and projects to `(B, T_proj, 4096)`.
  4. **Verification**:
     - `tests/test_audio_projector.py`: Verify forward pass, shape verification `(B, T_downsampled, 4096)`, and backward gradient flow.

---

### Phase 6: End-to-End STT Pipeline & Demonstration (`samvyas/pipelines/`)
- **Goal**: Integrate all blocks into a unified, executable pipeline.
- **Components to Implement**:
  1. `samvyas/pipelines/stt_pipeline.py`:
     - `STTPipeline`: Composite class uniting Denoiser + Semantic ASR + Paralinguistics + Audio Projector.
     - Method `process(audio_16k) -> STTPipelineOutput`:
       - `cleaned_audio`: Enhanced 16kHz audio tensor.
       - `transcription`: Recognized text string.
       - `projected_tokens`: Fused audio latents `(B, T_proj, 4096)` ready for LLM backbone.
  2. `tests/test_stt_pipeline.py`:
     - Full integration test running an end-to-end forward pass on [assets/audio/audio_for_inference.wav](./assets/audio/audio_for_inference.wav).
  3. Interactive verification notebook:
     - `notebooks/stt_pipeline_walkthrough.ipynb` demonstrating noisy audio input -> DeepFilterNet enhancement -> STT transcript -> projected embeddings visualization.

---

## 3. Hardware & Memory Safeguards (RTX 2050 4GB)

1. **Inference Precautions**:
   - Run both encoder backbones in `eval()` mode with `torch.no_grad()`.
   - Use `torch.autocast('cuda', dtype=torch.float16)` to halve tensor memory footprints.
2. **Sequential Execution**:
   - For memory-constrained environments, execute the denoiser and encoders sequentially, calling `torch.cuda.empty_cache()` between stages if necessary.
3. **Training Strategy (Future Phase)**:
   - Freeze the pre-trained Indic ASR and IndicWav2Vec backbones when training the Audio Projector.
   - Projector has only ~10-20M trainable parameters, making training fast and lightweight even on modest GPUs.

---

## 4. Verification Checklist

- [ ] Unit test: `DeepFilterDenoiser` cleans noisy audio and maintains shape `(B, T)`.
- [ ] Unit test: `IndicASREncoder` outputs transcription and `(B, T_frames, 768)` hidden states.
- [ ] Unit test: `ParalinguisticEncoder` outputs acoustic representations.
- [ ] Unit test: `AudioProjector` fuses inputs, downsamples temporally, and outputs `(B, T_proj, 4096)`.
- [ ] E2E integration test: `STTPipeline` processes raw 16kHz audio file from end to end without error.
