# Samvyas: STT & Dual-Stream Perception Implementation Plan

## Continuous Bengali collection (2026-10-10)

User-authorized scope: provide one sentence-per-line Bengali script and a
continuous story; extend the existing browser recorder with ten-minute modes.
Preserve the full master recording and exact editable script. Sentence mode
proposes cuts at long pauses with padding; count mismatches must never silently
assign transcripts. Save proposals separately for listening/transcript review,
not into the accepted training manifest. Story mode preserves the whole take
for later alignment. Verify segmentation, preservation, rates and HTTP/UI
integration with synthetic audio; preserve all prior STT/training progress.

Authorized Kaggle execution (2026-10-09): use the authenticated CLI to verify
`jeet065/jeet-voice`, upload/update a private training notebook with that dataset
and GPU/Internet enabled, then start execution and verify job status. Keep all
credentials out of notebook source and submission files. Preserve the dataset
and existing unrelated notebooks. Record submission identity and actual state.

Submission status: private `jeet065/jeet-indicf5-fine-tune` uses T4, Internet,
voice input and hash-checked hosted model input. Version 1 failed because the
Kaggle host runs Python 3.13. Version 2 creates an isolated Python 3.11 runtime
under /tmp and was successfully submitted (QUEUED). Training completion and
voice-quality review remain pending; no credentials are embedded.

Startup follow-up: version 2 verified the pinned model/vocabulary and GPU but
failed on an inherited Jupyter-only matplotlib backend before training.
Version 3 was submitted with a headless `Agg` subprocess backend.

Version 3 result: COMPLETE, 40 epochs and 480 steps; validation loss
0.663958 -> 0.586096. Checkpoints exported on Kaggle; reports and all base/adapted
listening previews downloaded and signal-checked. User listening review remains
the voice-quality acceptance step.

Approved Jeet adaptation workflow (2026-10-08): prepare a portable bundle of
the 50 accepted recordings with deterministic 45/5 train/validation splits,
keeping source recordings unchanged. Add a Kaggle notebook and isolated
training CLI for the pinned IndicF5 flow-matching objective. Initially adapt
the last two transformer blocks and output layers, use batch one with gradient
accumulation, validation loss and early stopping, and export a full checkpoint
compatible with the inference wrapper. Test data portability, gradients,
frozen layers and checkpoint compatibility locally. Actual production-scale
training runs on a Kaggle GPU; do not describe preparation as a trained model.

Workflow status (2026-10-09): dataset prepared, training/validation and export
code implemented, inference accepts adapted weights, and Kaggle notebook plus
private upload bundle generated. Full tests and pretrained forward/backward
smoke passed. Kaggle execution and comparative listening remain outstanding.

Jeet voice replication (2026-10-08): audit the completed `voice_users/jeet`
profile and generate unseen Bengali speech with its accepted conversational
reference. Package the selected reference explicitly for reuse. This step uses
frozen IndicF5; speaker checkpoint adaptation is a separate requested scope
pending clarification. Assess intelligibility and speaker likeness by listening
before claiming voice replication quality.

## Active extension: Bengali voice recorder and IndicF5 baseline

User-approved implementation (2026-10-07):
- Build a terminal recorder with 50 Bengali prompts, microphone check/selection,
  Enter-to-start/stop, a 20-second limit, playback/retry/transcript editing,
  explicit transcript confirmation, and resumable accepted recordings.
- Save unprocessed mono 16 kHz PCM WAVs and UTF-8 JSONL metadata under
  `assets/audio/voice_dataset/` (ignored by Git). Validate samples, report
  clipping/silence, never overwrite accepted takes, and report orphan files.
- Add an isolated IndicF5 inference wrapper/CLI taking target text, reference
  WAV, and verified reference transcript. Export native 24 kHz and canonical
  16 kHz outputs; frozen batch-one inference with explicit CPU fallback.
- Verify recorder/dataset behavior with mocked audio and TTS with a mocked
  backend. Actual recording and listening evaluation require the user's voice.
- Defer speaker fine-tuning, explicit vector adapters, and LLM integration.
  Collect a 50-clip pilot before expanding toward 30–60 usable minutes.

The earlier STT plan and progress are preserved below.

Extension status (2026-10-07): recorder, prompts, dataset helpers, TTS wrapper,
CLI, isolated dependency manifests, and documentation implemented. Non-heavy
suite: 74 passed / 4 skipped; lint/format and diff checks passed. User supplied
10 recordings and resolved IndicF5 access. Three baseline sentences were
generated on both CPU and RTX 2050 CUDA, with valid 24 kHz/16 kHz exports.
Runtime fixes cover public-vocoder authentication, Bengali terminal encoding,
BF16 transformer inference and float32 vocoder decoding. Listening review
remains pending. No speaker training or voice similarity claim has been made.

Recorder update (2026-10-08): simplified terminal interaction to one-letter
commands and added native Windows Unicode output for Bengali prompt display.

Web recorder extension (2026-10-08): implemented a local-only HTML recording
studio. It creates/opens a named user profile below `assets/audio/voice_users/`,
captures microphone audio in the browser, previews/retries/saves samples, and
keeps the existing 16 kHz mono WAV + JSONL contract. The UI shows one Bengali
prompt at a time with recording status, meter, timer, transcript editing, and
saved-progress feedback. Browser audio is transferred only to the local Python
server, validated/resampled and atomically stored through `VoiceDataset`.
Unit and API verification is complete; live microphone capture remains for the
user to assess on their chosen device.

Capture quality update (2026-10-08): browser recording now uses `AudioWorklet`
instead of the unstable legacy main-thread capture API, requests the best
available 48 kHz mono microphone stream, and reports the active rate. Existing
saved recordings are never modified automatically; any distorted take must be
recorded again after the update.

Recorder compatibility update (2026-10-08): fixed an unsupported-Web-Audio
capability guard that could make Record appear unresponsive. The preferred
worklet capture requests a 48 kHz context and batches PCM frames before passing
them to the page; a clearly labelled lossless compatibility path is available
when the browser cannot run the worklet.

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
# Modern Bengali story recording script (2026-10-10)

Replace the current story script with an original 5–10 minute contemporary,
everyday colloquial Bengali story with conversational dialogue. Aim for a
comfortable spoken pace and natural variation; avoid claiming that text alone
can create a regional accent, since that depends on the recorded speaker.

# New recordings and adaptation v2 (2026-10-11)

User corrected the scripts and confirmed they are the intended transcript
source. Preserve all M4A originals and v1 assets. The two sentence takes decode
to mono 16 kHz and segment to exactly 30 cuts at the established 1.2-second
threshold (16 + 14). Combined with 50 accepted clips, the isolated bundle has
80 clips / 506.23 sec, split 72/8. The long story is preserved locally, pending
utterance alignment for training. Private Kaggle T4 notebook version 6 completed at `jeet065/jeet-indicf5-fine-tune-v2`. It trained 72 clips / 453.28 sec with 8 validation clips / 52.95 sec for 40 epochs (720 updates). Validation loss improved from 0.65291 to 0.52525. Kaggle generated base/adapted previews for statement, question, and expressive text in 24 kHz and 16 kHz. The 1,286 MiB best checkpoint was retrieved locally. The current 1,037-word story was synthesized on the RTX 2050 in 87 chunks at 32 steps, then stitched with paragraph pauses. Mono PCM-24 outputs at 24 kHz and 16 kHz are in `outputs/tts/jeet_story_local_20261011/` (506.39 sec). WAV headers verified; listening review is pending, so do not infer voice similarity from loss or generation alone. Detailed audit:
docs/tts_recordings_audit.md.
