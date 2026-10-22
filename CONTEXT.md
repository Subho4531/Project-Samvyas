# 📖 Project Samvyas — Master Context & Technical Blueprint

> **System Purpose**: End-to-end conversational Bengali Voice AI & Digital Twin platform. Enables real-time, low-latency Bengali speech interaction with an authentic, emotion-steered voice clone of the user ("Jeet").

---

## 1. Executive Summary & Vision

Project Samvyas transforms traditional conversational voice AI by decoupling **cognitive directing** from **acoustic acting**:

```
[ User Speech (16kHz) ]
          │
          ▼
 [ 1. STT Worker ] ──> Bengali Text ("তুমি কাল কেমন ছিলে?")
          │
          ▼
 [ 2. LLM Director ] ──> Directs Content & Emotion Markup:
                         "[warm] শরীরটা একটু খারাপ ছিল, [warm] তাই আসতে পারিনি।"
          │
          ▼
 [ 3. Tag Parser & Chunker ] ──> Slices at clause boundaries ('।') into synthesis jobs
          │
          ├─────────────────────────────────────────┐
          ▼                                         ▼
 [ 4A. Bengali G2P & Normalizer ]       [ 4B. Curated Style Bank ]
   NFC, Bengali Digits -> Words,          64-dim Latent Style Vectors:
   IPA Juktakkhor Conjunct Resolution     [warm, excited, whisper, serious]
          │                                         │
          └────────────────────┬────────────────────┘
                               ▼
               [ 5. StyleTTS 2 Expressive Engine ]
                 • Timbre: Jeet Vocal Cord Weights (Fixed Identity)
                 • Prosody: F0 & Energy Modulated by Style Vector s
                 • Single-pass AdaIN Generator + Vocos Vocoder
                 ⚡ Latency: < 250 ms (RTF < 0.1, VRAM < 1.8 GB)
                               │
                               ▼
               [ 6. Chained Stream Playback Queue ]
                 Dual Export: 24 kHz Hi-Fi + 16 kHz Resampled PCM
                 🎯 Total Turnaround: < 600 ms
```

---

## 2. Architectural Evolution: 1.0 Baseline vs. 2.0 Target

| Dimension | Samvyas 1.0 (Baseline) | Samvyas 2.0 (Active Architecture) |
| :--- | :--- | :--- |
| **TTS Core** | AI4Bharat IndicF5 (Diffusion / Flow Matching) | **StyleTTS 2 (Non-Autoregressive AdaIN ResBlocks)** |
| **Synthesis Latency** | 12–15 seconds per sentence (32 ODE steps) | **220–260 ms per sentence (Single forward pass)** |
| **Emotion Control** | None (reference WAV fixes both timbre & prosody) | **Dynamic (Style Bank latent vectors steered by LLM)** |
| **Hardware Stability** | Frequent CUDA OOM on RTX 2050 (4 GB) | **Peak VRAM < 1.8 GB FP16 (Stable alongside STT)** |
| **Text Processing** | Character-level tokenization | **Bengali G2P: Orthography ➔ IPA with Conjunct Resolution** |
| **Serving Mode** | Blocking subprocess spawning | **Persistent in-memory worker with streaming queue** |

---

## 3. Repository Anatomy & Directory Structure

```
Project Samvyas/
├── samvyas/                         # Core Python package
│   ├── audio/                       # Audio processing front-end
│   │   ├── io.py                    # Audio file loading, saving & sample rate validation
│   │   ├── phonemize_bn.py          # Bengali normalizer, G2P phonemizer & IPA tokenizer
│   │   ├── recording.py             # AudioWorklet browser recording interface
│   │   ├── pause_segmentation.py    # Energy/silence-based utterance segmentation
│   │   └── denoiser/                # DeepFilterNet integration
│   ├── models/
│   │   ├── encoders/
│   │   │   ├── semantic/indic_asr.py # NeMo IndicConformer Bengali ASR
│   │   │   └── paralinguistic/      # Acoustic & prosody feature extractors
│   │   └── tts/
│   │       ├── style_bank.py        # 64-dim emotion style vector manager & interpolator
│   │       ├── styletts2.py         # StyleTTS 2 engine wrapper & prosody predictor
│   │       ├── indic_f5.py          # IndicF5 diffusion baseline wrapper
│   │       └── adaptation.py        # DiT block parameter adaptation
│   ├── data/
│   │   ├── voice_dataset.py         # Manifest.jsonl dataset parser & validator
│   │   └── tts_adaptation.py        # Train/val split bundle generator
│   └── web/
│       ├── voice_chat.py            # End-to-end voice chat service backend
│       ├── voice_recorder.py        # Voice studio recorder service backend
│       └── static/                  # Web interfaces (HTML/CSS/JS)
├── configs/
│   ├── tts/
│   │   ├── styletts2_bn.yaml        # StyleTTS 2 model & emotion configuration
│   │   ├── style_bank.pt            # Serialized 64-dim emotion style vectors
│   │   ├── continuous_sentences_bn.txt # 30-sentence structured training prompts
│   │   └── continuous_story_bn.txt  # 1,037-word narrative script for synthesis tests
├── docs/
│   ├── styletts2_architecture.md    # Master architecture specification
│   ├── styletts2_pipeline.excalidraw # End-to-end interactive canvas
│   ├── styletts2_pipeline.svg       # End-to-end high-res vector diagram
│   ├── styletts2_ml_architecture.excalidraw # TTS ML deep learning canvas
│   ├── styletts2_ml_architecture.svg # TTS ML deep learning vector diagram
│   └── tts_recordings_audit.md      # Detailed audit of voice takes & segments
├── scripts/
│   ├── voice_chat_web.py            # Launcher for Voice Chat web instance (port 8766)
│   ├── voice_recorder_web.py        # Launcher for Voice Studio recorder (port 8765)
│   ├── stt_worker.py                # Persistent stdio JSON STT worker
│   ├── extract_style_bank.py        # Style Bank calibration CLI
│   ├── generate_excalidraw_pipeline.py # Excalidraw pipeline diagram generator
│   ├── generate_tts_ml_diagram.py   # TTS ML architecture diagram generator
│   ├── test_styletts2_pipeline.py   # StyleTTS 2 latency & verification benchmark
│   └── patch_nemo_checkpoint.py     # Repacks NeMo 1.x checkpoint for NeMo 2.x
└── tests/                           # Automated pytest test suites
```

---

## 4. Environment Isolation Protocol ("Never Mix Them")

Because NeMo, Torch cu121, DeepFilterNet, and modern web dependencies conflict on Windows, environments are strictly partitioned:

| Environment | Path | Python | Packages & Role |
| :--- | :--- | :--- | :--- |
| **STT / Pipeline** | `.venv` | `3.11` | PyTorch 2.5.1+cu121, `nemo_toolkit[asr]`, DeepFilterNet, Transformers 4.49.0 |
| **Recorder & Web** | `.venv-recorder` | `3.11` | Fast HTTP servers, WebAudio handlers (no PyTorch, no GPU footprint) |
| **IndicF5 Baseline** | `.venv-tts` | `3.11` | AI4Bharat IndicF5, PyTorch 2.5.1+cu121 (isolated from NeMo) |
| **StyleTTS 2** | `.venv-styletts2` | `3.11` | StyleTTS 2, Bengali G2P, Librosa, Munch, Einops, Vocos |
| **Verification** | `.venv-verify` | `3.11` | Fast pytest runner, linting, and smoke tests |
| **Kaggle CLI** | `.venv-kaggle` | `3.11` | Kaggle kernel upload, monitoring, and checkpoint retrieval |

> **Rule**: Always invoke the environment's python explicitly (e.g. `.\.venv-recorder\Scripts\python.exe scripts/voice_chat_web.py`).

---

## 5. Audio & Tensor Contracts

* **Sample Rates**:
  * Internal pipeline & STT: **Mono 16,000 Hz**.
  * Native TTS output: **24,000 Hz**.
  * Resampled TTS export: **16,000 Hz**.
* **Tensor Shapes & Types**:
  * Audio Waveforms: `(B, T)` or `(1, T)`, `torch.float32` in `[-1.0, 1.0]`.
  * Semantic Latents: `(B, T_frames, 768)`.
  * Style Vectors: `(B, 64)`, $L_2$-normalized.
* **Peak Safety**:
  * Never destructively normalize raw collected mic recordings.
  * Apply a single shared peak attenuation ($-1.0 \text{ dB}$, peak factor `0.89`) at the TTS export boundary.

---

## 6. Datasets & Voice Adaptation Ledger

### The Voice Profile: "Jeet"
* **Accepted Short Takes**: 80 clips (506.23 seconds total duration).
  * Training Split: 72 clips / 453.28 s.
  * Validation Split: 8 clips / 52.95 s.
* **Continuous Story Audio**:
  * 1,037 words, 5,969 characters (~8.5 min take).
  * Kept isolated from training data; used as an unseen held-out synthesis evaluation.
* **Adapted Checkpoint v2**:
  * Checkpoint: `outputs/tts/jeet_adaptation_v2/jeet_adaptation_v2/checkpoints/best.safetensors` (1,286 MiB).
  * Trained on Kaggle T4 (40 epochs, 720 steps, validation loss dropped from 0.6529 to 0.5252).
* **Emotion Style Bank Calibration**:
  * Stored in `configs/tts/style_bank.pt`.
  * Supports 6 discrete emotion categories (`neutral`, `warm`, `excited`, `whisper`, `serious`, `inquisitive`) plus continuous interpolation.

---

## 7. Hardware & Deployment Constraints (RTX 2050 4 GB)

* **VRAM Ceiling**: Maximum 4,096 MB VRAM.
* **Memory Partitioning**:
  * STT Worker: ~1.2 GB VRAM FP16.
  * StyleTTS 2: ~1.6 GB VRAM FP16.
  * Total Concurrent Footprint: **~2.8 GB VRAM** (leaves > 1.2 GB headroom for system/display).
* **Fallback Rule**: `--cpu-fallback` is enabled on all TTS wrappers to prevent crash cascades during temporary memory pressure.

---

## 8. Operational Commands Cheat-Sheet

```powershell
# 1. Run End-to-End Voice Chat Web Instance (port 8766)
$env:SAMVYAS_NEMO_PATH = "$HOME\.cache\samvyas\indicconformer_bn_agg.nemo"
.\.venv-recorder\Scripts\python.exe scripts/voice_chat_web.py --port 8766 --cpu-fallback --checkpoint outputs/tts/jeet_adaptation_v2/jeet_adaptation_v2/checkpoints/best.safetensors

# 2. Run Voice Studio Browser Recorder (port 8765)
.\.venv-recorder\Scripts\python.exe scripts/voice_recorder_web.py

# 3. Test StyleTTS 2 Pipeline & Latency
$env:PYTHONPATH="."
.\.venv-verify\Scripts\python.exe scripts/test_styletts2_pipeline.py

# 4. Calibrate / Inspect Style Bank
$env:PYTHONPATH="."
.\.venv-verify\Scripts\python.exe scripts/extract_style_bank.py --show

# 5. Run Automated Unit Tests
.\.venv-verify\Scripts\python.exe -m pytest tests/test_styletts2.py tests/test_indic_f5.py -v

# 6. Verify Git Safety & Code Formatting
.\.venv-verify\Scripts\python.exe -m ruff check samvyas/
git diff --check
```
