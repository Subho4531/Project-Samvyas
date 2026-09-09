# Project Memory & Research Ledger: VoiceLLM

## 1. Project Overview & Mission
- **Project Name:** VoiceLLM
- **Mission:** Build a state-of-the-art, end-to-end Voice Large Language Model capable of high-fidelity, expressive zero-shot voice cloning, emotional mimicry, and natural conversational prosody.
- **Hardware Strategy:** Local development & modular research codebase; Model training executed on Kaggle GPU/TPU environments managed with packaged training recipes.
- **Architecture Direction:** Modern Transformer backbone with multimodal audio-text representation, neural audio codec tokenization (e.g., RVQ / SNAC / DAC / EnCodec), speaker conditioning embeddings, and controllable emotional/prosodic latents.

---

## 2. Core Architecture Tenets
1. **End-to-End Autoregressive + Hierarchical Modeling:** 
   - Semantic text and acoustic token unification.
   - Delay-pattern / multi-codebook autoregressive prediction or hybrid AR-NAR (Autoregressive prompt + Flow-Matching / Non-Autoregressive acoustic refinement).
2. **Zero-Shot Voice Cloning & Speaker Conditioning:**
   - Deep speaker representation via neural speaker encoders (ECAPA2 / CAMPPlus / x-vector) combined with acoustic prompt prefixing.
3. **Emotional & Prosodic Controllability:**
   - Explicit & implicit emotion latents (valence/arousal, reference audio style transfer, textual emotion tags).
4. **Kaggle Training Optimization:**
   - Gradient accumulation, BF16/FP16 mixed precision, activation checkpointing, compact batching with dynamic audio padding, and seamless Hugging Face / WandB checkpoint syncing.

---

## 3. Repository Architecture & Layout
```
VoiceLLM/
├── MEMORY.md                   # Live state, architectural decisions, and roadmap
├── README.md                   # Repository documentation & getting started
├── pyproject.toml              # Modern Python packaging & dependencies
├── requirements.txt            # Pinned research dependencies
├── configs/                    # YAML configs for models, codecs, and training
│   ├── model/
│   │   └── voicellm_base.yaml
│   ├── codec/
│   │   └── snac_24khz.yaml
│   └── train/
│       └── kaggle_t4_recipe.yaml
├── voicellm/
│   ├── __init__.py
│   ├── core/                   # Base classes, registry, device & precision utilities
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── utils.py
│   ├── audio/                  # Audio processing, normalization, and IO
│   │   ├── __init__.py
│   │   ├── io.py
│   │   └── processing.py
│   ├── models/                 # Neural architectures
│   │   ├── __init__.py
│   │   ├── backbones/          # Modern Transformer decoders (RoPE, SwiGLU, RMSNorm)
│   │   │   ├── __init__.py
│   │   │   └── transformer.py
│   │   ├── codecs/             # Codec adapters (SNAC, DAC, EnCodec)
│   │   │   ├── __init__.py
│   │   │   └── base_codec.py
│   │   ├── conditioning/       # Speaker & emotion embeddings
│   │   │   ├── __init__.py
│   │   │   └── speaker_encoder.py
│   │   └── voice_llm.py        # End-to-end Voice LLM model class
│   ├── data/                   # Dataset loaders, audio tokenizers, collators
│   │   ├── __init__.py
│   │   ├── tokenizer.py
│   │   └── dataset.py
│   ├── training/               # Losses, trainers, and schedulers
│   │   ├── __init__.py
│   │   └── losses.py
│   ├── inference/              # Generation engine, KV cache, streaming
│   │   ├── __init__.py
│   │   └── engine.py
│   ├── eval/                   # Benchmarks (SIM-O, WER, PESQ)
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── kaggle/                 # Kaggle notebook export & packaging harnesses
│       ├── __init__.py
│       └── packager.py
├── scripts/                    # CLI utilities
│   ├── preprocess.py
│   ├── train_kaggle_bundle.py
│   └── generate.py
└── tests/                      # Unit tests
    ├── __init__.py
    └── test_model_forward.py
```

---

## 4. Work Completed (Initialization Phase)
- [x] Initialized modular research repository structure.
- [x] Built `MEMORY.md` persistent tracking ledger.
- [x] Implemented core Transformer decoder with modern SOTA primitives (Rotary Position Embeddings / RoPE, RMSNorm, SwiGLU activations, KV Cache).
- [x] Implemented multi-codebook acoustic token handling & audio codec abstraction.
- [x] Implemented Speaker & Emotion conditioning module abstractions.
- [x] Implemented end-to-end VoiceLLM forward & generation logic with delay pattern / hierarchical codebook modeling.
- [x] Implemented Kaggle training packager utility to bundle repository code for single-click Kaggle execution.
- [x] Created baseline configuration files and test harnesses.
- [x] Configured Python 3.10 virtual environment and passed full unit test suite (7/7 tests passing: audio I/O, RMSNorm, SwiGLU, TransformerDecoder, forward loss, and KV-cache generation).

---

## 5. Next Planned Milestones
- [ ] **Architecture Discussion & Finalization:** Decide between pure multi-codebook AR (VALL-E / AudioPaLM / Moshi style) vs. Hybrid AR + Flow-Matching (CosyVoice / F5-TTS style).
- [ ] **Acoustic Codec Integration:** Benchmark 24kHz / 44.1kHz codecs (SNAC vs. DAC vs. Descript vs. Mimi) for token compression vs. reconstruction fidelity.
- [ ] **Data Pipeline & Formatting:** Build tokenization pipeline for audio-text aligned datasets (LibriTTS-R, Emilia, Common Voice).
- [ ] **Kaggle Training Script Polish:** Setup Kaggle notebook with Hugging Face Hub checkpoint syncing and WandB live telemetry.
