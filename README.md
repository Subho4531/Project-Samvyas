# 🎙️ VoiceLLM: End-to-End Multimodal Voice LLM & Emotional Speech Synthesis

VoiceLLM is a state-of-the-art research and execution framework designed for building **Zero-Shot Emotional Voice Mimicry and Speech-to-Speech LLMs**. 

Engineered with deep transformer design principles (Rotary Position Embeddings, SwiGLU activations, RMSNorm, KV-cache acceleration, Multi-Codebook Residual Vector Quantization modeling), the repository isolates research, modeling, preprocessing, and inference locally while providing streamlined export recipes for training on **Kaggle GPU/TPU environments**.

---

## 🌟 Key Architectural Pillars

1. **Modern Transformer Audio-Language Backbone**
   - Decoder-only architecture built with RoPE (Rotary Position Embeddings), SwiGLU non-linearities, and RMSNorm.
   - Unified text-audio vocabulary or interleaved delay-pattern multi-codebook modeling.

2. **Zero-Shot Voice Cloning & Speaker Conditioning**
   - High-capacity neural speaker representations (ECAPA/CAMPPlus/x-vector embeddings) coupled with acoustic reference context prefixing.

3. **Emotional & Prosodic Controllability**
   - Support for explicit emotion tokens, reference audio prosody transfer, and acoustic latent conditioning.

4. **Kaggle Training Decoupling & Export Harness**
   - Built-in bundler (`voicellm.kaggle.packager`) that compiles the codebase into self-contained training scripts or single-file notebooks ready to run on Kaggle's free T4/P100 GPUs with mixed precision (`torch.cuda.amp`), gradient accumulation, and Hugging Face / WandB checkpointing.

---

## 📁 Repository Structure

```
VoiceLLM/
├── MEMORY.md                   # Live architectural ledger & progress tracking
├── configs/                    # Model, codec, and training hyperparameter configs
├── voicellm/
│   ├── core/                   # Device handling, precision, and config schemas
│   ├── audio/                  # Audio I/O, spectrograms, loudness normalization
│   ├── models/                 # Transformer backbone, RVQ heads, conditioning
│   ├── data/                   # Tokenizers, dataset streaming, audio-text collators
│   ├── training/               # Multi-codebook cross-entropy losses & optimizers
│   ├── inference/              # KV-cached generation engine & voice cloning pipeline
│   ├── eval/                   # Benchmarks (Speaker similarity, WER, audio metrics)
│   └── kaggle/                 # Kaggle export tools & execution blueprints
├── scripts/                    # CLI tools for preprocessing, packaging, and inference
└── tests/                      # Unit tests & shape verification suites
```

---

## 🚀 Quickstart

### 1. Local Setup
```bash
# Clone and create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install in editable mode
pip install -e .
```

### 2. Packaging for Kaggle Training
To package the entire codebase and training recipe into a standalone script or notebook for Kaggle:
```bash
python scripts/train_kaggle_bundle.py --config configs/train/kaggle_t4_recipe.yaml --output kaggle_train_bundle.py
```

### 3. Local Verification & Tests
```bash
pytest tests/
```
