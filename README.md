# 🎙️ Samvyas

<p align="center">
  <img src="./public/stt.png" alt="Samvyas Architecture Blueprint" width="100%">
</p>

**Samvyas** is a modern research and development repository for Multimodal Voice & Audio Intelligence.

---

## 📁 Repository Overview

```
Samvyas/
├── assets/                     # Audio samples & media assets
│   └── audio/
├── configs/                    # Configuration files
├── notebooks/                  # Interactive experimentation & validation notebooks
│   └── indic_conformer_test.ipynb
├── public/                     # Architectural assets
│   └── stt.png
├── tests/                      # Testing harnesses
├── pyproject.toml              # Build configuration
├── requirements.txt            # Dependencies
└── README.md
```

---

## 🚀 Quickstart

```bash
# Install dependencies (use the project .venv on Windows)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Install the samvyas package editable (needed for scripts/)
.\.venv\Scripts\python.exe -m pip install -e . --no-deps

# Fetch model weights once (cached outside the repo)
.\.venv\Scripts\python.exe scripts/download_models.py

# Live mic demo: 5s clips from default mic, loop till Ctrl+C
# (STT transcript + 768-vector/PCA + paralinguistic acoustics/prosody)
.\.venv\Scripts\python.exe scripts/live_stt.py

# Mic-less dry run on a sample file
.\.venv\Scripts\python.exe scripts/live_stt.py --file assets/audio/vocaltest.m4a --no-denoise

# Run tests (SAMVYAS_RUN_HEAVY=1 enables real-model GPU tests)
.\.venv\Scripts\python.exe -m pytest tests/ -q
```
