# Project Memory & Research Ledger: Samvyas

## 1. Project Overview
- **Project Name:** Samvyas
- **Mission:** High-performance Multimodal Voice & Audio Intelligence.
- **Reference Diagram:** [public/stt.png](./public/stt.png)

---

## 2. Repository Layout
```
Samvyas/
├── assets/
│   └── audio/
│       ├── audio_for_inference.wav
│       └── vocaltest.m4a
├── configs/
├── notebooks/
│   └── indic_conformer_test.ipynb
├── public/
│   └── stt.png
├── tests/
├── .gitignore
├── MEMORY.md
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## 3. Current State
- [x] Renamed project to **Samvyas**.
- [x] Completely removed legacy `voicellm` folder.
- [x] Preserved reference research (`notebooks/indic_conformer_test.ipynb`) and audio samples (`assets/audio/`).
- [x] **Phase 1 (STT-only) implemented & verified 2026-09-10**: `samvyas/audio` (io + DeepFilterNet v2), `samvyas/models/encoders/semantic` (IndicConformer BN, Vdim 768), `samvyas/pipelines/stt_pipeline.py`, configs, `scripts/download_models.py`. 18/18 pytest pass (incl. 2 GPU heavy: real DeepFilter + real .nemo ≤5s clip) on RTX 2050.
- [ ] Next: Paralinguistic branch (Phase 4 of plan.md), then Audio Projector d=4096 (Phase 5).
