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
- [x] **Live demo dual-stream 2026-09-10**: `scripts/live_stt.py` remade — STT turn + para branch on cleaned audio (backbone/layer/device, acoustic pooled stats, prosody voiced-ratio/mean-F0/energy/ZCR, para→semantic alignment, per-stage timings). Para defaults to CPU (4GB guard); `--no-para/--para-device/--para-layer`. Dry-run verified with real IndicWav2Vec (stt 40.9s incl. load, para 12.6s CPU).
- [x] **Paralinguistic branch 2026-09-10**: `samvyas/models/encoders/paralinguistic/` (`base.py` contracts + `align_frames`, `indic_wav2vec.py` frozen SSL encoder default `ai4bharat/indicwav2vec_v1_bengali`, `prosody.py` F0/voicing/energy/ZCR @20ms hop). Config `configs/encoders/paralinguistic.yaml`; `download_models.py --include-para` (graceful gated message). Suite: 26 passed, 3 skipped. Real-HF path proven on tiny wav2vec2; real IndicWav2Vec weights **blocked on user HF access** (`huggingface-cli login`, then re-run fetch + `SAMVYAS_RUN_HEAVY=1 pytest`).
- [x] **HF auth wired 2026-09-10**: token in repo `.env` (`HF_ACESS_TOKEN`), `.env` git-ignored, `python-dotenv` added, `download_models.py` auto-loads it (verified `whoami: jeetxab`). Real IndicWav2Vec still 403 — account needs repo access grant (see next).
- [ ] **Waiting on user**: approve access at https://huggingface.co/ai4bharat/indicwav2vec_v1_bengali, then re-run fetch + heavy test (commands in plan).
- [x] **Para weights live 2026-09-10**: access granted, `indicwav2vec_v1_bengali` cached (wav2vec2-large, D=1024, 24 layers). Full suite with real weights: **29/29 pass** on RTX 2050 (incl. real DeepFilter, real .nemo, real IndicWav2Vec ≤5s clips).
