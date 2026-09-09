# Project Memory & Research Ledger: VoiceLLM

## 1. Project Overview & Mission
- **Project Name:** VoiceLLM
- **Mission:** Build a state-of-the-art End-to-End Multimodal Voice LLM supporting Bengali, Hindi, and English with zero-shot emotional mimicry, natural conversational prosody, and real-time audio perception.
- **Reference Architecture:** Dual-stream audio perception (DeepFilterNet v2 -> Semantic ASR + Paralinguistics -> Audio Projector -> LLM Backbone).

---

## 2. Core Architecture Pipeline
1. **Audio Enhancement Frontend:**
   - 16kHz microphone stream -> **DeepFilterNet v2** noise suppression.
2. **Dual-Stream Encoder:**
   - **Semantic / STT Branch:** Fine-tuned IndicWav2Vec / IndicConformer with CTC/RNNT head ($V_{\text{dim}} = 768$).
   - **Paralinguistic Branch:** IndicWav2Vec acoustic embeddings capturing prosody, timbre, cadence, and imperfections.
3. **Audio Projector & Downsampler:**
   - Dual-branch fusion, temporal downsampling, and MLP projection to $d_{\text{model}} = 4096$.
4. **LLM Backbone & Audio Synthesis:**
   - Decoder-only Transformer with RoPE, SwiGLU, and RMSNorm.
   - Speech synthesis via neural audio codec (e.g. SNAC).

---

## 3. Repository Layout (Clean & Organized)
```
VoiceLLM/
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
├── voicellm/
│   ├── audio/
│   │   └── denoiser/
│   ├── models/
│   │   ├── backbones/
│   │   ├── encoders/
│   │   └── projectors/
│   ├── data/
│   └── inference/
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 4. Current State & Next Steps
- [x] Cleaned repository of legacy files, build caches, and extracted checkpoints.
- [x] Preserved reference Jupyter notebook (`notebooks/indic_conformer_test.ipynb`) and audio assets (`assets/audio/`).
- [x] Established clean, modular directory structure mapped to the architecture diagram.
- [ ] **Step 1:** Implement audio I/O and DeepFilterNet v2 denoiser wrapper in `voicellm/audio/denoiser/`.
- [ ] **Step 2:** Implement dual-stream encoders (Indic ASR + Paralinguistic) in `voicellm/models/encoders/`.
- [ ] **Step 3:** Implement Audio Projector (Downsampler + MLP) in `voicellm/models/projectors/`.
- [ ] **Step 4:** Implement Transformer LLM Backbone in `voicellm/models/backbones/`.
- [ ] **Step 5:** Assemble end-to-end pipeline and test harness.
