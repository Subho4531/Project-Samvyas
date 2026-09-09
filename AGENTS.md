# 🤖 Samvyas Agent Directives & Workspace Instructions

Welcome to **Samvyas** — a research and production framework for Multilingual Multimodal Voice Intelligence.

All AI coding assistants and autonomous agents operating in this workspace must adhere strictly to the protocols, design constraints, and coding standards outlined below.

---

## 🎯 1. Architectural Mission & Blueprint

Samvyas implements an end-to-end voice intelligence pipeline based on the architecture blueprint in [`public/stt.png`](./public/stt.png):

```
Mic Input (16kHz Mono)
       │
       ▼
DeepFilterNet v2 (Real-time Speech Enhancement & Noise Removal)
       │
       ├──► Branch 1: STT / Semantic (Indic ASR: IndicWav2Vec / IndicConformer)
       │              ├── CTC Head + RNN-T Decoding (Text tokens)
       │              └── Semantic Bottleneck Latents (Vdim = 768)
       │
       └──► Branch 2: Paralinguistic Analysis (IndicWav2Vec Acoustic Extractor)
                      └── Prosody, timbre, hesitation, emotional cadence
       │
       ▼
Audio Projector
       ├── Temporal Downsampler (Strided Conv1d / Pooling)
       └── Multi-Layer Perceptron (MLP) Projection -> d_model = 4096
       │
       ▼
Multimodal LLM Decoder Backbone
```

---

## 🛡️ 2. Critical System Invariants & Safety Rules

1. **Zero Large Binary Commits**:
   - **NEVER** stage or commit large binary weights (`*.ckpt`, `*.pt`, `*.pth`, `*.bin`, `*.nemo`, `*.safetensors`, `*.onnx`) to Git.
   - Large model checkpoints must be fetched dynamically using `huggingface_hub.hf_hub_download` and cached locally in `~/.cache/huggingface/`.
   - Verify `.gitignore` before every commit.

2. **Audio Format Standard**:
   - The canonical audio format across the entire pipeline is **16 kHz, single-channel (mono), 32-bit floating point (`[-1.0, 1.0]`)**.
   - All input audio handlers must automatically validate and resample incoming streams or files to 16 kHz mono.

3. **Hardware & Device Agility**:
   - Never hardcode `cuda:0` or assume a GPU is available.
   - Always allow target device selection: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
   - Support FP16 / BF16 autocast where performance matters, with fallback to FP32 on CPU.

4. **Explicit Tensor Shape Annotations**:
   - Every module's `forward()` method must document tensor shapes in docstrings and comments (e.g., `(B, T_samples)`, `(B, T_frames, 768)`, `(B, T_downsampled, 4096)`).
   - Use shape assertions or tests to catch dimension mismatches early.

---

## 📁 3. Workspace Layout & Component Boundaries

```
Samvyas/
├── .agents/                    # Agent rules, workflows, and skills
│   ├── rules/                  # Specialized behavioral and coding rules
│   └── instructions.md         # Detailed agent development procedures
├── assets/                     # Small reference audio samples and test fixtures
│   └── audio/
├── configs/                    # YAML configuration files for models and components
├── notebooks/                  # Interactive experimentation & research notebooks
├── public/                     # Static diagrams and project assets (e.g., stt.png)
├── samvyas/                    # Primary Python package
│   ├── audio/                  # Audio I/O, spectrograms, DeepFilterNet denoiser
│   │   ├── denoiser/
│   │   ├── io.py
│   │   └── processing.py
│   ├── models/                 # Neural architectures
│   │   ├── encoders/           # Dual-stream STT and Paralinguistic encoders
│   │   ├── projectors/         # Audio downsampler and MLP projection (d=4096)
│   │   └── backbones/          # LLM Transformer decoders
│   ├── data/                   # Dataset streaming, collators, and tokenizers
│   └── inference/              # Real-time microphone capture & streaming pipeline
├── tests/                      # Pytest verification suites
├── AGENTS.md                   # This instruction file
├── plan.md                     # Active engineering implementation plan
├── pyproject.toml              # Build & dependency metadata
└── requirements.txt            # Core environment dependencies
```

---

## 🛠️ 4. Development & Verification Workflow

When executing any task on this repository, agents must:

1. **Review Context**: Check [`MEMORY.md`](./MEMORY.md) and [`plan.md`](./plan.md) before making changes.
2. **Modular Implementation**:
   - Write clean, self-contained classes with abstract base classes where appropriate.
   - Implement unit tests under `tests/` alongside every new component.
3. **Automated Verification**:
   - Run `pytest tests/` after modifying code to guarantee no regressions.
   - Validate tensor shapes with dummy data in unit tests.
4. **Update Memory & Plan**:
   - Update [`MEMORY.md`](./MEMORY.md) with completed milestones and architectural decisions.
   - Update [`plan.md`](./plan.md) as phases are checked off.
