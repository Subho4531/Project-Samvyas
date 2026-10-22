# Samvyas 2.0: StyleTTS 2 Architecture for Bengali Expressive Voice Twin

## 1. Executive Summary

This document specifies the technical architecture for transitioning Samvyas from its initial **IndicF5** baseline to **StyleTTS 2**. The objective is an end-to-end, ultra-low-latency conversational Bengali voice twin of the user ("Jeet") capable of dynamic emotional prosody controlled in real time by an LLM orchestrator.

### Why This Pivot is Necessary

| Constraint | IndicF5 (Current Baseline) | StyleTTS 2 (Target Pipeline) |
| :--- | :--- | :--- |
| **Synthesis Latency** | 12–15 seconds per sentence (32-step flow matching ODE) | **250–400 ms per sentence** (Single-pass AdaIN Generator) |
| **Emotion / Prosody** | Monolithic: reference WAV fixes both timbre and tone | **Decoupled**: Timbre (Jeet identity) + Style Vector (Emotion) |
| **VRAM Consumption** | 3.2 GB+ (frequent CUDA OOM on 4 GB RTX 2050) | **1.6–1.9 GB** during FP16 inference |
| **Steerability** | Black-box in-context copy; no explicit pitch/speed dials | Explicit pitch ($F_0$), energy, duration, and style modulation |

---

## 2. High-Level System Architecture: The "Director & Actor" Paradigm

The pipeline separates conversational intelligence from acoustic performance:

```
[ User Audio Input ]
        │
        ▼ (Mono 16 kHz PCM)
[ STT Worker: IndicConformer CTC / Agg (Bengali) ]
        │
        ▼ Bengali Text (e.g. "তুমি কাল কেন আসোনি?")
[ LLM Orchestrator: Gemini 2.5 Flash / Local LLM ]
        │
        │ Directs dialogue with semantic prosody markup:
        │ e.g. "[concerned] কাল শরীরটা খারাপ ছিল, [warm] তাই আসতে পারিনি।"
        ▼
[ Stream Chunking & Emotion Tag Parser ]
        ├── Chunk 1: Text = "কাল শরীরটা খারাপ ছিল,", Emotion = "concerned"
        └── Chunk 2: Text = "তাই আসতে পারিনি।", Emotion = "warm"
        │
        ▼
[ Bengali Phonemization & Normalization (G2P) ]
        │ Bengali Text ──> Clean IPA Phoneme Sequence
        ▼
[ StyleTTS 2 Expressive Engine ]
        ├── Speaker Identity Weights: Jeet Pretrained Timbre Base
        ├── Style Bank: Injects 64-dim style vector for "concerned" / "warm"
        ├── Prosody Predictor: Computes $F_0$, energy, and durations conditioned on Style
        └── Acoustic Decoder (Vocos/HiFi-GAN): Generates 24 kHz waveform (< 350 ms)
        │
        ▼
[ Audio Stream Queue & Web Client Playback ]
```

---

## 3. Detailed Component Specifications

### 3.1. Bengali Text Normalization & Phonemization (G2P)
Stock StyleTTS 2 expects English phonemes via eSpeak. Bengali requires custom phonological normalization:
1. **Script Normalization:**
   * Unicode canonical normalization (`NFC`).
   * Digit conversion: Bengali numerals (`০, ১, ২...`) expanded to phonetically spelled Bengali words (`শূন্য, এক, দুই...`).
   * Punctuation sanitization: Handling Bengali *Dari* (`।`) and standard pause markers (`?`, `,`, `!`).
2. **Grapheme-to-Phoneme (G2P) Converter:**
   * Converts Bengali orthography to International Phonetic Alphabet (IPA).
   * Resolves inherent vowel retention / schwa deletion (*hoshonto* nuances).
   * Resolves Bengali consonant conjuncts (*Juktakkhor*, e.g., ক্ষ `kʰkʰ`, জ্ঞ `g̃g̃`, ঙ্ক `ŋk`).
   * Supports fallback mapping to unified IPA vocabulary tokens understood by the phoneme embedding table.

### 3.2. Disentangled Style & Emotion Architecture

In StyleTTS 2, speech representation is factorized as:
$$\text{Speech} = \mathcal{G}(\text{Content}(ph), \text{Timbre}(spk), \text{Style}(s), F_0, N)$$

where:
* $ph$ = Input phoneme sequence.
* $spk$ = Speaker embedding vector (locks model to Jeet's unique vocal cords).
* $s \in \mathbb{R}^{64}$ or $\mathbb{R}^{128}$ = Acoustic style vector containing prosody, pitch distribution, tempo, and vocal tension.
* $F_0, N$ = Fundamental frequency contour and pitch energy predicted by the prosody module conditioned on $s$.

#### The Style Bank Manager
Instead of relying on random diffusion sampling at runtime (which can cause inconsistent emotional delivery), Samvyas utilizes a **Curated Style Bank**:
* **Anchor Recordings:** Pre-recorded short clips of Jeet speaking in distinct expressive states:
  * `neutral`: Standard conversational cadence.
  * `warm`: Lower pitch, gentle onset, friendly tone.
  * `excited`: Higher pitch range, faster tempo, elevated energy.
  * `whisper`: Compressed dynamic range, unvoiced fricatives, lowered energy.
  * `serious`: Measured tempo, descending terminal pitch.
  * `inquisitive`: Terminal rising pitch inflection.
* **Extraction:** The Style Encoder extracts the latent style vector $s_e$ for each anchor clip once and stores them in `configs/tts/style_bank.pt`.
* **Interpolation:** The pipeline supports convex combination of styles:
  $$s_{\text{blended}} = \alpha s_A + (1 - \alpha) s_B$$
  Allowing subtle gradients (e.g., 70% warm + 30% excited).

### 3.3. Acoustic Generator & Inference Latency Budget

* **Generator:** AdaIN (Adaptive Instance Normalization) Residual Blocks feed directly into a neural vocoder (Vocos or HiFi-GAN).
* **Latency Profile on RTX 2050 (4 GB):**
  * Phonemization + Text Processing: ~5 ms
  * Prosody & Style Prediction ($F_0$, Duration): ~30 ms
  * AdaIN Waveform Generation: ~220 ms
  * Total Synthesis Latency per sentence: **~260 ms** (Real-Time Factor < 0.1).
* **Memory Footprint:** Peak VRAM < 1.8 GB FP16, allowing concurrent execution with the STT worker without GPU memory paging crashes.

---

## 4. LLM Directing Protocol

The LLM is prompted with a structured system rule instructing it to produce conversation text with lightweight bracket tags:

```
Rule: Always precede your responses or sentences with an emotion tag from:
[neutral], [warm], [excited], [whisper], [serious], [inquisitive]

Example:
User: "কালকের ম্যাচটা কেমন হলো?"
Assistant: "[excited] অসাধারণ ম্যাচ হয়েছে! [warm] শেষ ওভারে চার মেরে দলটা জিতে গেল।"
```

The streaming web server parses these tags on-the-fly, splits sentences at punctuation boundaries, retrieves the requested style vector from the Style Bank, and initiates immediate TTS synthesis before the LLM has even finished generating the full paragraph.

---

## 5. Dataset Adaptation & Training Plan

### 5.1. Data Requirements
1. **Existing Base:** The 80 accepted clips (~500s) from `voice_users/jeet/` provide clean foundational vocal timbre.
2. **Emotion Calibration Set:** 30–40 new short takes (5–8 clips per emotion category) recorded via the web recorder to populate the Style Bank.
3. **Phoneme Alignment:** Montreal Forced Aligner (MFA) or CTC alignment for duration training.

### 5.2. Training Regimen
* **Base Model:** Pretrained multilingual StyleTTS 2 checkpoint.
* **Stage 1 (Transfer):** Freeze text encoder; fine-tune duration predictor and $F_0$ predictor on Jeet's voice.
* **Stage 2 (Identity Adaptation):** Fine-tune AdaIN decoder layers with adversarial loss on Kaggle T4 GPU (batch size 4, gradient accumulation 2, ~3 hours).
* **Stage 3 (Style Bank Calibration):** Pass emotion takes through the trained style encoder; serialize `style_bank.pt`.

---

## 6. Implementation Deliverables

1. `samvyas/audio/phonemize_bn.py`: Production-grade Bengali normalizer & IPA phonemizer.
2. `samvyas/models/tts/style_bank.py`: Style vector extractor, storage, and interpolation module.
3. `samvyas/models/tts/styletts2.py`: Clean runtime engine implementing Samvyas audio tensor contracts.
4. `requirements-styletts2.txt`: Isolated dependency specification for the StyleTTS 2 environment.
5. `scripts/test_styletts2_pipeline.py`: End-to-end verification and latency benchmark script.
6. `tests/test_style_tts.py`: Automated unit tests for phonemization, style vector interpolation, and tensor shapes.
