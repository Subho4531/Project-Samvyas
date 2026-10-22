# Project Memory & Research Ledger: Samvyas

## End-to-end voice merge — 2026-10-13

- Merged STT + Gemini + TTS into one window: browser Talk → `POST /api/voice`
  → persistent `scripts/stt_worker.py` (DeepFilter + NeMo, stdio JSON) →
  streamed Gemini replies → per-sentence IndicF5 TTS with chained playback.
  Talk prefers server STT, falls back to browser recognition; typing always works.
- Rebuilt `.venv` on Python 3.11 (was pointing at deleted 3.10). NeMo fought
  three rounds: v3 CPU-torch overwrite (force-reinstalled cu121 `--no-deps`),
  transformers v5 (pinned 4.49), datasets/hub-v1 (datasets 3.x). Final:
  NeMo 2.7.3 + torch 2.5.1+cu121 + CUDA green.
- Checkpoint is NeMo 1.x-era: `scripts/patch_nemo_checkpoint.py` repacks it
  for NeMo 2.x (tokenizer `multilingual`→`agg`, drop stale `multisoftmax` /
  `multilingual` / `language_keys`; weights untouched) to
  `~/.cache/samvyas/indicconformer_bn_agg.nemo` (`SAMVYAS_NEMO_PATH`,
  auto-detected). `IndicASREncoder(lang="bn")` default-decodes aggregate CTC.
- Verified E2E on RTX 2050: real 5.8 s clip → word-perfect transcript vs
  manifest (punctuation aside) → Gemini Bengali reply → 2 CUDA TTS wavs.
  STT worker holds GPU while TTS loads/unloads; no OOM this run
  (`--cpu-fallback` armed). Worker stdout hardened (loguru off stdout,
  capped non-JSON skip). Chat suite 12/12; full suite has 9 pre-existing
  fixture failures (deleted wav + absent prompts file).

## New recording audit — 2026-10-11

- Found three user source takes in `configs/tts/recordings/`: sentence parts
  (2:02, 1:42) and a story (8:52), all 48 kHz stereo AAC. Temporary mono 16 kHz
  conversion measured peaks 0.467–0.501 and RMS 0.078–0.081; no silence or
  clipping concern. Original files remain untouched.
- The 30-line sentence script yields 16 + 15 pause candidates at a 0.8-second
  gap, or 16 + 14 at 1.2 seconds. Do not assign line transcripts by order yet.
  Corrected two visible script typos: `ক্যামেরার` → `কলমের`, `বসে` → `বাসে`.
- The 1,037-word story matches the current story script and lasts 8:52. Pauses
  yield 43 pieces (0.8 sec; 1.8–40.4s) or 15 pieces (1.2 sec; several longer
  than the current 15s clip limit). Needs transcript alignment and sentence
  crops before training.
- Corrected sentence file timestamp is 2026-10-11 07:36. User confirmed the
  corrected scripts are the transcript source. At the established 1.2-second
  pause threshold, sentence takes yield exactly 30 ordered cuts (16 + 14).
- Prepared `outputs/tts/jeet_adaptation_v2/`: 80 clips total / 506.23 sec,
  72 train / 8 validation / 453.28 train sec / 52.95 validation sec. This
  includes the prior 50 accepted samples plus 30 new sentence crops. The
  8:52 story is preserved locally pending utterance alignment and excluded.
- Created private Kaggle input dataset `jeet065/jeet-voice-adaptation-v2`; the
  source M4As were not uploaded. Updated the existing private T4 notebook and
  pushed versions 4–6 under canonical slug `jeet065/jeet-indicf5-fine-tune-v2`.
  v4 failed installing PyTorch 2.5.1 on Kaggle Python 3.13; v5 reused Kaggle
  torch 2.11 but hit IndicF5’s stale NumPy<=1.26.4 cap. v6 used the Kaggle CUDA
  stack and completed. It trained 72 clips / 453.28 sec against 8 validation
  clips / 52.95 sec for 40 epochs / 720 updates. Validation loss: 0.65291 base,
  0.52525 best. Base/adapted listening previews (3 texts each, 24 kHz and 16 kHz)
  and reports were downloaded under `outputs/tts/jeet_adaptation_v2/jeet_adaptation_v2/`.
  Retrieved the 1,286 MiB v2 best checkpoint locally. Ran the full 1,037-word /
  9-paragraph story on the RTX 2050 in 87 chunks at 32 diffusion steps; completed
  in 12.4 minutes on CUDA. Saved stitched mono PCM-24 WAVs at 24 kHz and 16 kHz,
  duration 506.39 sec, under `outputs/tts/jeet_story_local_20261011/`. WAV headers
  were verified. Voice intelligibility, similarity, prosody, and artifacts still
  need listening review. Original v1 dataset/checkpoint remain preserved. Detailed
  audit is in `docs/tts_recordings_audit.md`.

## Continuous collection — 2026-10-10

- Added original UTF-8 scripts: 30 Bengali sentences (one per line) and one
  continuous Bengali story under `configs/tts/continuous_*_bn.txt`.
- Browser recorder now offers sentence and story modes up to ten minutes,
  keeping the exact input WAV, canonical 16 kHz mono PCM master and script in
  unique per-user `sessions/` folders. Existing accepted audio is preserved.
- Sentence mode proposes cuts with 20 ms RMS analysis, a 1.2-second minimum
  quiet gap and 200 ms padding. Readers should leave two-second inter-line
  pauses. Matching counts are only suggested text alignment; mismatches leave
  transcripts blank. Every accepted crop requires an explicit listening and
  exact-transcript confirmation. Whole stories are pending later alignment.
- Verification: 29 recorder/segmentation/dataset tests passed, including HTTP
  upload/playback/acceptance, silence handling, short-pause preservation,
  duplicate protection and master/crop consistency. Browser JavaScript passed
  `node --check`. No real microphone/listening assessment was performed in
  this change; cut quality in ambient noise still needs user review.
- Review cards currently live in the open page. Finish acceptance before
  refreshing; saved masters/crops remain available on disk afterward.


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
- [x] **Jeet fine-tuning completed on Kaggle, version 3 (2026-10-09)**:
  job status COMPLETE. Finished 40 epochs / 480 optimizer updates on 45 train
  and 5 validation clips. Held-out flow loss improved from 0.6639581680 to
  0.5860964179 (11.73% reduction). Exported best/last safetensors and generated
  three base/adapted Bengali comparisons at 24 kHz and 16 kHz. Best checkpoint
  response has size 1,348,435,680 bytes and a valid-length safetensors header;
  checkpoint remains on Kaggle, not yet downloaded locally. Reports and all
  12 preview WAVs downloaded under `outputs/tts/jeet_trained_kaggle/jeet_adaptation`;
  audio checks pass (finite mono, correct sample rates/amplitude). Independent
  voice-likeness and intelligibility listening review remains pending. Validation
  loss reduction does not establish equivalent improvement in speaker likeness.
- [x] **Kaggle startup fix 2026-10-09**: version 2 completed dependency setup
  under Python 3.11, found Tesla T4, and verified exact hashes for base weights
  and vocabulary. It then failed before optimization because the subprocess
  inherited `MPLBACKEND=module://matplotlib_inline.backend_inline`, unavailable
  in the isolated environment. Notebook now sets `MPLBACKEND=Agg` for CLI
  imports. Corrected version 3 successfully submitted and API reports RUNNING.
  Local F5 import with Agg also passes. No completed training or adapted
  checkpoint is claimed yet.
- [x] **Kaggle job submitted 2026-10-09** through authenticated Kaggle CLI:
  `https://www.kaggle.com/code/jeet065/jeet-indicf5-fine-tune`, kernel ID
  137726006. Remote metadata confirms private visibility, GPU T4, Internet,
  and `jeet065/jeet-voice` input. Version 1 reached RUNNING but failed at the
  dependency install: Kaggle host Python 3.13 cannot install torch 2.5.1.
  Corrected version 2 was pushed successfully and is QUEUED, creating an
  isolated Python 3.11 environment under /tmp before installing pinned wheels.
  Training completion, exported weights and listening quality remain unverified.
  The notebook embeds current training code to supersede the older input
  bundle. Model input is `amit27kumar/models--ai4bharat--indicf5/Transformers/1/1`;
  the notebook requires exact SHA-256 matches against the pinned original
  checkpoint and vocabulary before any optimizer update. No model-source code
  is imported, and no HF/Kaggle credential is embedded. A private cached-weight
  upload was started as a fallback, then cancelled once the hosted model input
  allowed submission. Remote hash validation results are pending run output.
- [x] **Local dataset weight loader**: `SAMVYAS_INDICF5_DIR` supports verified
  Kaggle-mounted model/vocabulary files without gated Hub authentication.
  Loader/inference tests: 22 passed, 1 skipped; lint and formatting pass.
- [x] **Kaggle dataset supplied 2026-10-09**: user linked
  `https://www.kaggle.com/datasets/jeet065/jeet-voice`. Notebook setup now targets
  the `jeet-voice` mount, supports extracted or zipped prepared bundles, checks
  required code/split metadata, and protects against ZIP path traversal.
  Remote file listing could not be verified (page inaccessible, API HTTP 403,
  browser unavailable). Training has not been started remotely.
- [x] **Jeet fine-tuning workflow prepared 2026-10-09**: portable split of 45
  train clips (329.027 seconds) and 5 validation clips (36.163 seconds), with
  checksums, original 16 kHz WAVs, no transcript overlap, and native 24 kHz
  conversion at the model boundary. All 50 transcripts fit the pinned IndicF5
  vocabulary. `scripts/train_indic_f5.py` adapts the last two DiT blocks and
  output layers (31,588,452 parameters), using batch one, accumulation four,
  learning rate 1e-5, at most 40 epochs, fixed-randomness validation and patience
  five. Exports full best/last online weights in safetensors; CLI inference now
  accepts `--checkpoint`. Best retains base weights if validation never improves.
- [x] **Kaggle handoff prepared**: `notebooks/jeet_indic_f5_finetune.ipynb`
  installs pinned dependencies, reads a Kaggle HF_TOKEN secret, trains, and
  synthesizes base/adapted listening comparisons with a training-only reference.
  `outputs/tts/jeet_kaggle_bundle.zip` contains 50 audio clips and the required
  notebook/code (9,324,987 bytes), verified without secrets, caches or weights.
  No Kaggle upload or training job has been executed; no adapted Jeet checkpoint
  or improved voice identity is claimed yet.
- [x] **Adaptation verification 2026-10-09**: full suite 82 passed / 4 skipped;
  lint, formatting, notebook syntax, bundle integrity and diff checks passed.
  Tiny upstream CFM tests verify gradient flow, unchanged frozen layers,
  native-rate tensor shapes, final partial gradient accumulation, and strict
  checkpoint reloading. The full pretrained model also completed a real Jeet
  clip forward/backward on RTX 2050 with finite loss 0.5272, prediction shape
  (1,667,100), and peak allocated GPU memory 1,545,674,752 bytes. This smoke test
  made no optimizer update and is not a training or voice-similarity result.
- [x] **Jeet reference voice preview 2026-10-08**: audited all 50 accepted
  recordings in `assets/audio/voice_users/jeet` (365.190 seconds). All WAVs
  are mono 16 kHz, with no detected clipping/near-silence or orphan files.
  Selected conversational reference `76f1edc2b50a49f19901fc83baf5bf2f`.
  Frozen IndicF5 generated three unseen Bengali sentences on CUDA at 32 steps
  into `outputs/tts/jeet_pilot_20261008`, with 24 kHz/16 kHz exports.
  Audit and explicit reference metadata are in `outputs/tts/jeet_dataset_audit.json`
  and `outputs/tts/jeet_voice_preset.json`. No speaker checkpoint was trained;
  independent transcript and voice-similarity listening assessment is pending.
- [x] **Bengali recording pilot implemented 2026-10-07**: `scripts/record_voice.py`,
  reusable microphone/session helpers, resumable WAV/JSONL dataset store, and
  50 Bengali prompts. Enter start/stop, 20-second bound, mic check/selection,
  playback/retry/transcript editing, explicit confirmation, quality measurements,
  orphan reporting, and interruption-safe manifest publication are covered.
  `.venv-recorder` is ready (Python 3.11, NumPy/sounddevice/soundfile; no PyTorch).
  Default microphone settings validated for mono 16 kHz; no live voice recording
  or listening assessment was performed by the agent.
- [x] **Recorder UX simplified 2026-10-08**: Windows console input/output code
  pages are set to UTF-8 and Bengali prompts use `WriteConsoleW` when available,
  avoiding legacy terminal mojibake. Recording controls are now single letters:
  `r` record, `y` save/confirm, `n` retry, `p` play, `e` edit, `s` skip, and
  `q` quit. The old multi-letter `yes` confirmation was removed.
- [x] **Local browser recorder added 2026-10-08**: `scripts/voice_recorder_web.py`
  serves a loopback-only HTML Voice Studio. It creates or resumes per-person
  profiles under `assets/audio/voice_users/<safe-name>/`, displays Bengali
  prompts in a browser, and provides Record, Stop, Play take, Record again,
  transcript edit, and Save sample controls. Browser WAV input is downmixed,
  resampled through SciPy, and stored through the existing atomic `VoiceDataset`
  writer as mono 16 kHz PCM WAV plus JSONL metadata. The generated profiles are
  Git-ignored. The UI was opened locally and Bengali rendering, profile opening,
  prompt display, progress, and controls were visually checked; microphone
  permission, live capture, and listening were intentionally not exercised.
- [x] **Web recorder verification 2026-10-08**: service and HTTP route tests
  cover profile creation, 48 kHz browser-WAV conversion, duplicate prompt
  protection, API prompt delivery, and malformed-WAV rejection. Full test suite:
  **78 passed, 4 skipped**. SciPy is installed in `.venv-recorder`.
- [x] **Web capture quality update 2026-10-08**: replaced the legacy
  main-thread `ScriptProcessor` browser capture with `AudioWorklet` PCM frame
  capture, requesting 48 kHz mono input and retaining a lossless WAV path to
  the canonical 16 kHz store. The UI displays the active rate and asks for a
  better microphone below 44.1 kHz. Existing saved browser clips are preserved
  unchanged; distorted takes need a new recording because their lost detail
  cannot be recovered after capture.
- [x] **Recorder compatibility and packet fix 2026-10-08**: fixed the Record
  button capability guard, which could throw before reporting an unsupported
  Web Audio environment. The recorder now prefers `AudioWorklet`, uses an
  explicit 48 kHz audio context when available, batches 4,096 PCM samples per
  worklet transfer, flushes the final partial packet before stopping, and uses
  a visible compatibility capture fallback when worklets are unavailable. A
  diagnostic start/stop capture on this machine completed without saving a
  file; its original context rate was 192 kHz, confirming why batching and
  the requested 48 kHz context matter.
- [x] **IndicF5 baseline integration implemented 2026-10-07**: isolated
  `requirements-tts.txt`, frozen transformer/Vocos wrapper, explicit accepted
  clip selection or reference/transcript arguments, three unseen Bengali texts,
  24 kHz + 16 kHz exports, run metadata, and opt-in CUDA OOM CPU fallback.
  Upstream source and model revision pinned; real checkpoint compatibility is
  verified by CPU and CUDA generation. Voice quality awaits listening review.
  Acoustic/prosody vector conditioning and
  fine-tuning are intentionally deferred.
- [x] **Verification 2026-10-07**: complete non-heavy suite in `.venv-verify`:
  **74 passed, 4 skipped** (three existing STT heavy tests and the new opt-in
  real TTS test). Ruff lint/format and `git diff --check` pass. TTS dependencies
  installed in this separate CPU verification environment; real upstream
  `load_model`/`infer_batch_process` imports and signatures verified. A tiny
  randomly initialized upstream flow decoder also produced finite
  `(1, 12, 100)` mel output with frozen parameters; this does not validate the
  pretrained checkpoint or speech quality.
- [x] **Quick baseline recordings checked 2026-10-07**: user collected 10
  accepted clips totaling 89.28 seconds, all mono 16 kHz, 7.85–11.34 seconds,
  with no near-silent clips or detected sample clipping and no orphan files.
  Clip `6dc2569c5e7a487e8937bda22156e163` selected for the baseline. No additional
  recordings are needed for this reference-conditioned test. Spoken transcript
  accuracy and voice similarity have not been independently listening-verified.
- [x] **IndicF5 access granted 2026-10-07**: after user accepted access,
  authenticated vocabulary/config/source downloads succeeded. Earlier HTTP
  403 is resolved. The 1,402,789,408-byte checkpoint and pinned Vocos weights
  are cached outside Git. `.venv-tts` is installed with PyTorch 2.5.1+cu121
  and detects the NVIDIA GeForce RTX 2050.
- [x] **Real Bengali baseline generated 2026-10-07**: all three target sentences
  synthesized at 32 steps using clip `6dc2569c5e7a487e8937bda22156e163` on CPU
  and CUDA. Outputs in `outputs/tts/indicf5_pilot_20261007_cpu/` and
  `outputs/tts/indicf5_pilot_20261007_cuda/`, each with six WAVs and run metadata.
  Generated durations are 5.21, 5.09, and 5.32 seconds; both sample rates have
  finite, non-silent audio in [-1, 1]. These are reference-conditioned outputs,
  not speaker fine-tuning results.
- [x] **Real-run fixes**: public Vocos download explicitly bypasses the expired
  cached OAuth login; terminal output handles Bengali UTF-8. FP16 decoding
  produced invalid audio on CUDA, so the transformer now uses BF16 when
  supported (otherwise float32), while Vocos always decodes in float32.
  Non-finite mel inputs are rejected. Regression tests cover vocoder precision,
  forward shape, input gradient flow, invalid mels, and anonymous public fetches.
- [ ] **Baseline listening evaluation**: review the three generated sentences for intelligibility,
  voice identity, delivery, and artifacts by listening.
- [ ] **Environment note 2026-10-07**: the original `.venv` launcher references
  a removed Python 3.10 installation. It was left unchanged. Use
  `.venv-recorder` for collection and the installed `.venv-tts` for CUDA
  synthesis. README documents setup with CUDA/CPU wheel options. Verification
  used `.venv-verify`.
- [x] Renamed project to **Samvyas**.
- [x] Completely removed legacy `voicellm` folder.
- [x] Preserved reference research (`notebooks/indic_conformer_test.ipynb`) and audio samples (`assets/audio/`).
- [x] **Phase 1 (STT-only) implemented & verified 2026-09-10**: `samvyas/audio` (io + DeepFilterNet v2), `samvyas/models/encoders/semantic` (IndicConformer BN, Vdim 768), `samvyas/pipelines/stt_pipeline.py`, configs, `scripts/download_models.py`. 18/18 pytest pass (incl. 2 GPU heavy: real DeepFilter + real .nemo ≤5s clip) on RTX 2050.
- [x] **Live demo dual-stream 2026-09-10**: `scripts/live_stt.py` remade — STT turn + para branch on cleaned audio (backbone/layer/device, acoustic pooled stats, prosody voiced-ratio/mean-F0/energy/ZCR, para→semantic alignment, per-stage timings). Para defaults to CPU (4GB guard); `--no-para/--para-device/--para-layer`. Dry-run verified with real IndicWav2Vec (stt 40.9s incl. load, para 12.6s CPU).
- [x] **Paralinguistic branch 2026-09-10**: `samvyas/models/encoders/paralinguistic/` (`base.py` contracts + `align_frames`, `indic_wav2vec.py` frozen SSL encoder default `ai4bharat/indicwav2vec_v1_bengali`, `prosody.py` F0/voicing/energy/ZCR @20ms hop). Config `configs/encoders/paralinguistic.yaml`; `download_models.py --include-para` (graceful gated message). Suite: 26 passed, 3 skipped. Real-HF path proven on tiny wav2vec2; real IndicWav2Vec weights **blocked on user HF access** (`huggingface-cli login`, then re-run fetch + `SAMVYAS_RUN_HEAVY=1 pytest`).
- [x] **HF auth wired 2026-09-10**: token in repo `.env` (`HF_ACESS_TOKEN`), `.env` git-ignored, `python-dotenv` added, `download_models.py` auto-loads it (verified `whoami: jeetxab`). Real IndicWav2Vec still 403 — account needs repo access grant (see next).
- [ ] **Waiting on user**: approve access at https://huggingface.co/ai4bharat/indicwav2vec_v1_bengali, then re-run fetch + heavy test (commands in plan).
- [x] **Para weights live 2026-09-10**: access granted, `indicwav2vec_v1_bengali` cached (wav2vec2-large, D=1024, 24 layers). Full suite with real weights: **29/29 pass** on RTX 2050 (incl. real DeepFilter, real .nemo, real IndicWav2Vec ≤5s clips).
