# AGENTS.md — Samvyas

STT dual-stream research repo (denoiser + semantic ASR + paralinguistic branch) plus a **separate frozen IndicF5 reference-conditioned TTS baseline**. `public/stt.png` is the target diagram, not the current code: there are **no** `samvyas/models/projectors/` or `backbones/` modules — do not invent them. TTS takes target text + reference WAV + verified transcript only; it does not accept acoustic/prosody vectors.

Source of truth order: executable config/scripts > `README.md` (setup/usage) > `MEMORY.md` (ledger) / `plan.md` (active plan) / `docs/tts_recordings_audit.md` / `outputs/tts/jeet_story_local_20261011/FULL_CONTEXT.md` (long-form TTS run recipe).

## Layout (real entrypoints)

- `samvyas/audio/` (`io.py`, `recording.py`, `pause_segmentation.py`, `denoiser/`), `samvyas/models/encoders/{semantic,paralinguistic}/`, `samvyas/models/tts/` (`indic_f5.py`, `adaptation.py`), `samvyas/pipelines/stt_pipeline.py`, `samvyas/web/voice_recorder.py`
- `scripts/`: `live_stt.py`, `download_models.py`, `record_voice.py`, `voice_recorder_web.py`, `test_indic_f5.py`, `train_indic_f5.py`, `prepare_tts_adaptation.py`
- `configs/{encoders,frontend,tts}/`, `tests/`, `notebooks/jeet_indic_f5_finetune.ipynb`, voice data under `assets/audio/voice_{dataset,users}/` (git-ignored)

## Envs — never mix them (Windows)

- `.venv` = STT (Python 3.11 via `uv venv --python 3.11`; needs `nemo_toolkit[asr]` which is NOT in requirements.txt). Install order matters: torch cu121 → `-r requirements.txt` → `nemo_toolkit[asr]` → **force-reinstall torch+torchaudio cu121 `--no-deps` again** (NeMo pulls CPU torch) → pin `transformers==4.49.0` (v5 breaks NeMo) and `datasets>=3.6,<4` (hub-v1 compat). Verify with `import nemo.collections.asr` + `torch.cuda.is_available()`.
- `.venv-recorder` = mic collection + chat servers (no torch/GPU); `.venv-tts` = IndicF5 only (Python 3.11, `torch==2.5.1+cu121`); `.venv-verify` = fast CPU checks; `.venv-kaggle` = Kaggle CLI. Never install TTS deps into `.venv`.
- Always call the venv python explicitly, e.g. `.\.venv\Scripts\python.exe -m pytest tests/ -q`. Create TTS env with `uv venv --python 3.11 .venv-tts` then cu121 torch wheel first, then `-r requirements-tts.txt` (CPU-only: replace `cu121` with `cpu`).
- Secrets: `.env` (git-ignored) holds `HF_TOKEN`; legacy `HF_ACESS_TOKEN` spelling also read. Accept `ai4bharat/IndicF5` access conditions first. Never print tokens, never bundle them into Kaggle datasets (use Kaggle Secrets).

## Commands

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps   # required for scripts/
.\.venv\Scripts\python.exe scripts/download_models.py      # weights -> HF cache, not repo
.\.venv\Scripts\python.exe scripts/live_stt.py             # 5s mic loop till Ctrl+C
.\.venv\Scripts\python.exe scripts/live_stt.py --file assets/audio/vocaltest.m4a --no-denoise
.\.venv-recorder\Scripts\python.exe scripts/record_voice.py --list-devices
.\.venv-recorder\Scripts\python.exe scripts/voice_recorder_web.py  # loopback http://127.0.0.1:8765 only
.\.venv-tts\Scripts\python.exe scripts/test_indic_f5.py --clip-id YOUR_CLIP_ID --cpu-fallback
.\.venv-verify\Scripts\python.exe -m pytest tests/test_voice_dataset.py tests/test_voice_recording.py tests/test_indic_f5.py -q
```

## Audio / tensor contract

- All internal audio: mono 16 kHz, float32 `[-1.0, 1.0]`; shapes `(B,T)` or `(B,1,T)`; semantic latents `(B,T_frames,768)`. TTS exports native 24 kHz + resampled 16 kHz; resample to 24 kHz only at the IndicF5 boundary.
- Never normalize collected mic audio; TTS wrapper applies one shared peak attenuation to both exports and records `peak_scale` if over range.
- Voice datasets: unprocessed mono 16 kHz PCM-16 WAV + UTF-8 `manifest.jsonl`. One recorder process per dataset folder. Resume by re-running same command; orphans are reported, never auto-deleted. TTS reference must be an accepted 1–15 s clip; longer takes are kept but rejected as references without silent truncation.

## TTS adaptation + long-form synthesis (from FULL_CONTEXT.md)

- Direction: long-term goal is STT → LLM → TTS voice twin; current TTS has no STT-vector conditioning and no streaming — do not build those unasked.
- Fine-tunes start from the pinned pretrained base (`MODEL_REVISION` in `samvyas/models/tts/indic_f5.py`), never from a prior adapted checkpoint; scope is last 2 DiT blocks + output layers (`configure_trainable_layers` in `adaptation.py`). Preserve prior datasets/checkpoints (v1 kept alongside v2).
- Dataset rule: sentence takes → pause-segmented transcript-aligned clips for training; the continuous story recording is excluded from training (pause-transcript alignment unreliable) and used as unseen synthesis text instead. Provenance (`recording_provenance.json`, `preparation_report.json`, `training_report.json`, `history.json`) lives beside each `outputs/tts/` run.
- Long-form synthesis has no script — follow the FULL_CONTEXT.md recipe: split text at punctuation/word boundaries into ≤110-char chunks, 32 diffusion steps, stitch in order with short inter-chunk pauses + longer paragraph pauses (~12 min per ~500 s on RTX 2050 CUDA). Story WAVs are mono 24-bit PCM at 24 kHz + resampled 16 kHz, same duration; record peak amplitude in `run.json`.
- Never claim voice replication from loss drops or successful generation; human listening review (base vs adapted previews in each run's `listening/`) is the voice-quality gate.

## Hardware (RTX 2050 4 GB)

- Para branch defaults to CPU; `--no-para` / `--para-device` override, GPU para risks OOM. Heavy encoders: `eval()` + `torch.no_grad()`, BF16 transformer / float32 Vocos decode, `torch.cuda.empty_cache()` between stages, heavy tests use ≤5 s clips.
- `scripts/test_indic_f5.py` needs `--output-dir` pointing at a new/empty folder (existing outputs never overwritten); `--cpu-fallback` only retries CUDA OOM on CPU.

## End-to-end voice loop (merged STT + LLM + TTS)

- Browser Talk → `POST /api/voice` (16 kHz WAV ≤16 MB/30 s) → `scripts/stt_worker.py` persistent worker in `.venv` (loads DeepFilter+NeMo once, stdio JSON protocol) → `start_stream` → per-sentence TTS in `.venv-tts` → chained playback. Talk prefers server STT, falls back to browser recognition, typing always works. `STTWorker`/`worker_transcriber` in `samvyas/web/voice_chat.py`; `--no-stt/--stt-python/--stt-device/--stt-lang/--stt-no-denoise` on the launcher.
- The `.nemo` checkpoint is NeMo 1.x-era: `scripts/patch_nemo_checkpoint.py` repacks it for NeMo 2.x (tokenizer `multilingual`→`agg`, drop stale `multisoftmax`/`multilingual`/`language_keys`; weights untouched) to `~/.cache/samvyas/indicconformer_bn_agg.nemo`. Point `SAMVYAS_NEMO_PATH` at it (launcher auto-detects). `IndicASREncoder(lang="bn")` default-decodes the aggregate tokenizer in that language. Verified: real Bengali clip transcribes with one word error vs manifest.
- Worker hardening: third-party loguru sinks are forced off stdout and the client skips non-JSON stdout lines (capped), so lazy library logging can't corrupt the protocol.
- VRAM: STT worker holds GPU while per-sentence TTS subprocesses load/unload around it — on 4 GB, use `--cpu-fallback` (TTS) or `--stt-device cpu` at the first OOM.
- Pre-existing suite gaps (not regressions): `assets/audio/audio_for_inference.wav` deleted and `configs/tts/bengali_prompts.json` absent → 9 fixture failures in audio/segmentation/dataset/recorder tests.

## Tests

- Default `pytest tests/ -q` skips heavy/real-model tests. `SAMVYAS_RUN_HEAVY=1` enables real DeepFilter/`.nemo`/IndicWav2Vec GPU tests (needs weights + access). TTS real-model smoke needs `SAMVYAS_RUN_TTS_REAL=1` + `SAMVYAS_TTS_REFERENCE` + `SAMVYAS_TTS_REFERENCE_TEXT` in `.venv-tts`; listening review is the voice-quality gate, not loss values.
- New modules need a `tests/test_*.py` covering forward pass, output shapes, and gradient flow if trainable.

## Git safety / style

- Never commit `.venv*`, `assets/audio/voice_dataset/`, `assets/audio/voice_users/`, `outputs/tts/`, `.env`, or weights (`*.ckpt/.pt/.pth/.bin/.safetensors/.onnx/.nemo` — see `.gitignore`).
- Python ≥3.10, full type hints + tensor-shape docstrings, `ruff` line-length 100 (`ruff check` / `ruff format --check`, `git diff --check` clean).
- Kaggle note: notebook builds a Python 3.11 env under `/tmp` (host is 3.13) and sets `MPLBACKEND=Agg` for CLI imports; check with `.venv-kaggle` `kaggle.exe kernels status|logs`.
