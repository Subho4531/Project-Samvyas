# Emotion LoRA for IndicF5

Status: scaffold + trainer implemented; Kaggle smoke run pending data and a
pushed revision. This is not a claim of emotional quality or voice likeness.

## Objective and non-goals

Learn four controllable delivery tags — neutral, happy, sad, angry — on top of
the pinned `ai4bharat/IndicF5` base. The adapter is reference-conditioned like
the base: target text + a same-speaker reference WAV + verified transcript.

Non-goals for this phase: STT-vector conditioning, streaming synthesis, an LLM
director, a vocoder, or replacing the frozen Jeet v2 baseline. Jeet v2 remains a
separate voice-likeness comparison.

## Data

The canonical bundle is portable:

```text
bundle/
  train.jsonl
  validation.jsonl
  clips.jsonl
  report.json
  wavs/<source>/<clip_id>.wav
```

Every pair row contains a target clip and a same-speaker prompt:

```json
{
  "clip_id": "subesco_...",
  "audio_path": "wavs/subesco/....wav",
  "text": "target Bengali transcript",
  "emotion": "happy",
  "speaker_id": "subesco:F_01",
  "sentence_id": "S_3",
  "source": "subesco",
  "duration": 2.8,
  "sha256": "...",
  "prompt_clip_id": "subesco_...",
  "prompt_audio_path": "wavs/subesco/....wav",
  "prompt_text": "neutral reference transcript",
  "prompt_emotion": "neutral",
  "prompt_speaker_id": "subesco:F_01",
  "prompt_sentence_id": "S_7",
  "prompt_duration": 3.1,
  "tag_id": 2
}
```

Rules enforced by `samvyas/data/emotion_dataset.py`:

- audio is mono 16 kHz PCM-16 WAV; the dataset resamples to 24 kHz and extracts
  100-bin Vocos mels;
- prompt and target must share a speaker and cannot be the same clip;
- a target cannot be its own prompt;
- paths are relative and may not escape the bundle;
- only neutral/happy/sad/angry are accepted (labels are normalized);
- pairs are rejected if `prompt + target` exceeds the configured total duration.

`scripts/prepare_emotion_manifest.py` normalizes SUBESCO integer labels
(0 angry, 3 happy, 4 neutral, 5 sad; other classes are skipped), accepts a
curated IndicVoices-R neutral slice from JSONL/CSV/parquet, and preserves the
existing Jeet v2 train/validation split. It holds out speakers and, for the
remaining speakers, sentence IDs. The report records held-out speakers,
held-out sentences, skipped rows, unknown vocabulary characters, neutral-hour
limiting, and pair statistics.

Prompt policy: 80% neutral prompts (100% for neutral targets) and 20% other
emotions for non-neutral targets. Prefer a different sentence; fall back to a
different take only when necessary. Neutral prompts are the default because the
reference must not be the only source of the target emotion.

Jeet is only ~7.5 minutes, so the builder replays each Jeet training pair 8
times (`--jeet-replay 8`) and caps the neutral slice at 5 hours. With roughly
4.4 hours of SUBESCO, that is about 42% SUBESCO, 48% neutral, and 10% Jeet by
sampling. Validation pairs are never replayed.

## Model

`samvyas/models/tts/emotion_lora.py` freezes the complete CFM and adds:

- LoRA matrices on `to_q`, `to_k`, `to_v`, `to_out.0`, `ff.ff.0.0`, `ff.ff.2`
  in every DiT block;
- a learned tag embedding added to the DiT time embedding. Row 0 is a learned
  null tag; rows 1-4 are neutral/happy/sad/angry. Tag dropout replaces 10% of
  rows with index 0 during training, so all rows (including null) are used under
  DDP.

`B` starts at zero, so the adapter is exactly the base model at step zero. Only
`lora_a`, `lora_b`, and `tag_embed.embed.weight` are trainable. Adapter exports
contain only those tensors plus a JSON sidecar; the 1.3 GB base is never copied.

## Objective

For each pair:

```text
x1 = [prompt_mel | target_mel]          # (B, T, 100)
cond = [prompt_mel | 0]                 # target frames are generated
t ~ U(0,1)
x0 ~ N(0,I)
phi = (1-t) x0 + t x1
pred = DiT(phi, cond, text, t + tag_embedding(tag))
loss = mean_{target frames} ||pred - (x1 - x0)||^2
```

Validity padding is excluded. The prompt stays visible; the target suffix is
masked from `cond` and is the only region contributing to the loss. This differs
from upstream F5's random-span `frac_lengths_mask`.

Optional `--attention-mask` passes the validity mask through attention. It is
off by default to match F5 training behavior; the loss mask is always applied.

## Training

```powershell
.\.venv-tts\Scripts\python.exe scripts/prepare_emotion_manifest.py `
  --subesco PATH_TO_SUBESCO `
  --neutral PATH_TO_INDICVOICES_NEUTRAL `
  --jeet-root outputs/tts/jeet_adaptation_v2/dataset `
  --vocab PATH_TO_INDICF5/vocab.txt `
  --output-dir outputs/emotion_data/v1

accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 `
  scripts/train_emotion_lora.py `
  --data-root outputs/emotion_data/v1 `
  --output-dir outputs/emotion_lora/v1 `
  --gradient-checkpointing
```

`run.py` on Kaggle uses `--max-frames 1800 --max-batch-size 2
--accumulation 4` (effective batch 16) with Accelerate DDP.
The frame sampler packs similar-length pairs under the frame budget; F5 memory
is dominated by sequence length. Checkpointing saves only LoRA/tag tensors plus
optimizer/scheduler state on rank 0. `best`, `last`, and the newest two
`step_*` directories are retained. `--resume DIR` loads `DIR/adapter.safetensors`
and `DIR/trainer_state.pt`.

The first run is a 50-step Jeet-only smoke test (`SAMVYAS_MODE=smoke`). It
validates DDP, fp16, gradient checkpointing, checkpoint export, and resume
without waiting for SUBESCO/IndicVoices-R. It does not establish emotion
quality.

## Evaluation gates

Training loss alone is not evidence. A real acceptance run needs:

1. a held-out emotion classifier or human emotion labels;
2. STT CER on generated target text (the STT pipeline is separate);
3. per-emotion F0, energy, and speaking-rate deltas versus neutral;
4. a listening review comparing base vs adapter and checking Jeet voice likeness.

BANSpEmo can cross-check angry/happy/sad but has no neutral class. No emotion
classifier exists in this repository yet; `paralinguistic/indic_wav2vec.py` is
a frozen feature extractor, not a classifier.

## Integration boundary

Inference integration is intentionally not implemented. The future path is:
LLM director → an emotion tag aligned to `EMOTION_TAGS` → `indic_f5.py` passes
the tag to `EmotionCFM.sample` alongside the reference audio/transcript. The
current `IndicF5Backend.generate` has no emotion parameter and must not be
changed until the adapter passes listening review.
