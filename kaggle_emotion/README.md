# Samvyas emotion LoRA on Kaggle

This kernel trains a frozen-base IndicF5 adapter on prompt/target pairs:

```text
mel = [neutral prompt frames | emotional target frames]
loss = flow matching on target frames only
trainable = LoRA on to_q/to_k/to_v/to_out.0/ff.ff.0.0/ff.ff.2 + 4 tag rows + null row
```

`run.py` starts with `nvidia-smi`, requires exactly two T4s, clones the repo,
installs the TTS-only dependencies, builds a portable manifest, launches
`accelerate --multi_gpu --num_processes 2 --mixed_precision fp16`, and packages
the resumable `last` adapter.

## First run: 50-step smoke test

The default mode is `smoke`. It uses only the existing private Jeet v2 dataset,
so it does not need SUBESCO or IndicVoices-R yet. It runs 50 optimizer steps
over up to 6 epochs (two-process sharding plus four gradient-accumulation
microbatches per step) and validates the data path, DDP, fp16, gradient
checkpointing, adapter export, and resume state.

1. Push the scaffold to GitHub. `run.py` clones `SAMVYAS_REPO` at
   `SAMVYAS_COMMIT` (default `main`). Alternatively, upload the repository as a
   private Kaggle code dataset; `run.py` auto-detects
   `scripts/train_emotion_lora.py`, or set `SAMVYAS_CODE_DIR` to its root.
2. Check `kernel-metadata.json`; the `id` must match your Kaggle account.
3. Push the kernel:

   ```powershell
   .\.venv-kaggle\Scripts\kaggle.exe kernels push -p kaggle_emotion
   .\.venv-kaggle\Scripts\kaggle.exe kernels status jeet065/samvyas-indicf5-emotion-lora
   .\.venv-kaggle\Scripts\kaggle.exe kernels logs jeet065/samvyas-indicf5-emotion-lora
   ```

4. Download `emotion_lora_smoke_bundle.zip` from the kernel output before the
   12-hour session ends. Kaggle `/kaggle/working` does not survive forever.

## Full run

1. Accept the `ai4bharat/IndicF5` access conditions and add `HF_TOKEN` as a
   Kaggle Secret if the model is not attached as a local model source.
2. Upload the official CC BY 4.0 SUBESCO parquet bundle as a private dataset.
3. Upload the curated IndicVoices-R Bengali neutral slice as a private dataset
   (JSONL/CSV/parquet with audio, text, speaker_id, sentence_id, duration).
4. Add both dataset sources to `kernel-metadata.json` and set
   `SAMVYAS_MODE=full` in the kernel environment (Kaggle: **Settings → Environment
   variables**, or edit the default in `run.py`).
5. The manifest builder keeps at most 5 neutral hours, holds out 2 speakers and
   1 sentence per speaker, writes 80% neutral prompts, and filters pairs above
   19 seconds total.
6. Re-push the kernel. The trainer resumes from `last/trainer_state.pt` when
   the working directory still has it.

Dataset paths can be overridden with `JEET_DIR`, `SUBESCO_DIR`, `NEUTRAL_DIR`,
`SAMVYAS_DATA_DIR`, `SAMVYAS_CODE_DIR`, and `SAMVYAS_INDICF5_DIR`.

## Outputs

```text
/kaggle/working/emotion_lora_smoke/
  last/adapter.safetensors
  last/adapter.json
  last/trainer_state.pt
  best/...
  step_0000025/...
  history.json
  training_report.json
/kaggle/working/emotion_lora_smoke_bundle.zip
```

`training_report.json` explicitly marks listening review as pending. Loss values
are not evidence of emotional quality or voice likeness.

## Licensing

SUBESCO is CC BY 4.0; retain attribution and do not redistribute the private
Jeet voice data. IndicVoices-R is gated and CC BY 4.0; follow its terms. The
adapter contains no base weights, but it is still derived from a gated model and
personal voice references.

## Compute notes

Kaggle currently gives T4 x2 for `machine_shape: "NvidiaTeslaT4"` with about
30 GPU-hours/week and a 12-hour session cap. The trainer checkpoints on rank 0
only, keeps `best` plus the newest two `step_*` directories, and always writes
`last` for resume. Use `SAMVYAS_MODE=full` only after the smoke run passes.

`run.py` sets `USE_TF=0` / `USE_FLAX=0` / `TRANSFORMERS_NO_TF=1` before any
import: Kaggle's base image ships TensorFlow and JAX, and `transformers` 4.49
eagerly imports TensorFlow when it is detected, which pulls in JAX and needs
numpy>=2 — colliding with the F5 stack's numpy 1.26.4 pin. The smoke run also
uses `--num-workers 0` because Windows spawn pickles the dataset.

The frozen DiT keeps float32 weights, so the flow target is cast to the model
dtype/device inside `forward_emotion` before the DiT sees it. Without that cast
fp16 produces a dtype mismatch and NaN validation loss at step 0.
