# Samvyas emotion LoRA on Kaggle

This kernel trains a frozen-base IndicF5 adapter on prompt/target pairs:

```text
mel = [neutral prompt frames | emotional target frames]
loss = flow matching on target frames only
trainable = LoRA on to_q/to_k/to_v/to_out.0/ff.ff.0.0/ff.ff.2 + 4 tag rows + null row
```

`run.py` starts with `nvidia-smi`, requires two T4s, clones the repo into `/tmp`,
installs the TTS-only dependencies, builds a portable manifest, launches
`accelerate --multi_gpu --num_processes 2 --mixed_precision bf16`, runs the
fixed 5-sentence unseen-text evaluation grid (`32, 16, 8` ODE steps), and
packages checkpoints + evaluation audio into a single bundle zip.

## Smoke run override (`SAMVYAS_MODE=smoke`)

Set `SAMVYAS_MODE=smoke` to run a 50-step Jeet-only sanity check (6 epochs max,
two-process sharding plus four gradient-accumulation microbatches per step)
without fetching SUBESCO or IndicVoices-R.

1. Push changes to GitHub (`SAMVYAS_REPO` at `SAMVYAS_COMMIT`, default `main`),
   or attach the repo as a private Kaggle dataset.
2. Check `kernel-metadata.json`; the `id` must match your Kaggle account.
3. Push the kernel:

   ```powershell
   .\.venv-kaggle\Scripts\kaggle.exe kernels push -p kaggle_emotion
   .\.venv-kaggle\Scripts\kaggle.exe kernels status jeet065/samvyas-indicf5-emotion-lora
   .\.venv-kaggle\Scripts\kaggle.exe kernels logs jeet065/samvyas-indicf5-emotion-lora
   ```

4. Download `emotion_lora_full_bundle.zip` from the kernel output before the
   12-hour session ends. Kaggle `/kaggle/working` does not survive forever.

`run.py` launches `accelerate --multi_gpu --num_processes 2 --mixed_precision
bf16`. IndicF5's own inference path autocasts to bfloat16, and the 22-block DiT
overflows fp16 (NaN validation loss at step 0), so bf16 is used for training
and evaluation alike.

## Full run (default: `SAMVYAS_MODE=full`)

1. Accept `ai4bharat/IndicF5` and `ai4bharat/indicvoices_r` access conditions on
   Hugging Face, and enable the `HF_TOKEN` secret on the Kaggle kernel.
2. In `full` mode, `run.py` automatically downloads `sajid73/SUBESCO-audio-dataset`
   (`data/*.parquet`) and strided `ai4bharat/indicvoices_r` Bengali shards
   (`Bengali/train-*.parquet`, stopping around 6.5 pairable hours / >= 12 speakers)
   into `/tmp`, builds the manifest with `--limit-neutral-hours 5 --holdout-speakers 2
   --holdout-sentences 1 --neutral-prompt-ratio 0.8 --jeet-replay 8`, and deletes
   the raw 48 kHz parquet downloads before training begins.
3. Training runs up to 10 epochs (with early stopping `--patience 5` and a 9.5h
   wall-clock cap `--max-seconds 34200` so DDP exits cleanly in time for evaluation
   and zipping before Kaggle's 12-hour timeout).
4. After training, `scripts/evaluate_emotion_lora.py --mode unseen --steps 32,16,8`
   synthesizes 5 unseen Bengali sentences across `<neutral>/<happy>/<sad>/<angry>`
   conditioned on Jeet's canonical neutral clip.

Dataset paths can be overridden with `JEET_DIR`, `SUBESCO_DIR`, `NEUTRAL_DIR`,
`SAMVYAS_DATA_DIR`, `SAMVYAS_CODE_DIR`, and `SAMVYAS_INDICF5_DIR`.

## Outputs

```text
/kaggle/working/emotion_lora_full/
  best/adapter.safetensors
  last/adapter.safetensors
  last/adapter.json
  last/trainer_state.pt
  step_0000200/...
  history.json
  training_report.json
  manifest_metadata/{report.json,clips.jsonl,train.jsonl,validation.jsonl}
  eval_unseen/{evaluation_report.json,listening_review.jsonl,audio/*_{24k,16k}.wav}
/kaggle/working/emotion_lora_full_bundle.zip
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
