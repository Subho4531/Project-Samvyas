"""Kaggle dual-T4 job for the IndicF5 emotion LoRA adapter.

Modes (``SAMVYAS_MODE``):
  smoke: Jeet v2 only, 50 optimizer steps, validates DDP/checkpoint plumbing.
  full:  SUBESCO + curated IndicVoices-R Bengali neutral + Jeet, Stage A training
         (up to 10 epochs / 9.5h cap) followed by unseen-sentence evaluation.

The script starts with nvidia-smi and fails unless exactly two GPUs are visible.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# Kaggle's base image ships TensorFlow + JAX. transformers 4.49 eagerly
# imports TensorFlow when it is detected, and that pulls in JAX, which needs
# numpy>=2 (np.dtypes.StringDType). The F5 stack pins numpy 1.26.4, so disable
# TF/Flax detection before anything touches transformers.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
# Keep Hugging Face downloads out of /kaggle/working so kernel outputs stay small.
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")

# This must be the first executable action in the kernel.
SMI = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
print(SMI.stdout)
if SMI.returncode != 0:
    raise SystemExit("nvidia-smi failed; no GPU worker is attached.")

import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available on this worker.")
if torch.cuda.device_count() != 2:
    raise SystemExit(f"Expected exactly 2 GPUs, found {torch.cuda.device_count()}.")
print(
    json.dumps(
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(index) for index in range(2)],
        }
    ),
    flush=True,
)

WORK = Path("/kaggle/working")
TMP_WORK = Path("/tmp/samvyas_work")
INPUT = Path("/kaggle/input")
MODE = os.environ.get("SAMVYAS_MODE", "full").strip().lower()
if MODE not in ("smoke", "full"):
    raise SystemExit("SAMVYAS_MODE must be smoke or full.")
REPO_URL = os.environ.get("SAMVYAS_REPO", "https://github.com/Subho4531/Project-Samvyas.git")
COMMIT = os.environ.get("SAMVYAS_COMMIT", "main")


def _disk_report(label: str) -> None:
    """Log available disk space in /kaggle/working and /tmp."""
    info: dict[str, dict[str, float]] = {}
    for name, path in (("working", WORK), ("tmp", Path("/tmp"))):
        if path.exists():
            usage = shutil.disk_usage(path)
            info[name] = {
                "total_gb": round(usage.total / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
            }
    print(f"Disk [{label}]: {json.dumps(info)}", flush=True)


def secret(name: str) -> str | None:
    """Read a Kaggle secret without failing when the client is unavailable."""
    try:
        from kaggle_secrets import UserSecretsClient

        value = UserSecretsClient().get_secret(name)
        return str(value) if value else None
    except Exception as exc:  # noqa: BLE001 - Kaggle may not expose secrets outside kernels
        print(f"Kaggle secret {name!r} unavailable ({type(exc).__name__}).", flush=True)
        return None


def base_vocab(base: Path) -> Path | None:
    """Return the vocab file for root or checkpoints/ model layouts."""
    for candidate in (base / "vocab.txt", base / "checkpoints" / "vocab.txt"):
        if candidate.is_file():
            return candidate
    return None


def find_local_base() -> Path | None:
    """Find an attached IndicF5 model directory with vocab and weights."""
    explicit = os.environ.get("SAMVYAS_INDICF5_DIR")
    if explicit:
        root = Path(explicit)
        if base_vocab(root) is not None and (root / "model.safetensors").is_file():
            return root
    for vocab in INPUT.rglob("vocab.txt"):
        root = (
            vocab.parent if (vocab.parent / "model.safetensors").is_file() else vocab.parent.parent
        )
        if (root / "model.safetensors").is_file():
            return root
    return None


def find_input(*needles: str) -> Path | None:
    """Return the first input directory whose path contains all needles.

    Kaggle now mounts datasets as ``/kaggle/input/datasets/<owner>/<slug>``
    rather than directly under ``/kaggle/input``.  Search both layouts.
    """
    explicit = os.environ.get("SAMVYAS_DATA_DIR")
    if explicit and Path(explicit).is_dir():
        return Path(explicit)
    roots: list[Path] = [INPUT]
    datasets = INPUT / "datasets"
    if datasets.is_dir():
        roots.append(datasets)
        roots.extend(path for path in datasets.rglob("*") if path.is_dir())
    roots.extend(path for path in INPUT.iterdir() if path.is_dir() and path != datasets)
    seen: set[Path] = set()
    for directory in sorted(roots):
        if directory in seen:
            continue
        seen.add(directory)
        name = directory.name.lower()
        if all(needle in name for needle in needles):
            return directory
    return None


def env_dir(name: str) -> Path | None:
    """Return an existing directory from an environment override."""
    value = os.environ.get(name)
    if value and Path(value).is_dir():
        return Path(value)
    return None


def resolve_jeet_dir() -> Path:
    """Locate the attached Jeet v2 dataset directory under /kaggle/input."""
    jeet = env_dir("JEET_DIR") or find_input("jeet", "voice", "adaptation")
    if jeet is None:
        for candidate in (
            INPUT / "jeet-voice-adaptation-v2",
            INPUT / "datasets" / "jeet065" / "jeet-voice-adaptation-v2",
        ):
            if candidate.is_dir():
                jeet = candidate
                break
    if jeet is None or not jeet.is_dir():
        raise SystemExit(f"Jeet input dataset not found under {INPUT}.")
    return jeet


def clone_code() -> Path:
    """Clone or reuse the repository in /tmp and check out the requested revision."""
    if os.environ.get("SAMVYAS_CODE_DIR"):
        code = Path(os.environ["SAMVYAS_CODE_DIR"]).resolve()
        if not (code / "scripts" / "train_emotion_lora.py").is_file():
            raise SystemExit(f"SAMVYAS_CODE_DIR is not the Samvyas repo: {code}")
        return code
    for script in INPUT.rglob("train_emotion_lora.py"):
        candidate = script.parent.parent
        if (candidate / "samvyas" / "models" / "tts" / "emotion_lora.py").is_file():
            print("Using attached code dataset:", candidate, flush=True)
            return candidate
    TMP_WORK.mkdir(parents=True, exist_ok=True)
    code = TMP_WORK / "VoiceLLM"
    if code.exists():
        shutil.rmtree(code)
    token = secret("GITHUB_TOKEN")
    url = REPO_URL
    if token and REPO_URL.startswith("https://github.com/"):
        url = REPO_URL.replace("https://", f"https://x-access-token:{token}@", 1)
    subprocess.run(
        ["git", "clone", "--filter=blob:none", url, str(code)],
        check=True,
    )
    subprocess.run(["git", "-C", str(code), "checkout", COMMIT], check=True)
    if token:
        subprocess.run(
            ["git", "-C", str(code), "remote", "set-url", "origin", REPO_URL],
            check=True,
        )
    return code


def install(code: Path) -> None:
    """Install the TTS-only package set without replacing Kaggle's CUDA torch."""
    packages = [
        "f5_tts @ git+https://github.com/AI4Bharat/IndicF5.git@13f7c4d627cc10111aea8fe9c0039462cacacdc7",
        "accelerate>=1.0",
        "safetensors>=0.4",
        "pyarrow>=15",
        "python-dotenv>=1.0",
        "torchdiffeq>=0.2.4",
        # F5's unpinned transitive install can pull transformers v5 / numpy 2,
        # which break the pinned IndicF5 inference code.
        "numpy==1.26.4",
        "transformers==4.49.0",
        "datasets>=3.6,<4",
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
        check=True,
    )
    eval_check = subprocess.run(
        [sys.executable, "-c", "import librosa, sklearn"],
        capture_output=True,
        check=False,
    )
    if eval_check.returncode != 0:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "librosa>=0.10",
                "scikit-learn>=1.3",
                "numpy==1.26.4",
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import f5_tts, accelerate, safetensors, pyarrow, librosa, sklearn; print('tts+eval deps ok')",
        ],
        check=True,
    )
    print("Repository:", code, flush=True)


def fetch_subesco(dst: Path) -> Path:
    """Download public sajid73/SUBESCO-audio-dataset parquet shards into /tmp."""
    from huggingface_hub import snapshot_download

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    print("Downloading sajid73/SUBESCO-audio-dataset (data/*.parquet)...", flush=True)
    snapshot_download(
        repo_id="sajid73/SUBESCO-audio-dataset",
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
        local_dir=str(dst),
        token=False,
    )
    shards = sorted(dst.rglob("*.parquet"))
    if not shards:
        raise SystemExit(f"No parquet shards downloaded for SUBESCO under {dst}.")
    print(f"Downloaded {len(shards)} SUBESCO parquet shards to {dst}.", flush=True)
    return dst


def _strided_order(items: list[str], limit: int) -> list[str]:
    """Select paths evenly strided across the shard list for speaker diversity."""
    if len(items) <= limit:
        return list(items)
    order: list[str] = []
    used: set[int] = set()
    step = max(1, len(items) // limit)
    for offset in range(step):
        for idx in range(offset, len(items), step):
            if idx not in used:
                order.append(items[idx])
                used.add(idx)
            if len(order) >= limit:
                return order
    return order


def fetch_indicvoices_neutral(
    dst: Path,
    token: str,
    *,
    target_hours: float = 6.5,
    min_speakers: int = 12,
    max_shards: int = 24,
) -> Path:
    """Download gated ai4bharat/indicvoices_r Bengali shards until target hours/speakers are met."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)
    print("Listing ai4bharat/indicvoices_r Bengali/train-*.parquet shards...", flush=True)
    repo_files = sorted(
        f
        for f in api.list_repo_files("ai4bharat/indicvoices_r", repo_type="dataset")
        if f.startswith("Bengali/train-") and f.endswith(".parquet")
    )
    if not repo_files:
        raise SystemExit("No Bengali/train-*.parquet shards found in ai4bharat/indicvoices_r.")

    candidates = _strided_order(repo_files, max_shards)
    speaker_clips: Counter[str] = Counter()
    speaker_seconds: Counter[str] = Counter()
    downloaded = 0

    for shard in candidates:
        local_file = Path(
            hf_hub_download(
                repo_id="ai4bharat/indicvoices_r",
                filename=shard,
                repo_type="dataset",
                token=token,
                local_dir=str(dst),
            )
        )
        downloaded += 1
        pf = pq.ParquetFile(local_file)
        for batch in pf.iter_batches(columns=["duration", "speaker_id", "text"]):
            cols = batch.to_pydict()
            for dur, sp, txt in zip(cols["duration"], cols["speaker_id"], cols["text"]):
                if dur is None or not math.isfinite(float(dur)):
                    continue
                dur_f = float(dur)
                if not 1.0 <= dur_f <= 15.0:
                    continue
                if not isinstance(txt, str) or not txt.strip():
                    continue
                sp_str = str(sp or "").strip()
                if not sp_str:
                    continue
                speaker_clips[sp_str] += 1
                speaker_seconds[sp_str] += dur_f

        pairable_speakers = [sp for sp, cnt in speaker_clips.items() if cnt >= 2]
        pairable_hours = sum(speaker_seconds[sp] for sp in pairable_speakers) / 3600.0
        print(
            f"IV-R shard {downloaded}/{len(candidates)} ({shard}): "
            f"{pairable_hours:.2f}h across {len(pairable_speakers)} pairable speakers",
            flush=True,
        )
        if pairable_hours >= target_hours and len(pairable_speakers) >= min_speakers:
            break
        free_gb = shutil.disk_usage("/tmp").free / (1024**3)
        if free_gb < 4.5 and pairable_hours >= 3.0 and len(pairable_speakers) >= 6:
            print(
                f"Stopping IV-R shard fetch early with {free_gb:.2f} GB free in /tmp "
                f"and {pairable_hours:.2f}h collected.",
                flush=True,
            )
            break

    if not any(dst.rglob("*.parquet")):
        raise SystemExit(f"No IndicVoices-R parquet files present in {dst}.")
    return dst


def prepare_data(code: Path, base: Path | None, jeet: Path) -> Path:
    """Fetch corpora if needed, build manifest under /tmp, and free raw HF shards."""
    TMP_WORK.mkdir(parents=True, exist_ok=True)
    data = TMP_WORK / f"emotion_data_{MODE}"
    if data.exists():
        shutil.rmtree(data)
    prep = code / "scripts" / "prepare_emotion_manifest.py"
    command = [
        sys.executable,
        str(prep),
        "--jeet-root",
        str(jeet),
        "--output-dir",
        str(data),
        "--seed",
        "42",
        "--min-duration",
        "1.0",
        "--max-duration",
        "15.0",
    ]
    vocab = base_vocab(base) if base is not None else None
    if vocab is not None:
        command.extend(["--vocab", str(vocab)])

    raw_downloads: Path | None = None
    if MODE == "full":
        raw_downloads = TMP_WORK / "hf_datasets"
        subesco = env_dir("SUBESCO_DIR") or find_input("subesco")
        if subesco is None:
            subesco = fetch_subesco(raw_downloads / "subesco")
        neutral = env_dir("NEUTRAL_DIR") or find_input("indicvoices") or find_input("neutral")
        if neutral is None:
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise SystemExit(
                    "HF_TOKEN is required to download gated ai4bharat/indicvoices_r in full mode. "
                    "Enable HF_TOKEN under Kaggle Secrets -> Add-ons."
                )
            neutral = fetch_indicvoices_neutral(
                raw_downloads / "indicvoices_r",
                token,
                target_hours=float(os.environ.get("SAMVYAS_NEUTRAL_FETCH_HOURS", "6.5")),
                max_shards=int(os.environ.get("SAMVYAS_MAX_IVR_SHARDS", "24")),
            )
        command.extend(
            [
                "--subesco",
                str(subesco),
                "--neutral",
                str(neutral),
                "--limit-neutral-hours",
                "5",
                "--holdout-speakers",
                "2",
                "--holdout-sentences",
                "1",
                "--neutral-prompt-ratio",
                "0.8",
                "--jeet-replay",
                "8",
            ]
        )
    else:
        command.extend(
            [
                "--limit-neutral-hours",
                "0",
                "--holdout-speakers",
                "0",
                "--holdout-sentences",
                "0",
                "--neutral-prompt-ratio",
                "1.0",
                "--jeet-replay",
                "1",
            ]
        )

    env = {**os.environ, "PYTHONPATH": str(code)}
    subprocess.run(command, check=True, env=env)
    if not (data / "train.jsonl").is_file() or not (data / "validation.jsonl").is_file():
        raise SystemExit("Manifest build did not produce train/validation JSONL.")
    report = json.loads((data / "report.json").read_text(encoding="utf-8"))
    print("Manifest report:", json.dumps(report, indent=2), flush=True)

    # Free multi-GB 48 kHz HF parquet downloads now that normalized 16 kHz WAVs exist.
    if raw_downloads is not None and raw_downloads.exists():
        shutil.rmtree(raw_downloads)
        print("Cleaned up raw HF dataset shards under", raw_downloads, flush=True)
    _disk_report("after prepare_data")
    return data


def _copy_manifest_metadata(data: Path, output: Path) -> None:
    """Copy lightweight JSON/JSONL manifest provenance into the output folder."""
    meta_dir = output / "manifest_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for name in ("report.json", "clips.jsonl", "train.jsonl", "validation.jsonl"):
        src = data / name
        if src.is_file():
            shutil.copy2(src, meta_dir / name)


def train(code: Path, base: Path | None, data: Path) -> Path:
    """Launch Accelerate DDP on both T4s and return the run directory."""
    output = WORK / f"emotion_lora_{MODE}"
    if output.exists() and any(output.iterdir()):
        resume = output / "last" / "trainer_state.pt"
        if not resume.is_file():
            output = output.with_name(output.name + "_" + time.strftime("%Y%m%d_%H%M%S"))
    resume = output / "last"
    script = code / "scripts" / "train_emotion_lora.py"
    accelerate_bin = shutil.which("accelerate")
    launcher = (
        [accelerate_bin, "launch"]
        if accelerate_bin
        else [sys.executable, "-m", "accelerate.commands.launch"]
    )
    max_seconds = "0" if MODE == "smoke" else os.environ.get("SAMVYAS_MAX_SECONDS", "34200")
    command = [
        *launcher,
        "--multi_gpu",
        "--num_processes",
        "2",
        # IndicF5's own inference path autocasts to bfloat16 (see indic_f5.py);
        # the 22-block DiT overflows fp16 and produces NaN validation loss.
        "--mixed_precision",
        "bf16",
        str(script),
        "--data-root",
        str(data),
        "--output-dir",
        str(output),
        "--mel-cache",
        str(TMP_WORK / "emotion_mel_cache"),
        "--epochs",
        "6" if MODE == "smoke" else "10",
        "--max-steps",
        "50" if MODE == "smoke" else "0",
        "--max-seconds",
        max_seconds,
        "--learning-rate",
        "2e-4",
        "--accumulation",
        "4",
        "--lora-rank",
        "8",
        "--lora-alpha",
        "16",
        "--tag-dropout",
        "0.1",
        "--max-frames",
        "1800",
        "--max-batch-size",
        "2",
        "--eval-batch-size",
        "1",
        "--num-workers",
        "0" if MODE == "smoke" else "2",
        "--gradient-checkpointing",
        "--mixed-precision",
        "bf16",
        "--save-every",
        "25" if MODE == "smoke" else "200",
        "--log-every",
        "5",
        "--patience",
        "0" if MODE == "smoke" else "5",
        "--metadata",
        json.dumps(
            {
                "kaggle_mode": MODE,
                "repo": REPO_URL,
                "commit": COMMIT,
                "manifest": str(data),
            }
        ),
    ]
    if resume.is_dir() and (resume / "trainer_state.pt").is_file():
        command.extend(["--resume", str(resume)])
        print("Resuming from", resume, flush=True)
    env = {
        **os.environ,
        "PYTHONPATH": str(code),
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if base is not None:
        env["SAMVYAS_INDICF5_DIR"] = str(base)
    subprocess.run(command, check=True, env=env)
    _copy_manifest_metadata(data, output)
    return output


def evaluate(code: Path, base: Path | None, data: Path, jeet: Path, output: Path) -> None:
    """Synthesize the fixed 5 unseen Bengali sentences x 4 tags x (32,16,8) steps grid."""
    best_adapter = output / "best" / "adapter.safetensors"
    last_adapter = output / "last" / "adapter.safetensors"
    adapter = best_adapter if best_adapter.is_file() else last_adapter
    eval_dir = output / "eval_unseen"
    script = code / "scripts" / "evaluate_emotion_lora.py"
    command = [
        sys.executable,
        str(script),
        "--adapter",
        str(adapter),
        "--data-root",
        str(data),
        "--jeet-root",
        str(jeet),
        "--mode",
        "unseen",
        "--steps",
        "32,16,8",
        "--output-dir",
        str(eval_dir),
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(code),
        "CUDA_VISIBLE_DEVICES": "0",
    }
    if base is not None:
        env["SAMVYAS_INDICF5_DIR"] = str(base)
    print(f"Running unseen evaluation with adapter {adapter}...", flush=True)
    subprocess.run(command, check=True, env=env)


def _verify_indicvoices_access(token: str) -> None:
    """Fail fast before pip install if HF_TOKEN cannot read gated IndicVoices-R shards."""
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
    except ImportError:
        return
    probe_url = hf_hub_url(
        repo_id="ai4bharat/indicvoices_r",
        filename="Bengali/train-00000-of-00155.parquet",
        repo_type="dataset",
    )
    try:
        meta = get_hf_file_metadata(probe_url, token=token)
        print(f"Verified IndicVoices-R gated access (probe size={meta.size}).", flush=True)
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403) or "gated" in type(exc).__name__.lower() or "403" in str(exc):
            raise SystemExit(
                "HF_TOKEN lacks access to gated dataset ai4bharat/indicvoices_r (HTTP "
                f"{status or 403}). Visit https://huggingface.co/datasets/ai4bharat/indicvoices_r "
                "signed in as your HF account, click 'Agree and access repository', and ensure "
                "fine-grained tokens allow 'Read access to contents of all public gated repos "
                "you can access'."
            ) from exc
        raise


def main() -> None:
    """Prepare data, train on two GPUs, evaluate unseen audio, and package the bundle."""
    _disk_report("startup")
    token = (
        secret("HF_TOKEN")
        or secret("HF_ACESS_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HF_ACESS_TOKEN")
    )
    if token:
        os.environ["HF_TOKEN"] = token
    need_ivr_download = (
        MODE == "full" and env_dir("NEUTRAL_DIR") is None and find_input("indicvoices") is None
    )
    if need_ivr_download:
        if not token:
            raise SystemExit(
                "HF_TOKEN secret is missing for full mode. Kaggle CLI `kernels push` does not carry "
                "Add-ons -> Secrets; open the pushed notebook in Kaggle Web UI, enable Add-ons -> "
                "Secrets -> HF_TOKEN, and run Save Version -> Save & Run All (or provide NEUTRAL_DIR)."
            )
        _verify_indicvoices_access(token)
    base = find_local_base()
    if base is not None:
        os.environ["SAMVYAS_INDICF5_DIR"] = str(base)
        print("Local IndicF5 base:", base, flush=True)
    elif not os.environ.get("HF_TOKEN"):
        print("No local base or HF_TOKEN; the gated HF download may fail.", flush=True)

    jeet = resolve_jeet_dir()
    code = clone_code()
    install(code)
    data = prepare_data(code, base, jeet)
    output = train(code, base, data)
    report_path = output / "training_report.json"
    if report_path.is_file():
        print("Training report:", report_path.read_text(encoding="utf-8"), flush=True)
    last_adapter = output / "last" / "adapter.safetensors"
    if not last_adapter.is_file():
        raise SystemExit(f"Training did not produce a last adapter: {last_adapter}")

    evaluate(code, base, data, jeet, output)
    archive = shutil.make_archive(str(WORK / f"emotion_lora_{MODE}_bundle"), "zip", output)
    _disk_report("finished")
    print("Bundle:", archive, flush=True)
    print("Last adapter:", last_adapter, flush=True)
    print("Download the bundle before the Kaggle session ends.", flush=True)


if __name__ == "__main__":
    main()
