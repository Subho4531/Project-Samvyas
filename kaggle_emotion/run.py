"""Kaggle dual-T4 job for the IndicF5 emotion LoRA scaffold.

Modes (``SAMVYAS_MODE``):
  smoke: Jeet v2 only, 50 optimizer steps, validates DDP/checkpoint plumbing.
  full:  SUBESCO + curated IndicVoices-R neutral + Jeet, full training.

The script starts with nvidia-smi and fails unless exactly two GPUs are visible.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

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
INPUT = Path("/kaggle/input")
MODE = os.environ.get("SAMVYAS_MODE", "smoke").strip().lower()
if MODE not in ("smoke", "full"):
    raise SystemExit("SAMVYAS_MODE must be smoke or full.")
REPO_URL = os.environ.get("SAMVYAS_REPO", "https://github.com/Subho4531/Project-Samvyas.git")
COMMIT = os.environ.get("SAMVYAS_COMMIT", "main")


def secret(name: str) -> str | None:
    """Read a Kaggle secret without failing when the client is unavailable."""
    try:
        from kaggle_secrets import UserSecretsClient

        value = UserSecretsClient().get_secret(name)
        return str(value) if value else None
    except Exception:  # noqa: BLE001 - Kaggle may not expose secrets outside kernels
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


def clone_code() -> Path:
    """Clone or reuse the repository and check out the requested revision."""
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
    code = WORK / "VoiceLLM"
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
    subprocess.run(
        [sys.executable, "-c", "import f5_tts, accelerate, safetensors; print('tts deps ok')"],
        check=True,
    )
    print("Repository:", code, flush=True)


def prepare_data(code: Path, base: Path | None) -> Path:
    """Run the manifest builder for smoke or full mode."""
    data = WORK / f"emotion_data_{MODE}"
    if data.exists():
        shutil.rmtree(data)
    prep = code / "scripts" / "prepare_emotion_manifest.py"
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
    if MODE == "full":
        subesco = env_dir("SUBESCO_DIR") or find_input("subesco")
        neutral = env_dir("NEUTRAL_DIR") or find_input("indicvoices") or find_input("neutral")
        if subesco is None:
            raise SystemExit(
                "SUBESCO input missing. Upload the official CC BY 4.0 parquet and add "
                "its dataset source to kernel-metadata.json (or set SUBESCO_DIR)."
            )
        if neutral is None:
            raise SystemExit(
                "Curated IndicVoices-R neutral input missing. Upload the JSONL/parquet "
                "bundle and add its dataset source (or set NEUTRAL_DIR)."
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
    return data


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
        [accelerate_bin] if accelerate_bin else [sys.executable, "-m", "accelerate.commands.launch"]
    )
    command = [
        *launcher,
        "--multi_gpu",
        "--num_processes",
        "2",
        "--mixed_precision",
        "fp16",
        str(script),
        "--data-root",
        str(data),
        "--output-dir",
        str(output),
        "--mel-cache",
        str(WORK / "emotion_mel_cache"),
        "--epochs",
        "6" if MODE == "smoke" else "20",
        "--max-steps",
        "50" if MODE == "smoke" else "0",
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
        "2",
        "--gradient-checkpointing",
        "--mixed-precision",
        "fp16",
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
    return output


def main() -> None:
    """Prepare data, train on two GPUs, and package the resumable output."""
    token = secret("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
    base = find_local_base()
    if base is not None:
        os.environ["SAMVYAS_INDICF5_DIR"] = str(base)
        print("Local IndicF5 base:", base, flush=True)
    elif not os.environ.get("HF_TOKEN"):
        print("No local base or HF_TOKEN; the gated HF download may fail.", flush=True)
    code = clone_code()
    install(code)
    data = prepare_data(code, base)
    output = train(code, base, data)
    report_path = output / "training_report.json"
    if report_path.is_file():
        print("Training report:", report_path.read_text(encoding="utf-8"), flush=True)
    last_adapter = output / "last" / "adapter.safetensors"
    if not last_adapter.is_file():
        raise SystemExit(f"Training did not produce a last adapter: {last_adapter}")
    archive = shutil.make_archive(str(WORK / f"emotion_lora_{MODE}_bundle"), "zip", output)
    print("Bundle:", archive, flush=True)
    print("Last adapter:", last_adapter, flush=True)
    print("Download the bundle before the Kaggle session ends.", flush=True)


if __name__ == "__main__":
    main()
