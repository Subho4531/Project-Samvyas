# setup.ps1 - create the .venv for the Mimi realtime demo and install deps.
#
#   .\setup.ps1              # CPU build of torch (default, fastest to install)
#   .\setup.ps1 -Cuda        # CUDA 12.9 build of torch
#   .\setup.ps1 -Force       # wipe and recreate .venv
#
# Uses `uv` when available (much faster), otherwise the py launcher + pip.

[CmdletBinding()]
param(
    [string]$Python = "3.11",
    [switch]$Cuda,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$venvDir   = Join-Path $PSScriptRoot ".venv"
$pythonExe = Join-Path $venvDir "Scripts\python.exe"
$hasUv     = [bool](Get-Command uv -ErrorAction SilentlyContinue)

# --- 1. virtualenv -----------------------------------------------------------
if ($Force -and (Test-Path $venvDir)) {
    Write-Host "Removing existing virtualenv..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venvDir
}

if (-not (Test-Path $pythonExe)) {
    if ($hasUv) {
        Write-Host "Creating .venv with uv (Python $Python)..." -ForegroundColor Cyan
        uv venv --python $Python $venvDir
    }
    else {
        Write-Host "Creating .venv with the py launcher (Python $Python)..." -ForegroundColor Cyan
        py -$Python -m venv $venvDir
    }
}
else {
    Write-Host "Reusing existing virtualenv." -ForegroundColor Cyan
}

if (-not (Test-Path $pythonExe)) { throw "Python executable not found at $pythonExe" }

# --- 2. installer ------------------------------------------------------------
function Install-Pkgs {
    param([string[]]$Packages, [string[]]$Extra = @())
    if ($hasUv) {
        uv pip install --python $pythonExe @Extra @Packages
    }
    else {
        & $pythonExe -m pip install --upgrade @Extra @Packages
    }
    if ($LASTEXITCODE -ne 0) { throw "Failed to install: $($Packages -join ', ')" }
}

if (-not $hasUv) {
    & $pythonExe -m pip --version *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Bootstrapping pip..." -ForegroundColor Cyan
        & $pythonExe -m ensurepip --upgrade
    }
    & $pythonExe -m pip install --upgrade pip --quiet
}

# --- 3. packages -------------------------------------------------------------
if ($Cuda) {
    Write-Host "Installing torch (CUDA 12.9)..." -ForegroundColor Cyan
    Install-Pkgs -Packages @("torch") -Extra @("--index-url", "https://download.pytorch.org/whl/cu129")
}
else {
    Write-Host "Installing torch (CPU)..." -ForegroundColor Cyan
    Install-Pkgs -Packages @("torch")
}

Write-Host "Installing transformers / audio deps..." -ForegroundColor Cyan
Install-Pkgs -Packages @(
    "transformers>=4.57.0",
    "numpy",
    "scipy",
    "sounddevice",
    "safetensors",
    "huggingface-hub"
)

# --- 4. verify ---------------------------------------------------------------
Write-Host "`nVerifying install..." -ForegroundColor Cyan
$check = @'
import sys, torch, transformers, sounddevice, scipy, numpy
print('python      ', sys.version.split()[0])
print('torch       ', torch.__version__, '| cuda:', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('numpy       ', numpy.__version__)
print('scipy       ', scipy.__version__)
print('sounddevice ', sounddevice.__version__)
from transformers.models.mimi.modeling_mimi import MimiConv1dPaddingCache
print('streaming   OK (MimiConv1dPaddingCache available)')
'@
& $pythonExe -c $check
if ($LASTEXITCODE -ne 0) { throw "Verification failed." }

Write-Host "`nDone. Try:  .\run.ps1 --bench" -ForegroundColor Green
