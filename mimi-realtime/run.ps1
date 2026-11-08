# run.ps1 - thin wrapper that runs the realtime demo inside the venv.
#
#   .\run.ps1                              # mic -> speakers
#   .\run.ps1 --list-devices
#   .\run.ps1 --wav input.wav --out out.wav
#   .\run.ps1 --bench

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"
$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Missing .venv - run .\setup.ps1 first."
}

& $pythonExe (Join-Path $PSScriptRoot "realtime_mimi.py") @RemainingArgs
exit $LASTEXITCODE
