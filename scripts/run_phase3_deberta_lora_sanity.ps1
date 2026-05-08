$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

python src/training/train_encoder.py --config configs/phase3/lora/deberta_lora_r4.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
