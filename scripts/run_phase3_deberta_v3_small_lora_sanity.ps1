$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

python src/training/train_encoder.py --config configs/phase3/lora/deberta_v3_small_lora_r4_sanity.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
