$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

python src/training/train_encoder.py --config configs/phase3/roberta_large.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
