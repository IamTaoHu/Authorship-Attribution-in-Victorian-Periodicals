$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

python src/training/train_encoder.py --config configs/phase3/modernbert_base_256.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python src/training/train_encoder.py --config configs/phase3/modernbert_base_512.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python src/training/train_encoder.py --config configs/phase3/modernbert_base_1024.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
