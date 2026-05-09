$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

python src/inference/run_prompting.py --select-few-shot --config configs/phase4/smoke_tinyllama_few_shot.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python src/inference/run_prompting.py --config configs/phase4/smoke_tinyllama_zero_shot.yaml --dry-run
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python src/inference/run_prompting.py --config configs/phase4/smoke_tinyllama_zero_shot.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python src/inference/run_prompting.py --config configs/phase4/smoke_tinyllama_few_shot.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
