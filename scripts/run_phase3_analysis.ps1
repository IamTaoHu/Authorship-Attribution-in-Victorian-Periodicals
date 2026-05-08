$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

python src/evaluation/evaluate_encoder_outputs.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python src/visualization/plot_phase3_results.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$EmbeddingFiles = Get-ChildItem -Path "outputs/phase3" -Filter "embeddings.npy" -Recurse -File -ErrorAction SilentlyContinue
foreach ($EmbeddingFile in $EmbeddingFiles) {
    $ExperimentName = Split-Path -Leaf $EmbeddingFile.DirectoryName
    python src/visualization/plot_embeddings.py --experiment $ExperimentName
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

python src/analysis/error_analysis_phase3.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
