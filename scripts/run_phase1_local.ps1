param(
    [string]$Config = "configs/phase1/dataset_pipeline.yaml",
    [int]$MaxSamples = 500,
    [switch]$FullTokenization
)

$ErrorActionPreference = "Stop"

Write-Host "Phase 1 Dataset Pipeline"
Write-Host "Config: $Config"
Write-Host "Phase 1 does not train or fine-tune any model."

$tokenizationArgs = @("--config", $Config, "--skip_missing_tokenizers")
if (-not $FullTokenization) {
    $tokenizationArgs += @("--max_samples", $MaxSamples)
}

python -m py_compile `
    src/data/load_hf_datasets.py `
    src/data/prepare_phase1_dataset.py `
    src/data/validate_periad.py `
    src/data/dataset_statistics.py `
    src/data/tokenization_analysis.py `
    src/visualization/plot_phase1_dataset.py `
    scripts/check_phase1_outputs.py

python scripts/check_project_structure.py
python src/data/validate_periad.py --config $Config
python src/data/prepare_phase1_dataset.py --config $Config
python src/data/dataset_statistics.py --config $Config
python src/data/tokenization_analysis.py @tokenizationArgs
python src/visualization/plot_phase1_dataset.py --config $Config
python scripts/check_phase1_outputs.py --config $Config

Write-Host ""
Write-Host "Phase 1 outputs:"
Write-Host "- Processed PERIAD: dataset_dir('processed', 'periad')"
Write-Host "- VEAA metadata: dataset_dir('processed', 'veaa')"
Write-Host "- Tables: phase_artifact_dir('phase1', 'tables')"
Write-Host "- Reports: phase_artifact_dir('phase1', 'reports')"
Write-Host "- Plots: phase_artifact_dir('phase1', 'plots')"
