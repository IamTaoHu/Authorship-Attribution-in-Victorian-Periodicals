$ErrorActionPreference = "Stop"

function Invoke-Phase5Step {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Command
    )

    & $Command[0] @($Command[1..($Command.Length - 1)])
    if ($LASTEXITCODE -ne 0) {
        throw "Phase 5 diagnostic step failed with exit code ${LASTEXITCODE}: $($Command -join ' ')"
    }
}

Invoke-Phase5Step @("python", "src/training/train_decoder_qlora.py", "--config", "configs/phase5/diagnostic_tinyllama_qlora.yaml")
Invoke-Phase5Step @("python", "src/evaluation/evaluate_decoder_qlora.py", "--config", "configs/phase5/diagnostic_tinyllama_qlora.yaml")
Invoke-Phase5Step @("python", "src/visualization/plot_phase5_results.py", "--phase5_dir", "outputs/phase5")
Invoke-Phase5Step @("python", "scripts/check_phase5_outputs.py", "--phase5_dir", "outputs/phase5", "--checkpoint_dir", "checkpoints/phase5")
