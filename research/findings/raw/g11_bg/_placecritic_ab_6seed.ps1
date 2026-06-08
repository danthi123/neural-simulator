# 6-seed A/B gate: place-code NEURAL value critic vs STAGE-A host scaffold.
# SERIAL, --deterministic, NO --emit-activity. Acceptance: NEURAL summed
# final-quarter distance <= STAGE-A (no regression; lower = better).
#
# Param $CriticWindow: if "1", the NEURAL condition adds --critic-window
# (Stage 2 reward-window-gated GABA_B); else Stage 1 (gate held open).
# Set from the Stage-1 smoke verdict.
param(
    [string]$CriticWindow = "0",
    [int]$LeadSteps = 120
)
$ErrorActionPreference = "Stop"
$env:SIM_BACKEND = "cupy"
$seeds = @(42, 43, 44, 100, 101, 102)
$out = "research/findings/raw/g11_bg"

# Shared flagship A+E+G v2.5 stack (32x32).
$flagship = @(
    "--moving-goal", "--goal-schedule", "multi", "--deterministic",
    "--enable-msn-lateral-inhibition", "--enable-d1-d2-asymmetry", "--enable-striatal-pv-fsi",
    "--enable-cluster-a-closed-loop", "--enable-cluster-e-topography",
    "--enable-dlpfc-wm", "--enable-pfc-nmda",
    "--enable-visual-cortex", "--visual-cortex-action-warmup-steps", "600",
    "--enable-place-goal-readout",
    "--grid-size", "32", "--n-steps", "1800"
)

$windowArgs = @()
if ($CriticWindow -eq "1") { $windowArgs = @("--critic-window", "--critic-lead-steps", "$LeadSteps") }

foreach ($seed in $seeds) {
    # --- STAGE-A (host value scaffold) ---
    $sa = "$out/_placecritic_stagea_s$seed.json"
    if (Test-Path $sa) {
        Write-Host "[skip] STAGE-A seed $seed exists"
    } else {
        Write-Host "[run ] STAGE-A seed $seed -> $sa"
        python -m research.runners.g11_bg_runner @flagship --spiking-snc `
            --seed $seed --out $sa 2>&1 | Tee-Object -FilePath "$out/_placecritic_stagea_s$seed.log" | Select-Object -Last 2
    }

    # --- NEURAL (place-code critic) ---
    $nu = "$out/_placecritic_neural_s$seed.json"
    if (Test-Path $nu) {
        Write-Host "[skip] NEURAL seed $seed exists"
    } else {
        Write-Host "[run ] NEURAL seed $seed -> $nu  (window=$CriticWindow)"
        python -m research.runners.g11_bg_runner @flagship --spiking-snc --enable-neural-critic @windowArgs `
            --seed $seed --out $nu 2>&1 | Tee-Object -FilePath "$out/_placecritic_neural_s$seed.log" | Select-Object -Last 2
    }
}
Write-Host "=== 6-seed A/B complete ==="
