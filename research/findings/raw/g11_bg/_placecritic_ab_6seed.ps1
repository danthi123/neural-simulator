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

# Critic drive calibration (2026-06-08, runner-side; diag*.py). The MSN-D1 critic
# was silent in nav. Decisive cause (OU is OFF in nav -> deterministic): the
# sensor_place_readout place code fires <1 Hz at the default 600 pA, far too sparse
# to drive ANY striatal critic. Calibration that fires it in-context: RS critic
# (lower rheobase) + raised afferent weight (25) + wider sigma (2.5) + STRONGER
# place drive (1500 pA, ~2x default). NB the stronger drive doubles the actor's
# place+goal drive too, so it is applied to BOTH A/B conditions (sigmaArg/driveArg
# below) and nav is validated. (A teacher current was REJECTED: post>>pre LTD
# collapses the weight, diag6.)
$criticCal = @(
    "--critic-neuron-type", "IZH2007_RS_CORTICAL_PYRAMIDAL",
    "--critic-afferent-weight", "25"
)

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

# Fair A/B: BOTH conditions use the widened place sigma AND the stronger place
# drive (they change the actor's place/goal code for both, so the only A/B-isolated
# variable is the neural critic + its critic-type/weight). The critic-type/weight
# flags are inert without --enable-neural-critic.
$sigmaArg = @("--hippocampus-drive-sigma", "2.5", "--hippocampus-drive-max-pa", "1500")

foreach ($seed in $seeds) {
    # --- STAGE-A (host value scaffold) ---
    $sa = "$out/_placecritic_stagea_s$seed.json"
    if (Test-Path $sa) {
        Write-Host "[skip] STAGE-A seed $seed exists"
    } else {
        Write-Host "[run ] STAGE-A seed $seed -> $sa"
        python -m research.runners.g11_bg_runner @flagship --spiking-snc @sigmaArg `
            --seed $seed --out $sa 2>&1 | Tee-Object -FilePath "$out/_placecritic_stagea_s$seed.log" | Select-Object -Last 2
    }

    # --- NEURAL (place-code critic, calibrated) ---
    $nu = "$out/_placecritic_neural_s$seed.json"
    if (Test-Path $nu) {
        Write-Host "[skip] NEURAL seed $seed exists"
    } else {
        Write-Host "[run ] NEURAL seed $seed -> $nu  (window=$CriticWindow)"
        python -m research.runners.g11_bg_runner @flagship --spiking-snc --enable-neural-critic @criticCal @sigmaArg @windowArgs `
            --seed $seed --out $nu 2>&1 | Tee-Object -FilePath "$out/_placecritic_neural_s$seed.log" | Select-Object -Last 2
    }
}
Write-Host "=== 6-seed A/B complete ==="
