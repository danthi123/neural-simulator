# 6-seed A/B: NEURAL value critic (dense vs_place_context afferent + per-region homeostasis,
# 2026-06-09 VALIDATED) vs STAGE-A (host _V_scaffold scaffold). SERIAL, --deterministic, GPU/CuPy.
# Acceptance: NEURAL summed final-quarter distance <= STAGE-A (no nav regression).
# An honest nav regression IS a valid deliverable (it maps a cost) -- do NOT hide/brute-force it.
#
# Run: pwsh research/findings/raw/g11_bg/_navcritic_ab_6seed.ps1
$ErrorActionPreference = "Stop"
$env:SIM_BACKEND = "cupy"
$seeds = @(42, 43, 44, 100, 101, 102)
$flagBase = @(
  "--moving-goal", "--goal-schedule", "multi", "--deterministic",
  "--enable-msn-lateral-inhibition", "--enable-d1-d2-asymmetry", "--enable-striatal-pv-fsi",
  "--enable-cluster-a-closed-loop", "--enable-cluster-e-topography",
  "--enable-dlpfc-wm", "--enable-pfc-nmda",
  "--enable-visual-cortex", "--visual-cortex-action-warmup-steps", "600",
  "--grid-size", "32", "--spiking-snc", "--n-steps", "1800"
)
$outDir = "research/findings/raw/g11_bg"

foreach ($seed in $seeds) {
  # STAGE-A (host scaffold): --spiking-snc only.
  $outA = "$outDir/_navcritic_stagea_s$seed.json"
  Write-Host "=== STAGE-A seed $seed -> $outA ==="
  python -m research.runners.g11_bg_runner @flagBase --seed $seed --out $outA `
    > "$outDir/_navcritic_stagea_s$seed.log" 2>&1
  Write-Host "    stagea seed $seed exit=$LASTEXITCODE"

  # NEURAL (validated mechanism): + --enable-neural-critic --enable-place-goal-readout --enable-critic-homeostasis.
  $outN = "$outDir/_navcritic_neural_s$seed.json"
  Write-Host "=== NEURAL seed $seed -> $outN ==="
  python -m research.runners.g11_bg_runner @flagBase --enable-neural-critic `
    --enable-place-goal-readout --enable-critic-homeostasis --seed $seed --out $outN `
    > "$outDir/_navcritic_neural_s$seed.log" 2>&1
  Write-Host "    neural seed $seed exit=$LASTEXITCODE"
}
Write-Host "=== 6-seed A/B COMPLETE ==="
