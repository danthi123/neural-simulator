# 6-seed A/B (WARM-UP variant) — fires only if the N9 warm-up smoke confirms the
# critic WAKES in deployed nav. NEURAL value critic (dense vs_place_context afferent +
# per-region homeostasis + --critic-warmup-trials 20, the 2026-06-09 deadlock fix) vs
# STAGE-A (host _V_scaffold). SERIAL, --deterministic, GPU/CuPy.
# Acceptance: NEURAL summed final-quarter distance <= STAGE-A (no nav regression).
# An honest nav regression IS a valid deliverable (it maps a cost) -- do NOT hide/brute-force it.
#
# Run (ONLY after _n9_warmup_smoke_s42 shows CRITIC_WAKES_NAV_SANE):
#   pwsh research/findings/raw/g11_bg/_navcritic_ab_6seed_warmup.ps1
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
  # STAGE-A (host scaffold): --spiking-snc only (no warm-up needed; host value).
  $outA = "$outDir/_navcritic_wu_stagea_s$seed.json"
  Write-Host "=== STAGE-A seed $seed -> $outA ==="
  python -m research.runners.g11_bg_runner @flagBase --seed $seed --out $outA `
    > "$outDir/_navcritic_wu_stagea_s$seed.log" 2>&1
  Write-Host "    stagea seed $seed exit=$LASTEXITCODE"

  # NEURAL + WARM-UP: the validated deadlock fix (--critic-warmup-trials 20).
  $outN = "$outDir/_navcritic_wu_neural_s$seed.json"
  Write-Host "=== NEURAL+WARMUP seed $seed -> $outN ==="
  python -m research.runners.g11_bg_runner @flagBase --enable-neural-critic `
    --enable-place-goal-readout --enable-critic-homeostasis --critic-warmup-trials 20 `
    --seed $seed --out $outN `
    > "$outDir/_navcritic_wu_neural_s$seed.log" 2>&1
  Write-Host "    neural seed $seed exit=$LASTEXITCODE"
}
Write-Host "=== 6-seed A/B (warm-up variant) COMPLETE ==="
