# SC-orienting reflex de-risk multi-seed: condition A (SC reflex, heuristic OFF)
# and condition C (floor, both OFF) at seeds 43, 44 (seed 42 done in the smoke).
# Decisive comparison: does A navigate (beat C's floor) robustly across seeds?
$ErrorActionPreference = "Continue"
Set-Location "E:\Documents\Projects\sim"

$common = @(
  "--moving-goal","--goal-schedule","multi","--deterministic",
  "--enable-msn-lateral-inhibition","--enable-d1-d2-asymmetry","--enable-striatal-pv-fsi",
  "--enable-cluster-a-closed-loop","--enable-cluster-e-topography",
  "--enable-dlpfc-wm","--enable-pfc-nmda",
  "--enable-visual-cortex","--visual-cortex-action-warmup-steps","600",
  "--genuine-thal-disinhibition","--genuine-gpi-tonic-pa","1300","--genuine-thal-tonic-pa","750",
  "--readout-source","spiking_wta","--urgency-max-pa","180",
  "--grid-size","8","--n-steps","1800"
)

foreach ($seed in @(43, 44)) {
  Write-Output "=== A seed $seed (SC reflex, heuristic OFF) ==="
  python -m research.runners.g11_bg_runner @common --seed $seed --heuristic-strength 0 --sc-orienting-reflex --out "research/findings/raw/_sc_reflex_A_s$seed.json"
  Write-Output "=== C seed $seed (floor, both OFF) ==="
  python -m research.runners.g11_bg_runner @common --seed $seed --heuristic-strength 0 --out "research/findings/raw/_sc_reflex_C_floor_s$seed.json"
}
Write-Output "=== SC REFLEX MULTISEED DONE ==="
