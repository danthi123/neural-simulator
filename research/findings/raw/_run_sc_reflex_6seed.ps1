# SC-orienting reflex 6-seed extension: A (reflex) + C (floor) at seeds 100,101,102
# (42/43/44 already done -> canonical 6-seed set).
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

foreach ($seed in @(100, 101, 102)) {
  Write-Output "=== A seed $seed (SC reflex, heuristic OFF) ==="
  python -m research.runners.g11_bg_runner @common --seed $seed --heuristic-strength 0 --sc-orienting-reflex --out "research/findings/raw/_sc_reflex_A_s$seed.json"
  Write-Output "=== C seed $seed (floor, both OFF) ==="
  python -m research.runners.g11_bg_runner @common --seed $seed --heuristic-strength 0 --out "research/findings/raw/_sc_reflex_C_floor_s$seed.json"
}
Write-Output "=== SC REFLEX 6-SEED DONE ==="
