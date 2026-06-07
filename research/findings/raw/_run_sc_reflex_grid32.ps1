# SC-orienting reflex grid-32 production confirm: A (reflex) + C (floor), seed 42.
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
  "--grid-size","32","--seed","42","--n-steps","1800"
)
Write-Output "=== A grid-32 seed 42 (SC reflex, heuristic OFF) ==="
python -m research.runners.g11_bg_runner @common --heuristic-strength 0 --sc-orienting-reflex --out "research/findings/raw/_sc_reflex_A_grid32_s42.json"
Write-Output "=== C grid-32 seed 42 (floor, both OFF) ==="
python -m research.runners.g11_bg_runner @common --heuristic-strength 0 --out "research/findings/raw/_sc_reflex_C_floor_grid32_s42.json"
Write-Output "=== SC REFLEX GRID-32 DONE ==="
