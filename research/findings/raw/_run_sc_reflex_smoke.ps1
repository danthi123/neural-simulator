# SC-orienting reflex de-risk smoke: 3 controlled conditions, seed 42, grid-8,
# multi-goal (cheat-5 benchmark), inside the N8+N6 biologized back-end.
# A = SC reflex (heuristic OFF); B = heuristic-on cheat baseline; C = floor (both off).
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
  "--grid-size","8","--seed","42","--n-steps","1800"
)

Write-Output "=== A: SC reflex (heuristic OFF) ==="
python -m research.runners.g11_bg_runner @common --heuristic-strength 0 --sc-orienting-reflex --out research/findings/raw/_sc_reflex_A_s42.json
Write-Output "=== B: heuristic-on cheat baseline ==="
python -m research.runners.g11_bg_runner @common --out research/findings/raw/_sc_reflex_B_heuron_s42.json
Write-Output "=== C: floor (heuristic OFF, reflex OFF) ==="
python -m research.runners.g11_bg_runner @common --heuristic-strength 0 --out research/findings/raw/_sc_reflex_C_floor_s42.json
Write-Output "=== SC REFLEX SMOKE DONE ==="
