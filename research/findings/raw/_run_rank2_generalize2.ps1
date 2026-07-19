# Rank 2 multi-goal GENERALIZATION test: train plain R2 (learned-from-vision) on ONE
# goal through the reflex wean (0-3000 on `far`), then THREE NEW goals AFTER the wean
# (3000/4000/5000). Does the learned goal-agnostic (dx,dy)->action map navigate to goals
# it was NEVER taught on, with the reflex OFF? Position-preserving code predicts YES.
# Seeds 42,43,44. Per-phase final_quarter over phases 1-3 (post-wean, NEW goals) = the metric.
$ErrorActionPreference = "Continue"
Set-Location "E:\Documents\Projects\sim"

$common = @(
  "--moving-goal","--goal-schedule","generalize2","--deterministic",
  "--enable-msn-lateral-inhibition","--enable-d1-d2-asymmetry","--enable-striatal-pv-fsi",
  "--enable-cluster-a-closed-loop","--enable-cluster-e-topography",
  "--enable-dlpfc-wm","--enable-pfc-nmda",
  "--enable-visual-cortex","--visual-cortex-action-warmup-steps","600",
  "--genuine-thal-disinhibition","--genuine-gpi-tonic-pa","1300","--genuine-thal-tonic-pa","750",
  "--readout-source","spiking_wta","--urgency-max-pa","180",
  "--heuristic-strength","0","--sc-orienting-reflex",
  "--sc-reflex-wean-start","2000","--sc-reflex-wean-steps","1000",
  "--learned-perception","--learned-perception-from-vision",
  "--grid-size","8","--n-steps","6000"
)
foreach ($seed in @(42, 43, 44)) {
  Write-Output "=== generalize seed $seed (plain R2, NEW goals post-wean) ==="
  python -m research.runners.g11_bg_runner @common --seed $seed --out "research/findings/raw/_rank2_generalize2_s$seed.json"
}
Write-Output "=== RANK 2 GENERALIZE DONE ==="
