# Plain Rank 2 (learned-from-vision, NO teacher) 6-seed extension: seeds 100,101,102
# (42/43/44 already done: 3.93/2.25/2.60). The REAL Rank 2 result (the teacher was a
# seed-42 artifact + counterproductive). Single-goal, reflex weaned @2000, 6000 steps.
$ErrorActionPreference = "Continue"
Set-Location "E:\Documents\Projects\sim"

$common = @(
  "--moving-goal","--goal-schedule","single","--deterministic",
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
foreach ($seed in @(100, 101, 102)) {
  Write-Output "=== plain R2 seed $seed (no teacher) ==="
  python -m research.runners.g11_bg_runner @common --seed $seed --out "research/findings/raw/_rank2_R2_s$seed.json"
}
Write-Output "=== RANK 2 PLAIN 6-SEED DONE ==="
