# Rank 2 supervised-teacher multi-seed: paired comparison teacher vs plain R2 at
# seeds 43, 44 (seed 42: teacher 3.30 vs plain 3.93). Single-goal, reflex weaned @2000.
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
foreach ($seed in @(43, 44)) {
  Write-Output "=== TEACHER seed $seed (sensory-cortex-teacher-pA 1500) ==="
  python -m research.runners.g11_bg_runner @common --seed $seed --sensory-cortex-teacher-pA 1500 --out "research/findings/raw/_rank2_teacher_s$seed.json"
  Write-Output "=== PLAIN R2 seed $seed (no teacher) ==="
  python -m research.runners.g11_bg_runner @common --seed $seed --out "research/findings/raw/_rank2_R2_s$seed.json"
}
Write-Output "=== RANK 2 TEACHER MULTISEED DONE ==="
