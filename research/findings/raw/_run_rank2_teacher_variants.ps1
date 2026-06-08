# Rank 2 cheap teacher-lever variants (seed 42) to push the learned read-out below
# the teacher baseline (~3.3): (V1) LONGER supervised teaching — unlike reward-STDP,
# supervised learning should keep tightening with more clean examples (the teacher's
# TAUGHT phase is already ~1.8 ~ reflex; only post-wean consolidation lags); (V2)
# STRONGER teacher clamp (2500) — a cleaner desired-output label.
$ErrorActionPreference = "Continue"
Set-Location "E:\Documents\Projects\sim"

$base = @(
  "--moving-goal","--goal-schedule","single","--deterministic",
  "--enable-msn-lateral-inhibition","--enable-d1-d2-asymmetry","--enable-striatal-pv-fsi",
  "--enable-cluster-a-closed-loop","--enable-cluster-e-topography",
  "--enable-dlpfc-wm","--enable-pfc-nmda",
  "--enable-visual-cortex","--visual-cortex-action-warmup-steps","600",
  "--genuine-thal-disinhibition","--genuine-gpi-tonic-pa","1300","--genuine-thal-tonic-pa","750",
  "--readout-source","spiking_wta","--urgency-max-pa","180",
  "--heuristic-strength","0","--sc-orienting-reflex",
  "--learned-perception","--learned-perception-from-vision",
  "--grid-size","8","--seed","42"
)
Write-Output "=== V1: teacher 1500 + LONGER teaching (wean@3000, 9000 steps) ==="
python -m research.runners.g11_bg_runner @base --sensory-cortex-teacher-pA 1500 --sc-reflex-wean-start 3000 --sc-reflex-wean-steps 1500 --n-steps 9000 --out research/findings/raw/_rank2_teacher_longteach_s42.json
Write-Output "=== V2: STRONGER teacher 2500 (wean@2000, 6000 steps) ==="
python -m research.runners.g11_bg_runner @base --sensory-cortex-teacher-pA 2500 --sc-reflex-wean-start 2000 --sc-reflex-wean-steps 1000 --n-steps 6000 --out research/findings/raw/_rank2_teacher_strong_s42.json
Write-Output "=== RANK 2 TEACHER VARIANTS DONE ==="
