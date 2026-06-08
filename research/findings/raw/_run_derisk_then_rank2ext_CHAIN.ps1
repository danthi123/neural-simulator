# CHAINED autonomous GPU batch (2026-06-08), run from the ISOLATED worktree
# E:\Documents\Projects\sim-derisk (clean committed runner @ ae150246) so the frontend
# subagent's uncommitted WIP edits to g11_bg_runner.py / sim/progress.py CANNOT contaminate
# these results. Outputs are written by ABSOLUTE path back into the MAIN tree's raw dir.
#
# Part 1 (priority): neural reward+DA nav de-risk — N5 coordinate-free perceived-approach
#   reward + the SPIKING-SNc actor-critic dopamine (Stage A protected edit) vs the cheat
#   baseline (coord Manhattan reward + raw-scalar DA). Flagship multi-goal, seeds 42/43/44.
#   Acceptance: neural reward+DA does NOT regress the nav score. First FULL-NAV test of --spiking-snc.
# Part 2: Rank 2 generalize2 6-seed extension — seeds 100/101/102 (42/43/44 already 3/3 GO).
$ErrorActionPreference = "Continue"
Set-Location "E:\Documents\Projects\sim-derisk"
$OUT = "E:\Documents\Projects\sim\research\findings\raw"

# ---- Part 1: de-risk (flagship multi-goal, 1800 steps) ----
$derisk = @(
  "--moving-goal","--goal-schedule","multi","--deterministic",
  "--enable-msn-lateral-inhibition","--enable-d1-d2-asymmetry","--enable-striatal-pv-fsi",
  "--enable-cluster-a-closed-loop","--enable-cluster-e-topography",
  "--enable-dlpfc-wm","--enable-pfc-nmda",
  "--enable-visual-cortex","--visual-cortex-action-warmup-steps","600",
  "--genuine-thal-disinhibition","--genuine-gpi-tonic-pa","1300","--genuine-thal-tonic-pa","750",
  "--readout-source","spiking_wta","--urgency-max-pa","180",
  "--sc-orienting-reflex",
  "--grid-size","8","--n-steps","1800"
)
foreach ($seed in @(42, 43, 44)) {
  Write-Output "=== [P1] NEURAL reward+DA seed $seed (N5 perceived + spiking-SNc) ==="
  python -m research.runners.g11_bg_runner @derisk --seed $seed --perceived-approach-reward --spiking-snc --out "$OUT\_biorda_neural_s$seed.json"
  Write-Output "=== [P1] CHEAT reward+DA seed $seed (coord Manhattan + raw scalar DA) ==="
  python -m research.runners.g11_bg_runner @derisk --seed $seed --out "$OUT\_biorda_cheat_s$seed.json"
}
Write-Output "=== PART 1 (BIO-REWARD-DA DE-RISK) DONE ==="

# ---- Part 2: Rank 2 generalize2 6-seed extension (6000 steps) ----
$rank2 = @(
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
foreach ($seed in @(100, 101, 102)) {
  Write-Output "=== [P2] Rank2 generalize2 seed $seed (NEW goals post-wean, reflex OFF) ==="
  python -m research.runners.g11_bg_runner @rank2 --seed $seed --out "$OUT\_rank2_generalize2_s$seed.json"
}
Write-Output "=== PART 2 (RANK2 GENERALIZE2 6-SEED EXT) DONE ==="
Write-Output "=== CHAIN COMPLETE ==="
