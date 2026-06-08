# N5+N9 de-risk: the fully coordinate-free reward + dopamine (perceived-approach reward N5 +
# actor-critic RPE dopamine N9) vs the cheat baseline (coordinate Manhattan reward + raw-reward
# DA), in the biologized flagship multi-goal config (SC reflex + N8 + N6). Acceptance: the
# biologized reward+DA does NOT regress the nav score (8/8 CPU label-agreement predicts parity;
# the win is coordinate-freeness + actor-critic provenance). Seeds 42,43,44 (extend to 6 if GO).
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
  "--sc-orienting-reflex",
  "--grid-size","8","--n-steps","1800"
)
foreach ($seed in @(42, 43, 44)) {
  Write-Output "=== BIOLOGIZED reward+DA seed $seed (N5 perceived + N9 actor-critic RPE) ==="
  python -m research.runners.g11_bg_runner @common --seed $seed --perceived-approach-reward --rpe-dopamine --out "research/findings/raw/_n5n9_bio_s$seed.json"
  Write-Output "=== CHEAT reward+DA seed $seed (coord Manhattan + raw DA) ==="
  python -m research.runners.g11_bg_runner @common --seed $seed --out "research/findings/raw/_n5n9_cheat_s$seed.json"
}
Write-Output "=== N5N9 DE-RISK DONE ==="
