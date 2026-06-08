# Fully-biologized reward+dopamine de-risk: N5 perceived-approach reward + the SPIKING-SNc
# neural dopamine RPE (Stage A) vs the cheat baseline (coord Manhattan reward + raw-scalar DA),
# in the biologized flagship multi-goal config (SC reflex + N8 + N6). Acceptance: the neural
# reward+dopamine does NOT regress the nav score. This is the first FULL-NAV test of --spiking-snc
# (CPU smoke + Pavlovian already passed; this is the nav-score regression gate). Seeds 42,43,44.
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
  Write-Output "=== NEURAL reward+DA seed $seed (N5 perceived + spiking-SNc) ==="
  python -m research.runners.g11_bg_runner @common --seed $seed --perceived-approach-reward --spiking-snc --out "research/findings/raw/_biorda_neural_s$seed.json"
  Write-Output "=== CHEAT reward+DA seed $seed (coord Manhattan + raw scalar DA) ==="
  python -m research.runners.g11_bg_runner @common --seed $seed --out "research/findings/raw/_biorda_cheat_s$seed.json"
}
Write-Output "=== BIO-REWARD-DA DE-RISK DONE ==="
