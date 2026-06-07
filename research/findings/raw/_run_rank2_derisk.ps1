# Rank 2 de-risk: does the LEARNED position-preserving circuit (sensory<-vision),
# taught by the SC reflex then WEANED, navigate self-sufficiently post-wean?
# R2  = reflex (weaned@2000) teaches the learned-from-vision sensory->cortex.
# CTRL= reflex (weaned@2000) teaches IT->cortex only (no learned-from-vision) =
#       the position-INVARIANT path that the N1 scaffold found fragile.
# Single-goal, 6000 steps (full reflex-off at 3000 -> last-quarter [4500-6000] is
# the durable post-wean metric). seed 42.
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
  "--grid-size","8","--seed","42","--n-steps","6000"
)
Write-Output "=== R2: reflex teaches LEARNED-from-vision, then weans ==="
python -m research.runners.g11_bg_runner @common --enable-learned-perception --learned-perception-from-vision --out "research/findings/raw/_rank2_R2_s42.json"
Write-Output "=== CTRL: reflex teaches IT-only (no learned-from-vision), then weans ==="
python -m research.runners.g11_bg_runner @common --out "research/findings/raw/_rank2_CTRL_itonly_s42.json"
Write-Output "=== RANK 2 DE-RISK DONE ==="
