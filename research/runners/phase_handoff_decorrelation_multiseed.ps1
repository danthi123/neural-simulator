# OPTION (a) PHASE-handoff — DECISIVE multi-seed run.
# 3 model seeds (the whitening) x 3 bench/projection seeds (the composer) so RATE + PHASE are both robust to the
# composer's random projection (the seed-42 pilot showed RATE is projection-seed-sensitive: 84.6% @ proj 42 vs the
# boundary's 72% @ its proj; the membrane degradation is identical, coh ~0.088). ONE process runs all 3 model seeds
# internally (the runner loops --seeds); heavy + SEQUENTIAL by construction (no parallel agent-benchmark OOM).
$env:REALOBJ_CIFAR = "data/cifar10/cifar-10-batches-py/data_batch_1"
$env:PYTHONUNBUFFERED = "1"
python -m research.runners.phase_handoff_decorrelation_compose `
    --seeds 42 43 44 --bench-seeds 42 43 44 --K 300 --lam 0.01 --period 400 `
    --out "research/findings/raw/_phase_handoff_3seed.json" 2>&1 |
    Tee-Object -FilePath "research/findings/raw/_phase_handoff_3seed.log"
Write-Host "decisive 3-seed done" -ForegroundColor Green
