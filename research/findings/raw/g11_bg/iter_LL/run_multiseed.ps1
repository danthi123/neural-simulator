# Iter LL 6-seed multi-seed validation script.
# Iter AA recipe (weak pool dynamics 0.05/0.3/0.8) at biological scale
# (n_per_wernicke_pool=500, n_per_lang_out_pool=500, n_lang_input=2048).
# Multi-trial averaging at recognition (n=5).
#
# Compare to iter AA 4/6 bidirectional at toy scale (100-neuron pools).
# Tests the hypothesis: biological scale alone (no canon dynamics change)
# pushes 4/6 to 6/6 by averaging out per-seed structural asymmetry.
#
# Wall clock estimate: ~10-12 min/seed * 6 = ~60-75 min total.

$env:PYTHONIOENCODING = "utf-8"
$ErrorActionPreference = "Continue"

$seeds = @(42, 43, 44, 100, 101, 102)
$outDir = "research/findings/raw/g11_bg/iter_LL"

foreach ($seed in $seeds) {
    if ($seed -eq 42) {
        Write-Host "[iter_LL] seed 42 already run, skipping" -ForegroundColor Gray
        continue
    }
    Write-Host "[iter_LL] Starting seed $seed..." -ForegroundColor Cyan
    $t0 = Get-Date
    python -u -m research.runners.validate_ventral_semantic `
        --seed $seed --n-train-events 400 --n-replay-cycles 40 `
        --n-lang-input 2048 `
        --enable-multi-pool-wernicke --n-wernicke-pools 2 `
        --n-per-wernicke-pool 500 --n-per-wernicke-pool-fs 60 `
        --interleaved-training `
        --enable-per-concept-lang-out-pools --n-per-lang-out-pool 500 `
        --apply-wernicke-topographic `
        --n-recognition-trials 5 --inter-trial-rest-steps 100 `
        --out "$outDir/iter_LL_seed$seed.json" `
        > "$outDir/iter_LL_seed$seed.log" 2>&1
    $dt = (Get-Date) - $t0
    Write-Host "[iter_LL] seed $seed done in $($dt.TotalMinutes.ToString('F1')) min" -ForegroundColor Green
}

Write-Host "[iter_LL] All seeds 43,44,100,101,102 complete." -ForegroundColor Yellow

# Aggregate
python -m research.runners.aggregate_p5_pool_readout `
    --raw-root $outDir `
    --prefix iter_LL_seed --seeds 42,43,44,100,101,102 `
    --out research/findings/2026-05-12-P5-iterLL-multiseed.md `
    --label "iter LL (biological scale, weak dynamics, multi-trial avg)"
