# Iter KK 6-seed multi-seed validation script.
# Tier 1 cortical canon (internal_density=0.10, exc=2.0, inh=4.0) applied
# to wernicke_pool + lang_output_pool, at biological scale
# (n_per_wernicke_pool=500, n_per_lang_out_pool=500, n_lang_input=2048).
#
# Multi-trial averaging at recognition (n=5 trials per stim) — addresses
# iter AA seed-44/101 borderline failures per the iter AA findings doc.
#
# Compare to iter AA 4/6 bidirectional at toy scale (100-neuron pools,
# weak cortical params, n=1 trial). Hypothesis: biology-correct canon
# + scale + multi-trial averaging gives 6/6 matching Tier 1 success.
#
# Wall clock estimate: ~15-20 min/seed × 6 = ~90-120 min total.

$env:PYTHONIOENCODING = "utf-8"
$ErrorActionPreference = "Continue"

$seeds = @(42, 43, 44, 100, 101, 102)
$outDir = "research/findings/raw/g11_bg/iter_KK"

foreach ($seed in $seeds) {
    Write-Host "[iter_KK] Starting seed $seed..." -ForegroundColor Cyan
    $t0 = Get-Date
    python -m research.runners.validate_ventral_semantic `
        --seed $seed --n-train-events 400 --n-replay-cycles 40 `
        --n-lang-input 2048 `
        --enable-multi-pool-wernicke --n-wernicke-pools 2 `
        --n-per-wernicke-pool 500 --n-per-wernicke-pool-fs 60 `
        --interleaved-training `
        --enable-per-concept-lang-out-pools --n-per-lang-out-pool 500 `
        --apply-wernicke-topographic `
        --n-recognition-trials 5 --inter-trial-rest-steps 100 `
        --out "$outDir/iter_KK_seed$seed.json" `
        > "$outDir/iter_KK_seed$seed.log" 2>&1
    $dt = (Get-Date) - $t0
    Write-Host "[iter_KK] seed $seed done in $($dt.TotalMinutes.ToString('F1')) min" -ForegroundColor Green
}

Write-Host "[iter_KK] All 6 seeds complete." -ForegroundColor Yellow
