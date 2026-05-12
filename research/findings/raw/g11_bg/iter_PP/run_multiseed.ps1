# Iter PP 6-seed multi-seed validation script.
# Cluster K v2 sensory grounding + multimodal_hub + lang_output FS pools.
# Tests whether output-layer cross-inhibition addresses the structural
# pool bias that iter OO_visual partially exposed (apple flipped to
# correct +23, river flipped to wrong -24).
#
# Architecture: 13.1K neurons, ~2.8M synapses growing to ~6.3M.
# Wall clock estimate: ~10 min/seed * 6 = ~60 min total.

$env:PYTHONIOENCODING = "utf-8"
$ErrorActionPreference = "Continue"

$seeds = @(42, 43, 44, 100, 101, 102)
$outDir = "research/findings/raw/g11_bg/iter_PP"

foreach ($seed in $seeds) {
    if ($seed -eq 42) {
        Write-Host "[iter_PP] seed 42 already run, skipping" -ForegroundColor Gray
        continue
    }
    Write-Host "[iter_PP] Starting seed $seed..." -ForegroundColor Cyan
    $t0 = Get-Date
    python -u -m research.runners.validate_ventral_semantic `
        --seed $seed --n-train-events 400 --n-replay-cycles 40 `
        --n-lang-input 2048 `
        --enable-multi-pool-wernicke --n-wernicke-pools 2 `
        --n-per-wernicke-pool 500 --n-per-wernicke-pool-fs 60 `
        --interleaved-training `
        --enable-per-concept-lang-out-pools --n-per-lang-out-pool 500 `
        --enable-lang-out-fs-pools --n-per-lang-out-fs-pool 60 `
        --apply-wernicke-topographic `
        --enable-visual-cortex --enable-multimodal-hub `
        --pair-visual-during-training `
        --n-recognition-trials 5 --inter-trial-rest-steps 100 `
        --out "$outDir/iter_PP_seed$seed.json" `
        > "$outDir/iter_PP_seed$seed.log" 2>&1
    $dt = (Get-Date) - $t0
    Write-Host "[iter_PP] seed $seed done in $($dt.TotalMinutes.ToString('F1')) min" -ForegroundColor Green
}

Write-Host "[iter_PP] All seeds 43,44,100,101,102 complete." -ForegroundColor Yellow

# Aggregate
python -m research.runners.aggregate_p5_pool_readout `
    --raw-root $outDir `
    --prefix iter_PP_seed --seeds 42,43,44,100,101,102 `
    --out research/findings/2026-05-12-P5-iterPP-multiseed.md `
    --label "iter PP (sensory grounded + lang_output FS @ bio scale)"
