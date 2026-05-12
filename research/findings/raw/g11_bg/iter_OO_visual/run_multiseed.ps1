# Iter OO_visual 6-seed multi-seed validation script.
# Sensory grounding via Cluster K v2 visual cortex + multimodal_hub.
# Per-concept geometric images drive retina alongside lang_input during
# training (multimodal co-firing).
#
# Architecture: 13K neurons, 2.75M synapses, ~6-8GB GPU at peak.
# Wall clock estimate: ~15-20 min/seed * 6 = ~2 hr total.

$env:PYTHONIOENCODING = "utf-8"
$ErrorActionPreference = "Continue"

$seeds = @(42, 43, 44, 100, 101, 102)
$outDir = "research/findings/raw/g11_bg/iter_OO_visual"

foreach ($seed in $seeds) {
    if ($seed -eq 42) {
        Write-Host "[iter_OO_visual] seed 42 already run, skipping" -ForegroundColor Gray
        continue
    }
    Write-Host "[iter_OO_visual] Starting seed $seed..." -ForegroundColor Cyan
    $t0 = Get-Date
    python -u -m research.runners.validate_ventral_semantic `
        --seed $seed --n-train-events 400 --n-replay-cycles 40 `
        --n-lang-input 2048 `
        --enable-multi-pool-wernicke --n-wernicke-pools 2 `
        --n-per-wernicke-pool 500 --n-per-wernicke-pool-fs 60 `
        --interleaved-training `
        --enable-per-concept-lang-out-pools --n-per-lang-out-pool 500 `
        --apply-wernicke-topographic `
        --enable-visual-cortex --enable-multimodal-hub `
        --pair-visual-during-training `
        --n-recognition-trials 5 --inter-trial-rest-steps 100 `
        --out "$outDir/iter_OO_visual_seed$seed.json" `
        > "$outDir/iter_OO_visual_seed$seed.log" 2>&1
    $dt = (Get-Date) - $t0
    Write-Host "[iter_OO_visual] seed $seed done in $($dt.TotalMinutes.ToString('F1')) min" -ForegroundColor Green
}

Write-Host "[iter_OO_visual] All seeds 43,44,100,101,102 complete." -ForegroundColor Yellow

# Aggregate
python -m research.runners.aggregate_p5_pool_readout `
    --raw-root $outDir `
    --prefix iter_OO_visual_seed --seeds 42,43,44,100,101,102 `
    --out research/findings/2026-05-12-P5-iterOOvisual-multiseed.md `
    --label "iter OO_visual (sensory grounded multimodal training)"
