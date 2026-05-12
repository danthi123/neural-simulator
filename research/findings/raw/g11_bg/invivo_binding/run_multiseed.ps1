# In-vivo new-vocab binding 6-seed multi-seed validation script.
# Tests biology-grounded variants (V0 vanilla, V_HIPPO_BIO, V_SCHEMA)
# on the main_hippo lineage. PASS criterion: >= 4/6 seeds with all 4
# novel bindings correct per variant.
#
# Each variant forks main_hippo and trains independently. Test bindings:
#   apple -> north, river -> east, mountain -> south, forest -> west
#
# Wall clock estimate: ~15-25 min/seed * 6 = ~90-150 min total.

$env:PYTHONIOENCODING = "utf-8"
$ErrorActionPreference = "Continue"

$seeds = @(42, 43, 44, 100, 101, 102)
$outDir = "research/findings/raw/g11_bg/invivo_binding"

foreach ($seed in $seeds) {
    if ($seed -eq 42) {
        Write-Host "[invivo] seed 42 already run, skipping" -ForegroundColor Gray
        continue
    }
    Write-Host "[invivo] Starting seed $seed..." -ForegroundColor Cyan
    $t0 = Get-Date
    python -u -m research.runners.investigate_invivo_binding_fix `
        --base-lineage main_hippo --seed $seed `
        --n-events 200 `
        --variants v0_vanilla,v_hippo_bio,v_schema `
        --out "$outDir/invivo_seed$seed.json" `
        > "$outDir/invivo_seed$seed.log" 2>&1
    $dt = (Get-Date) - $t0
    Write-Host "[invivo] seed $seed done in $($dt.TotalMinutes.ToString('F1')) min" -ForegroundColor Green
}

Write-Host "[invivo] All seeds 43,44,100,101,102 complete." -ForegroundColor Yellow
