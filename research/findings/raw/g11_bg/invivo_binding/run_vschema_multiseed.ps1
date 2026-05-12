# V_SCHEMA-only 5-seed multi-seed validation.
# V_SCHEMA showed the ONLY genuine novel-key bind success on seed 42
# (mountain->south via schema-supported anchor reinforcement).
# Tests whether the success generalizes — does each seed get >=1 true
# bind, and which direction?
#
# Wall clock estimate: ~8 min/seed * 5 = ~40 min total.

$env:PYTHONIOENCODING = "utf-8"
$ErrorActionPreference = "Continue"

$seeds = @(43, 44, 100, 101, 102)
$outDir = "research/findings/raw/g11_bg/invivo_binding"

foreach ($seed in $seeds) {
    Write-Host "[v_schema] Starting seed $seed..." -ForegroundColor Cyan
    $t0 = Get-Date
    # Clean prior fork if exists
    $fork = "bridges/lineage/invivo_fix_v_schema_seed$seed"
    if (Test-Path $fork) { Remove-Item -Path $fork -Recurse -Force }

    python -u -m research.runners.investigate_invivo_binding_fix `
        --base-lineage main_hippo --seed $seed `
        --n-events 200 `
        --variants v_schema `
        --out "$outDir/invivo_seed$($seed)_v_schema.json" `
        > "$outDir/invivo_seed$($seed)_v_schema.log" 2>&1
    $dt = (Get-Date) - $t0
    Write-Host "[v_schema] seed $seed done in $($dt.TotalMinutes.ToString('F1')) min" -ForegroundColor Green
}

Write-Host "[v_schema] All 5 seeds complete." -ForegroundColor Yellow
