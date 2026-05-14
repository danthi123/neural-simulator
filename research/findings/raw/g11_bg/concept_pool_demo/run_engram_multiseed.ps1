# Multi-seed engram-based composition (seeds 43-46)
# Uses existing v16 bridges (any v14/v16 bridge works - engrams don't
# require v16 wiring). ~30s per seed.

$ErrorActionPreference = "Stop"
Set-Location "E:\Documents\Projects\sim"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$OutDir = "research\findings\raw\g11_bg\concept_pool_demo"

function Run-Engram-Seed {
    param([int]$Seed)
    Write-Host "[engram-MS] seed $Seed encoding + recall..."
    python -m research.runners.compose_engram_demo `
        --load-bridge "$OutDir\seed${Seed}_v16.simstate.h5" `
        --seed $Seed `
        --encoding-steps 200 `
        --top-k 100 `
        --recall-stim-pA 1500 `
        --recall-steps 100 `
        --save-bridge "$OutDir\seed${Seed}_engram.simstate.h5" `
        --out "$OutDir\seed${Seed}_engram.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_engram.log"

    $r = Get-Content "$OutDir\seed${Seed}_engram.json" -Raw | ConvertFrom-Json
    Write-Host "[engram-MS] seed $Seed : PASS=$($r.n_pass)/4, TRUE_rank=$($r.true_rank)/24"
}

foreach ($seed in 43, 44, 45, 46) {
    Run-Engram-Seed -Seed $seed
}

Write-Host ""
Write-Host "=== engram-composition multi-seed summary ==="
foreach ($s in 42, 43, 44, 45, 46) {
    $jp = "$OutDir\seed${s}_engram.json"
    if (Test-Path $jp -PathType Leaf) {
        $r = Get-Content $jp -Raw | ConvertFrom-Json
        Write-Host "  seed $s : PASS=$($r.n_pass)/4, TRUE_rank=$($r.true_rank)/24, TRUE_pass=$($r.true_pass), best_perm=$($r.best_perm_pass)"
    }
}
