# Multi-seed engram 8-pair scaling test
# Tests: does engram-composition scale to 8 (verb+noun) pairs across seeds?

$ErrorActionPreference = "Stop"
Set-Location "E:\Documents\Projects\sim"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$OutDir = "research\findings\raw\g11_bg\concept_pool_demo"
$Pairs = "go:north,come:south,stop:west,look:east,apple:north,river:south,dog:west,cat:east"

function Run-Engram-8pair-Seed {
    param([int]$Seed)
    Write-Host "[engram-8pair-MS] seed $Seed..."
    python -m research.runners.compose_engram_demo `
        --load-bridge "$OutDir\seed${Seed}_v16.simstate.h5" `
        --seed $Seed `
        --compose-pairs $Pairs `
        --encoding-steps 200 `
        --top-k 100 `
        --recall-stim-pA 1500 `
        --recall-steps 100 `
        --out "$OutDir\seed${Seed}_engram_8pair.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_engram_8pair.log" | Out-Null

    $r = Get-Content "$OutDir\seed${Seed}_engram_8pair.json" -Raw | ConvertFrom-Json
    Write-Host "[engram-8pair-MS] seed $Seed : PASS=$($r.n_pass)/8"
}

foreach ($seed in 42, 43, 44, 45, 46) {
    Run-Engram-8pair-Seed -Seed $seed
}

Write-Host ""
Write-Host "=== engram 8-pair multi-seed summary ==="
$total = 0
$max = 0
foreach ($s in 42, 43, 44, 45, 46) {
    $jp = "$OutDir\seed${s}_engram_8pair.json"
    if (Test-Path $jp -PathType Leaf) {
        $r = Get-Content $jp -Raw | ConvertFrom-Json
        Write-Host "  seed $s : PASS=$($r.n_pass)/8"
        $total += $r.n_pass
        $max += 8
    }
}
Write-Host "TOTAL: $total/$max = $([math]::Round(100*$total/$max, 1))%"
