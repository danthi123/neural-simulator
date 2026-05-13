# Sequentially run v7 multi-seed validation seeds 44, 45, 46.
# Seed 44 is already running; wait for it then launch 45, 46.

$ErrorActionPreference = "Stop"
Set-Location "E:\Documents\Projects\sim"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$OutDir = "research\findings\raw\g11_bg\concept_pool_demo"

function Wait-Seed {
    param([int]$Seed)
    $jsonPath = "$OutDir\seed${Seed}_v7.json"
    while (-not (Test-Path $jsonPath)) {
        Start-Sleep -Seconds 30
        $progress = (Select-String -Path "$OutDir\seed${Seed}_v7.log" -Pattern "VERDICT" -ErrorAction SilentlyContinue).Count
        if ($progress -gt 0) {
            Start-Sleep -Seconds 3  # wait for JSON write
            return
        }
    }
}

function Run-Seed {
    param([int]$Seed)
    Write-Host "[$Seed] launching..."
    python -m research.runners.concept_pool_demo `
        --seed $Seed `
        --n-train-events 200 `
        --n-lang-input 2048 `
        --n-per-pool 200 `
        --n-fs-per-pool 24 `
        --weak-concept-dynamics `
        --interleaved `
        --topographic-factor 3.0 `
        --off-target-factor 0.3 `
        --save-bridge "$OutDir\seed${Seed}_v7.simstate.h5" `
        --out "$OutDir\seed${Seed}_v7.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_v7.log"
    Write-Host "[$Seed] done"
}

# Wait for seed 44 (already launched)
Write-Host "Waiting for seed 44 to complete..."
Wait-Seed -Seed 44
$r44 = Get-Content "$OutDir\seed44_v7.json" -Raw | ConvertFrom-Json
Write-Host "Seed 44 result: $($r44.n_pass)/$($r44.n_words)"

# Launch 45
Run-Seed -Seed 45
$r45 = Get-Content "$OutDir\seed45_v7.json" -Raw | ConvertFrom-Json
Write-Host "Seed 45 result: $($r45.n_pass)/$($r45.n_words)"

# Launch 46
Run-Seed -Seed 46
$r46 = Get-Content "$OutDir\seed46_v7.json" -Raw | ConvertFrom-Json
Write-Host "Seed 46 result: $($r46.n_pass)/$($r46.n_words)"

# Summary
Write-Host ""
Write-Host "=== Multi-seed v7 summary ==="
foreach ($s in 42, 43, 44, 45, 46) {
    $p = "$OutDir\seed${s}_v7.json"
    if (Test-Path $p) {
        $r = Get-Content $p -Raw | ConvertFrom-Json
        Write-Host "  seed $s : $($r.n_pass)/$($r.n_words) ($($r.wall_clock_s) s)"
    }
}
