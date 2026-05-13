# v9 multi-seed validation (seeds 43, 44, 45, 46).
# Runs sequentially. Each seed: train then A->W readout.
# ETA: ~13 min training + 30s A->W per seed = ~13.5 min/seed, 54 min total.

$ErrorActionPreference = "Stop"
Set-Location "E:\Documents\Projects\sim"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$OutDir = "research\findings\raw\g11_bg\concept_pool_demo"

function Run-V9-Seed {
    param([int]$Seed)
    Write-Host "[v9-MS] launching seed $Seed training..."
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
        --save-bridge "$OutDir\seed${Seed}_v9.simstate.h5" `
        --out "$OutDir\seed${Seed}_v9.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_v9.log"

    Write-Host "[v9-MS] running A->W readout seed $Seed..."
    python -m research.runners.concept_speak_demo `
        --seed $Seed `
        --n-lang-input 2048 `
        --n-per-pool 200 `
        --n-fs-per-pool 24 `
        --weak-concept-dynamics `
        --load-bridge "$OutDir\seed${Seed}_v9.simstate.h5" `
        --out "$OutDir\seed${Seed}_v9_speak.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_v9_speak.log"

    $r = Get-Content "$OutDir\seed${Seed}_v9.json" -Raw | ConvertFrom-Json
    $s = Get-Content "$OutDir\seed${Seed}_v9_speak.json" -Raw | ConvertFrom-Json
    Write-Host "[v9-MS] seed $Seed : Phase 1 = $($r.n_pass)/12, A->W = $($s.a_to_w_pass)/12"
}

foreach ($seed in 43, 44, 45, 46) {
    Run-V9-Seed -Seed $seed
}

Write-Host ""
Write-Host "=== v9 multi-seed summary ==="
foreach ($s in 42, 43, 44, 45, 46) {
    $jp = "$OutDir\seed${s}_v9.json"
    $sp = "$OutDir\seed${s}_v9_speak.json"
    if (Test-Path $jp -PathType Leaf) {
        $r = Get-Content $jp -Raw | ConvertFrom-Json
        $speak_str = ""
        if (Test-Path $sp -PathType Leaf) {
            $s_data = Get-Content $sp -Raw | ConvertFrom-Json
            $speak_str = "A->W=$($s_data.a_to_w_pass)/12"
        }
        Write-Host "  seed $s : Phase 1 = $($r.n_pass)/12  $speak_str"
    }
}
