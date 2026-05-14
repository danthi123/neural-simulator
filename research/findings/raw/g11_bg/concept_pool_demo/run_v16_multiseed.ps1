# v16 multi-seed validation (seeds 43-46)
# Direct verb_pool -> motor pathways, zero-init.
# Seed 42 single-seed: P1 13/16 + A->W 16/16 = 29/32 (91%).

$ErrorActionPreference = "Stop"
Set-Location "E:\Documents\Projects\sim"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$OutDir = "research\findings\raw\g11_bg\concept_pool_demo"

function Run-V16-Seed {
    param([int]$Seed)
    Write-Host "[v16-MS] launching seed $Seed training..."
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
        --enable-adjective `
        --orthogonal-codes `
        --sparsity 0.05 `
        --enable-direct-verb-to-motor `
        --save-bridge "$OutDir\seed${Seed}_v16.simstate.h5" `
        --out "$OutDir\seed${Seed}_v16.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_v16.log"

    Write-Host "[v16-MS] running A->W readout seed $Seed..."
    python -m research.runners.concept_speak_demo `
        --seed $Seed `
        --n-lang-input 2048 `
        --n-per-pool 200 `
        --n-fs-per-pool 24 `
        --weak-concept-dynamics `
        --enable-adjective `
        --orthogonal-codes `
        --sparsity 0.05 `
        --load-bridge "$OutDir\seed${Seed}_v16.simstate.h5" `
        --out "$OutDir\seed${Seed}_v16_speak.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_v16_speak.log"

    $r = Get-Content "$OutDir\seed${Seed}_v16.json" -Raw | ConvertFrom-Json
    $s = Get-Content "$OutDir\seed${Seed}_v16_speak.json" -Raw | ConvertFrom-Json
    Write-Host "[v16-MS] seed $Seed : Phase 1 = $($r.n_pass)/16, A->W = $($s.a_to_w_pass)/16"
}

foreach ($seed in 43, 44, 45, 46) {
    Run-V16-Seed -Seed $seed
}

Write-Host ""
Write-Host "=== v16 multi-seed summary ==="
foreach ($s in 42, 43, 44, 45, 46) {
    $jp = "$OutDir\seed${s}_v16.json"
    $sp = "$OutDir\seed${s}_v16_speak.json"
    if (Test-Path $jp -PathType Leaf) {
        $r = Get-Content $jp -Raw | ConvertFrom-Json
        $s_str = ""
        if (Test-Path $sp -PathType Leaf) {
            $s_d = Get-Content $sp -Raw | ConvertFrom-Json
            $s_str = "  A->W=$($s_d.a_to_w_pass)/16"
        }
        Write-Host "  seed $s : Phase 1 = $($r.n_pass)/16  $s_str"
    }
}
