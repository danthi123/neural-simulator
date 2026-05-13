# v14 multi-seed validation (16 pools + orthogonal codes)
# Seeds 43, 44, 45, 46. ~17 min training + ~30s A->W per seed.

$ErrorActionPreference = "Stop"
Set-Location "E:\Documents\Projects\sim"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$OutDir = "research\findings\raw\g11_bg\concept_pool_demo"

function Run-V14-Seed {
    param([int]$Seed)
    Write-Host "[v14-MS] launching seed $Seed training..."
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
        --save-bridge "$OutDir\seed${Seed}_v14.simstate.h5" `
        --out "$OutDir\seed${Seed}_v14.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_v14.log"

    Write-Host "[v14-MS] running A->W readout seed $Seed..."
    python -m research.runners.concept_speak_demo `
        --seed $Seed `
        --n-lang-input 2048 `
        --n-per-pool 200 `
        --n-fs-per-pool 24 `
        --weak-concept-dynamics `
        --enable-adjective `
        --orthogonal-codes `
        --sparsity 0.05 `
        --load-bridge "$OutDir\seed${Seed}_v14.simstate.h5" `
        --out "$OutDir\seed${Seed}_v14_speak.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_v14_speak.log"

    $r = Get-Content "$OutDir\seed${Seed}_v14.json" -Raw | ConvertFrom-Json
    $s = Get-Content "$OutDir\seed${Seed}_v14_speak.json" -Raw | ConvertFrom-Json
    Write-Host "[v14-MS] seed $Seed : Phase 1 = $($r.n_pass)/16, A->W = $($s.a_to_w_pass)/16"
}

foreach ($seed in 43, 44, 45, 46) {
    Run-V14-Seed -Seed $seed
}

Write-Host ""
Write-Host "=== v14 multi-seed summary ==="
foreach ($s in 42, 43, 44, 45, 46) {
    $jp = "$OutDir\seed${s}_v14.json"
    $sp = "$OutDir\seed${s}_v14_speak.json"
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
