# v11 multi-seed validation (16 pools: 4 motor + 4 noun + 4 verb + 4 adj)
# Seeds 43, 44, 45, 46. Each: train + A->W readout (~17 min/seed).

$ErrorActionPreference = "Stop"
Set-Location "E:\Documents\Projects\sim"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$OutDir = "research\findings\raw\g11_bg\concept_pool_demo"

function Run-V11-Seed {
    param([int]$Seed)
    Write-Host "[v11-MS] launching seed $Seed training..."
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
        --save-bridge "$OutDir\seed${Seed}_v11.simstate.h5" `
        --out "$OutDir\seed${Seed}_v11.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_v11.log"

    Write-Host "[v11-MS] running A->W readout seed $Seed..."
    python -m research.runners.concept_speak_demo `
        --seed $Seed `
        --n-lang-input 2048 `
        --n-per-pool 200 `
        --n-fs-per-pool 24 `
        --weak-concept-dynamics `
        --load-bridge "$OutDir\seed${Seed}_v11.simstate.h5" `
        --out "$OutDir\seed${Seed}_v11_speak.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_v11_speak.log"

    $r = Get-Content "$OutDir\seed${Seed}_v11.json" -Raw | ConvertFrom-Json
    $s = Get-Content "$OutDir\seed${Seed}_v11_speak.json" -Raw | ConvertFrom-Json
    Write-Host "[v11-MS] seed $Seed : Phase 1 = $($r.n_pass)/16, A->W = $($s.a_to_w_pass)/12"
}

foreach ($seed in 43, 44, 45, 46) {
    Run-V11-Seed -Seed $seed
}

Write-Host ""
Write-Host "=== v11 multi-seed summary ==="
foreach ($s in 42, 43, 44, 45, 46) {
    $jp = "$OutDir\seed${s}_v11.json"
    $sp = "$OutDir\seed${s}_v11_speak.json"
    if (Test-Path $jp -PathType Leaf) {
        $r = Get-Content $jp -Raw | ConvertFrom-Json
        $speak_str = ""
        if (Test-Path $sp -PathType Leaf) {
            $s_data = Get-Content $sp -Raw | ConvertFrom-Json
            $speak_str = "A->W=$($s_data.a_to_w_pass)/12"
        }
        Write-Host "  seed $s : Phase 1 = $($r.n_pass)/16  $speak_str"
    }
}
