# Multi-seed compose-training at 400 events/pair (seeds 43-46)
# Compose-training is FRESH (starts from seed{N}_v16.simstate.h5),
# does NOT chain from the 100-event composed bridge.
# Each seed: ~8 min compose + 30s A->W = ~8.5 min
# Total: ~34 min for 4 seeds.

$ErrorActionPreference = "Stop"
Set-Location "E:\Documents\Projects\sim"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$OutDir = "research\findings\raw\g11_bg\concept_pool_demo"

function Run-V16-Compose400-Seed {
    param([int]$Seed)
    Write-Host "[v16-compose400-MS] seed $Seed compose-training (400 events/pair)..."
    python -m research.runners.concept_compose_train `
        --load-bridge "$OutDir\seed${Seed}_v16.simstate.h5" `
        --seed $Seed `
        --compose-pairs "go:north,come:south,stop:west,look:east" `
        --n-events-per-pair 400 `
        --orthogonal-codes `
        --sparsity 0.05 `
        --save-bridge "$OutDir\seed${Seed}_v16_composed400.simstate.h5" `
        --out "$OutDir\seed${Seed}_v16_compose400.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_v16_compose400.log"

    Write-Host "[v16-compose400-MS] A->W readout seed $Seed..."
    python -m research.runners.concept_speak_demo `
        --seed $Seed `
        --n-lang-input 2048 `
        --n-per-pool 200 `
        --n-fs-per-pool 24 `
        --weak-concept-dynamics `
        --enable-adjective `
        --orthogonal-codes `
        --sparsity 0.05 `
        --load-bridge "$OutDir\seed${Seed}_v16_composed400.simstate.h5" `
        --out "$OutDir\seed${Seed}_v16_composed400_speak.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_v16_composed400_speak.log"

    $c = Get-Content "$OutDir\seed${Seed}_v16_compose400.json" -Raw | ConvertFrom-Json
    $s = Get-Content "$OutDir\seed${Seed}_v16_composed400_speak.json" -Raw | ConvertFrom-Json
    Write-Host "[v16-compose400-MS] seed $Seed : Compose = $($c.n_pass)/$($c.n_total), A->W = $($s.a_to_w_pass)/16"
}

foreach ($seed in 43, 44, 45, 46) {
    Run-V16-Compose400-Seed -Seed $seed
}

Write-Host ""
Write-Host "=== v16+compose@400 multi-seed summary ==="
foreach ($s in 42, 43, 44, 45, 46) {
    $cp = "$OutDir\seed${s}_v16_compose400.json"
    $sp = "$OutDir\seed${s}_v16_composed400_speak.json"
    if (Test-Path $cp -PathType Leaf) {
        $c = Get-Content $cp -Raw | ConvertFrom-Json
        $s_str = ""
        if (Test-Path $sp -PathType Leaf) {
            $s_d = Get-Content $sp -Raw | ConvertFrom-Json
            $s_str = "  A->W=$($s_d.a_to_w_pass)/16"
        }
        Write-Host "  seed $s : Compose = $($c.n_pass)/$($c.n_total)  $s_str"
    }
}
