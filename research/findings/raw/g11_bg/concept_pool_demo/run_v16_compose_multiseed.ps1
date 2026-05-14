# Multi-seed compose-training on existing v16 bridges (seeds 43-46)
# Each seed: ~2 min for 100 events/pair x 4 pairs + 30s inference
# Total: ~12 min for 4 seeds

$ErrorActionPreference = "Stop"
Set-Location "E:\Documents\Projects\sim"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$OutDir = "research\findings\raw\g11_bg\concept_pool_demo"

function Run-V16-Compose-Seed {
    param([int]$Seed)
    Write-Host "[v16-compose-MS] launching seed $Seed compose-training..."
    python -m research.runners.concept_compose_train `
        --load-bridge "$OutDir\seed${Seed}_v16.simstate.h5" `
        --seed $Seed `
        --compose-pairs "go:north,come:south,stop:west,look:east" `
        --n-events-per-pair 100 `
        --orthogonal-codes `
        --sparsity 0.05 `
        --save-bridge "$OutDir\seed${Seed}_v16_composed.simstate.h5" `
        --out "$OutDir\seed${Seed}_v16_compose.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_v16_compose.log"

    Write-Host "[v16-compose-MS] running A->W readout seed $Seed (composed bridge)..."
    python -m research.runners.concept_speak_demo `
        --seed $Seed `
        --n-lang-input 2048 `
        --n-per-pool 200 `
        --n-fs-per-pool 24 `
        --weak-concept-dynamics `
        --enable-adjective `
        --orthogonal-codes `
        --sparsity 0.05 `
        --load-bridge "$OutDir\seed${Seed}_v16_composed.simstate.h5" `
        --out "$OutDir\seed${Seed}_v16_composed_speak.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_v16_composed_speak.log"

    $c = Get-Content "$OutDir\seed${Seed}_v16_compose.json" -Raw | ConvertFrom-Json
    $s = Get-Content "$OutDir\seed${Seed}_v16_composed_speak.json" -Raw | ConvertFrom-Json
    Write-Host "[v16-compose-MS] seed $Seed : Compose = $($c.n_pass)/$($c.n_total), A->W = $($s.a_to_w_pass)/16"
}

foreach ($seed in 43, 44, 45, 46) {
    Run-V16-Compose-Seed -Seed $seed
}

Write-Host ""
Write-Host "=== v16+compose multi-seed summary ==="
foreach ($s in 42, 43, 44, 45, 46) {
    $cp = "$OutDir\seed${s}_v16_compose.json"
    $sp = "$OutDir\seed${s}_v16_composed_speak.json"
    $cp_alt = "$OutDir\seed${s}_v16_compose_test.json"
    if (Test-Path $cp -PathType Leaf) {
        $c = Get-Content $cp -Raw | ConvertFrom-Json
        $s_str = ""
        if (Test-Path $sp -PathType Leaf) {
            $s_d = Get-Content $sp -Raw | ConvertFrom-Json
            $s_str = "  A->W=$($s_d.a_to_w_pass)/16"
        }
        Write-Host "  seed $s : Compose = $($c.n_pass)/$($c.n_total)  $s_str"
    } elseif (Test-Path $cp_alt -PathType Leaf) {
        $c = Get-Content $cp_alt -Raw | ConvertFrom-Json
        Write-Host "  seed $s : Compose = $($c.n_pass)/$($c.n_total) (seed42 from test runner)"
    }
}
