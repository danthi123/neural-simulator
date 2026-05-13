# v15c multi-seed validation: seeds 43, 44, 45, 46 (~70 min total)
# v14 recipe + --enable-dlpfc-verb-unidirectional. v15c uses weak
# dlpfc dynamics + skip lang_input->dlpfc + zero-init pathway weights.

$ErrorActionPreference = "Stop"
Set-Location "E:\Documents\Projects\sim"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$OutDir = "research\findings\raw\g11_bg\concept_pool_demo"

function Run-V15c-Seed {
    param([int]$Seed)
    Write-Host "[v15c-MS] launching seed $Seed training..."
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
        --enable-dlpfc-verb-unidirectional `
        --save-bridge "$OutDir\seed${Seed}_v15.simstate.h5" `
        --out "$OutDir\seed${Seed}_v15.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_v15.log"

    Write-Host "[v15c-MS] running A->W readout seed $Seed..."
    python -m research.runners.concept_speak_demo `
        --seed $Seed `
        --n-lang-input 2048 `
        --n-per-pool 200 `
        --n-fs-per-pool 24 `
        --weak-concept-dynamics `
        --enable-adjective `
        --orthogonal-codes `
        --sparsity 0.05 `
        --load-bridge "$OutDir\seed${Seed}_v15.simstate.h5" `
        --out "$OutDir\seed${Seed}_v15_speak.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_v15_speak.log"

    $r = Get-Content "$OutDir\seed${Seed}_v15.json" -Raw | ConvertFrom-Json
    $s = Get-Content "$OutDir\seed${Seed}_v15_speak.json" -Raw | ConvertFrom-Json
    Write-Host "[v15c-MS] seed $Seed : Phase 1 = $($r.n_pass)/16, A->W = $($s.a_to_w_pass)/16"
}

foreach ($seed in 43, 44, 45, 46) {
    Run-V15c-Seed -Seed $seed
}

Write-Host ""
Write-Host "=== v15c multi-seed summary ==="
foreach ($s in 42, 43, 44, 45, 46) {
    $jp = "$OutDir\seed${s}_v15.json"
    $sp = "$OutDir\seed${s}_v15_speak.json"
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
