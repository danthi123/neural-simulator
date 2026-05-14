#!/usr/bin/env pwsh
# v19 multi-seed validation: cross_pool_concept gate frozen during Phase 1,
# opened only during concept-concept engram encoding. If 25% top-1 ceiling
# breaks (≥3/8 multi-seed), this is real progress beyond v16/v18.
#
# Usage:
#   pwsh research/runners/v19_multiseed.ps1
#
# Each seed: ~20 min train + ~3 min strict eval = ~23 min/seed
# Total: 5 seeds × 23 min = ~2 hr (sequential)

$ErrorActionPreference = 'Continue'
$Seeds = 42, 43, 44, 45, 46
$BridgeDir = "research/findings/raw/g11_bg/concept_pool_demo"
$StrictDir = "research/findings/raw/g11_bg/compose_concept_strict"
New-Item -ItemType Directory -Force -Path $StrictDir | Out-Null

foreach ($Seed in $Seeds) {
    $Bridge = "$BridgeDir/seed${Seed}_v19.simstate.h5"
    $TrainOut = "$BridgeDir/seed${Seed}_v19.json"
    $TrainLog = "$BridgeDir/seed${Seed}_v19.log"
    $StrictOut = "$StrictDir/seed${Seed}_v19_strict.json"
    $StrictLog = "$StrictDir/seed${Seed}_v19_strict.log"

    if (-not (Test-Path $Bridge)) {
        Write-Host "[v19 seed=$Seed] Training Phase 1 bridge..."
        $T = Get-Date
        python -m research.runners.concept_pool_demo `
            --seed $Seed --n-train-events 200 --n-lang-input 2048 --n-per-pool 200 `
            --n-fs-per-pool 24 --weak-concept-dynamics --interleaved `
            --topographic-factor 3.0 --off-target-factor 0.3 `
            --enable-adjective --orthogonal-codes --sparsity 0.05 `
            --enable-cross-pool-concept-pathways `
            --save-bridge $Bridge --out $TrainOut `
            *> $TrainLog
        $DT = (Get-Date) - $T
        Write-Host ("[v19 seed=$Seed] trained in {0:0.0} min" -f $DT.TotalMinutes)
    } else {
        Write-Host "[v19 seed=$Seed] bridge exists, skipping train"
    }

    Write-Host "[v19 seed=$Seed] Running strict top-1 eval..."
    $T = Get-Date
    python -m research.runners.compose_concept_strict `
        --load-bridge $Bridge --seed $Seed `
        --n-lang-input 2048 --n-per-pool 200 --n-fs-per-pool 24 `
        --n-words-for-orthogonal 16 --encoding-steps 200 --sparsity 0.05 `
        --enable-cross-pool-concept-pathways `
        --out $StrictOut `
        *> $StrictLog
    $DT = (Get-Date) - $T
    Write-Host ("[v19 seed=$Seed] strict eval in {0:0.0} min" -f $DT.TotalMinutes)
}

Write-Host ""
Write-Host "=== v19 multi-seed strict summary ==="
foreach ($Seed in $Seeds) {
    $StrictOut = "$StrictDir/seed${Seed}_v19_strict.json"
    if (Test-Path $StrictOut) {
        $J = Get-Content $StrictOut | ConvertFrom-Json
        Write-Host ("  seed={0}: top1={1}/{2} top3={3}/{2}" -f $Seed, $J.n_top1_pass, $J.n_total, $J.n_top3_pass)
    } else {
        Write-Host "  seed=$Seed: MISSING"
    }
}
