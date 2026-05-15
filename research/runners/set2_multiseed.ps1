#!/usr/bin/env pwsh
# Multi-seed retrain for set2 bridge (12 NEW concept words).
# After set2 seed 42 validates, this trains seeds 43-46 for multi-seed coverage.

$ErrorActionPreference = 'Continue'
$Seeds = 43, 44, 45, 46  # 42 already done
$BridgeDir = "research/findings/raw/g11_bg/concept_pool_demo"

foreach ($Seed in $Seeds) {
    $Bridge = "$BridgeDir/seed${Seed}_set2.simstate.h5"
    $Out = "$BridgeDir/seed${Seed}_set2.json"
    $Log = "$BridgeDir/seed${Seed}_set2.log"
    if (Test-Path $Bridge) {
        Write-Host "[seed $Seed] bridge exists, skipping"
        continue
    }
    Write-Host "[seed $Seed] training set2 bridge..."
    $T = Get-Date
    python -m research.runners.concept_pool_demo_set2 `
        --seed $Seed --n-train-events 200 --n-lang-input 2048 --n-per-pool 200 `
        --n-fs-per-pool 24 --weak-concept-dynamics --interleaved `
        --topographic-factor 3.0 --off-target-factor 0.3 `
        --enable-adjective --orthogonal-codes --sparsity 0.05 `
        --save-bridge $Bridge --out $Out `
        *> $Log
    $DT = (Get-Date) - $T
    Write-Host ("[seed $Seed] done in {0:0.0} min" -f $DT.TotalMinutes)
}

Write-Host ""
Write-Host "=== Set2 Phase 1 summary ==="
foreach ($Seed in @(42) + $Seeds) {
    $Out = "$BridgeDir/seed${Seed}_set2.json"
    if (Test-Path $Out) {
        $J = Get-Content $Out | ConvertFrom-Json
        Write-Host ("  seed={0}: n_pass={1}/{2}" -f $Seed, $J.n_pass, $J.n_words)
    } else {
        Write-Host "  seed=$Seed: MISSING"
    }
}
