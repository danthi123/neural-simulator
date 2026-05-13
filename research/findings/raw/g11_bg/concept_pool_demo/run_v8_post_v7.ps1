# v8 batch: after v7 multi-seed completes, re-apply the
# concept_to_language_output_weight 0.5 -> 2.0 fix and run v8 batch
# (single seed first, then chain if v8 works).

$ErrorActionPreference = "Stop"
Set-Location "E:\Documents\Projects\sim"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$OutDir = "research\findings\raw\g11_bg\concept_pool_demo"

function Wait-Seed {
    param([int]$Seed, [string]$Variant)
    $jsonPath = "$OutDir\seed${Seed}_${Variant}.json"
    while (-not (Test-Path $jsonPath)) {
        Start-Sleep -Seconds 30
        $progress = (Select-String -Path "$OutDir\seed${Seed}_${Variant}.log" -Pattern "VERDICT" -ErrorAction SilentlyContinue).Count
        if ($progress -gt 0) {
            Start-Sleep -Seconds 3
            return
        }
    }
}

# Wait for v7 seed 46 (last in chain)
Write-Host "[v8] Waiting for v7 seed 46 to complete..."
Wait-Seed -Seed 46 -Variant "v7"
$r46 = Get-Content "$OutDir\seed46_v7.json" -Raw | ConvertFrom-Json
Write-Host "[v8] V7 seed 46 done: $($r46.n_pass)/$($r46.n_words)"

# Apply weight fix
Write-Host "[v8] Applying concept_to_language_output_weight fix..."
git checkout d21efae~ -- research/runners/text_minimal_isolation.py 2>$null  # revert undone
$content = Get-Content "research/runners/text_minimal_isolation.py" -Raw
$content = $content -replace "concept_to_language_output_weight: float = 0.5,", "concept_to_language_output_weight: float = 2.0,"
Set-Content "research/runners/text_minimal_isolation.py" $content
git diff --stat -- research/runners/text_minimal_isolation.py
git add research/runners/text_minimal_isolation.py
git commit -m "fix(concept-pools): v8 - concept_to_language_output_weight 0.5->2.0 (re-apply)"
Write-Host "[v8] Fix re-applied"

# Run v8 seed 42 first (single-seed validation)
Write-Host "[v8] Launching seed 42 v8..."
python -m research.runners.concept_pool_demo `
    --seed 42 `
    --n-train-events 200 `
    --n-lang-input 2048 `
    --n-per-pool 200 `
    --n-fs-per-pool 24 `
    --weak-concept-dynamics `
    --interleaved `
    --topographic-factor 3.0 `
    --off-target-factor 0.3 `
    --save-bridge "$OutDir\seed42_v8.simstate.h5" `
    --out "$OutDir\seed42_v8.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed42_v8.log"
$r42v8 = Get-Content "$OutDir\seed42_v8.json" -Raw | ConvertFrom-Json
Write-Host "[v8] Seed 42 v8 result: $($r42v8.n_pass)/$($r42v8.n_words)"

# Test A->W readout on v8 bridge (the key test for the weight fix)
Write-Host "[v8] Running A->W readout on v8 bridge..."
python -m research.runners.concept_speak_demo `
    --seed 42 `
    --n-lang-input 2048 `
    --n-per-pool 200 `
    --n-fs-per-pool 24 `
    --weak-concept-dynamics `
    --load-bridge "$OutDir\seed42_v8.simstate.h5" `
    --out "$OutDir\seed42_v8_speak.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed42_v8_speak.log"

Write-Host ""
Write-Host "=== v7 vs v8 comparison ==="
$r42v7 = Get-Content "$OutDir\seed42_v7.json" -Raw | ConvertFrom-Json
$v7speak = Get-Content "$OutDir\seed42_v7_speak.json" -Raw | ConvertFrom-Json
$v8speak = Get-Content "$OutDir\seed42_v8_speak.json" -Raw | ConvertFrom-Json
Write-Host "  Phase 1 isolation: v7=$($r42v7.n_pass)/12  v8=$($r42v8.n_pass)/12"
Write-Host "  Phase 3 A->W:       v7=$($v7speak.a_to_w_pass)/12  v8=$($v8speak.a_to_w_pass)/12"
