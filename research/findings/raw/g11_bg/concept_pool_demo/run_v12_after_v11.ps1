# v12 launch: after v11 multi-seed completes, run v12 single seed.
# v12 = v11 architecture + --enable-dlpfc-verb-holding for sequential
# composition.

$ErrorActionPreference = "Stop"
Set-Location "E:\Documents\Projects\sim"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$OutDir = "research\findings\raw\g11_bg\concept_pool_demo"

# Wait for v11 multi-seed completion
while (-not (Test-Path "$OutDir\seed46_v11_speak.json")) {
    Start-Sleep -Seconds 60
}
Start-Sleep -Seconds 5

# Launch v12 seed 42 (12-pool baseline first, no adjective)
Write-Host "[v12] Training seed 42 with dlpfc_verb holding..."
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
    --enable-dlpfc-verb-holding `
    --save-bridge "$OutDir\seed42_v12.simstate.h5" `
    --out "$OutDir\seed42_v12.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed42_v12.log"

# Phase 3 A->W (no adjective, so 12 pools)
Write-Host "[v12] A->W readout..."
python -m research.runners.concept_speak_demo `
    --seed 42 `
    --n-lang-input 2048 `
    --n-per-pool 200 `
    --n-fs-per-pool 24 `
    --weak-concept-dynamics `
    --load-bridge "$OutDir\seed42_v12.simstate.h5" `
    --out "$OutDir\seed42_v12_speak.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed42_v12_speak.log"

# Phase 2 composition (the v12 target test)
Write-Host "[v12] Composition test..."
python -m research.runners.concept_compose_demo `
    --seed 42 `
    --n-lang-input 2048 `
    --n-per-pool 200 `
    --n-fs-per-pool 24 `
    --weak-concept-dynamics `
    --load-bridge "$OutDir\seed42_v12.simstate.h5" `
    --out "$OutDir\seed42_v12_compose.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed42_v12_compose.log"

# Compare
$r12 = Get-Content "$OutDir\seed42_v12.json" -Raw | ConvertFrom-Json
$s12 = Get-Content "$OutDir\seed42_v12_speak.json" -Raw | ConvertFrom-Json
$c12 = Get-Content "$OutDir\seed42_v12_compose.json" -Raw | ConvertFrom-Json

Write-Host ""
Write-Host "=== v9 vs v12 (seed 42, 12-pool) ==="
$r9 = Get-Content "$OutDir\seed42_v9.json" -Raw | ConvertFrom-Json
$s9 = Get-Content "$OutDir\seed42_v9_speak.json" -Raw | ConvertFrom-Json
Write-Host "  Phase 1 W->A:    v9=$($r9.n_pass)/12  v12=$($r12.n_pass)/12"
Write-Host "  Phase 3 A->W:    v9=$($s9.a_to_w_pass)/12  v12=$($s12.a_to_w_pass)/12"
Write-Host "  Compose seq:     v12=$($c12.n_sequential_pass)/$($c12.n_sequential_pairs)"
Write-Host "  Compose cofire:  v12=$($c12.n_cofire_pass)/$($c12.n_cofire_pairs)"
