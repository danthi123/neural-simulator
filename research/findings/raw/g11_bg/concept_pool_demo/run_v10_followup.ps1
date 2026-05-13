# v10 follow-up: after training, run A->W readout + compose test
# with extended NMDA tau (250ms).

$ErrorActionPreference = "Stop"
Set-Location "E:\Documents\Projects\sim"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$OutDir = "research\findings\raw\g11_bg\concept_pool_demo"

# Wait for v10 training to complete
while (-not (Test-Path "$OutDir\seed42_v10.json")) {
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 3

# A->W readout
Write-Host "[v10-followup] A->W readout..."
python -m research.runners.concept_speak_demo `
    --seed 42 `
    --n-lang-input 2048 `
    --n-per-pool 200 `
    --n-fs-per-pool 24 `
    --weak-concept-dynamics `
    --nmda-tau-decay-ms 250.0 `
    --load-bridge "$OutDir\seed42_v10.simstate.h5" `
    --out "$OutDir\seed42_v10_speak.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed42_v10_speak.log"

# Compose (sequential + co-fire) — this is the key v10 test
Write-Host "[v10-followup] Composition test (NMDA tau 250ms)..."
python -m research.runners.concept_compose_demo `
    --seed 42 `
    --n-lang-input 2048 `
    --n-per-pool 200 `
    --n-fs-per-pool 24 `
    --weak-concept-dynamics `
    --load-bridge "$OutDir\seed42_v10.simstate.h5" `
    --out "$OutDir\seed42_v10_compose.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed42_v10_compose.log"

# Compare
$r10 = Get-Content "$OutDir\seed42_v10.json" -Raw | ConvertFrom-Json
$s10 = Get-Content "$OutDir\seed42_v10_speak.json" -Raw | ConvertFrom-Json
$c10 = Get-Content "$OutDir\seed42_v10_compose.json" -Raw | ConvertFrom-Json

Write-Host ""
Write-Host "=== v9 vs v10 (seed 42) ==="
$r9 = Get-Content "$OutDir\seed42_v9.json" -Raw | ConvertFrom-Json
$s9 = Get-Content "$OutDir\seed42_v9_speak.json" -Raw | ConvertFrom-Json
Write-Host "  Phase 1 W->A:     v9=$($r9.n_pass)/12  v10=$($r10.n_pass)/12"
Write-Host "  Phase 3 A->W:     v9=$($s9.a_to_w_pass)/12  v10=$($s10.a_to_w_pass)/12"
Write-Host "  Compose seq:      v10=$($c10.n_sequential_pass)/$($c10.n_sequential_pairs)"
Write-Host "  Compose cofire:   v10=$($c10.n_cofire_pass)/$($c10.n_cofire_pairs)"
