# v9 follow-up: wait for v9 training, then run A->W readout.

$ErrorActionPreference = "Stop"
Set-Location "E:\Documents\Projects\sim"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$OutDir = "research\findings\raw\g11_bg\concept_pool_demo"
$Seed = 42

# Wait for v9 JSON
while (-not (Test-Path "$OutDir\seed${Seed}_v9.json")) {
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 3

# Run A->W readout
Write-Host "[v9-followup] Running A->W readout..."
python -m research.runners.concept_speak_demo `
    --seed $Seed `
    --n-lang-input 2048 `
    --n-per-pool 200 `
    --n-fs-per-pool 24 `
    --weak-concept-dynamics `
    --load-bridge "$OutDir\seed${Seed}_v9.simstate.h5" `
    --out "$OutDir\seed${Seed}_v9_speak.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_v9_speak.log"

# Compare with v7 and v8
$r9 = Get-Content "$OutDir\seed${Seed}_v9.json" -Raw | ConvertFrom-Json
$r7 = Get-Content "$OutDir\seed${Seed}_v7.json" -Raw | ConvertFrom-Json
$r8 = Get-Content "$OutDir\seed${Seed}_v8.json" -Raw | ConvertFrom-Json
$s9 = Get-Content "$OutDir\seed${Seed}_v9_speak.json" -Raw | ConvertFrom-Json
$s7 = Get-Content "$OutDir\seed${Seed}_v7_speak.json" -Raw | ConvertFrom-Json
$s8 = Get-Content "$OutDir\seed${Seed}_v8_speak.json" -Raw | ConvertFrom-Json

Write-Host ""
Write-Host "=== v7 vs v8 vs v9 comparison (seed 42) ==="
Write-Host "  Phase 1 isolation: v7=$($r7.n_pass)/12  v8=$($r8.n_pass)/12  v9=$($r9.n_pass)/12"
Write-Host "  Phase 3 A->W:       v7=$($s7.a_to_w_pass)/12  v8=$($s8.a_to_w_pass)/12  v9=$($s9.a_to_w_pass)/12"
