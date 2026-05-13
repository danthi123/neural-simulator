# v13 follow-up: after training, A->W + composition test.

$ErrorActionPreference = "Stop"
Set-Location "E:\Documents\Projects\sim"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$OutDir = "research\findings\raw\g11_bg\concept_pool_demo"

while (-not (Test-Path "$OutDir\seed42_v13.json")) {
    Start-Sleep -Seconds 60
}
Start-Sleep -Seconds 3

Write-Host "[v13] A->W readout..."
python -m research.runners.concept_speak_demo `
    --seed 42 --n-lang-input 2048 --n-per-pool 200 --n-fs-per-pool 24 `
    --weak-concept-dynamics `
    --load-bridge "$OutDir\seed42_v13.simstate.h5" `
    --out "$OutDir\seed42_v13_speak.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed42_v13_speak.log"

Write-Host "[v13] Composition..."
python -m research.runners.concept_compose_demo `
    --seed 42 --n-lang-input 2048 --n-per-pool 200 --n-fs-per-pool 24 `
    --weak-concept-dynamics `
    --load-bridge "$OutDir\seed42_v13.simstate.h5" `
    --out "$OutDir\seed42_v13_compose.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed42_v13_compose.log"

$r13 = Get-Content "$OutDir\seed42_v13.json" -Raw | ConvertFrom-Json
$s13 = Get-Content "$OutDir\seed42_v13_speak.json" -Raw | ConvertFrom-Json
$c13 = Get-Content "$OutDir\seed42_v13_compose.json" -Raw | ConvertFrom-Json

$r9 = Get-Content "$OutDir\seed42_v9.json" -Raw | ConvertFrom-Json
$s9 = Get-Content "$OutDir\seed42_v9_speak.json" -Raw | ConvertFrom-Json

Write-Host ""
Write-Host "=== v9 vs v13 (seed 42) ==="
Write-Host "  Phase 1 W->A:    v9=$($r9.n_pass)/12  v13=$($r13.n_pass)/12"
Write-Host "  Phase 3 A->W:    v9=$($s9.a_to_w_pass)/12  v13=$($s13.a_to_w_pass)/12"
Write-Host "  Compose seq:     v13=$($c13.n_sequential_pass)/$($c13.n_sequential_pairs)"
Write-Host "  Compose cofire:  v13=$($c13.n_cofire_pass)/$($c13.n_cofire_pairs)"
