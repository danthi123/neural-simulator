# v3 launch script: 4 verb pools + tighter topographic + longer reset (NMDA decay)
# Use if v2 hangs or fails. Adds reset_steps=300 (~150ms) to let NMDA fully decay
# between training events.

$ErrorActionPreference = "Stop"
Set-Location "E:\Documents\Projects\sim"
$env:PYTHONIOENCODING = "utf-8"

$OutDir = "research\findings\raw\g11_bg\concept_pool_demo"
$Seed = 42

Write-Host "[v3] Launching seed $Seed with reset_steps=300 (NMDA decay fix)..."
python -m research.runners.concept_pool_demo `
    --seed $Seed `
    --n-train-events 200 `
    --n-lang-input 4096 `
    --n-per-pool 500 `
    --n-fs-per-pool 60 `
    --reset-steps 300 `
    --out "$OutDir\seed${Seed}_v3.json" `
    --save-bridge "$OutDir\seed${Seed}_v3.simstate.h5" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_v3.log"
Write-Host "[v3] Done (exit $LASTEXITCODE)"
