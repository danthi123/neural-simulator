# v11 follow-up: re-run A->W with --enable-adjective for 16-pool readout.
# After v11 multi-seed chain completes, this iterates seeds 43-46 and
# saves seed{N}_v11_speak16.json with the full 16-pool A->W.

$ErrorActionPreference = "Stop"
Set-Location "E:\Documents\Projects\sim"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$OutDir = "research\findings\raw\g11_bg\concept_pool_demo"

# Wait for v11 multi-seed completion
while (-not (Test-Path "$OutDir\seed46_v11_speak.json")) {
    Start-Sleep -Seconds 60
}
Start-Sleep -Seconds 3

# Re-run A->W with adjective for each seed
foreach ($seed in 42, 43, 44, 45, 46) {
    $bridge = "$OutDir\seed${seed}_v11.simstate.h5"
    if (-not (Test-Path $bridge)) {
        Write-Host "[speak16] seed $seed bridge missing, skipping"
        continue
    }
    Write-Host "[speak16] seed $seed A->W with adjective..."
    python -m research.runners.concept_speak_demo `
        --seed $seed `
        --n-lang-input 2048 `
        --n-per-pool 200 `
        --n-fs-per-pool 24 `
        --weak-concept-dynamics `
        --enable-adjective `
        --load-bridge $bridge `
        --out "$OutDir\seed${seed}_v11_speak16.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${seed}_v11_speak16.log" | Select-String -Pattern "VERDICT" | ForEach-Object { Write-Host $_ }
}

Write-Host ""
Write-Host "=== v11 16-pool A->W summary ==="
foreach ($s in 42, 43, 44, 45, 46) {
    $sp = "$OutDir\seed${s}_v11_speak16.json"
    if (Test-Path $sp -PathType Leaf) {
        $d = Get-Content $sp -Raw | ConvertFrom-Json
        Write-Host "  seed $s : A->W = $($d.a_to_w_pass)/$($d.a_to_w_total)"
    }
}
