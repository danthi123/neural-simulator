# Multi-seed validator for concept_pool_demo.
# Waits for seed 42 to finish, then launches 43-46 sequentially.
# Per autonomous-runs principle #2: don't sit idle, do the next thing.

$ErrorActionPreference = "Stop"
$SimRoot = "E:\Documents\Projects\sim"
$OutDir = "$SimRoot\research\findings\raw\g11_bg\concept_pool_demo"
Set-Location $SimRoot

function Wait-Seed42 {
    # 2026-05-13: seed42 v1 FAILed (0/10). v2 uses suffix _v2 for fresh run.
    Write-Host "[multiseed] Waiting for seed 42 v2 to finish..."
    while ($true) {
        $jsonPath = "$OutDir\seed42_v2.json"
        if (Test-Path $jsonPath) {
            $size = (Get-Item $jsonPath).Length
            if ($size -gt 100) {
                Write-Host "[multiseed] Seed 42 v2 done ($size bytes). Verifying..."
                $log = Get-Content "$OutDir\seed42_v2.log" -Tail 30
                if ($log -match "VERDICT") {
                    Write-Host "[multiseed] Seed 42 v2 VERDICT line found. Proceeding."
                    return $true
                }
            }
        }
        $progress = (Select-String -Path "$OutDir\seed42_v2.log" -Pattern "^  trained '" -ErrorAction SilentlyContinue).Count
        Write-Host "[multiseed] still waiting... $progress/12 words trained"
        Start-Sleep -Seconds 60
    }
}

function Run-Seed {
    param([int]$Seed)
    $outPath = "$OutDir\seed${Seed}_v2.json"
    $logPath = "$OutDir\seed${Seed}_v2.log"
    $savePath = "$OutDir\seed${Seed}_v2.simstate.h5"
    Write-Host "[multiseed] Launching seed $Seed (v2)..."
    $env:PYTHONIOENCODING = "utf-8"
    python -m research.runners.concept_pool_demo `
        --seed $Seed `
        --n-train-events 200 `
        --n-lang-input 4096 `
        --n-per-pool 500 `
        --n-fs-per-pool 60 `
        --out $outPath `
        --save-bridge $savePath 2>&1 | Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[multiseed] Seed $Seed FAILED (exit $LASTEXITCODE)"
        return $false
    }
    Write-Host "[multiseed] Seed $Seed v2 done (bridge saved to $savePath)"
    return $true
}

# Wait for seed 42 first
Wait-Seed42 | Out-Null

# Read seed 42 v2 result to decide whether multi-seed is worth it
$seed42 = Get-Content "$OutDir\seed42_v2.json" -Raw | ConvertFrom-Json
$pass = $seed42.n_pass
$total = $seed42.n_words
Write-Host ""
Write-Host "[multiseed] Seed 42 v2 result: $pass/$total cross-category PASS"

# 12 words total in v2; threshold scales (was 5/10 v1, now 6/12 v2)
$threshold = [math]::Ceiling($total * 0.5)
if ($pass -lt $threshold) {
    Write-Host "[multiseed] Seed 42 v2 below $threshold/$total threshold. Skipping multi-seed."
    Write-Host "[multiseed] Architecture needs refinement before multi-seed validation."
    exit 0
}

Write-Host "[multiseed] Seed 42 above threshold. Launching seeds 43-46..."
foreach ($seed in 43, 44, 45, 46) {
    Run-Seed -Seed $seed | Out-Null
}

# Aggregate
Write-Host ""
Write-Host "[multiseed] All seeds complete. Aggregating..."
$results = @()
foreach ($seed in 42, 43, 44, 45, 46) {
    $path = "$OutDir\seed${seed}_v2.json"
    if (Test-Path $path) {
        $r = Get-Content $path -Raw | ConvertFrom-Json
        $results += [PSCustomObject]@{
            Seed = $seed
            Pass = $r.n_pass
            Total = $r.n_words
            WallClock = [math]::Round($r.wall_clock_s / 60, 1)
        }
    }
}

$results | Format-Table -AutoSize
Write-Host ""
$means = ($results | Measure-Object Pass -Average).Average
$min = ($results | Measure-Object Pass -Minimum).Minimum
$max = ($results | Measure-Object Pass -Maximum).Maximum
Write-Host "[multiseed] mean $($means.ToString('F1'))/10  range $min-$max"
