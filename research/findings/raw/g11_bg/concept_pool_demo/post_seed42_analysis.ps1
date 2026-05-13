# post-seed42 analysis: wait for completion, then dispatch next action
# Per autonomous-runs principle #2: dispatch the next thing automatically.

$ErrorActionPreference = "Stop"
$SimRoot = "E:\Documents\Projects\sim"
$OutDir = "$SimRoot\research\findings\raw\g11_bg\concept_pool_demo"
Set-Location $SimRoot

# Wait for seed 42 JSON
$jsonPath = "$OutDir\seed42.json"
$logPath = "$OutDir\seed42.log"

Write-Host "[post-seed42] Waiting for seed42.json..."
while (-not (Test-Path $jsonPath)) {
    Start-Sleep -Seconds 30
    $progress = (Select-String -Path $logPath -Pattern "^  trained '" -ErrorAction SilentlyContinue).Count
    Write-Host "[post-seed42] still waiting... $progress/10 words trained"
}

# Allow a few seconds for the JSON to fully write
Start-Sleep -Seconds 3

# Parse result
$result = Get-Content $jsonPath -Raw | ConvertFrom-Json
$pass = $result.n_pass
$total = $result.n_words
$wallMin = [math]::Round($result.wall_clock_s / 60, 1)

Write-Host ""
Write-Host "=" * 60
Write-Host "[post-seed42] SEED 42 RESULT: $pass/$total in $wallMin min"
Write-Host "=" * 60

# Per-word breakdown
Write-Host ""
Write-Host "Per-word results:"
foreach ($word in $result.results.PSObject.Properties.Name) {
    $r = $result.results.$word
    $marker = if ($r.target_rate -gt $r.max_off_target) { "PASS" } else { "FAIL" }
    $ratio = [math]::Round($r.target_rate / [math]::Max($r.max_off_target, 0.001), 2)
    Write-Host "  $word -> $($r.target): target=$([math]::Round($r.target_rate, 3))  max_off=$([math]::Round($r.max_off_target, 3))  ratio=${ratio}x  [$marker]"
}

# Decide next action
Write-Host ""
if ($pass -ge 8) {
    Write-Host "[post-seed42] VERDICT: GO ($pass/$total). Recommended next step:"
    Write-Host "  1. Launch multi-seed validation (43, 44, 45, 46)"
    Write-Host "  2. Run: powershell $OutDir\launch_multiseed.ps1"
    Write-Host ""
    Write-Host "  After multi-seed PASS: scale to 16+ noun pools for real diversity"
}
elseif ($pass -ge 5) {
    Write-Host "[post-seed42] VERDICT: PARTIAL ($pass/$total). Recommended next step:"
    Write-Host "  1. Diagnose failing words with concept_weight_probe"
    Write-Host "  2. Consider: longer training (400 events), stronger topographic"
    Write-Host "     prior (2.0/0.5), or weak cross-kind FS"
}
else {
    Write-Host "[post-seed42] VERDICT: FAIL ($pass/$total). Architecture needs work:"
    Write-Host "  1. Check structural bias (which pool dominates everything?)"
    Write-Host "  2. Run concept_weight_probe to see if STDP converged"
    Write-Host "  3. Consider per-pool gain rebalancing"
}
