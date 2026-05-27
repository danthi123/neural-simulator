# Direction 7 PARALLEL production launcher.
#
# 5 parallel python processes, one per bridge (A_nouns / B_verbs / C_adj /
# D_spatial / E_functional), each running all 3 seeds (42/43/44).
# Each process: --skip-probe (KILL-SAFE per-cell caches written
# independently). After all 5 complete, run the cross-bridge probe once
# from the controller via the existing direction_7_cross_bridge_probe.
#
# Future runs (D8 V=640+, etc) can call this script as a template.
#
# ETA: each parallel process trains 3 cells. Sequential per-cell wall
# ~110 min at V=64 production scale (n_lang=4096, n_per_pool=200,
# events=200) -> 5.5 hr per process if compute scales perfectly. Realistic
# 3-4x speedup with 5 parallel processes on RTX 3090 (24 GB) -> ~7-9 hr
# wall for all 15 cells. Compared to ~27-32 hr sequential.
#
# Usage:
#   pwsh research/findings/raw/direction_7_parallel_production.ps1
#
# Discipline:
# - Each child process is isolated; no shared state except per-cell cache.
# - KILL-SAFE: re-launching skips any cell whose cache already exists.
# - No protected/frozen/moat module touched.

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path "$PSScriptRoot/../../..").Path
Set-Location $RepoRoot

$Bridges = @("A_nouns", "B_verbs", "C_adj", "D_spatial", "E_functional")
$Seeds = @("42", "43", "44")
$CacheDir = "research/findings/raw/direction_7_cache"
$LogDir = "research/findings/raw/direction_7_parallel_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$StartTime = Get-Date
Write-Host "[D7-parallel-launcher] Starting at $($StartTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host "[D7-parallel-launcher] Bridges: $($Bridges -join ', ')"
Write-Host "[D7-parallel-launcher] Seeds: $($Seeds -join ', ')"
Write-Host "[D7-parallel-launcher] Cache dir: $CacheDir"
Write-Host ""

# Launch 5 parallel processes, one per bridge.
$Procs = @()
foreach ($bridge in $Bridges) {
    $logPath = Join-Path $LogDir "$bridge.log"
    $errPath = Join-Path $LogDir "$bridge.err"
    $outJson = "research/findings/raw/direction_7_5bridge_production_$bridge.json"

    Write-Host "[D7-parallel-launcher] Launching $bridge (all seeds) -> log $logPath"
    $proc = Start-Process -FilePath "python" `
        -ArgumentList "-u", "-m", "research.findings.raw.direction_7_5bridge_runner",
                      "--bridges", $bridge,
                      "--seeds", "42", "43", "44",
                      "--skip-probe",
                      "--out", $outJson `
        -RedirectStandardOutput $logPath `
        -RedirectStandardError $errPath `
        -WorkingDirectory $RepoRoot `
        -NoNewWindow `
        -PassThru
    $Procs += @{ Bridge = $bridge; Process = $proc; LogPath = $logPath; OutJson = $outJson }
    Start-Sleep -Seconds 2  # small stagger to avoid GPU init thundering herd
}

Write-Host ""
Write-Host "[D7-parallel-launcher] All 5 processes launched. PIDs:"
foreach ($p in $Procs) { Write-Host "  $($p.Bridge): PID $($p.Process.Id)" }
Write-Host ""

# Poll until all complete (or timeout at 24 hr safety)
$pollSec = 120
$timeoutHr = 24
Write-Host "[D7-parallel-launcher] Polling every $pollSec sec; safety timeout $timeoutHr hr."

while ($true) {
    $running = $Procs | Where-Object { -not $_.Process.HasExited }
    $done = $Procs | Where-Object { $_.Process.HasExited }
    $elapsed = ((Get-Date) - $StartTime).TotalMinutes

    if ($running.Count -eq 0) {
        Write-Host "[D7-parallel-launcher] All 5 processes complete at $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss')); wall $([Math]::Round($elapsed, 1)) min"
        break
    }
    if ($elapsed -gt ($timeoutHr * 60)) {
        Write-Host "[D7-parallel-launcher] TIMEOUT after $timeoutHr hr; $($running.Count) processes still running."
        break
    }

    # Status line every poll
    $runningNames = ($running | ForEach-Object { $_.Bridge }) -join ", "
    Write-Host "[D7-parallel-launcher] $([Math]::Round($elapsed, 1)) min: $($done.Count)/5 done, running: $runningNames"
    Start-Sleep -Seconds $pollSec
}

# Verify all expected cells exist
$expectedCells = @()
foreach ($b in $Bridges) {
    foreach ($s in $Seeds) {
        $expectedCells += "$CacheDir/activity_full_${b}_seed${s}.npz"
    }
}
$missing = $expectedCells | Where-Object { -not (Test-Path $_) }
if ($missing.Count -gt 0) {
    Write-Host "[D7-parallel-launcher] MISSING $($missing.Count) expected cells:"
    $missing | ForEach-Object { Write-Host "  $_" }
    Set-Content -Path "research/findings/raw/direction_7_parallel_blocked.txt" `
        -Value "Missing cells after parallel run: $($missing -join '; ')" -Encoding ASCII
    exit 5
}

Write-Host ""
Write-Host "[D7-parallel-launcher] All 15 cells exist. Running cross-bridge probe..."

# Run the cross-bridge probe once (CPU-only)
$probeOut = "research/findings/raw/direction_7_cross_bridge_production.json"
$probeLog = "research/findings/raw/direction_7_cross_bridge_production.log"
& python -u -m research.findings.raw.direction_7_cross_bridge_probe `
    --seeds 42 43 44 --cache-dir $CacheDir --out $probeOut 2>&1 | Tee-Object -FilePath $probeLog

Write-Host ""
Write-Host "[D7-parallel-launcher] Cross-bridge probe done. Verdict in $probeOut"
$probe = Get-Content $probeOut -Raw | ConvertFrom-Json
Write-Host "[D7-parallel-launcher] Verdict: $($probe.verdict)"
foreach ($loadKey in @("2", "3", "5")) {
    $cell = $probe.aggregate.$loadKey
    if ($null -ne $cell) {
        Write-Host ("  L={0}: OB mean={1:F3}  OI mean={2:F3}" -f $loadKey, $cell.order_bearing_mean, $cell.order_invariant_mean)
    }
}

$totalElapsed = ((Get-Date) - $StartTime).TotalMinutes
Write-Host "[D7-parallel-launcher] TOTAL WALL: $([Math]::Round($totalElapsed, 1)) min"
exit 0
