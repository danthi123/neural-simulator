# Direction 8 V=640 speedup-smoke launcher.
#
# Purpose: A/B test the optimization combo (--use-fp16 + reduced
# stim_steps_per_event) against the D7 V=320 smoke baseline to measure
# real per-event speedup BEFORE committing to D8 production.
#
# Pre-staged 2026-05-27. Do NOT launch while D7 production (PID 30216,
# started 18:08:40) is still running -- the parallel-launch NEGATIVE
# finding (commit 49e2d58) shows Windows WDDM time-slices any 2nd CUDA
# process to ~50% throughput, so the smoke wall measurement would be
# contaminated by D7 production contention.
#
# Recommended launch sequence:
#   1. Wait for D7 production complete (~Friday 29 May 22:00 EDT)
#      OR verify GPU is idle:
#        Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
#          Where-Object {$_.CommandLine -match 'direction_7|direction_8'}
#   2. Launch D8 SMOKE with optimization combo:
#        pwsh research/findings/raw/direction_8_speedup_smoke.ps1
#   3. Compare wall vs D7 V=64 smoke baseline:
#        D7 smoke per-cell: 28.5 min (V=64, n_lang=2048, events=50,
#                                       stim_steps=100, fp16=False)
#        D8 unoptimized per-cell predicted: ~57 min (V doubled +
#                                                     n_lang doubled)
#        D8 optimized per-cell goal: <30 min (target 2x speedup)
#   4. If smoke verdict PASS at 0.80 bar AND wall reduction >= 30%,
#      optimization combo validated -> apply to D8 production.
#
# CONFIG (frozen for this A/B test):
#   --smoke (n_lang=4096, n_per_pool=100, events=50, M_OBS=8, V=128)
#   --use-fp16 (cfg.fp16_synapse_state = True; ~0.5% noise per 1000
#                events per sim/config.py; STDP w_max stays fp32)
#   --stim-steps-per-event 50 (halved from D6/D7 default of 100)
#   --reset-steps 50 (unchanged from default; reset must stay >=25
#                      for NMDA decay window per concept_pool_demo.py)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path "$PSScriptRoot/../../..").Path
Set-Location $RepoRoot

# Safety check: refuse to launch if D7 production is still running.
$d7Procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'direction_7_5bridge_runner' }
if ($d7Procs) {
    Write-Host "[D8-speedup-smoke] BLOCKED: D7 production is still running:"
    $d7Procs | ForEach-Object { Write-Host "  PID $($_.ProcessId): $($_.CommandLine.Substring(0, [Math]::Min(120, $_.CommandLine.Length)))" }
    Write-Host "[D8-speedup-smoke] Wait for D7 production to complete (or kill it manually) before launching D8 smoke."
    Write-Host "[D8-speedup-smoke] Parallel CUDA contention on Windows WDDM contaminates wall measurements (per commit 49e2d58 finding)."
    exit 1
}

# Also refuse if any D8 producer process is already running.
$d8Procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'direction_8_5bridge_runner' }
if ($d8Procs) {
    Write-Host "[D8-speedup-smoke] BLOCKED: D8 runner already running:"
    $d8Procs | ForEach-Object { Write-Host "  PID $($_.ProcessId)" }
    Write-Host "[D8-speedup-smoke] Kill manually if intended to restart."
    exit 1
}

$SmokeOut = "research/findings/raw/direction_8_5bridge_smoke_optimized.json"
$SmokeLog = "research/findings/raw/direction_8_5bridge_smoke_optimized.log"

Write-Host "[D8-speedup-smoke] Starting at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "[D8-speedup-smoke] Config: --smoke --use-fp16 --stim-steps-per-event 50 --reset-steps 50"
Write-Host "[D8-speedup-smoke] Out: $SmokeOut"
Write-Host "[D8-speedup-smoke] Log: $SmokeLog"
Write-Host ""

$StartTime = Get-Date
$proc = Start-Process -FilePath "python" `
    -ArgumentList "-u", "-m", "research.findings.raw.direction_8_5bridge_runner",
                  "--smoke",
                  "--seeds", "42", "43", "44",
                  "--use-fp16",
                  "--stim-steps-per-event", "50",
                  "--reset-steps", "50",
                  "--out", $SmokeOut `
    -RedirectStandardOutput $SmokeLog `
    -RedirectStandardError "$SmokeLog.err" `
    -WorkingDirectory $RepoRoot `
    -NoNewWindow `
    -PassThru

Write-Host "[D8-speedup-smoke] D8 SMOKE launched. PID: $($proc.Id)"
Write-Host "[D8-speedup-smoke] Tail log with: Get-Content -Path $SmokeLog -Wait -Tail 20"
Write-Host "[D8-speedup-smoke] Expected baseline (D7 V=64 smoke): 28.5 min/cell x 15 cells = 427 min total"
Write-Host "[D8-speedup-smoke] D8 unoptimized prediction (V=128, n_lang=4096): ~57 min/cell x 15 cells = ~855 min"
Write-Host "[D8-speedup-smoke] D8 optimized target: <30 min/cell -> ~450 min total (2x speedup vs D8 unoptimized; close to D7 smoke wall)"
Write-Host ""
Write-Host "[D8-speedup-smoke] When smoke completes, compare:"
Write-Host "  - verdict in $SmokeOut (must be DIRECTION_8_PASS at 0.80 bar)"
Write-Host "  - training_wall_clock_minutes in $SmokeOut"
Write-Host "  - cross_bridge probe output in research/findings/raw/direction_8_cross_bridge_smoke.json"
exit 0
