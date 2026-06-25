# Develop-loop TRAINING watchdog — keeps the 24/7 foundational-curriculum training alive.
#
# THE OWNER'S BAR: "Run it 24/7 in a fashion that won't get accidentally killed no matter what, but can be
# paused when I want to game and recover without too much lost in the event of a crash, restart, etc."
#
# This is the supervisor-of-the-supervisor. It is itself the long-lived process (launch it run-DETACHED via
# scripts/start_develop_training.ps1 so it survives a terminal close). In a loop it:
#   1. If the PAUSE sentinel exists -> the owner wants the GPU (gaming). Do NOT run training. Sleep, re-check.
#   2. Else, run `research.runners.develop_loop_supervisor` IN THE FOREGROUND (blocking) — that process is the
#      training. It persists+fsyncs the lineage every day and resumes on every launch.
#   3. When it exits:
#        - exit code 42  => the supervisor saw the PAUSE sentinel and stopped CLEANLY. Do NOT restart; loop back
#          to the pause-wait.
#        - any other code (crash, OOM, hard kill of the child, transient error) => RESTART it (it resumes from
#          the last durably-persisted day; AT MOST the in-flight day is re-done). A short backoff prevents a
#          tight crash-loop.
#
# Distinct from scripts/autonomous_watchdog.ps1 (which re-invokes the Claude CLI). THIS watchdog owns the
# TRAINING process specifically and respects the PAUSE sentinel.
#
# Usage (foreground, for testing the loop logic):
#   pwsh scripts/develop_train_watchdog.ps1 -LineageRoot bridges/developed/curriculum `
#        -BundleRoot bridges/developed/curriculum/bundles -PerDayBundles -MaxWindowsPerDay 2500
#
# Usage (the real 24/7 deploy = launch it DETACHED so it survives terminal close):
#   pwsh scripts/start_develop_training.ps1     # wraps this in a detached, terminal-independent process
#
# PAUSE / RESUME (owner):
#   New-Item -ItemType File bridges/PAUSE       # pause (training stops cleanly after the current day; GPU freed)
#   Remove-Item bridges/PAUSE                   # resume (the watchdog relaunches training, resuming the brain)

param(
  [string]$LineageRoot   = "bridges/developed/curriculum",
  [string]$LineageName   = "curriculum",
  [int]$Seed             = 42,
  [int]$MaxWindowsPerDay = 2500,
  [int]$NHub             = 200,
  [int]$NPer             = 12,
  [int]$D                = 128,
  [string]$BundleRoot    = "",
  [switch]$PerDayBundles,
  [string]$CorpusPath    = "",
  [string]$PauseFile     = "bridges/PAUSE",
  [string]$Backend       = "cupy",
  [int]$BackoffSeconds   = 15,        # backoff after a crash before relaunch (prevents a tight crash-loop)
  [int]$PausePollSeconds = 30,        # how often to re-check the PAUSE sentinel while paused
  [int]$MaxRestarts      = 0,         # 0 = unlimited (24/7). >0 caps restarts (for testing).
  [double]$MaxRuntimeS   = 0          # 0 = no per-invocation runtime cap (the supervisor recycles itself if set)
)

$ErrorActionPreference = "Stop"
$Repo   = "E:\Documents\Projects\sim"
$Python = "python"
$Log    = Join-Path $Repo "research\findings\raw\develop_train_watchdog.log"
$PAUSE_EXIT_CODE = 42

function Note($m) {
  $line = "[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $m
  Write-Output $line
  try { Add-Content -Path $Log -Value $line -ErrorAction SilentlyContinue } catch {}
}

# Resolve the pause sentinel to an absolute path so we test the SAME file the supervisor polls.
$PauseAbs = if ([System.IO.Path]::IsPathRooted($PauseFile)) { $PauseFile } else { Join-Path $Repo $PauseFile }

Note ("watchdog start: lineage=$LineageRoot/$LineageName backend=$Backend max_windows/day=$MaxWindowsPerDay " +
      "pause_file=$PauseAbs")

# Build the supervisor argument list once.
$supArgs = @(
  "-u", "-m", "research.runners.develop_loop_supervisor",
  "--lineage-root", $LineageRoot, "--lineage-name", $LineageName,
  "--seed", $Seed, "--max-windows-per-day", $MaxWindowsPerDay,
  "--n-hub", $NHub, "--n-per", $NPer, "--D", $D,
  "--pause-file", $PauseFile
)
if ($BundleRoot -ne "")   { $supArgs += @("--bundle-root", $BundleRoot) }
if ($PerDayBundles)       { $supArgs += "--per-day-bundles" }
if ($CorpusPath -ne "")   { $supArgs += @("--corpus-path", $CorpusPath) }
if ($MaxRuntimeS -gt 0)   { $supArgs += @("--max-runtime-s", $MaxRuntimeS) }

$env:SIM_BACKEND = $Backend
$restarts = 0

while ($true) {
  # 1. Honor PAUSE: while the sentinel exists, do not run training (the owner has the GPU).
  if (Test-Path $PauseAbs) {
    Note ("PAUSE sentinel present -> training held (gaming). Re-check in ${PausePollSeconds}s.")
    Start-Sleep -Seconds $PausePollSeconds
    continue
  }

  # 2. Run the training supervisor in the FOREGROUND (blocking). It is the training; it persists every day.
  Note ("launching training supervisor (restart #$restarts)")
  $exit = -999
  try {
    # Set-Location so `-m research.runners...` resolves; the supervisor inherits SIM_BACKEND.
    Push-Location $Repo
    & $Python @supArgs
    $exit = $LASTEXITCODE
  } catch {
    Note ("supervisor launch error: {0}" -f $_.Exception.Message)
    $exit = -1
  } finally {
    Pop-Location
  }
  Note ("supervisor exited code=$exit")

  # 3. Decide: pause-exit => do NOT restart (loop back to pause-wait); else => restart from the last checkpoint.
  if ($exit -eq $PAUSE_EXIT_CODE) {
    Note ("clean PAUSE/runtime-cap exit -> not restarting; returning to pause-wait")
    # If it was a pause, the sentinel is set and the top-of-loop pause-wait handles it. If it was a runtime cap
    # (no sentinel), loop straight back and relaunch (lossless resume).
    if (-not (Test-Path $PauseAbs)) { Note "runtime-cap recycle -> relaunching immediately"; continue }
    continue
  }

  # crash / unexpected exit / hard-kill of the child -> resume after a short backoff.
  $restarts++
  if ($MaxRestarts -gt 0 -and $restarts -ge $MaxRestarts) {
    Note ("reached MaxRestarts=$MaxRestarts -> watchdog stopping (testing cap)")
    break
  }
  Note ("non-pause exit (crash/kill) -> resuming from last checkpoint after ${BackoffSeconds}s backoff")
  Start-Sleep -Seconds $BackoffSeconds
}

Note "watchdog exit"
