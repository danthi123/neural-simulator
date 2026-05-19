# Autonomous continuation watchdog (LOCAL, GPU-capable).
#
# Concrete anti-stall guarantee: a Windows Scheduled Task runs this on an
# interval. If the autonomous research arc has genuinely stalled (no new git
# commit for > StalledMinutes AND no GPU sim / claude process active AND no
# fresh watchdog lock), it re-invokes the local Claude CLI headless on the
# repo with a continuation prompt that reads research/findings/AUTONOMOUS_STATE.md
# and continues the exact next concrete action -- including GPU runs, because
# this runs on the owner's machine (unlike the remote cloud safety-net routine).
#
# Conservative by design: it must NEVER collide with a healthy session or an
# in-flight background subagent. Stall is inferred only from a long commit gap
# plus no active python(sim)/claude process plus no recent lock.
#
# Usage:
#   pwsh scripts/autonomous_watchdog.ps1 -DryRun     # print decision only
#   pwsh scripts/autonomous_watchdog.ps1             # act if stalled
#
# Registered as a Scheduled Task (see scripts/register_watchdog.ps1 logic in
# AUTONOMOUS_STATE.md). The owner authorized 24/7 unattended autonomy and
# explicitly required this guarantee.

param(
  [int]$StalledMinutes = 40,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Repo   = "E:\Documents\Projects\sim"
$Claude = "C:\Users\dant123\.local\bin\claude.exe"
$Lock   = Join-Path $Repo ".watchdog.lock"
$Log    = Join-Path $Repo "research\findings\raw\autonomous_watchdog.log"
$now    = Get-Date

function Note($m) {
  $line = "[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f $now, $m
  Write-Output $line
  try { Add-Content -Path $Log -Value $line -ErrorAction SilentlyContinue } catch {}
}

# --- 1. commit recency (the primary health signal: the arc commits often) ---
$lastCommitUnix = [int](git -C $Repo log -1 --format=%ct).Trim()
$ageMin = [math]::Round((([DateTimeOffset]$now).ToUnixTimeSeconds() - $lastCommitUnix) / 60.0, 1)

# --- 2. active work signals (do NOT fire if real work is in flight) ---
$claudeProcs = @(Get-Process claude -ErrorAction SilentlyContinue)
# python sim using real memory (a GPU run) = work in flight
$pyBusy = @(Get-Process python -ErrorAction SilentlyContinue |
            Where-Object { $_.WorkingSet64 -gt 800MB })

# --- 3. fresh lock (a prior watchdog launch still plausibly working) ---
$lockFresh = $false
if (Test-Path $Lock) {
  $lockAgeMin = ((Get-Date) - (Get-Item $Lock).LastWriteTime).TotalMinutes
  if ($lockAgeMin -lt 90) { $lockFresh = $true }
}

$stalled = ($ageMin -ge $StalledMinutes) -and
           ($claudeProcs.Count -eq 0) -and
           ($pyBusy.Count -eq 0) -and
           (-not $lockFresh)

Note ("check: commit_age={0}min threshold={1} claude_procs={2} py_busy={3} lock_fresh={4} -> stalled={5}" -f `
      $ageMin, $StalledMinutes, $claudeProcs.Count, $pyBusy.Count, $lockFresh, $stalled)

if (-not $stalled) { Note "no action (healthy / work in flight / lock fresh)"; exit 0 }
if ($DryRun)       { Note "DRYRUN: would re-invoke local Claude continuation now"; exit 0 }

# --- 4. re-invoke local Claude headless to continue the arc (GPU-capable) ---
Set-Content -Path $Lock -Value ("{0:o}" -f $now)
$prompt = @'
Autonomous continuation (local watchdog re-trigger). The session stalled.
Read research/findings/AUTONOMOUS_STATE.md NOW and continue the exact next
concrete action it specifies, including any PENDING-LOCAL-GPU step (you ARE
on the owner's machine with the RTX 3090 / CuPy). HARD RULES: never end on a
future-tense promise (the next-action tool call is in the same turn); no
self-imposed stop/checkpoints; GPU/CuPy for real runs, numpy only for tiny
smoke; never weaken a frozen bar or the no-confabulation moat; reuse-by-import
only; honest propagation of every outcome to BOTH git remotes (origin+gitea);
iterate following the project reference biology, no hand-back, no declare-unfit.
Update research/findings/AUTONOMOUS_STATE.md every cycle.
'@
Note "STALLED -> launching local Claude continuation"
try {
  & $Claude -p $prompt --permission-mode bypassPermissions --add-dir $Repo 2>&1 |
    ForEach-Object { Note "claude> $_" }
  Note ("claude exited code={0}" -f $LASTEXITCODE)
} catch {
  Note ("claude launch error: {0}" -f $_.Exception.Message)
} finally {
  Remove-Item $Lock -ErrorAction SilentlyContinue
}
exit 0
