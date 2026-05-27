# Direction 7 post-smoke chain — auto-launch production on smoke PASS.
#
# Pre-staged 2026-05-27 to run automatically when D7 SMOKE completes.
# Logic:
#   1. Wait for research/findings/raw/direction_7_5bridge_smoke.json to appear.
#   2. Parse smoke verdict from probe_result.verdict field.
#   3. If verdict is DIRECTION_7_PASS OR DIRECTION_7_PARTIAL with any L cell
#      above the 0.80 bar:
#        - Launch D7 PRODUCTION decisive (--full / no --smoke flag).
#        - Production has its OWN KILL-SAFE per-cell caches.
#   4. If verdict is DIRECTION_7_NEGATIVE OR DIRECTION_7_VOID_MALFORMED:
#        - Do NOT launch production (controller manually inspects).
#        - Write status_blocked.txt with details.
#
# Usage:
#   pwsh research/findings/raw/direction_7_post_smoke_chain.ps1
#
# Discipline:
# - Polls smoke completion every 60s (not aggressively).
# - Does NOT touch any protected/frozen/moat module.
# - Does NOT re-run smoke if smoke result already exists.
# - All output ASCII-only.

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path "$PSScriptRoot/../../..").Path
Set-Location $RepoRoot

$SmokeJson = "research/findings/raw/direction_7_5bridge_smoke.json"
$ProductionJson = "research/findings/raw/direction_7_5bridge_production.json"
$ProductionLog = "research/findings/raw/direction_7_5bridge_production.log"
$StatusBlocked = "research/findings/raw/direction_7_post_smoke_blocked.txt"

Write-Host "[D7-post-smoke-chain] Starting watcher at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "[D7-post-smoke-chain] Waiting for $SmokeJson..."

$pollSec = 60
$startTime = Get-Date

while (-not (Test-Path $SmokeJson)) {
    $elapsed = ((Get-Date) - $startTime).TotalMinutes
    if ($elapsed -gt 600) {
        # 10 hr safety bound; if smoke hasn't completed in 10 hr something is wrong.
        Write-Host "[D7-post-smoke-chain] TIMEOUT after 10 hr; smoke never completed. Inspect manually."
        Set-Content -Path $StatusBlocked -Value "TIMEOUT after 10 hr at $(Get-Date)" -Encoding ASCII
        exit 2
    }
    Start-Sleep -Seconds $pollSec
}

Write-Host "[D7-post-smoke-chain] $SmokeJson exists; parsing verdict..."

$smoke = Get-Content $SmokeJson -Raw | ConvertFrom-Json
$probe = $smoke.probe_result

if ($null -eq $probe) {
    Write-Host "[D7-post-smoke-chain] BLOCK: smoke ran but probe_result is null (probe skipped or failed)."
    Set-Content -Path $StatusBlocked -Value "probe_result null in smoke at $(Get-Date)" -Encoding ASCII
    exit 3
}

$verdict = $probe.verdict
if ($null -eq $verdict) {
    Write-Host "[D7-post-smoke-chain] BLOCK: smoke probe_result missing verdict field."
    Set-Content -Path $StatusBlocked -Value "verdict null in smoke probe at $(Get-Date)" -Encoding ASCII
    exit 4
}

Write-Host "[D7-post-smoke-chain] Smoke verdict: $verdict"

# Compute aggregate cell PASS count for diagnosis (regardless of verdict tag).
$nCellsAbove = 0
if ($null -ne $probe.aggregate -and $null -ne $probe.aggregate.per_load) {
    foreach ($load in @("L=2", "L=3", "L=5")) {
        $cell = $probe.aggregate.per_load.$load
        if ($null -ne $cell) {
            if ($cell.OB -ge 0.80) { $nCellsAbove++ }
            if ($cell.OI -ge 0.80) { $nCellsAbove++ }
        }
    }
}
Write-Host "[D7-post-smoke-chain] aggregate cells above 0.80 bar: $nCellsAbove / 6"

# Launch production if smoke is PASS or PARTIAL with reasonable cell count.
# NEGATIVE / VOID -> manual inspection required.
if ($verdict -eq "DIRECTION_7_PASS") {
    $launchOk = $true
    $reason = "smoke PASS multi-seed"
} elseif ($verdict -eq "DIRECTION_7_PARTIAL" -and $nCellsAbove -ge 3) {
    $launchOk = $true
    $reason = "smoke PARTIAL with $nCellsAbove/6 cells above bar (sufficient signal to commit to production)"
} else {
    $launchOk = $false
    $reason = "verdict=$verdict; nCellsAbove=$nCellsAbove. Production NOT launched; manual inspection needed."
}

Write-Host "[D7-post-smoke-chain] launch_decision: $($launchOk); reason: $reason"

if (-not $launchOk) {
    Set-Content -Path $StatusBlocked -Value "verdict=$verdict; n_cells_above=$nCellsAbove. $reason" -Encoding ASCII
    exit 5
}

if (Test-Path $ProductionJson) {
    Write-Host "[D7-post-smoke-chain] Production result already exists; not re-launching."
    exit 0
}

Write-Host "[D7-post-smoke-chain] Launching D7 PRODUCTION decisive in background..."
Write-Host "[D7-post-smoke-chain] Command: python -u -m research.findings.raw.direction_7_5bridge_runner --seeds 42 43 44 --out $ProductionJson"
Write-Host "[D7-post-smoke-chain] ETA ~27-32 hr GPU on CuPy/RTX 3090; KILL-SAFE per-cell caches."

# Launch as detached background process; log to direction_7_5bridge_production.log
$cmd = "python -u -m research.findings.raw.direction_7_5bridge_runner --seeds 42 43 44 --out $ProductionJson"
Start-Process -FilePath "python" `
    -ArgumentList "-u", "-m", "research.findings.raw.direction_7_5bridge_runner", "--seeds", "42", "43", "44", "--out", $ProductionJson `
    -RedirectStandardOutput $ProductionLog `
    -RedirectStandardError "$ProductionLog.err" `
    -WorkingDirectory $RepoRoot `
    -NoNewWindow `
    -PassThru | Out-Null

Write-Host "[D7-post-smoke-chain] Production launched at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "[D7-post-smoke-chain] Log: $ProductionLog"
exit 0
