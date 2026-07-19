# Direction 7 post-smoke chain — auto-launch production on smoke PASS.
#
# v2 (2026-05-27): robust to two probe-output schemas
#   A) Inline: direction_7_5bridge_smoke.json has non-null probe_result
#   B) Separate: direction_7_cross_bridge_smoke.json written by a separate
#      probe invocation (matches D6 actual smoke pattern)
#
# Schema (verified against D6 cross_bridge probe output):
#   verdict (top-level): "DIRECTION_7_PASS" / "_PARTIAL" / "_NEGATIVE" / "_VOID_MALFORMED"
#   aggregate (dict keyed by load number as string "2"/"3"/"5"):
#     order_bearing_mean (float)
#     order_invariant_mean (float)
#     order_{bearing,invariant}_per_seed (list of floats)
#
# Logic:
#   1. Wait for direction_7_5bridge_smoke.json to appear.
#   2. Try inline probe_result first; fall back to direction_7_cross_bridge_smoke.json.
#   3. If verdict is PASS, or PARTIAL with >=3/6 cells above 0.80 bar:
#      launch D7 PRODUCTION decisive (KILL-SAFE per-cell caches).
#   4. Otherwise: write blocked status, exit non-zero.

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path "$PSScriptRoot/../../..").Path
Set-Location $RepoRoot

$SmokeJson = "research/findings/raw/direction_7_5bridge_smoke.json"
$CrossBridgeSmokeJson = "research/findings/raw/direction_7_cross_bridge_smoke.json"
$ProductionJson = "research/findings/raw/direction_7_5bridge_production.json"
$ProductionLog = "research/findings/raw/direction_7_5bridge_production.log"
$StatusBlocked = "research/findings/raw/direction_7_post_smoke_blocked.txt"

Write-Host "[D7-post-smoke-chain-v2] Starting watcher at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "[D7-post-smoke-chain-v2] Waiting for $SmokeJson..."

$pollSec = 60
$startTime = Get-Date

while (-not (Test-Path $SmokeJson)) {
    $elapsed = ((Get-Date) - $startTime).TotalMinutes
    if ($elapsed -gt 600) {
        Write-Host "[D7-post-smoke-chain-v2] TIMEOUT after 10 hr; smoke never completed."
        Set-Content -Path $StatusBlocked -Value "TIMEOUT after 10 hr at $(Get-Date)" -Encoding ASCII
        exit 2
    }
    Start-Sleep -Seconds $pollSec
}

Write-Host "[D7-post-smoke-chain-v2] $SmokeJson exists; parsing verdict..."

$smoke = Get-Content $SmokeJson -Raw | ConvertFrom-Json
$probe = $smoke.probe_result

if ($null -eq $probe) {
    Write-Host "[D7-post-smoke-chain-v2] probe_result null in 5bridge file; checking separate cross_bridge file..."
    # Wait up to 5 minutes for separate cross_bridge probe file to appear
    # (the smoke runner may have launched the probe as a sub-process)
    $crossWait = 0
    while (-not (Test-Path $CrossBridgeSmokeJson) -and $crossWait -lt 300) {
        Start-Sleep -Seconds 30
        $crossWait += 30
    }
    if (Test-Path $CrossBridgeSmokeJson) {
        Write-Host "[D7-post-smoke-chain-v2] Found separate $CrossBridgeSmokeJson; using as probe source."
        $probe = Get-Content $CrossBridgeSmokeJson -Raw | ConvertFrom-Json
    } else {
        Write-Host "[D7-post-smoke-chain-v2] BLOCK: no probe output (inline or separate) after smoke completed."
        Set-Content -Path $StatusBlocked -Value "No probe output (inline OR separate cross_bridge file) found after smoke completed at $(Get-Date)" -Encoding ASCII
        exit 3
    }
}

$verdict = $probe.verdict
if ($null -eq $verdict) {
    Write-Host "[D7-post-smoke-chain-v2] BLOCK: probe output missing verdict field."
    Set-Content -Path $StatusBlocked -Value "verdict null in probe output at $(Get-Date)" -Encoding ASCII
    exit 4
}

Write-Host "[D7-post-smoke-chain-v2] Smoke verdict: $verdict"

# Compute aggregate cell PASS count (aggregate is dict keyed by load number as string).
$nCellsAbove = 0
$nCellsTotal = 0
if ($null -ne $probe.aggregate) {
    foreach ($loadKey in @("2", "3", "5")) {
        $cell = $probe.aggregate.$loadKey
        if ($null -ne $cell) {
            $nCellsTotal += 2
            if ($cell.order_bearing_mean -ge 0.80) { $nCellsAbove++ }
            if ($cell.order_invariant_mean -ge 0.80) { $nCellsAbove++ }
        }
    }
}
Write-Host "[D7-post-smoke-chain-v2] aggregate cells above 0.80 bar: $nCellsAbove / $nCellsTotal"

if ($verdict -eq "DIRECTION_7_PASS") {
    $launchOk = $true
    $reason = "smoke PASS multi-seed (all $nCellsTotal cells above bar)"
} elseif ($verdict -eq "DIRECTION_7_PARTIAL" -and $nCellsAbove -ge 3) {
    $launchOk = $true
    $reason = "smoke PARTIAL with $nCellsAbove/$nCellsTotal cells above bar"
} else {
    $launchOk = $false
    $reason = "verdict=$verdict; nCellsAbove=$nCellsAbove. Production NOT launched."
}

Write-Host "[D7-post-smoke-chain-v2] launch_decision: $($launchOk); reason: $reason"

if (-not $launchOk) {
    Set-Content -Path $StatusBlocked -Value "verdict=$verdict; n_cells_above=$nCellsAbove. $reason" -Encoding ASCII
    exit 5
}

if (Test-Path $ProductionJson) {
    Write-Host "[D7-post-smoke-chain-v2] Production result already exists; not re-launching."
    exit 0
}

Write-Host "[D7-post-smoke-chain-v2] Launching D7 PRODUCTION decisive in background..."
Write-Host "[D7-post-smoke-chain-v2] ETA ~27-32 hr GPU on CuPy/RTX 3090; KILL-SAFE per-cell caches."

Start-Process -FilePath "python" `
    -ArgumentList "-u", "-m", "research.findings.raw.direction_7_5bridge_runner", "--seeds", "42", "43", "44", "--out", $ProductionJson `
    -RedirectStandardOutput $ProductionLog `
    -RedirectStandardError "$ProductionLog.err" `
    -WorkingDirectory $RepoRoot `
    -NoNewWindow `
    -PassThru | Out-Null

Write-Host "[D7-post-smoke-chain-v2] Production launched at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "[D7-post-smoke-chain-v2] Log: $ProductionLog"
exit 0
