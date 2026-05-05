# Polls bio-three-factor-orthogonal master log for COMPLETE marker,
# aggregates results, then auto-launches step 3 32×32 gridworld smoke
# (GPU will be free after orthogonal completes).
#
# Created 2026-05-05 in the post-verdict autonomous arc:
#   step 1 ✅ gradient validates (3/3 NESW aligned)
#   step 2a ✅ rule not LR-limited (0/3 at 5x AND 10x)
#   step 2b 🔄 orthogonal cues sweep (this script polls for finish)
#   step 3 → 32×32 smoke test (this script auto-launches)
#   step 4 → conditional dendritic learning Week 0

$outDir = "research/findings/raw/g11_bg"
$findingsDir = "research/findings"
$logFile = "$outDir/post_orthogonal_step3.log"
$orthMaster = "$outDir/bio-three-factor-orthogonal.master.log"

"=== Post-orthogonal step3 orchestrator started $(Get-Date) ===" | Out-File -FilePath $logFile

# Step A: poll for orthogonal completion
"Polling for bio-three-factor-orthogonal COMPLETE..." | Out-File -Append $logFile
while ($true) {
    if (Test-Path $orthMaster) {
        $text = Get-Content $orthMaster -Raw -ErrorAction SilentlyContinue
        if ($text -match "bio-three-factor-orthogonal COMPLETE") {
            "Orthogonal sweep done at $(Get-Date)" | Out-File -Append $logFile
            break
        }
    }
    Start-Sleep -Seconds 60
}

# 30s grace for last seed's JSON to flush
Start-Sleep -Seconds 30

# Step B: run permuted-label control on orthogonal results
"" | Out-File -Append $logFile
"Aggregating orthogonal results at $(Get-Date)..." | Out-File -Append $logFile

$orthOut = & "C:\python312\python.exe" -m research.runners.permuted_label_check `
    --pattern "text_eval_3factor_tf_orthogonal_seed*.json" 2>&1 | Out-String
"$orthOut" | Out-File -Append $logFile

# Parse aligned ratio (look for "X/6" pattern in summary table)
$alignedN = 0
foreach ($line in ($orthOut -split "`n")) {
    if ($line -match "tf_orthogonal\s*\|\s*6\s*\|.*?\|.*?\|.*?\|\s*(\d+)/6") {
        $alignedN = [int]$Matches[1]
        break
    }
}
"Orthogonal aligned: $alignedN/6" | Out-File -Append $logFile

# Step C: GPU-idle gate before step 3 smoke
"" | Out-File -Append $logFile
"Waiting for GPU-idle (10% util for 5 min consecutive)..." | Out-File -Append $logFile
$consecutiveIdle = 0
$idleTarget = 10
$idleNeeded = 10  # 10 checks × 30s = 5 min
while ($consecutiveIdle -lt $idleNeeded) {
    Start-Sleep -Seconds 30
    try {
        $util = (& nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) -as [int]
    } catch { $util = 99 }
    if ($util -lt $idleTarget) {
        $consecutiveIdle++
        "  GPU util ${util}% — idle ($consecutiveIdle/$idleNeeded)" | Out-File -Append $logFile
    } else {
        $consecutiveIdle = 0
    }
}
"GPU idle confirmed at $(Get-Date)" | Out-File -Append $logFile

# Step D: run 32×32 smoke test (single seed)
"" | Out-File -Append $logFile
"Running 32×32 smoke test at $(Get-Date)..." | Out-File -Append $logFile

$smokeOutFile = "$outDir/scale_32x32_seed42.json"
$smokeProc = Start-Process -FilePath "C:\python312\python.exe" `
    -ArgumentList "-u","-m","research.runners.g11_bg_runner",
                  "--moving-goal","--goal-schedule","multi","--deterministic",
                  "--enable-msn-lateral-inhibition","--enable-d1-d2-asymmetry",
                  "--enable-striatal-pv-fsi","--enable-cluster-a-closed-loop",
                  "--enable-cluster-e-topography","--enable-dlpfc-wm",
                  "--enable-pfc-nmda","--enable-visual-cortex",
                  "--visual-cortex-action-warmup-steps","600",
                  "--grid-size","32","--seed","42","--n-steps","1800",
                  "--out",$smokeOutFile `
    -RedirectStandardOutput "$outDir/scale_32x32_seed42.log" `
    -RedirectStandardError "$outDir/scale_32x32_seed42.log.err" `
    -WindowStyle Hidden -PassThru
"Smoke test launched as PID $($smokeProc.Id)" | Out-File -Append $logFile
$smokeProc.WaitForExit()
"Smoke test finished (exit $($smokeProc.ExitCode)) at $(Get-Date)" | Out-File -Append $logFile

# Step E: report decision context for tomorrow
"" | Out-File -Append $logFile
"=== Decision context for user ===" | Out-File -Append $logFile
if ($alignedN -ge 4) {
    "*** ORTHOGONAL CUES RESCUE: $alignedN/6 aligned ***" | Out-File -Append $logFile
    "*** Input encoding ambiguity was the W→A bottleneck. ***" | Out-File -Append $logFile
    "*** Dendritic learning may NOT be needed for this task. ***" | Out-File -Append $logFile
} elseif ($alignedN -ge 2) {
    "*** PARTIAL: $alignedN/6 aligned ***" | Out-File -Append $logFile
    "*** Some input ambiguity helped, but not full rescue. ***" | Out-File -Append $logFile
    "*** Mixed signal — re-evaluate before dendritic. ***" | Out-File -Append $logFile
} else {
    "*** NO RESCUE: $alignedN/6 aligned ***" | Out-File -Append $logFile
    "*** Input encoding NOT the bottleneck. ***" | Out-File -Append $logFile
    "*** Rule fundamentally inadequate at biological scale. ***" | Out-File -Append $logFile
    "*** Dendritic learning Week 1+ now well-justified. ***" | Out-File -Append $logFile
}

"" | Out-File -Append $logFile
"32×32 smoke result: $smokeOutFile" | Out-File -Append $logFile
"  Random walk baseline ~21 (1/3 of 32×4)" | Out-File -Append $logFile
"  16×16 result was 2.97 ± 0.12; predicted 32×32: 4-6 if scaling holds" | Out-File -Append $logFile

"" | Out-File -Append $logFile
"=== Post-orthogonal step3 orchestrator COMPLETE at $(Get-Date) ===" | Out-File -Append $logFile
