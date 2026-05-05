# Polls bio-three-factor.master.log for COMPLETE marker, THEN waits for
# GPU to be genuinely idle (downstream chain stages may launch additional
# work — e.g. post_three_factor_decision.ps1 fires graded-DA at parallel=6
# after the partial-aligned branch). Without the GPU-idle gate, the bench
# runs UNDER CONTENTION (see 2026-05-05-bench-phase1-contamination.md).
#
# After GPU is free for `idleCheckCount` consecutive checks, runs:
#   1. tests/test_fp16_drift.py + test_three_factor_update.py
#   2. research/runners/bench_perf_stack.py --quick
#   3. Aggregates 3-factor sweep results (informational)
# Then writes a unified findings doc.

$outDir = "research/findings/raw/g11_bg"
$findingsDir = "research/findings"
$logFile = "$outDir/post_chain_bench.log"
$tfMaster = "$outDir/bio-three-factor.master.log"

# GPU-idle gate parameters
$idleThresholdPct = 10           # GPU util below this counts as idle
$idleCheckIntervalSec = 30       # check every N seconds
$idleCheckCount = 10             # consecutive idle checks needed (5 min default)
$idleMaxWaitSec = 7200           # absolute timeout (2 hr) — bail to bench anyway

"=== Post-chain benchmark started $(Get-Date) ===" | Out-File -FilePath $logFile

# Step 1: poll for 3-factor completion (existing logic)
"Polling for bio-three-factor COMPLETE..." | Out-File -Append $logFile
while ($true) {
    if (Test-Path $tfMaster) {
        $text = Get-Content $tfMaster -Raw -ErrorAction SilentlyContinue
        if ($text -match "bio-three-factor COMPLETE") {
            "Three-factor done at $(Get-Date)" | Out-File -Append $logFile
            break
        }
    }
    Start-Sleep -Seconds 60
}

# 30s grace for last seed's JSON to flush
Start-Sleep -Seconds 30

# Step 2: aggregate 3-factor sweep results (cheap, no GPU)
"" | Out-File -Append $logFile
"Aggregating bio_three_factor at $(Get-Date)" | Out-File -Append $logFile
$tfOutFile = "$findingsDir/2026-05-05-bio-three-factor-results.md"
$tfOutput = & "C:\python312\python.exe" -m research.result_aggregator `
    --config bio_three_factor `
    --out $tfOutFile 2>&1
"$tfOutput" | Out-File -Append $logFile

# Step 3: NEW — wait for GPU to be genuinely idle before benchmarking.
# Downstream chain stages (post_three_factor_decision.ps1) may launch
# additional sweeps that the master log can't predict.
"" | Out-File -Append $logFile
"Waiting for GPU-idle ($idleThresholdPct% util for $idleCheckCount checks @ $idleCheckIntervalSec sec)..." | Out-File -Append $logFile
$consecutiveIdle = 0
$startWait = Get-Date
while ($consecutiveIdle -lt $idleCheckCount) {
    Start-Sleep -Seconds $idleCheckIntervalSec
    try {
        $util = (& nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) -as [int]
    } catch {
        "  warn: nvidia-smi failed at $(Get-Date), assuming busy" | Out-File -Append $logFile
        $util = 99
    }
    if ($util -lt $idleThresholdPct) {
        $consecutiveIdle++
        "  GPU util ${util}% — idle ($consecutiveIdle/$idleCheckCount)" | Out-File -Append $logFile
    } else {
        if ($consecutiveIdle -gt 0) {
            "  GPU util ${util}% — busy, resetting idle counter" | Out-File -Append $logFile
        }
        $consecutiveIdle = 0
    }
    $elapsedSec = ((Get-Date) - $startWait).TotalSeconds
    if ($elapsedSec -gt $idleMaxWaitSec) {
        "  WARNING: idle wait timeout ($idleMaxWaitSec sec) — proceeding to bench anyway" | Out-File -Append $logFile
        break
    }
}
"GPU idle confirmed at $(Get-Date) — running tests + bench" | Out-File -Append $logFile

# Step 4: run FP16 drift + 3-factor update tests
"" | Out-File -Append $logFile
"Running FP16 drift tests at $(Get-Date)" | Out-File -Append $logFile
$pyTestOutput = & "C:\python312\python.exe" -m pytest `
    "tests/test_fp16_drift.py" `
    "tests/test_three_factor_update.py" `
    "-v" 2>&1 | Out-String
"$pyTestOutput" | Out-File -Append $logFile

# Step 5: run bench_perf_stack quick mode
"" | Out-File -Append $logFile
"Running perf stack benchmark at $(Get-Date)" | Out-File -Append $logFile
$benchOutput = & "C:\python312\python.exe" -m research.runners.bench_perf_stack `
    --quick 2>&1 | Out-String
"$benchOutput" | Out-File -Append $logFile

"" | Out-File -Append $logFile
"=== Post-chain benchmark COMPLETE at $(Get-Date) ===" | Out-File -Append $logFile
"" | Out-File -Append $logFile
"Suggested next steps:" | Out-File -Append $logFile
"  - Review $tfOutFile for 3-factor verdict" | Out-File -Append $logFile
"  - Review research/findings/raw/g11_bg/bench_perf_stack.json for Phase 1+2 speedups" | Out-File -Append $logFile
"  - If gradient_works: launch bio_b3_validation.yaml at parallel=6" | Out-File -Append $logFile
"  - Cloud H100 sweep ready when needed (scripts/deploy_to_cloud.sh)" | Out-File -Append $logFile
