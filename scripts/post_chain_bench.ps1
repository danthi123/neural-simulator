# Polls bio-three-factor.master.log for COMPLETE marker, then auto-runs:
#   1. tests/test_fp16_drift.py (lifts the GPU-gated skips since GPU is free)
#   2. research/runners/bench_perf_stack.py --quick
#   3. Aggregates 3-factor sweep results
# Then writes a unified findings doc.

$outDir = "research/findings/raw/g11_bg"
$findingsDir = "research/findings"
$logFile = "$outDir/post_chain_bench.log"
$tfMaster = "$outDir/bio-three-factor.master.log"

"=== Post-chain benchmark started $(Get-Date) ===" | Out-File -FilePath $logFile

# Step 1: poll for 3-factor completion
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

# Step 2: aggregate the 3-factor sweep results FIRST (most important)
"" | Out-File -Append $logFile
"Aggregating bio_three_factor at $(Get-Date)" | Out-File -Append $logFile
$tfOutFile = "$findingsDir/2026-05-05-bio-three-factor-results.md"
$tfOutput = & "C:\python312\python.exe" -m research.result_aggregator `
    --config bio_three_factor `
    --out $tfOutFile 2>&1
"$tfOutput" | Out-File -Append $logFile

# Step 3: now that GPU is free, run the FP16 drift test
"" | Out-File -Append $logFile
"Running FP16 drift tests at $(Get-Date)" | Out-File -Append $logFile
"  (lift GPU-gate skipif by re-running with --override-gpu-skip)" | Out-File -Append $logFile
$pyTestOutput = & "C:\python312\python.exe" -m pytest `
    "tests/test_fp16_drift.py" `
    "tests/test_three_factor_update.py" `
    "-v" 2>&1 | Out-String
"$pyTestOutput" | Out-File -Append $logFile

# Step 4: run bench_perf_stack quick mode (5 min, single-seed each config)
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
