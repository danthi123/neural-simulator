# Polls eval-sanity-check.master.log for completion, parses verdict,
# auto-fires B2 if eval is sound. Replaces wait_biology_then_decide.ps1
# for the B-branch sub-chain since that waiter already exited.

$outDir = "research/findings/raw/g11_bg"
$scMaster = "$outDir/eval-sanity-check.master.log"
$waitLog = "$outDir/wait_b1_then_b2.log"
$pidFile = "$outDir/wait_b1_then_b2.orchestrator-pid"

$PID | Out-File -FilePath $pidFile -Encoding ASCII

"=== Wait-B1-then-B2 started $(Get-Date) ===" | Out-File -FilePath $waitLog
"Polling $scMaster every 60s for 'eval-sanity-check COMPLETE'" | Out-File -Append $waitLog

while ($true) {
    if (Test-Path $scMaster) {
        $text = Get-Content $scMaster -Raw -ErrorAction SilentlyContinue
        if ($text -match "eval-sanity-check COMPLETE") {
            "B1 sanity check completion marker detected at $(Get-Date)" | Out-File -Append $waitLog
            break
        }
    }
    Start-Sleep -Seconds 60
}

"" | Out-File -Append $waitLog
"--- Aggregating sanity_check results at $(Get-Date) ---" | Out-File -Append $waitLog

$scOutput = & python -m research.result_aggregator --config sanity_check `
    --out research/findings/2026-05-04-eval-sanity-check-results-v2.md 2>&1
"$scOutput" | Out-File -Append $waitLog

# Parse verdict
$b1Verdict = "unknown"
if ($scOutput -match "Real word-action learning achieved") {
    $b1Verdict = "eval_sound"
} elseif ($scOutput -match "Partial signal") {
    $b1Verdict = "eval_partial"
} elseif ($scOutput -match "No real learning") {
    $b1Verdict = "eval_broken"
}

"" | Out-File -Append $waitLog
"B1 VERDICT: $b1Verdict" | Out-File -Append $waitLog

if ($b1Verdict -eq "eval_sound") {
    "Eval is sound -> bottleneck is plasticity. Launching B2 (sparse codes)." | Out-File -Append $waitLog
    $b2Path = "experiments/b2_sparse_codes.yaml"
    if (Test-Path $b2Path) {
        $b2Proc = Start-Process -FilePath "python.exe" -ArgumentList @(
            "-m", "research.experiment_runner", $b2Path
        ) -RedirectStandardOutput "$outDir/b2_sparse_codes.stdout.log" `
          -RedirectStandardError "$outDir/b2_sparse_codes.stderr.log" `
          -PassThru -NoNewWindow
        "Launched b2_sparse_codes as PID $($b2Proc.Id)" | Out-File -Append $waitLog
        $b2Proc.WaitForExit()
        "b2_sparse_codes finished (exit $($b2Proc.ExitCode)) at $(Get-Date)" | Out-File -Append $waitLog

        "" | Out-File -Append $waitLog
        "--- Aggregating b2_sparse_codes results ---" | Out-File -Append $waitLog
        $b2Output = & python -m research.result_aggregator --config b2_sparse_codes `
            --out research/findings/2026-05-04-b2-sparse-codes-results.md 2>&1
        "$b2Output" | Out-File -Append $waitLog
    } else {
        "ERROR: $b2Path not found. Skipping B2." | Out-File -Append $waitLog
    }
} elseif ($b1Verdict -eq "eval_broken") {
    "Eval methodology BROKEN -> stopping chain. Manual review needed." | Out-File -Append $waitLog
} else {
    "B1 verdict not actionable for auto-chain ($b1Verdict). Stopping; manual review." | Out-File -Append $waitLog
}

if (Test-Path $pidFile) {
    Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
}

"" | Out-File -Append $waitLog
"=== B1->B2 CHAIN COMPLETE at $(Get-Date) ===" | Out-File -Append $waitLog
