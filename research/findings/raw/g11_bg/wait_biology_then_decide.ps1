# Polls the biology sweep master log for completion, then decides
# A-branch (aligned >= 4/6 → minimum biology sweep) vs B-branch (all
# 0/6 → eval sanity check). Launches the chosen follow-up.
#
# Decision logic mirrors research/findings/2026-05-04-biology-sweep-followup-plan.md.
#
# Outcome A (real learning emerged):
#   - Run minimum_biology.yaml: vary topographic factor + FS pool size
#     to identify minimum sufficient biology
#   - Then build cascade-reintro experiment
#
# Outcome B (no condition aligned):
#   - Run eval_sanity_check.py: hardcoded weights that should trivially
#     align. If yes, eval is correct. If no, eval methodology is broken.

$outDir = "research/findings/raw/g11_bg"
$bioMaster = "$outDir/biology-sweep.master.log"  # set by experiment_runner
$bioMasterAlt = "$outDir/run_biology_sweep.master.log"  # set by powershell version
$waitLog = "$outDir/wait_biology_then_decide.log"
$pidFile = "$outDir/wait_biology_then_decide.orchestrator-pid"

$PID | Out-File -FilePath $pidFile -Encoding ASCII

"=== Wait-biology-then-decide started $(Get-Date) ===" | Out-File -FilePath $waitLog
"Polling for biology sweep completion every 60s" | Out-File -Append $waitLog

while ($true) {
    $done = $false
    foreach ($masterPath in @($bioMaster, $bioMasterAlt)) {
        if (Test-Path $masterPath) {
            $text = Get-Content $masterPath -Raw -ErrorAction SilentlyContinue
            if ($text -match "BIOLOGY SWEEP COMPLETE" -or `
                $text -match "biology-sweep COMPLETE" -or `
                $text -match "ALL BATCHES COMPLETE") {
                "Biology sweep completion marker detected at $(Get-Date) in $masterPath" | Out-File -Append $waitLog
                $done = $true
                break
            }
        }
    }
    if ($done) { break }
    Start-Sleep -Seconds 60
}

# Run the result aggregator to get aligned ratios per condition
"" | Out-File -Append $waitLog
"--- Running result aggregator at $(Get-Date) ---" | Out-File -Append $waitLog

$aggOutput = & python -m research.result_aggregator --config biology 2>&1
"$aggOutput" | Out-File -Append $waitLog

# Save full aggregation to a findings doc
& python -m research.result_aggregator --config biology `
    --out research/findings/2026-05-04-biology-sweep-results.md 2>&1 | `
    Out-File -Append $waitLog

# Parse aggregator output for the verdict line
$verdict = "unknown"
if ($aggOutput -match "Real word-action learning achieved") {
    $verdict = "A"  # aligned >= 4/6
} elseif ($aggOutput -match "Partial signal") {
    $verdict = "A_weak"  # aligned 2-3/6
} elseif ($aggOutput -match "No real learning") {
    $verdict = "B"  # all 0-1/6
}

"" | Out-File -Append $waitLog
"VERDICT: $verdict" | Out-File -Append $waitLog

if ($verdict -eq "A") {
    "Outcome A: real learning achieved. Launching minimum_biology sweep (24 runs)." | Out-File -Append $waitLog
    $followupPath = "experiments/minimum_biology.yaml"
    if (Test-Path $followupPath) {
        $proc = Start-Process -FilePath "python.exe" -ArgumentList @(
            "-m", "research.experiment_runner", $followupPath
        ) -RedirectStandardOutput "$outDir/minimum_biology.stdout.log" `
          -RedirectStandardError "$outDir/minimum_biology.stderr.log" `
          -PassThru -NoNewWindow
        "Launched minimum_biology as PID $($proc.Id)" | Out-File -Append $waitLog
        $proc.WaitForExit()
        "minimum_biology finished (exit $($proc.ExitCode)) at $(Get-Date)" | Out-File -Append $waitLog

        # Auto-aggregate minimum_biology results
        "" | Out-File -Append $waitLog
        "--- Aggregating minimum_biology results ---" | Out-File -Append $waitLog
        $minbioOutput = & python -m research.result_aggregator --config minimum_biology `
            --out research/findings/2026-05-04-minimum-biology-results.md 2>&1
        "$minbioOutput" | Out-File -Append $waitLog
    } else {
        "ERROR: $followupPath not found. Skipping A-branch follow-up." | Out-File -Append $waitLog
    }
} elseif ($verdict -eq "A_weak") {
    "Outcome A_weak: partial signal. Documenting + waiting for user." | Out-File -Append $waitLog
} elseif ($verdict -eq "B") {
    "Outcome B: no real learning across any condition. Running eval sanity check (6 seeds x 2 densities)." | Out-File -Append $waitLog
    $followupPath = "experiments/eval_sanity_check.yaml"
    if (Test-Path $followupPath) {
        $proc = Start-Process -FilePath "python.exe" -ArgumentList @(
            "-m", "research.experiment_runner", $followupPath
        ) -RedirectStandardOutput "$outDir/eval_sanity_check.stdout.log" `
          -RedirectStandardError "$outDir/eval_sanity_check.stderr.log" `
          -PassThru -NoNewWindow
        "Launched eval_sanity_check sweep as PID $($proc.Id)" | Out-File -Append $waitLog
        $proc.WaitForExit()
        "eval_sanity_check sweep finished (exit $($proc.ExitCode)) at $(Get-Date)" | Out-File -Append $waitLog

        # Auto-aggregate sanity_check results so the user sees the verdict
        # in the waiter log without running result_aggregator manually.
        "" | Out-File -Append $waitLog
        "--- Aggregating sanity_check results ---" | Out-File -Append $waitLog
        $scOutput = & python -m research.result_aggregator --config sanity_check `
            --out research/findings/2026-05-04-eval-sanity-check-results.md 2>&1
        "$scOutput" | Out-File -Append $waitLog

        # B1 -> B2 chain extension: if B1 verdict shows "Real word-action
        # learning achieved" (i.e., perfect mode aligned >= 4/6), the eval
        # is sound and the bottleneck is plasticity. Auto-launch B2
        # (sparse codes) to test sparse-code-overlap hypothesis.
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
            "B1 verdict not actionable for auto-chain. Stopping; manual review." | Out-File -Append $waitLog
        }
    } else {
        "ERROR: $followupPath not found. Skipping B-branch." | Out-File -Append $waitLog
    }
} else {
    "Unknown verdict; manual review needed." | Out-File -Append $waitLog
}

# Move pid to done
if (Test-Path $pidFile) {
    Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
}

"" | Out-File -Append $waitLog
"=== POST-BIOLOGY DECISION CHAIN COMPLETE at $(Get-Date) ===" | Out-File -Append $waitLog
