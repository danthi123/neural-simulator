# Biological-scale test chain launcher.
#
# Run AFTER REBOOT for fresh GPU state. Launches:
#   1. Webapp (uvicorn on port 8765, no --reload)
#   2. bio_sanity_check sweep (24 runs, parallel=1, ~3 hours)
#   3. Wait for bio_sanity_check to complete
#   4. Aggregate results + parse verdict
#   5. If perfect mode aligns >= 4/6: launch bio_proof_of_concept
#      (~2.5 hours)
#   6. Aggregate proof-of-concept results
#
# All processes run with parallelism=1 to maximize GPU per single run
# (RTX 3090 / 24 GB VRAM, biological architecture uses ~1-2 GB peak).
#
# Logs:
#   - Webapp: stdout suppressed (uvicorn --log-level warning)
#   - This script: research/findings/raw/g11_bg/launch_bio_chain.log
#   - Per-experiment: written by experiment_runner
#   - Per-condition aggregation: research/findings/2026-05-04-bio-*-results.md
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/launch_bio_chain.ps1
#
# Stop any running waiter / chain:
#   Get-Process powershell | Where-Object { $_.MainWindowTitle -like '*bio*' } | Stop-Process

$outDir = "research/findings/raw/g11_bg"
$findingsDir = "research/findings"
$launchLog = "$outDir/launch_bio_chain.log"

# Ensure outDir exists
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

"=== Bio chain launcher started $(Get-Date) ===" | Out-File -FilePath $launchLog
"" | Out-File -Append $launchLog

# Step 1: clean any stale python processes (best-effort)
"Step 1: cleaning stale processes" | Out-File -Append $launchLog
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Step 2: launch webapp on port 8765
"Step 2: launching webapp on port 8765" | Out-File -Append $launchLog
$webProc = Start-Process -FilePath "C:\python312\python.exe" `
    -ArgumentList "-m","uvicorn","webapp.server:app",
                  "--host","127.0.0.1","--port","8765",
                  "--log-level","warning" `
    -WindowStyle Hidden -PassThru
"Webapp launched as PID $($webProc.Id)" | Out-File -Append $launchLog
Start-Sleep -Seconds 5

# Step 3: launch bio_sanity_check sweep
"" | Out-File -Append $launchLog
"Step 3: launching bio_sanity_check at $(Get-Date)" | Out-File -Append $launchLog
$sanityProc = Start-Process -FilePath "C:\python312\python.exe" `
    -ArgumentList "-m","research.experiment_runner",
                  "experiments/bio_sanity_check.yaml" `
    -RedirectStandardOutput "$outDir/bio_sanity_check.stdout.log" `
    -RedirectStandardError "$outDir/bio_sanity_check.stderr.log" `
    -WindowStyle Hidden -PassThru
"bio_sanity_check launched as PID $($sanityProc.Id)" | Out-File -Append $launchLog

# Step 4: wait for bio_sanity_check to finish (parallel=1, expect ~3 hours)
$sanityProc.WaitForExit()
"" | Out-File -Append $launchLog
"bio_sanity_check finished (exit $($sanityProc.ExitCode)) at $(Get-Date)" | Out-File -Append $launchLog

# Step 5: aggregate sanity_check results
"" | Out-File -Append $launchLog
"Step 5: aggregating bio_sanity_check at $(Get-Date)" | Out-File -Append $launchLog
$bioOutFile = "$findingsDir/2026-05-04-bio-sanity-check-results.md"
$aggOutput = & "C:\python312\python.exe" -m research.result_aggregator `
    --config bio_sanity_check `
    --out $bioOutFile 2>&1
"$aggOutput" | Out-File -Append $launchLog

# Step 6: parse verdict from aggregator stdout (Headline: line)
$bioVerdict = "unknown"
if ($aggOutput -match "Real word-action learning achieved") {
    $bioVerdict = "eval_works"
} elseif ($aggOutput -match "Partial signal") {
    $bioVerdict = "eval_partial"
} elseif ($aggOutput -match "No real learning") {
    $bioVerdict = "eval_broken"
}

"" | Out-File -Append $launchLog
"BIO SANITY VERDICT: $bioVerdict" | Out-File -Append $launchLog

# Step 7: conditionally launch bio_proof_of_concept
if ($bioVerdict -eq "eval_works") {
    "" | Out-File -Append $launchLog
    "Eval works at biological scale -> launching proof-of-concept STDP training." | Out-File -Append $launchLog
    "Step 7: bio_proof_of_concept at $(Get-Date) (parallel=3, 6 seeds x 2 conds = 12 runs)" | Out-File -Append $launchLog
    $pocProc = Start-Process -FilePath "C:\python312\python.exe" `
        -ArgumentList "-m","research.experiment_runner",
                      "experiments/bio_proof_of_concept.yaml" `
        -RedirectStandardOutput "$outDir/bio_proof_of_concept.stdout.log" `
        -RedirectStandardError "$outDir/bio_proof_of_concept.stderr.log" `
        -WindowStyle Hidden -PassThru
    "bio_proof_of_concept launched as PID $($pocProc.Id)" | Out-File -Append $launchLog
    $pocProc.WaitForExit()
    "bio_proof_of_concept finished (exit $($pocProc.ExitCode)) at $(Get-Date)" | Out-File -Append $launchLog

    # Aggregate PoC results + parse verdict
    "" | Out-File -Append $launchLog
    "Step 8: aggregating bio_proof_of_concept" | Out-File -Append $launchLog
    $pocOut = & "C:\python312\python.exe" -m research.result_aggregator `
        --config bio_proof_of_concept `
        --out "$findingsDir/2026-05-04-bio-proof-of-concept-results.md" 2>&1
    "$pocOut" | Out-File -Append $launchLog

    # Stage 3: autonomous decision based on PoC verdict
    $pocVerdict = "unknown"
    if ($pocOut -match "Real word-action learning achieved") {
        $pocVerdict = "stdp_works_at_bio"
    } elseif ($pocOut -match "Partial signal") {
        $pocVerdict = "stdp_partial"
    } elseif ($pocOut -match "No real learning") {
        $pocVerdict = "stdp_fails_at_bio"
    }
    "" | Out-File -Append $launchLog
    "POC VERDICT: $pocVerdict" | Out-File -Append $launchLog

    if ($pocVerdict -eq "stdp_works_at_bio") {
        # SUCCESS: cortical canon (and maybe biology fix) enables W->A
        # at bio scale with STDP. Multi-seed already done at 6 seeds.
        # No further auto-experiment needed; this is the headline result.
        "" | Out-File -Append $launchLog
        "STDP works at biological scale. Headline finding documented." | Out-File -Append $launchLog
        "No further auto-experiments. Manual review for next directions:" | Out-File -Append $launchLog
        "  - scale-up validation (lang=4096, motor=1000)" | Out-File -Append $launchLog
        "  - harder benchmark (longer training, larger vocabulary)" | Out-File -Append $launchLog
        "  - ablations (canon vs biology fix vs scale)" | Out-File -Append $launchLog
    } elseif ($pocVerdict -eq "stdp_partial") {
        # MIXED: some seeds align. Run additional seeds OR sparse-code variation
        # to clarify. Auto-launching b3 gradient is premature.
        "" | Out-File -Append $launchLog
        "Partial signal. Stopping for manual review (more seeds vs sparser codes vs B3 gradient)." | Out-File -Append $launchLog
    } elseif ($pocVerdict -eq "stdp_fails_at_bio") {
        # FAILURE: STDP can't find the mapping even at bio scale with biology fix.
        # Auto-launch B3 (supervised gradient) at bio scale to test
        # "is plasticity rule the bottleneck?"
        "" | Out-File -Append $launchLog
        "STDP fails at biological scale. Launching B3 (supervised gradient) at bio scale" | Out-File -Append $launchLog
        "to test if ANY learning rule succeeds here." | Out-File -Append $launchLog
        "Step 9: bio_b3_gradient at $(Get-Date)" | Out-File -Append $launchLog
        $b3Proc = Start-Process -FilePath "C:\python312\python.exe" `
            -ArgumentList "-m","research.experiment_runner",
                          "experiments/bio_b3_gradient.yaml" `
            -RedirectStandardOutput "$outDir/bio_b3_gradient.stdout.log" `
            -RedirectStandardError "$outDir/bio_b3_gradient.stderr.log" `
            -WindowStyle Hidden -PassThru
        "bio_b3_gradient launched as PID $($b3Proc.Id)" | Out-File -Append $launchLog
        $b3Proc.WaitForExit()
        "bio_b3_gradient finished (exit $($b3Proc.ExitCode)) at $(Get-Date)" | Out-File -Append $launchLog

        "" | Out-File -Append $launchLog
        "Step 10: aggregating bio_b3_gradient" | Out-File -Append $launchLog
        $b3Out = & "C:\python312\python.exe" -m research.result_aggregator `
            --config b3_supervised_gradient `
            --out "$findingsDir/2026-05-04-bio-b3-gradient-results.md" 2>&1
        "$b3Out" | Out-File -Append $launchLog
    } else {
        "" | Out-File -Append $launchLog
        "PoC verdict $pocVerdict not actionable for auto-chain. Stopping." | Out-File -Append $launchLog
    }
} elseif ($bioVerdict -eq "eval_broken") {
    "" | Out-File -Append $launchLog
    "Eval broken even at bio scale -> deeper investigation needed." | Out-File -Append $launchLog
    "Skipping proof-of-concept; manual review required." | Out-File -Append $launchLog
} else {
    "" | Out-File -Append $launchLog
    "Verdict $bioVerdict not actionable for auto-chain." | Out-File -Append $launchLog
    "Skipping proof-of-concept; manual review." | Out-File -Append $launchLog
}

"" | Out-File -Append $launchLog
"=== Bio chain launcher COMPLETE at $(Get-Date) ===" | Out-File -Append $launchLog
"" | Out-File -Append $launchLog
"Webapp PID $($webProc.Id) still running. Check results at:" | Out-File -Append $launchLog
"  $bioOutFile" | Out-File -Append $launchLog
"  http://127.0.0.1:8765" | Out-File -Append $launchLog
"  python -m research.runners.morning_briefing" | Out-File -Append $launchLog
