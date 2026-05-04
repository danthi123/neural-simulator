# Post-sanity-check orchestrator for the bio chain.
#
# Replaces the original launch_bio_chain.ps1 stages 4-10 (the parts
# after sanity check launches). The original is killed because we
# upgraded the YAML to parallel=3 / 6 seeds AFTER it loaded the old
# script. PowerShell parses up-front, so live edits to launch_bio_chain.ps1
# don't take effect on the running process.
#
# This script:
#   1. Polls bio-sanity-check.master.log for "eval-sanity-check COMPLETE"
#      (or accepts "bio-sanity-check COMPLETE" since experiment_runner
#      uses the YAML name)
#   2. Aggregates bio_sanity_check + parses verdict
#   3. If eval works: launches bio_proof_of_concept (12 runs / parallel=3)
#   4. After PoC: aggregates + parses verdict
#   5. If STDP fails at bio: launches bio_b3_gradient (9 runs / parallel=2)
#   6. If STDP works: documents headline + stops (no further auto-runs)

$outDir = "research/findings/raw/g11_bg"
$findingsDir = "research/findings"
$logFile = "$outDir/orchestrate_bio_post_sanity.log"
$sanityMaster = "$outDir/bio-sanity-check.master.log"

"=== Post-sanity orchestrator started $(Get-Date) ===" | Out-File -FilePath $logFile

# Step 1: poll for sanity completion
"" | Out-File -Append $logFile
"Step 1: polling $sanityMaster every 30s" | Out-File -Append $logFile
while ($true) {
    if (Test-Path $sanityMaster) {
        $text = Get-Content $sanityMaster -Raw -ErrorAction SilentlyContinue
        # experiment_runner emits "=== {name} COMPLETE ===" with name = YAML's "name" field
        if ($text -match "bio-sanity-check COMPLETE") {
            "Sanity check completion marker detected at $(Get-Date)" | Out-File -Append $logFile
            break
        }
    }
    Start-Sleep -Seconds 30
}

# Step 2: aggregate sanity check
"" | Out-File -Append $logFile
"Step 2: aggregating bio_sanity_check" | Out-File -Append $logFile
$bioOutFile = "$findingsDir/2026-05-04-bio-sanity-check-results.md"
$aggOutput = & "C:\python312\python.exe" -m research.result_aggregator `
    --config bio_sanity_check `
    --out $bioOutFile 2>&1
"$aggOutput" | Out-File -Append $logFile

$bioVerdict = "unknown"
if ($aggOutput -match "Real word-action learning achieved") {
    $bioVerdict = "eval_works"
} elseif ($aggOutput -match "Partial signal") {
    $bioVerdict = "eval_partial"
} elseif ($aggOutput -match "No real learning") {
    $bioVerdict = "eval_broken"
}
"" | Out-File -Append $logFile
"BIO SANITY VERDICT: $bioVerdict" | Out-File -Append $logFile

if ($bioVerdict -ne "eval_works") {
    "" | Out-File -Append $logFile
    "Verdict $bioVerdict not actionable for auto-chain. Stopping." | Out-File -Append $logFile
    "=== Orchestrator COMPLETE at $(Get-Date) ===" | Out-File -Append $logFile
    exit 0
}

# Step 3: launch bio_proof_of_concept (parallel=3, 6 seeds, 12 runs)
"" | Out-File -Append $logFile
"Step 3: launching bio_proof_of_concept at $(Get-Date) (parallel=3, 6 seeds x 2 conds = 12 runs, ~5 hours)" | Out-File -Append $logFile
$pocProc = Start-Process -FilePath "C:\python312\python.exe" `
    -ArgumentList "-m","research.experiment_runner",
                  "experiments/bio_proof_of_concept.yaml" `
    -RedirectStandardOutput "$outDir/bio_proof_of_concept.stdout.log" `
    -RedirectStandardError "$outDir/bio_proof_of_concept.stderr.log" `
    -WindowStyle Hidden -PassThru
"bio_proof_of_concept launched as PID $($pocProc.Id)" | Out-File -Append $logFile
$pocProc.WaitForExit()
"bio_proof_of_concept finished (exit $($pocProc.ExitCode)) at $(Get-Date)" | Out-File -Append $logFile

# Step 4: aggregate PoC
"" | Out-File -Append $logFile
"Step 4: aggregating bio_proof_of_concept" | Out-File -Append $logFile
$pocOutFile = "$findingsDir/2026-05-04-bio-proof-of-concept-results.md"
$pocOutput = & "C:\python312\python.exe" -m research.result_aggregator `
    --config bio_proof_of_concept `
    --out $pocOutFile 2>&1
"$pocOutput" | Out-File -Append $logFile

$pocVerdict = "unknown"
if ($pocOutput -match "Real word-action learning achieved") {
    $pocVerdict = "stdp_works_at_bio"
} elseif ($pocOutput -match "Partial signal") {
    $pocVerdict = "stdp_partial"
} elseif ($pocOutput -match "No real learning") {
    $pocVerdict = "stdp_fails_at_bio"
}
"" | Out-File -Append $logFile
"POC VERDICT: $pocVerdict" | Out-File -Append $logFile

# Step 5: autonomous decision based on PoC verdict
if ($pocVerdict -eq "stdp_works_at_bio") {
    "" | Out-File -Append $logFile
    "STDP works at biological scale. Headline finding documented at $pocOutFile." | Out-File -Append $logFile
    "No further auto-experiments. Suggested next directions:" | Out-File -Append $logFile
    "  - Scale-up validation (lang=4096, motor=1000)" | Out-File -Append $logFile
    "  - Harder benchmark (longer training, larger vocabulary)" | Out-File -Append $logFile
    "  - Ablations (canon vs biology fix vs scale)" | Out-File -Append $logFile
} elseif ($pocVerdict -eq "stdp_partial") {
    "" | Out-File -Append $logFile
    "Partial signal. Stopping for manual review (more seeds vs sparser codes vs B3)." | Out-File -Append $logFile
} elseif ($pocVerdict -eq "stdp_fails_at_bio") {
    "" | Out-File -Append $logFile
    "STDP fails at biological scale. Launching B3 (supervised gradient) at bio scale" | Out-File -Append $logFile
    "to test if ANY learning rule succeeds here." | Out-File -Append $logFile
    "Step 6: bio_b3_gradient at $(Get-Date)" | Out-File -Append $logFile
    $b3Proc = Start-Process -FilePath "C:\python312\python.exe" `
        -ArgumentList "-m","research.experiment_runner",
                      "experiments/bio_b3_gradient.yaml" `
        -RedirectStandardOutput "$outDir/bio_b3_gradient.stdout.log" `
        -RedirectStandardError "$outDir/bio_b3_gradient.stderr.log" `
        -WindowStyle Hidden -PassThru
    "bio_b3_gradient launched as PID $($b3Proc.Id)" | Out-File -Append $logFile
    $b3Proc.WaitForExit()
    "bio_b3_gradient finished (exit $($b3Proc.ExitCode)) at $(Get-Date)" | Out-File -Append $logFile

    # Aggregate B3
    "" | Out-File -Append $logFile
    "Step 7: aggregating bio_b3_gradient" | Out-File -Append $logFile
    $b3Out = & "C:\python312\python.exe" -m research.result_aggregator `
        --config b3_supervised_gradient `
        --out "$findingsDir/2026-05-04-bio-b3-gradient-results.md" 2>&1
    "$b3Out" | Out-File -Append $logFile
} else {
    "" | Out-File -Append $logFile
    "PoC verdict $pocVerdict not actionable. Stopping." | Out-File -Append $logFile
}

"" | Out-File -Append $logFile
"=== Post-sanity orchestrator COMPLETE at $(Get-Date) ===" | Out-File -Append $logFile
