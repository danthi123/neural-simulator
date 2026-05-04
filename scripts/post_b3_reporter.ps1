# Polls bio-b3-gradient.master.log for COMPLETE marker, then runs
# the aggregator. Standalone replacement for the post-B3 step in
# orchestrate_bio_post_sanity.ps1 (which exited when its $b3Proc
# was killed for the --biological config restart).

$outDir = "research/findings/raw/g11_bg"
$findingsDir = "research/findings"
$logFile = "$outDir/post_b3_reporter.log"
$b3Master = "$outDir/bio-b3-gradient.master.log"

"=== Post-B3 reporter started $(Get-Date) ===" | Out-File -FilePath $logFile

# Step 1: poll for B3 completion
"Polling $b3Master every 30s for 'bio-b3-gradient COMPLETE'" | Out-File -Append $logFile
while ($true) {
    if (Test-Path $b3Master) {
        $text = Get-Content $b3Master -Raw -ErrorAction SilentlyContinue
        if ($text -match "bio-b3-gradient COMPLETE") {
            "B3 completion marker detected at $(Get-Date)" | Out-File -Append $logFile
            break
        }
    }
    Start-Sleep -Seconds 30
}

# Step 2: aggregate B3 results
"" | Out-File -Append $logFile
"Aggregating bio_b3_gradient at $(Get-Date)" | Out-File -Append $logFile
$b3OutFile = "$findingsDir/2026-05-04-bio-b3-gradient-results.md"
$b3Output = & "C:\python312\python.exe" -m research.result_aggregator `
    --config b3_supervised_gradient `
    --out $b3OutFile 2>&1
"$b3Output" | Out-File -Append $logFile

# Step 3: parse verdict
$b3Verdict = "unknown"
if ($b3Output -match "Real word-action learning achieved") {
    $b3Verdict = "gradient_works"
} elseif ($b3Output -match "Partial signal") {
    $b3Verdict = "gradient_partial"
} elseif ($b3Output -match "No real learning") {
    $b3Verdict = "gradient_fails"
}
"" | Out-File -Append $logFile
"B3 VERDICT: $b3Verdict" | Out-File -Append $logFile

if ($b3Verdict -eq "gradient_works") {
    "" | Out-File -Append $logFile
    "Gradient learning succeeds where STDP fails -> plasticity rule IS the bottleneck." | Out-File -Append $logFile
    "Future work: biology-grounded learning rules with better credit assignment" | Out-File -Append $logFile
    "  - Apical-basal feedback (Bono & Clopath 2017)" | Out-File -Append $logFile
    "  - Three-factor with eligibility (Fremaux & Gerstner 2016)" | Out-File -Append $logFile
} elseif ($b3Verdict -eq "gradient_fails") {
    "" | Out-File -Append $logFile
    "Even gradient fails -> architecture/training-dose ceiling." | Out-File -Append $logFile
    "Future work: sparser codes (token-sparsity 0.05/0.02), longer training," | Out-File -Append $logFile
    "  population vector decoding, alternative drive currents." | Out-File -Append $logFile
}

"" | Out-File -Append $logFile
"=== Post-B3 reporter COMPLETE at $(Get-Date) ===" | Out-File -Append $logFile
