# Polls bio-three-factor.master.log for COMPLETE marker, runs aggregator,
# writes findings doc with the headline scientific verdict.

$outDir = "research/findings/raw/g11_bg"
$findingsDir = "research/findings"
$logFile = "$outDir/post_three_factor_reporter.log"
$tfMaster = "$outDir/bio-three-factor.master.log"

"=== Post-three-factor reporter started $(Get-Date) ===" | Out-File -FilePath $logFile

while ($true) {
    if (Test-Path $tfMaster) {
        $text = Get-Content $tfMaster -Raw -ErrorAction SilentlyContinue
        if ($text -match "bio-three-factor COMPLETE") {
            "Three-factor completion detected at $(Get-Date)" | Out-File -Append $logFile
            break
        }
    }
    Start-Sleep -Seconds 60
}

# 30s grace for last seed's JSON to be fully flushed (B3 had this issue).
Start-Sleep -Seconds 30

"" | Out-File -Append $logFile
"Aggregating bio_three_factor at $(Get-Date)" | Out-File -Append $logFile
$tfOutFile = "$findingsDir/2026-05-05-bio-three-factor-results.md"
$tfOutput = & "C:\python312\python.exe" -m research.result_aggregator `
    --config bio_three_factor `
    --out $tfOutFile 2>&1
"$tfOutput" | Out-File -Append $logFile

$verdict = "unknown"
if ($tfOutput -match "Real word-action learning achieved") {
    $verdict = "three_factor_works"
} elseif ($tfOutput -match "Partial signal") {
    $verdict = "three_factor_partial"
} elseif ($tfOutput -match "No real learning") {
    $verdict = "three_factor_fails"
}
"" | Out-File -Append $logFile
"THREE-FACTOR VERDICT: $verdict" | Out-File -Append $logFile

if ($verdict -eq "three_factor_works") {
    "" | Out-File -Append $logFile
    "*** HEADLINE: biology-plausible learning rule matches gradient. ***" | Out-File -Append $logFile
    "*** The W->A bottleneck IS solvable with biology-grounded rules. ***" | Out-File -Append $logFile
    "Next: multi-task scaling, compositional language, real-image vision." | Out-File -Append $logFile
} elseif ($verdict -eq "three_factor_partial") {
    "" | Out-File -Append $logFile
    "*** HEADLINE: biology-plausible rule shows weak signal. ***" | Out-File -Append $logFile
    "*** STDP family has a ceiling; need dendritic or predictive coding. ***" | Out-File -Append $logFile
} elseif ($verdict -eq "three_factor_fails") {
    "" | Out-File -Append $logFile
    "*** HEADLINE: biology-plausible rule fails where gradient succeeds. ***" | Out-File -Append $logFile
    "*** STDP family insufficient. Next: apical-basal dendritic learning ***" | Out-File -Append $logFile
    "*** (Bono & Clopath 2017) or predictive coding (Rao & Ballard 1999). ***" | Out-File -Append $logFile
}

"" | Out-File -Append $logFile
"=== Post-three-factor reporter COMPLETE at $(Get-Date) ===" | Out-File -Append $logFile
