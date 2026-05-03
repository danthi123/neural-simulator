# Master super-orchestrator — runs H4 (isolation) then H1 (balanced) in
# sequence. Started after the v2+SWR 4-seed batch completes.
#
# This avoids any intervention needed during the autonomous night —
# results land continuously and aggregator can be re-run any time.

$outDir = "research/findings/raw/g11_bg"
$superLog = "$outDir/run_h4_then_h1.super.log"

"=== H4-then-H1 super-orchestrator started $(Get-Date) ===" | Out-File -FilePath $superLog

# --- Step 1: H4 (PFC bypass isolation, 6 seeds) ---
"" | Out-File -Append $superLog
"--- Launching H4 (PFC bypass isolation) at $(Get-Date) ---" | Out-File -Append $superLog
$h4 = Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-File", "research/findings/raw/g11_bg/run_h4_isolation_seeds.ps1" `
    -WorkingDirectory "E:/Documents/Projects/sim" `
    -PassThru -NoNewWindow -Wait
"H4 batch finished (exit $($h4.ExitCode)) at $(Get-Date)" | Out-File -Append $superLog

# Aggregate after H4
"--- Aggregating after H4 at $(Get-Date) ---" | Out-File -Append $superLog
& python -m research.runners.swr_aggregate --out "research/findings/2026-05-03-swr-multiseed-summary.md"
"Aggregator wrote summary at $(Get-Date)" | Out-File -Append $superLog

# --- Step 2: H1 (balanced replay, 6 seeds) ---
"" | Out-File -Append $superLog
"--- Launching H1 (balanced replay) at $(Get-Date) ---" | Out-File -Append $superLog
$h1 = Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-File", "research/findings/raw/g11_bg/run_h1_balanced_seeds.ps1" `
    -WorkingDirectory "E:/Documents/Projects/sim" `
    -PassThru -NoNewWindow -Wait
"H1 batch finished (exit $($h1.ExitCode)) at $(Get-Date)" | Out-File -Append $superLog

# Final aggregate
"--- Final aggregation at $(Get-Date) ---" | Out-File -Append $superLog
& python -m research.runners.swr_aggregate --out "research/findings/2026-05-03-swr-multiseed-summary.md"
"Final summary written at $(Get-Date)" | Out-File -Append $superLog

"" | Out-File -Append $superLog
"=== ALL BATCHES COMPLETE at $(Get-Date) ===" | Out-File -Append $superLog
