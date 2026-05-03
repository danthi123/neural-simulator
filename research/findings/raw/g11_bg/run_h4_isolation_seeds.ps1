# H4: PFC bypass isolation across 6 seeds
# Tests upper bound of language->motor direct pathway WITHOUT cascade
# interference. 100 paired-stim events per direction (400 total) is
# comparable in event count to v2 baseline + 500 SWR replays.

$seeds = @(42, 43, 44, 100, 101, 102)
$nEventsPerDir = 100
$outDir = "research/findings/raw/g11_bg"
$masterLog = "$outDir/run_h4.master.log"

"=== H4 PFC-bypass isolation 6-seed batch started $(Get-Date) ===" | Out-File -FilePath $masterLog
"Seeds: $($seeds -join ',')" | Out-File -Append $masterLog
"Events per direction: $nEventsPerDir (total $((4 * $nEventsPerDir)))" | Out-File -Append $masterLog

foreach ($seed in $seeds) {
    "" | Out-File -Append $masterLog
    "--- Starting H4 seed $seed at $(Get-Date) ---" | Out-File -Append $masterLog
    $logFile = "$outDir/h4_isolation_seed$seed.log"
    $errFile = "$outDir/h4_isolation_seed$seed.log.err"
    $pidFile = "$outDir/h4_isolation_seed$seed.pid"
    $outStats = "$outDir/text_eval_h4_isolation_seed$seed.json"

    $proc = Start-Process -FilePath "python.exe" -ArgumentList @(
        "-m", "research.runners.text_pfc_bypass_isolation",
        "--seed", "$seed",
        "--n-events-per-direction", "$nEventsPerDir",
        "--stim-steps-per-step", "200",
        "--reset-steps", "100",
        "--out-stats", $outStats
    ) -RedirectStandardOutput $logFile -RedirectStandardError $errFile -PassThru -NoNewWindow

    $proc.Id | Out-File -FilePath $pidFile -Encoding ASCII
    "Seed $seed launched as PID $($proc.Id)" | Out-File -Append $masterLog

    $proc.WaitForExit()
    "Seed $seed completed (exit $($proc.ExitCode)) at $(Get-Date)" | Out-File -Append $masterLog

    if (Test-Path $pidFile) {
        Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
    }
}

"" | Out-File -Append $masterLog
"=== H4 BATCH COMPLETE at $(Get-Date) ===" | Out-File -Append $masterLog
