# H1: balanced replay across 6 seeds
# Tests whether direction-balanced replay (sample N/4 events per direction
# rather than uniform-random over the buffer) rescues the W->A regression
# observed at n=3 with default frequency-weighted sampling.

$seeds = @(42, 43, 44, 100, 101, 102)
$outDir = "research/findings/raw/g11_bg"
$masterLog = "$outDir/run_h1.master.log"

"=== H1 balanced-replay 6-seed batch started $(Get-Date) ===" | Out-File -FilePath $masterLog
"Seeds: $($seeds -join ',')" | Out-File -Append $masterLog

foreach ($seed in $seeds) {
    "" | Out-File -Append $masterLog
    "--- Starting H1 seed $seed at $(Get-Date) ---" | Out-File -Append $masterLog
    $logFile = "$outDir/h1_balanced_seed$seed.log"
    $errFile = "$outDir/h1_balanced_seed$seed.log.err"
    $pidFile = "$outDir/h1_balanced_seed$seed.pid"
    $outStats = "$outDir/text_eval_h1_balanced_seed$seed.json"

    $proc = Start-Process -FilePath "python.exe" -ArgumentList @(
        "-m", "research.runners.text_train_curriculum",
        "--seed", "$seed",
        "--phase1-episodes", "0",
        "--phase2-episodes", "100",
        "--phase3-replays", "500",
        "--phase3-balanced-directions",
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
"=== H1 BATCH COMPLETE at $(Get-Date) ===" | Out-File -Append $masterLog
