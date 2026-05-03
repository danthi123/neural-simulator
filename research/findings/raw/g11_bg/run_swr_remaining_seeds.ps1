# 2026-05-03: complete the 6-seed SWR validation set.
# Already done: seeds 42, 43.
# This script: 44, 100, 101, 102 — sequentially, ~70 min each = ~5h total.
# Run detached so it survives terminal close.

# NOTE: master orchestrator PID is written to *.orchestrator-pid (NOT *.pid)
# so it doesn't appear in the webapp's inflight panel — that scanner globs
# for *.pid files and treats every match as a training run.
$seeds = @(44, 100, 101, 102)
$outDir = "research/findings/raw/g11_bg"
$masterLog = "$outDir/run_swr_remaining.master.log"

"=== SWR remaining-seeds run started $(Get-Date) ===" | Out-File -FilePath $masterLog
"Seeds: $($seeds -join ',')" | Out-File -Append $masterLog

foreach ($seed in $seeds) {
    "" | Out-File -Append $masterLog
    "--- Starting seed $seed at $(Get-Date) ---" | Out-File -Append $masterLog
    $logFile = "$outDir/v2_swr500_seed$seed.log"
    $errFile = "$outDir/v2_swr500_seed$seed.log.err"
    $pidFile = "$outDir/v2_swr500_seed$seed.pid"
    $outStats = "$outDir/text_eval_v2_swr500_seed$seed.json"

    # Launch foreground (within this script) so we wait for completion
    $proc = Start-Process -FilePath "python.exe" -ArgumentList @(
        "-m", "research.runners.text_train_curriculum",
        "--seed", "$seed",
        "--phase1-episodes", "0",
        "--phase2-episodes", "100",
        "--phase3-replays", "500",
        "--stim-steps-per-step", "200",
        "--reset-steps", "100",
        "--out-stats", $outStats
    ) -RedirectStandardOutput $logFile -RedirectStandardError $errFile -PassThru -NoNewWindow

    $proc.Id | Out-File -FilePath $pidFile -Encoding ASCII
    "Seed $seed launched as PID $($proc.Id)" | Out-File -Append $masterLog

    # Wait synchronously for this seed before moving to the next
    $proc.WaitForExit()
    "Seed $seed completed (exit $($proc.ExitCode)) at $(Get-Date)" | Out-File -Append $masterLog

    # Mark .pid as done so the webapp's inflight panel filters it out
    if (Test-Path $pidFile) {
        Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
    }
}

"" | Out-File -Append $masterLog
"=== ALL SEEDS COMPLETE at $(Get-Date) ===" | Out-File -Append $masterLog
