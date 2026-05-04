# Minimal language->motor isolation experiment, 6 seeds in parallel-3 at dt=1.0.
#
# THE decisive test: can the architecture learn word-action mapping
# AT ALL when stripped of cascade interference?
#
# Architecture: language_input (256) -> motor_X (25 each, 4 actions)
# NO cascade, NO PFC, NO retina, NO visuomotor
#
# Threshold for declaring real learning:
#   aligned ratio across 6 seeds >= 4/6 (random chance is 1/24/seed,
#   so 4/6 has joint p < 1e-3)
#
# If aligned >= 4/6: cascade IS the dominant interference. Path forward:
#   reduce cascade contribution, train language separately, scale up.
#
# If aligned < 4/6: fundamental issue (plasticity dose, soft-bound STDP,
#   sparse-code overlap, eval methodology). Need bigger rethink.
#
# Wall clock estimate: 1000 events/dir at dt=1.0 = ~50 min/seed at single
# process. Parallel-3 = ~80 min total for 6 seeds.

$outDir = "research/findings/raw/g11_bg"
$masterLog = "$outDir/run_minimal_isolation.master.log"

"=== Minimal isolation 6-seed (parallel-3, dt=1.0) started $(Get-Date) ===" | Out-File -FilePath $masterLog
"" | Out-File -Append $masterLog
"Architecture: language_input(256) -> motor_X(25 each, 4 actions)" | Out-File -Append $masterLog
"NO cascade, NO PFC, NO retina, NO visuomotor" | Out-File -Append $masterLog
"1000 paired-stim events per direction at dt=1.0" | Out-File -Append $masterLog
"" | Out-File -Append $masterLog

$seeds = @(42, 43, 44, 100, 101, 102)
$parallelism = 3

for ($i = 0; $i -lt $seeds.Count; $i += $parallelism) {
    $batchEnd = [Math]::Min($i + $parallelism - 1, $seeds.Count - 1)
    "" | Out-File -Append $masterLog
    "--- Batch (seeds $($seeds[$i])..$($seeds[$batchEnd])) at $(Get-Date) ---" | Out-File -Append $masterLog

    $procs = @()
    for ($j = $i; $j -le $batchEnd; $j++) {
        $seed = $seeds[$j]
        $logFile = "$outDir/minimal_iso_seed$seed.log"
        $errFile = "$outDir/minimal_iso_seed$seed.log.err"
        $pidFile = "$outDir/minimal_iso_seed$seed.pid"
        $outStats = "$outDir/text_eval_minimal_iso_seed$seed.json"

        $procArgs = @(
            "-m", "research.runners.text_minimal_isolation",
            "--seed", "$seed",
            "--n-events-per-direction", "1000",
            "--stim-steps-per-step", "100",
            "--reset-steps", "50",
            "--dt-ms", "1.0",
            "--out-stats", $outStats
        )

        $proc = Start-Process -FilePath "python.exe" -ArgumentList $procArgs `
            -RedirectStandardOutput $logFile -RedirectStandardError $errFile `
            -PassThru -NoNewWindow

        $proc.Id | Out-File -FilePath $pidFile -Encoding ASCII
        "Seed $seed launched as PID $($proc.Id)" | Out-File -Append $masterLog
        $procs += @{ Proc = $proc; PidFile = $pidFile; Seed = $seed }
    }

    foreach ($info in $procs) {
        $info.Proc.WaitForExit()
        "Seed $($info.Seed) finished (exit $($info.Proc.ExitCode)) at $(Get-Date)" | Out-File -Append $masterLog
        if (Test-Path $info.PidFile) {
            Move-Item -Path $info.PidFile -Destination "$($info.PidFile).done" -Force
        }
    }
}

# Final aligned summary
"" | Out-File -Append $masterLog
"--- Final aligned summary at $(Get-Date) ---" | Out-File -Append $masterLog
& python -m research.runners.permuted_label_check --pattern "text_eval_minimal_iso_seed*.json" 2>&1 | Out-File -Append $masterLog

"" | Out-File -Append $masterLog
"=== MINIMAL ISOLATION 6-SEED COMPLETE at $(Get-Date) ===" | Out-File -Append $masterLog
