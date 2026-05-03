# Replaces batch 2 + H4 dose of the running fundamentals sweep with
# dt=1.0 versions. Launched by `wait_batch1_then_kill_and_relaunch.ps1`
# after batch 1 of the original sweep completes.
#
# Why: batch 1 already ran at dt=0.5 (already in flight). Replacing
# the rest with dt=1.0 saves ~25 min on batch 2 + ~35 min on H4 dose
# = ~60 min wall clock. Combined with auto-followup at dt=1.0 (saves
# ~30 min), total savings ~90 min.
#
# Critical: emits "ARCH SWEEP SEED 42 COMPLETE" marker into the
# original master log so the existing wait_arch_then_followup waiter
# (PID 49476) triggers the auto-followup correctly.

$outDir = "research/findings/raw/g11_bg"
$masterLog = "$outDir/run_arch_sweep_seed42.master.log"

"" | Out-File -Append $masterLog
"--- INTERRUPTED: batch 2 + dose redirected to dt=1.0 at $(Get-Date) ---" | Out-File -Append $masterLog

# Variant args (same biology fixes, just paired with dt=1.0 + halved windows)
$variants = @(
    @{
        Name = "heb_drive_dt1"
        Args = @("--enable-hebbian", "--hebbian-weight-decay", "1e-7",
                 "--lang-input-drive-pA", "1000")
    },
    @{
        Name = "heb_stdp_dt1"
        Args = @("--enable-hebbian", "--hebbian-weight-decay", "1e-7",
                 "--stdp-w-max", "10")
    },
    @{
        Name = "drive_stdp_dt1"
        Args = @("--lang-input-drive-pA", "1000", "--stdp-w-max", "10")
    }
)

# Run batch 2 in parallel-3 at dt=1.0
"" | Out-File -Append $masterLog
"--- Batch 2 (dt=1.0, parallel-3) at $(Get-Date) ---" | Out-File -Append $masterLog
$procs = @()
foreach ($v in $variants) {
    $name = $v.Name
    $logFile = "$outDir/arch_$name.seed42.log"
    $errFile = "$outDir/arch_$name.seed42.log.err"
    $pidFile = "$outDir/arch_$name.seed42.pid"
    $outStats = "$outDir/text_eval_arch_$name`_seed42.json"

    $args = @(
        "-m", "research.runners.text_train_curriculum",
        "--seed", "42",
        "--phase1-episodes", "0",
        "--phase2-episodes", "100",
        "--phase3-replays", "0",
        "--steps-per-episode", "30",
        "--stim-steps-per-step", "100",
        "--reset-steps", "50",
        "--dt-ms", "1.0",
        "--out-stats", $outStats
    ) + $v.Args

    $proc = Start-Process -FilePath "python.exe" -ArgumentList $args `
        -RedirectStandardOutput $logFile -RedirectStandardError $errFile `
        -PassThru -NoNewWindow

    $proc.Id | Out-File -FilePath $pidFile -Encoding ASCII
    "Variant $name launched as PID $($proc.Id)" | Out-File -Append $masterLog
    $procs += @{ Proc = $proc; PidFile = $pidFile; Name = $name }
}
foreach ($info in $procs) {
    $info.Proc.WaitForExit()
    "Variant $($info.Name) finished (exit $($info.Proc.ExitCode)) at $(Get-Date)" | Out-File -Append $masterLog
    if (Test-Path $info.PidFile) {
        Move-Item -Path $info.PidFile -Destination "$($info.PidFile).done" -Force
    }
}

# H4 dose-1000 test at dt=1.0
"" | Out-File -Append $masterLog
"--- H4 dose-1000 test at dt=1.0 at $(Get-Date) ---" | Out-File -Append $masterLog
$h4Pid = "$outDir/h4_dose_test_dt1_seed42.pid"
$h4Proc = Start-Process -FilePath "python.exe" -ArgumentList @(
    "-m", "research.runners.text_pfc_bypass_isolation",
    "--seed", "42",
    "--n-events-per-direction", "1000",
    "--stim-steps-per-step", "100",
    "--reset-steps", "50",
    "--dt-ms", "1.0",
    "--out-stats", "$outDir/text_eval_h4_dose1000_dt1_seed42.json"
) -RedirectStandardOutput "$outDir/h4_dose_test_dt1_seed42.log" `
  -RedirectStandardError "$outDir/h4_dose_test_dt1_seed42.log.err" `
  -PassThru -NoNewWindow

$h4Proc.Id | Out-File -FilePath $h4Pid -Encoding ASCII
"H4 dose-dt1 launched as PID $($h4Proc.Id)" | Out-File -Append $masterLog
$h4Proc.WaitForExit()
"H4 dose-dt1 finished (exit $($h4Proc.ExitCode)) at $(Get-Date)" | Out-File -Append $masterLog
if (Test-Path $h4Pid) {
    Move-Item -Path $h4Pid -Destination "$h4Pid.done" -Force
}

# Emit COMPLETE marker so the existing waiter (PID 49476) triggers auto-followup
"" | Out-File -Append $masterLog
"=== ARCH SWEEP SEED 42 COMPLETE at $(Get-Date) ===" | Out-File -Append $masterLog
