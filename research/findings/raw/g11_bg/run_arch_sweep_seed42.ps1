# FUNDAMENTALS SWEEP — replaces the original v2-tweak arch sweep.
#
# Per the permuted-label finding (0/29 aligned across all v2-variant
# evals), more "tweak v2" experiments are unlikely to produce real
# learning. Instead, this sweep tests the FUNDAMENTAL hypotheses for
# why the architecture has unaligned structure:
#
#   1. hebbian_only       — re-enable Hebbian with reduced decay
#                            (Hebbian disable was the v2 shortcut; the
#                             biology-correct fix is reduced decay rate)
#   2. drive_5x           — language drive 1000 pA (vs 200) so language
#                            dominates cascade noise during training
#   3. stdp_wmax_10       — let STDP weights grow to 10 (vs 5) for more
#                            differentiation between word patterns
#   4. heb_drive          — Hebbian + strong drive (most biology-correct)
#   5. heb_stdp           — Hebbian + headroom
#   6. drive_stdp         — strong drive + headroom (no biology change)
#
# All run with 6-cheap-variants-x-1-seed = 6 runs.
# Parallel-3 execution: 6 runs / 3 parallel = 2 batches x ~30 min = ~60 min.
#
# ALSO (after the cheap sweep): single H4 plasticity-dose test at 1000
# events/dir on seed 42, to test whether H4's "below chance" was a
# plasticity-dose artifact. Adds ~70 min.
#
# Auto-followup (separate waiter, PID 49476) picks winner via
# auto_followup_arch_winner.ps1 — picks variant with HIGHEST true-label
# accuracy on seed 42, runs 6-seed validation in parallel-3 if >= 32%.

$outDir = "research/findings/raw/g11_bg"
$masterLog = "$outDir/run_arch_sweep_seed42.master.log"

"=== FUNDAMENTALS sweep (seed 42, parallel-3) started $(Get-Date) ===" | Out-File -FilePath $masterLog
"" | Out-File -Append $masterLog
"Variants test biology-correct fixes vs v2 (Hebbian off, stdp_wmax 5, drive 200)." | Out-File -Append $masterLog
"" | Out-File -Append $masterLog

# Each variant = (Name, Args[]) for text_train_curriculum
# All variants share v2 base config: phase1=0, phase2=100, phase3=0,
# stim 200, reset 100. Variant-specific args override.
$variants = @(
    @{
        Name = "heb_only"
        Args = @("--enable-hebbian", "--hebbian-weight-decay", "1e-7")
    },
    @{
        Name = "drive_5x"
        Args = @("--lang-input-drive-pA", "1000")
    },
    @{
        Name = "stdp_wmax_10"
        Args = @("--stdp-w-max", "10")
    },
    @{
        Name = "heb_drive"
        Args = @("--enable-hebbian", "--hebbian-weight-decay", "1e-7",
                 "--lang-input-drive-pA", "1000")
    },
    @{
        Name = "heb_stdp"
        Args = @("--enable-hebbian", "--hebbian-weight-decay", "1e-7",
                 "--stdp-w-max", "10")
    },
    @{
        Name = "drive_stdp"
        Args = @("--lang-input-drive-pA", "1000", "--stdp-w-max", "10")
    }
)

# Run 3 at a time
$parallelism = 3
$batchIndex = 0
for ($i = 0; $i -lt $variants.Count; $i += $parallelism) {
    $batchIndex++
    $batchEnd = [Math]::Min($i + $parallelism - 1, $variants.Count - 1)
    "" | Out-File -Append $masterLog
    "--- Batch $batchIndex ($($variants[$i].Name) .. $($variants[$batchEnd].Name)) at $(Get-Date) ---" | Out-File -Append $masterLog

    $procs = @()
    for ($j = $i; $j -le $batchEnd; $j++) {
        $v = $variants[$j]
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
            "--stim-steps-per-step", "200",
            "--reset-steps", "100",
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
}

# Bonus: H4 plasticity-dose test at 1000 events/dir, seed 42 only.
# Tests whether H4's below-chance result (n=6, 23%) is a dose artifact.
"" | Out-File -Append $masterLog
"--- H4 dose test (1000 events/dir, ~70 min) at $(Get-Date) ---" | Out-File -Append $masterLog
$h4Pid = "$outDir/h4_dose_test_seed42.pid"
$h4Proc = Start-Process -FilePath "python.exe" -ArgumentList @(
    "-m", "research.runners.text_pfc_bypass_isolation",
    "--seed", "42",
    "--n-events-per-direction", "1000",
    "--stim-steps-per-step", "200",
    "--reset-steps", "100",
    "--out-stats", "$outDir/text_eval_h4_dose1000_seed42.json"
) -RedirectStandardOutput "$outDir/h4_dose_test_seed42.log" `
  -RedirectStandardError "$outDir/h4_dose_test_seed42.log.err" `
  -PassThru -NoNewWindow

$h4Proc.Id | Out-File -FilePath $h4Pid -Encoding ASCII
"H4 dose test launched as PID $($h4Proc.Id)" | Out-File -Append $masterLog
$h4Proc.WaitForExit()
"H4 dose test finished (exit $($h4Proc.ExitCode)) at $(Get-Date)" | Out-File -Append $masterLog
if (Test-Path $h4Pid) {
    Move-Item -Path $h4Pid -Destination "$h4Pid.done" -Force
}

"" | Out-File -Append $masterLog
"=== ARCH SWEEP SEED 42 COMPLETE at $(Get-Date) ===" | Out-File -Append $masterLog
