# Architectural sweep — seed 42 only, exploration tier.
# Runs after H1 finishes. Tests 3 high-information architectural changes:
#
# A. n-motor-per-action 50  (motor pool SNR — population readout variance)
# B. token-sparsity 0.05    (orthogonal word codes — eliminates code overlap)
# C. A + B combined
#
# All run with v2 baseline config (Hebbian off, stdp_w_max=5, readout init 0.5)
# and standard 100-episode Phase 2 (no Phase 3 SWR — testing pure
# architectural deltas, not consolidation).
#
# If any variant gives W->A >= 35% on seed 42 (vs baseline 27%), that
# variant warrants full 6-seed validation on the next batch run.

$outDir = "research/findings/raw/g11_bg"
$masterLog = "$outDir/run_arch_sweep_seed42.master.log"

"=== Architectural sweep (seed 42) started $(Get-Date) ===" | Out-File -FilePath $masterLog
"" | Out-File -Append $masterLog
"Variant A: n-motor-per-action 50 (motor pool SNR)" | Out-File -Append $masterLog
"Variant B: token-sparsity 0.05 (orthogonal codes)" | Out-File -Append $masterLog
"Variant C: A + B combined" | Out-File -Append $masterLog
"" | Out-File -Append $masterLog
"All variants: 100 ep phase2, v2 config (Hebbian off, stdp_w_max=5, readout 0.5)" | Out-File -Append $masterLog

$variants = @(
    @{ Name = "motor50"; Args = @("--n-motor-per-action", "50") },
    @{ Name = "sparse005"; Args = @("--token-sparsity", "0.05") },
    @{ Name = "motor50_sparse005"; Args = @(
            "--n-motor-per-action", "50",
            "--token-sparsity", "0.05"
        ) }
)

foreach ($v in $variants) {
    $name = $v.Name
    "" | Out-File -Append $masterLog
    "--- Starting variant $name at $(Get-Date) ---" | Out-File -Append $masterLog
    $logFile = "$outDir/arch_$name.seed42.log"
    $errFile = "$outDir/arch_$name.seed42.log.err"
    $pidFile = "$outDir/arch_$name.seed42.pid"
    $outStats = "$outDir/text_eval_arch_$name`_seed42.json"

    # Standard v2 args + variant-specific.
    # v2 baseline uses phase1=0, phase2=100, phase3=0 (no SWR, since
    # SWR causes regression). Same config as text_eval_R3R6_100ep_HebOff_v2.
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

    $proc.WaitForExit()
    "Variant $name finished (exit $($proc.ExitCode)) at $(Get-Date)" | Out-File -Append $masterLog

    if (Test-Path $pidFile) {
        Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
    }
}

"" | Out-File -Append $masterLog
"=== ARCH SWEEP SEED 42 COMPLETE at $(Get-Date) ===" | Out-File -Append $masterLog
