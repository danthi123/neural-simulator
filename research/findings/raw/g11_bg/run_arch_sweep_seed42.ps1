# Architectural sweep — seed 42 only, exploration tier.
# Runs after H1 finishes. Tests 5 high-information architectural changes:
#
# A. motor50: --n-motor-per-action 50
#    Motor pool size 10 -> 50. Population-rate readout has lower variance
#    with larger population. If 28% ceiling is motor SNR, A breaks it.
# B. sparse005: --token-sparsity 0.05
#    12-13 active per word (vs 26 at default 0.1). Pairwise overlap drops
#    from 2-3 to 0-1. If 28% ceiling is code overlap interfering with
#    discrimination, B breaks it.
# C. lang512: --text-n-input-neurons 512 --text-n-output-neurons 512
#    Doubles language region capacity. More distinct dimensions for STDP
#    to sculpt word-specific weight patterns.
# D. motor50_sparse005: A + B
#    Combined upstream (orthogonal codes) + downstream (motor SNR). If
#    these bottlenecks compound, D > max(A, B).
# E. lang512_motor50: B + C
#    Bigger regions throughout. Tests whether "scale wins".
#
# All run with v2 baseline config (Hebbian off, stdp_w_max=5, readout
# init 0.5) and standard phase1=0, phase2=100, phase3=0 (no SWR — pure
# architectural deltas).
#
# If any variant gives W->A >= 35% on seed 42 (vs baseline 27%), that
# variant warrants full 6-seed validation in the next batch.

$outDir = "research/findings/raw/g11_bg"
$masterLog = "$outDir/run_arch_sweep_seed42.master.log"

"=== Architectural sweep (seed 42) started $(Get-Date) ===" | Out-File -FilePath $masterLog
"" | Out-File -Append $masterLog
"Variant A: motor50 (--n-motor-per-action 50)" | Out-File -Append $masterLog
"Variant B: sparse005 (--token-sparsity 0.05)" | Out-File -Append $masterLog
"Variant C: lang512 (--text-n-input-neurons 512 --text-n-output-neurons 512)" | Out-File -Append $masterLog
"Variant D: motor50_sparse005 (A + B combined)" | Out-File -Append $masterLog
"Variant E: lang512_motor50 (B + C combined)" | Out-File -Append $masterLog
"" | Out-File -Append $masterLog
"All variants: 100 ep phase2, v2 config (Hebbian off, stdp_w_max=5, readout 0.5)" | Out-File -Append $masterLog

$variants = @(
    @{ Name = "motor50"; Args = @("--n-motor-per-action", "50") },
    @{ Name = "sparse005"; Args = @("--token-sparsity", "0.05") },
    @{ Name = "lang512"; Args = @(
            "--text-n-input-neurons", "512",
            "--text-n-output-neurons", "512"
        ) },
    @{ Name = "motor50_sparse005"; Args = @(
            "--n-motor-per-action", "50",
            "--token-sparsity", "0.05"
        ) },
    @{ Name = "lang512_motor50"; Args = @(
            "--text-n-input-neurons", "512",
            "--text-n-output-neurons", "512",
            "--n-motor-per-action", "50"
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
"--- H4 dose-normalization test (1000 events/dir, ~70 min) at $(Get-Date) ---" | Out-File -Append $masterLog

# Bonus: H4 plasticity-dose test as 6th pseudo-variant. Tests whether
# H4's "below chance" result (n=4, 23% mean) is a plasticity-dose
# artifact. Uses text_pfc_bypass_isolation runner with --n-events-per-dir 1000.
$h4DoseLog = "$outDir/h4_dose_test_seed42.log"
$h4DoseErrFile = "$outDir/h4_dose_test_seed42.log.err"
$h4DosePidFile = "$outDir/h4_dose_test_seed42.pid"
$h4DoseOutStats = "$outDir/text_eval_h4_dose1000_seed42.json"

$doseProc = Start-Process -FilePath "python.exe" -ArgumentList @(
    "-m", "research.runners.text_pfc_bypass_isolation",
    "--seed", "42",
    "--n-events-per-direction", "1000",
    "--stim-steps-per-step", "200",
    "--reset-steps", "100",
    "--out-stats", $h4DoseOutStats
) -RedirectStandardOutput $h4DoseLog -RedirectStandardError $h4DoseErrFile -PassThru -NoNewWindow

$doseProc.Id | Out-File -FilePath $h4DosePidFile -Encoding ASCII
"H4 dose test launched as PID $($doseProc.Id)" | Out-File -Append $masterLog

$doseProc.WaitForExit()
"H4 dose test finished (exit $($doseProc.ExitCode)) at $(Get-Date)" | Out-File -Append $masterLog

if (Test-Path $h4DosePidFile) {
    Move-Item -Path $h4DosePidFile -Destination "$h4DosePidFile.done" -Force
}

"" | Out-File -Append $masterLog
"=== ARCH SWEEP SEED 42 COMPLETE at $(Get-Date) ===" | Out-File -Append $masterLog
