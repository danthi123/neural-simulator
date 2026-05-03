# H4 plasticity-dose test — same architecture, but 10x events.
# Tests whether the H4 inversion (n=4 mean 23%, below chance) is a
# plasticity-dose artifact rather than a real architecture-procedure
# inversion.
#
# Math: H4 standard 400 events x ~220 sub-steps = 88k total sub-step
# plasticity opportunities. v2 Phase 2 = 100 ep x 30 steps x 320
# sub-steps = 960k (11x more). So H4 is severely under-trained.
#
# This run: 1000 events/dir = 4000 events x 220 = 880k sub-steps.
# Comparable plasticity dose to Phase 2.
#
# Single seed (42) for ~70 min runtime.
# If result >= 40%: dose was the bottleneck. Re-run 6 seeds tomorrow.
# If result <= 30%: architecture-procedure inversion is real.
# If 30-40%: dose helps but architecture is still limiting.

$outDir = "research/findings/raw/g11_bg"
$masterLog = "$outDir/run_h4_dose_test.master.log"
$logFile = "$outDir/h4_dose_test_seed42.log"
$errFile = "$outDir/h4_dose_test_seed42.log.err"
$pidFile = "$outDir/h4_dose_test_seed42.pid"
$outStats = "$outDir/text_eval_h4_dose1000_seed42.json"

"=== H4 plasticity-dose test (1000 events/dir, seed 42) started $(Get-Date) ===" | Out-File -FilePath $masterLog
"" | Out-File -Append $masterLog
"Standard H4: 400 events/dir = 88k sub-steps" | Out-File -Append $masterLog
"This: 1000 events/dir = 880k sub-steps (matches v2 Phase 2 dose)" | Out-File -Append $masterLog

$proc = Start-Process -FilePath "python.exe" -ArgumentList @(
    "-m", "research.runners.text_pfc_bypass_isolation",
    "--seed", "42",
    "--n-events-per-direction", "1000",
    "--stim-steps-per-step", "200",
    "--reset-steps", "100",
    "--out-stats", $outStats
) -RedirectStandardOutput $logFile -RedirectStandardError $errFile -PassThru -NoNewWindow

$proc.Id | Out-File -FilePath $pidFile -Encoding ASCII
"H4-dose test launched as PID $($proc.Id)" | Out-File -Append $masterLog

$proc.WaitForExit()
"H4-dose test finished (exit $($proc.ExitCode)) at $(Get-Date)" | Out-File -Append $masterLog

if (Test-Path $pidFile) {
    Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
}

"" | Out-File -Append $masterLog
"=== H4 DOSE TEST COMPLETE at $(Get-Date) ===" | Out-File -Append $masterLog
