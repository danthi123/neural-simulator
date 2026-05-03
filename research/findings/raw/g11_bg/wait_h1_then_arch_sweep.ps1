# Polling waiter — waits for H1 batch complete marker then launches
# architectural sweep on seed 42.
#
# H1 marker: "H1 BATCH COMPLETE" in run_h1.master.log
# OR fallback: "ALL BATCHES COMPLETE" in run_h4_then_h1.super.log

$outDir = "research/findings/raw/g11_bg"
$h1Log = "$outDir/run_h1.master.log"
$superLog = "$outDir/run_h4_then_h1.super.log"
$waitLog = "$outDir/wait_arch_sweep.log"
$pidFile = "$outDir/wait_arch_sweep.orchestrator-pid"
$archScript = "research/findings/raw/g11_bg/run_arch_sweep_seed42.ps1"

$PID | Out-File -FilePath $pidFile -Encoding ASCII

"=== Waiting for H1/super marker $(Get-Date) ===" | Out-File -FilePath $waitLog
"Polling $h1Log and $superLog every 60s" | Out-File -Append $waitLog

while ($true) {
    $h1Done = $false
    $superDone = $false
    if (Test-Path $h1Log) {
        $h1Text = Get-Content $h1Log -Raw -ErrorAction SilentlyContinue
        if ($h1Text -match "H1 BATCH COMPLETE") {
            $h1Done = $true
        }
    }
    if (Test-Path $superLog) {
        $superText = Get-Content $superLog -Raw -ErrorAction SilentlyContinue
        if ($superText -match "ALL BATCHES COMPLETE") {
            $superDone = $true
        }
    }
    if ($h1Done -or $superDone) {
        if ($h1Done) {
            "H1 marker detected at $(Get-Date)" | Out-File -Append $waitLog
        }
        if ($superDone) {
            "Super-orch ALL-COMPLETE marker detected at $(Get-Date)" | Out-File -Append $waitLog
        }
        break
    }
    Start-Sleep -Seconds 60
}

"" | Out-File -Append $waitLog
"--- Launching architectural sweep at $(Get-Date) ---" | Out-File -Append $waitLog

$proc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-File", $archScript `
    -WorkingDirectory "E:/Documents/Projects/sim" `
    -PassThru -NoNewWindow

"Arch sweep launched as PID $($proc.Id) at $(Get-Date)" | Out-File -Append $waitLog
$proc.WaitForExit()
"Arch sweep finished (exit $($proc.ExitCode)) at $(Get-Date)" | Out-File -Append $waitLog

# Move our pid file to done
if (Test-Path $pidFile) {
    Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
}

"" | Out-File -Append $waitLog
"=== ARCH SWEEP CHAIN COMPLETE at $(Get-Date) ===" | Out-File -Append $waitLog
