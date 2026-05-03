# Polling waiter — waits for arch sweep complete marker then launches
# auto-followup (6-seed validation of winning variant if it clears 32%).

$outDir = "research/findings/raw/g11_bg"
$archMasterLog = "$outDir/run_arch_sweep_seed42.master.log"
$waitLog = "$outDir/wait_followup.log"
$pidFile = "$outDir/wait_followup.orchestrator-pid"
$followupScript = "research/findings/raw/g11_bg/auto_followup_arch_winner.ps1"

$PID | Out-File -FilePath $pidFile -Encoding ASCII

"=== Waiting for arch sweep marker $(Get-Date) ===" | Out-File -FilePath $waitLog
"Polling $archMasterLog every 60s for 'ARCH SWEEP SEED 42 COMPLETE'" | Out-File -Append $waitLog

while ($true) {
    if (Test-Path $archMasterLog) {
        $text = Get-Content $archMasterLog -Raw -ErrorAction SilentlyContinue
        if ($text -match "ARCH SWEEP SEED 42 COMPLETE") {
            "Arch sweep marker detected at $(Get-Date)" | Out-File -Append $waitLog
            break
        }
    }
    Start-Sleep -Seconds 60
}

"" | Out-File -Append $waitLog
"--- Launching auto-followup at $(Get-Date) ---" | Out-File -Append $waitLog

$proc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-File", $followupScript `
    -WorkingDirectory "E:/Documents/Projects/sim" `
    -PassThru -NoNewWindow

"Auto-followup launched as PID $($proc.Id) at $(Get-Date)" | Out-File -Append $waitLog
$proc.WaitForExit()
"Auto-followup finished (exit $($proc.ExitCode)) at $(Get-Date)" | Out-File -Append $waitLog

if (Test-Path $pidFile) {
    Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
}

"" | Out-File -Append $waitLog
"=== AUTO-FOLLOWUP CHAIN COMPLETE at $(Get-Date) ===" | Out-File -Append $waitLog
