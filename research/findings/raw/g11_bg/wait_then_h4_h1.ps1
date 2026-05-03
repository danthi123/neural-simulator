# Waits for the v2+SWR 4-seed batch to finish, then launches the H4
# super-orchestrator. Started in parallel with the existing batch so
# the chain runs automatically.

$outDir = "research/findings/raw/g11_bg"
$waitLog = "$outDir/wait_h4_h1.log"
$batchMasterLog = "$outDir/run_swr_remaining.master.log"
$completionMarker = "ALL SEEDS COMPLETE"

"=== Waiting for v2+SWR 4-seed batch at $(Get-Date) ===" | Out-File -FilePath $waitLog
"Polling $batchMasterLog every 60s for marker '$completionMarker'" | Out-File -Append $waitLog

while ($true) {
    if (Test-Path $batchMasterLog) {
        $content = Get-Content $batchMasterLog -Raw -ErrorAction SilentlyContinue
        if ($content -and $content.Contains($completionMarker)) {
            "Batch completion marker detected at $(Get-Date)" | Out-File -Append $waitLog
            break
        }
    }
    Start-Sleep -Seconds 60
}

# Launch the super-orchestrator (H4 then H1)
"--- Launching H4-then-H1 super-orchestrator at $(Get-Date) ---" | Out-File -Append $waitLog
$super = Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-File", "research/findings/raw/g11_bg/run_h4_then_h1.ps1" `
    -WorkingDirectory "E:/Documents/Projects/sim" `
    -PassThru -NoNewWindow

"Super-orchestrator launched as PID $($super.Id) at $(Get-Date)" | Out-File -Append $waitLog

# Don't WaitForExit on this — let it run independently. The waiter's
# job is done.
