# Polls run_minimal_isolation.master.log for completion, then launches
# the biology sweep (anti-cheat control + 3-condition x 6-seed).

$outDir = "research/findings/raw/g11_bg"
$minimalIsoMaster = "$outDir/run_minimal_isolation.master.log"
$waitLog = "$outDir/wait_biology_sweep.log"
$pidFile = "$outDir/wait_biology_sweep.orchestrator-pid"
$bioScript = "research/findings/raw/g11_bg/run_biology_sweep.ps1"

$PID | Out-File -FilePath $pidFile -Encoding ASCII

"=== Wait-for-minimal-iso started $(Get-Date) ===" | Out-File -FilePath $waitLog
"Polling $minimalIsoMaster every 30s for 'MINIMAL ISOLATION 6-SEED COMPLETE'" | Out-File -Append $waitLog

while ($true) {
    if (Test-Path $minimalIsoMaster) {
        $text = Get-Content $minimalIsoMaster -Raw -ErrorAction SilentlyContinue
        if ($text -match "MINIMAL ISOLATION 6-SEED COMPLETE") {
            "Minimal-iso completion marker detected at $(Get-Date)" | Out-File -Append $waitLog
            break
        }
    }
    Start-Sleep -Seconds 30
}

"" | Out-File -Append $waitLog
"--- Launching biology sweep at $(Get-Date) ---" | Out-File -Append $waitLog
$proc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-File", $bioScript `
    -WorkingDirectory "E:/Documents/Projects/sim" `
    -PassThru -NoNewWindow

"Biology sweep launched as PID $($proc.Id) at $(Get-Date)" | Out-File -Append $waitLog
$proc.WaitForExit()
"Biology sweep finished (exit $($proc.ExitCode)) at $(Get-Date)" | Out-File -Append $waitLog

if (Test-Path $pidFile) {
    Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
}

"" | Out-File -Append $waitLog
"=== BIOLOGY SWEEP CHAIN COMPLETE at $(Get-Date) ===" | Out-File -Append $waitLog
