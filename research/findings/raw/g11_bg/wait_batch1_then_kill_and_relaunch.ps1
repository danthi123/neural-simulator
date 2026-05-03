# Polls the running fundamentals sweep master log for batch 1 completion
# (last variant of batch 1 finishing), then kills the sweep process before
# batch 2 launches and replaces with the dt=1.0 batch 2 + dose script.
#
# Detection: batch 1 launches "heb_only", "drive_5x", "stdp_wmax_10" in
# parallel. Their finish messages are "Variant heb_only finished",
# "Variant drive_5x finished", "Variant stdp_wmax_10 finished" — when
# all three appear in run_arch_sweep_seed42.master.log, batch 1 is done.
#
# Right after batch 1, the sweep script's loop iterates to batch 2 and
# launches 3 more parallel variants. We need to kill BEFORE that iter
# starts. The window is short (a few hundred ms between batches in the
# sweep ps1), so we poll faster (every 5s) and check carefully.
#
# Safer approach: after detecting batch 1 done, check if any "heb_drive"
# or "heb_stdp" or "drive_stdp" PID files exist. If yes, batch 2
# already started — abort the swap to avoid killing in-flight work.
# If no, kill the sweep process and launch the replacement.

$outDir = "research/findings/raw/g11_bg"
$masterLog = "$outDir/run_arch_sweep_seed42.master.log"
$waitLog = "$outDir/wait_batch1_swap.log"
$pidFile = "$outDir/wait_batch1_swap.orchestrator-pid"
$replaceScript = "research/findings/raw/g11_bg/run_arch_sweep_batch2_dt1.ps1"

$PID | Out-File -FilePath $pidFile -Encoding ASCII

"=== Wait-batch1-swap waiter started $(Get-Date) ===" | Out-File -FilePath $waitLog
"Polling $masterLog every 5s for all 3 batch 1 variants finished" | Out-File -Append $waitLog

while ($true) {
    if (Test-Path $masterLog) {
        $text = Get-Content $masterLog -Raw -ErrorAction SilentlyContinue
        # Need to handle UTF-16 BOM and spaces from Out-File on Windows
        $textNorm = $text -replace '\s', ''
        $b1done = ($textNorm -match "Variantheb_onlyfinished") -and `
                  ($textNorm -match "Variantdrive_5xfinished") -and `
                  ($textNorm -match "Variantstdp_wmax_10finished")
        if ($b1done) {
            "Batch 1 detected complete at $(Get-Date)" | Out-File -Append $waitLog
            break
        }
    }
    Start-Sleep -Seconds 5
}

# Safety: check no batch-2 PID files exist yet
$batch2Started = (Test-Path "$outDir/arch_heb_drive.seed42.pid") -or `
                 (Test-Path "$outDir/arch_heb_stdp.seed42.pid") -or `
                 (Test-Path "$outDir/arch_drive_stdp.seed42.pid")
if ($batch2Started) {
    "ABORT: batch 2 already started before we could swap. Letting it run at dt=0.5." | Out-File -Append $waitLog
    if (Test-Path $pidFile) {
        Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
    }
    return
}

# Kill the sweep process. Find it by looking for python OR powershell
# processes, but the sweep ps1 runs as a powershell.exe.
"" | Out-File -Append $waitLog
"--- Killing original sweep process at $(Get-Date) ---" | Out-File -Append $waitLog

# Identify the sweep ps1 process by command line (less reliable on Windows
# without WMIC, but works via Get-CimInstance)
try {
    $sweepProc = Get-CimInstance -ClassName Win32_Process -Filter "Name='powershell.exe'" |
        Where-Object { $_.CommandLine -match "run_arch_sweep_seed42.ps1" } |
        Select-Object -First 1
    if ($sweepProc) {
        "Found sweep process PID $($sweepProc.ProcessId)" | Out-File -Append $waitLog
        Stop-Process -Id $sweepProc.ProcessId -Force
        Start-Sleep -Seconds 2
        "Sweep process stopped" | Out-File -Append $waitLog
    } else {
        "No sweep process found via CommandLine match - may have already exited" | Out-File -Append $waitLog
    }
} catch {
    "Error stopping sweep process: $_" | Out-File -Append $waitLog
    "Continuing anyway - replacement will run regardless" | Out-File -Append $waitLog
}

# Launch the dt=1.0 replacement
"" | Out-File -Append $waitLog
"--- Launching dt=1.0 replacement at $(Get-Date) ---" | Out-File -Append $waitLog
$repl = Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-File", $replaceScript `
    -WorkingDirectory "E:/Documents/Projects/sim" `
    -PassThru -NoNewWindow

"Replacement launched as PID $($repl.Id) at $(Get-Date)" | Out-File -Append $waitLog
$repl.WaitForExit()
"Replacement finished (exit $($repl.ExitCode)) at $(Get-Date)" | Out-File -Append $waitLog

if (Test-Path $pidFile) {
    Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
}

"" | Out-File -Append $waitLog
"=== Wait-batch1-swap COMPLETE at $(Get-Date) ===" | Out-File -Append $waitLog
