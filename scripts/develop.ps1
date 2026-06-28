# Self-driven longitudinal develop run -- start / pause / resume / stop / status.
# Drive the artificial-life brain-development run with NO Claude in the loop.
#
#   .\scripts\develop.ps1 start      # start (or resume) the run -- prints per-day progress
#   .\scripts\develop.ps1 pause      # stop cleanly at the next day boundary (zero work lost)
#   .\scripts\develop.ps1 resume     # remove the pause + continue from the saved day
#   .\scripts\develop.ps1 stop       # same as pause (graceful); or press Ctrl-C in the run window
#   .\scripts\develop.ps1 status     # show day / vocab / facts (no GPU needed; safe anytime)
#
# Run in the BACKGROUND so you can close the terminal:
#   Start-Process pwsh -ArgumentList '-NoProfile','-File','scripts\develop.ps1','start'
# Then pause / status from any other terminal.
#
# Extra flags after the verb pass through to the runner, e.g.:
#   .\scripts\develop.ps1 start --max-windows-per-day 4000

param([Parameter(Position = 0)][ValidateSet('start', 'resume', 'pause', 'stop', 'status')][string]$cmd = 'status')

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$pauseFile = Join-Path $repo 'bridges\developed\run3day\PAUSE'
$bundleDir = Join-Path $repo 'bridges\developed\run3day'

function New-PauseFile {
    $dir = Split-Path -Parent $pauseFile
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    if (-not (Test-Path $pauseFile)) { New-Item -ItemType File -Path $pauseFile | Out-Null }
}

switch ($cmd) {
    'pause' {
        New-PauseFile
        Write-Host "[develop] PAUSE requested -> the run stops at the next day boundary (zero completed work lost)."
        Write-Host "          resume with:  .\scripts\develop.ps1 resume"
    }
    'stop' {
        New-PauseFile
        Write-Host "[develop] STOP requested (graceful -- stops at the next day boundary)."
        Write-Host "          (or press Ctrl-C in the run window; the last completed day is always saved.)"
    }
    'status' {
        $env:SIM_BACKEND = 'numpy'   # status only reads the saved lineage -- no GPU needed
        python -m research.runners.develop_run --status
    }
    default {
        # start | resume
        if (Test-Path $pauseFile) { Remove-Item -Force $pauseFile; Write-Host "[develop] removed the PAUSE sentinel." }
        $env:SIM_BACKEND = 'cupy'
        Write-Host "[develop] starting/resuming the develop run."
        Write-Host "[develop] Ctrl-C stops cleanly (the last completed day is saved); per-day brains -> $bundleDir"
        python -m research.runners.develop_run @args
    }
}
