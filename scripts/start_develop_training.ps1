# Launch the 24/7 develop-loop TRAINING watchdog DETACHED so it survives a terminal close / parent exit.
#
# THE OWNER'S BAR: "...won't get accidentally killed no matter what..." This is requirement (2): the training
# (and its watchdog) must keep running after you close the terminal / log out of the SSH session / the launching
# shell exits. `Start-Process -WindowStyle Hidden` launches `develop_train_watchdog.ps1` as an INDEPENDENT process
# whose lifetime is NOT tied to this shell — closing this terminal does not kill it.
#
# It writes the watchdog PID to bridges/developed/.train_watchdog.pid so you can find + (deliberately) stop it.
#
# Usage:
#   pwsh scripts/start_develop_training.ps1
#   pwsh scripts/start_develop_training.ps1 -LineageRoot bridges/developed/curriculum `
#        -BundleRoot bridges/developed/curriculum/bundles -PerDayBundles -MaxWindowsPerDay 2500
#
# Verify it's alive:
#   Get-Content bridges/developed/curriculum/curriculum/heartbeat.json   # last completed day + liveness ts
#   Get-Content research/findings/raw/develop_train_watchdog.log -Tail 20
#
# PAUSE (free the GPU to game; resumable):   New-Item -ItemType File bridges/PAUSE
# RESUME:                                    Remove-Item bridges/PAUSE
# STOP it ENTIRELY (deliberate):             Stop-Process -Id (Get-Content bridges/developed/.train_watchdog.pid)

param(
  [string]$LineageRoot   = "bridges/developed/curriculum",
  [string]$LineageName   = "curriculum",
  [int]$Seed             = 42,
  [int]$MaxWindowsPerDay = 2500,
  [int]$NHub             = 200,
  [int]$NPer             = 12,
  [int]$D                = 128,
  [string]$BundleRoot    = "",
  [switch]$PerDayBundles,
  [string]$CorpusPath    = "",
  [string]$PauseFile     = "bridges/PAUSE",
  [string]$Backend       = "cupy"
)

$ErrorActionPreference = "Stop"
$Repo     = "E:\Documents\Projects\sim"
$Watchdog = Join-Path $Repo "scripts\develop_train_watchdog.ps1"
$PidFile  = Join-Path $Repo "bridges\developed\.train_watchdog.pid"

# Build the watchdog argument list (forward everything).
$wdArgs = @(
  "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Watchdog,
  "-LineageRoot", $LineageRoot, "-LineageName", $LineageName,
  "-Seed", $Seed, "-MaxWindowsPerDay", $MaxWindowsPerDay,
  "-NHub", $NHub, "-NPer", $NPer, "-D", $D,
  "-PauseFile", $PauseFile, "-Backend", $Backend
)
if ($BundleRoot -ne "") { $wdArgs += @("-BundleRoot", $BundleRoot) }
if ($PerDayBundles)     { $wdArgs += "-PerDayBundles" }
if ($CorpusPath -ne "") { $wdArgs += @("-CorpusPath", $CorpusPath) }

# Launch pwsh running the watchdog as a DETACHED, hidden, terminal-independent process.
$pwsh = (Get-Command pwsh -ErrorAction SilentlyContinue)?.Source
if (-not $pwsh) { $pwsh = (Get-Command powershell).Source }

$proc = Start-Process -FilePath $pwsh -ArgumentList $wdArgs -WindowStyle Hidden -PassThru
New-Item -ItemType Directory -Force (Split-Path $PidFile) | Out-Null
Set-Content -Path $PidFile -Value $proc.Id

Write-Output ("Started develop-training watchdog DETACHED (pid={0})." -f $proc.Id)
Write-Output ("  lineage : {0}/{1}" -f $LineageRoot, $LineageName)
Write-Output ("  log     : research/findings/raw/develop_train_watchdog.log")
Write-Output ("  pid file: {0}" -f $PidFile)
Write-Output ""
Write-Output "It survives this terminal closing. To PAUSE (free the GPU): New-Item -ItemType File $PauseFile"
Write-Output "To STOP entirely: Stop-Process -Id (Get-Content '$PidFile')"
