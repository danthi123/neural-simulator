# v3 variant launcher — try multiple architectural improvements if v2c is borderline.
#
# Three v3 options, in order of expected impact (lowest cost first):
#   v3a: interleaved training (matches Tier 1 pattern)
#   v3b: stronger topographic prior (3.0/0.3 = 10x ratio)
#   v3c: combined (interleaved + stronger topographic)
#
# Pick the one most likely to push borderline FAILs into PASS based on
# v2c failure pattern (probe via concept_weight_probe first).

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("v3a", "v3b", "v3c")]
    [string]$Variant,

    [int]$Seed = 42,
    [int]$Events = 200,
    [int]$LangIn = 2048,
    [int]$PerPool = 200,
    [int]$FsPerPool = 24
)

$ErrorActionPreference = "Stop"
Set-Location "E:\Documents\Projects\sim"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$OutDir = "research\findings\raw\g11_bg\concept_pool_demo"

# Pick CLI flags per variant
switch ($Variant) {
    "v3a" {
        Write-Host "[$Variant] interleaved training (default topographic 2.0/0.5)"
        $extraArgs = @("--interleaved")
    }
    "v3b" {
        Write-Host "[$Variant] stronger topographic (3.0/0.3 = 10x ratio)"
        $extraArgs = @("--topographic-factor", "3.0", "--off-target-factor", "0.3")
    }
    "v3c" {
        Write-Host "[$Variant] combined: interleaved + stronger topographic"
        $extraArgs = @("--interleaved", "--topographic-factor", "3.0", "--off-target-factor", "0.3")
    }
}

$outPath = "$OutDir\seed${Seed}_${Variant}.json"
$logPath = "$OutDir\seed${Seed}_${Variant}.log"
$savePath = "$OutDir\seed${Seed}_${Variant}.simstate.h5"

Write-Host "[$Variant] Launching seed $Seed..."
$baseArgs = @(
    "--seed", $Seed,
    "--n-train-events", $Events,
    "--n-lang-input", $LangIn,
    "--n-per-pool", $PerPool,
    "--n-fs-per-pool", $FsPerPool,
    "--save-bridge", $savePath,
    "--out", $outPath
)
python -m research.runners.concept_pool_demo @baseArgs @extraArgs 2>&1 | Tee-Object -FilePath $logPath
Write-Host "[$Variant] Done (exit $LASTEXITCODE)"
