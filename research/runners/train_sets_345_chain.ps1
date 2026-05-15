# Chain training for sets 3, 4, 5 -- waits for each to finish before the next.
# Set 3 is assumed already in flight (started manually).
# This script waits for set3 to complete, then auto-launches set4 and set5.

$ErrorActionPreference = "Stop"
$OUT_DIR = "research/findings/raw/g11_bg/concept_pool_demo"
$SEED = 42

# Common training args (matches v16 production recipe).
$COMMON_ARGS = @(
    "--seed", $SEED,
    "--n-train-events", "200",
    "--n-lang-input", "2048",
    "--n-per-pool", "200",
    "--n-fs-per-pool", "24",
    "--weak-concept-dynamics",
    "--interleaved",
    "--topographic-factor", "3.0",
    "--off-target-factor", "0.3",
    "--enable-adjective",
    "--orthogonal-codes",
    "--sparsity", "0.05"
)

function Wait-ForCompletion {
    param([string]$LogFile, [string]$SetName)
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Waiting on $SetName completion..." -ForegroundColor Cyan
    while ($true) {
        if (Test-Path $LogFile) {
            $tail = Get-Content $LogFile -Tail 5 -ErrorAction SilentlyContinue
            $verdict = $tail | Where-Object { $_ -match "VERDICT|PASS|FAIL|=== concept_pool_demo" } | Select-Object -Last 1
            if ($verdict -and $verdict -match "VERDICT") {
                Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $SetName done: $verdict" -ForegroundColor Green
                return
            }
        }
        Start-Sleep -Seconds 30
    }
}

function Train-Set {
    param([string]$SetName)
    $module = "research.runners.concept_pool_demo_$SetName"
    $bridge = "$OUT_DIR/seed${SEED}_${SetName}.simstate.h5"
    $json   = "$OUT_DIR/seed${SEED}_${SetName}.json"
    $log    = "$OUT_DIR/seed${SEED}_${SetName}.log"
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] === Launching $SetName ===" -ForegroundColor Yellow
    $args = $COMMON_ARGS + @("--save-bridge", $bridge, "--out", $json)
    & python -m $module @args 2>&1 | Tee-Object -FilePath $log
    Wait-ForCompletion -LogFile $log -SetName $SetName
}

# Wait for set3 (already in flight) to finish.
Wait-ForCompletion -LogFile "$OUT_DIR/seed${SEED}_set3.log" -SetName "set3"

# Launch set4 + set5 sequentially.
Train-Set -SetName "set4"
Train-Set -SetName "set5"

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] All 3 sets (set3, set4, set5) trained. 60-word vocab ready." -ForegroundColor Green
