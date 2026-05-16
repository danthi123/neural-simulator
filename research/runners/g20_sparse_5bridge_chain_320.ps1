# Train the 320-concept production tier: 5 sparse-distributed bridges
# x 64 concepts each = 320 unique concepts.
#
# Sparse-distributed is multi-seed validated at 64 concepts/bridge @
# 100% (288/288; 2026-05-15-sparse-distributed-capacity-curve.md), so
# this is a scaling/integration tier, not a new experiment.
#
# Differences vs the 160 chain (g20_sparse_5bridge_chain.ps1):
#   - --n-concepts 64           (was 32)
#   - --sparsity 0.007          (was 0.02) -- REQUIRED: orthogonal-drive
#       needs n_active < stride = n_lang/n_cues = 8192/64 = 128.
#       0.007*8192 = 57 active < 128  OK.  (0.02 -> 164 > 128 = crash.)
#   - reads g20_<name>_vocab64.txt (64-word files, NOT the 32-word ones)
#   - writes to g20_sparse_bridges_320/  (does NOT clobber the
#       validated 160-concept bridges)
$ErrorActionPreference = "Continue"
$BRIDGE_DIR = "research/findings/raw/g11_bg/g20_sparse_bridges_320"
New-Item -ItemType Directory -Force -Path $BRIDGE_DIR | Out-Null

Write-Host "[chain320] $(Get-Date) Waiting for GPU to free up..."
$maxWait = 9000
$elapsed = 0
while ($elapsed -lt $maxWait) {
    $py = Get-Process python -ErrorAction SilentlyContinue
    if (-not $py) {
        Write-Host "[chain320] $(Get-Date) GPU free, starting 320-concept tier"
        break
    }
    Start-Sleep -Seconds 60
    $elapsed += 60
}

foreach ($name in "bridgeA_nouns", "bridgeB_verbs", "bridgeC_adj", "bridgeD_spatial", "bridgeE_functional") {
    $vocabFile = "research/findings/raw/g11_bg/g20_${name}_vocab64.txt"
    $bridgeOut = "$BRIDGE_DIR/${name}_sparse64.simstate.h5"
    $jsonOut = "$BRIDGE_DIR/${name}_sparse64.json"
    $logOut = "$BRIDGE_DIR/${name}_sparse64.log"

    if (Test-Path $jsonOut) {
        Write-Host "[chain320] $name sparse64 already done, skipping"
        continue
    }
    $vocabContent = (Get-Content $vocabFile | Where-Object { $_ -and -not ($_ -match "^#") }) -join ","
    $n = ($vocabContent -split ',').Count

    Write-Host "[chain320] $(Get-Date) === sparse64 $name ($n concepts) ==="
    python -m research.runners.concept_pool_sparse_distributed `
        --seed 42 --n-concepts $n --n-train-events 400 `
        --n-lang-input 8192 --n-shared-pool 2000 --pattern-size 100 `
        --top-k 150 --sparsity 0.007 `
        --vocab $vocabContent `
        --save-bridge $bridgeOut --out $jsonOut 2>&1 | Tee-Object -FilePath $logOut
}

Write-Host "[chain320] $(Get-Date) DONE: 5 sparse64 bridges trained (320 concepts)"
