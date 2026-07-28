$ErrorActionPreference = "Continue"
$BRIDGE_DIR = "research/findings/raw/g11_bg/g20_sparse_bridges"
New-Item -ItemType Directory -Force -Path $BRIDGE_DIR | Out-Null

# Wait for the 256-concept test (and any other python) to finish
Write-Host "[chain] $(Get-Date) Waiting for GPU to free up..."
$maxWait = 9000
$elapsed = 0
while ($elapsed -lt $maxWait) {
    $py = Get-Process python -ErrorAction SilentlyContinue
    if (-not $py) {
        Write-Host "[chain] $(Get-Date) GPU free, starting 5-bridge sparse ensemble"
        break
    }
    Start-Sleep -Seconds 60
    $elapsed += 60
}

# Train 5 sparse-distributed bridges at 64-concept tier (validated 100%)
foreach ($name in "bridgeA_nouns", "bridgeB_verbs", "bridgeC_adj", "bridgeD_spatial", "bridgeE_functional") {
    $vocabFile = "research/findings/raw/g11_bg/g20_${name}_vocab.txt"
    $bridgeOut = "$BRIDGE_DIR/${name}_sparse.simstate.h5"
    $jsonOut = "$BRIDGE_DIR/${name}_sparse.json"
    $logOut = "$BRIDGE_DIR/${name}_sparse.log"

    if (Test-Path $jsonOut) {
        Write-Host "[chain] $name sparse already done, skipping"
        continue
    }
    # Each vocab file has 32 concepts; sparse-distributed handles them
    # at 100% in a 2000-pool. (Could extend to 64 by combining 2 vocabs
    # but keep 32 for direct comparison to contiguous production.)
    $vocabContent = (Get-Content $vocabFile | Where-Object { $_ -and -not ($_ -match "^#") }) -join ","
    $n = ($vocabContent -split ',').Count

    Write-Host "[chain] $(Get-Date) === sparse $name ($n concepts) ==="
    python -m research.runners.concept_pool_sparse_distributed `
        --seed 42 --n-concepts $n --n-train-events 400 `
        --n-lang-input 8192 --n-shared-pool 2000 --pattern-size 100 `
        --top-k 150 --sparsity 0.02 `
        --vocab $vocabContent `
        --save-bridge $bridgeOut --out $jsonOut 2>&1 | Tee-Object -FilePath $logOut
}

Write-Host "[chain] $(Get-Date) DONE: 5 sparse-distributed bridges trained"
