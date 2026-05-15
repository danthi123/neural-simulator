# Train 4 new G.20 bridges (B/C/D/E vocabs) sequentially.
# Bridge A (nouns) reuses the validated shared_pool_n32.simstate.h5 if it
# exists; otherwise trains it too.
#
# Each bridge:
#   - 32 concepts, 1600-neuron shared pool, 400 train events
#   - validated G.20 recipe: topographic_factor 10.0, off_target 0.1,
#     sparsity 0.03, slice_size 50, top_k 100
#   - ~30 min per bridge
#
# Total wall clock: 4 bridges x ~30 min = ~2 hours (or 5 x 30 if A also needed)
#
# Waits for current python (multi-seed chain) to finish before starting.
$ErrorActionPreference = "Continue"
$OUT_DIR = "research/findings/raw/g11_bg"
$BRIDGE_DIR = "$OUT_DIR/g20_bridges"
New-Item -ItemType Directory -Force -Path $BRIDGE_DIR | Out-Null

# Wait for multi-seed chain to finish
Write-Host "[chain] $(Get-Date) Waiting for multi-seed chain to complete..."
$maxWait = 5400  # 90 min max
$elapsed = 0
while ($elapsed -lt $maxWait) {
    $py = Get-Process python -ErrorAction SilentlyContinue
    if (-not $py) {
        Write-Host "[chain] $(Get-Date) Python died, proceeding"
        break
    }
    Start-Sleep -Seconds 60
    $elapsed += 60
}

function Train-G20-Bridge {
    param([string]$Name, [string]$VocabFile)
    $bridgeOut = "$BRIDGE_DIR/${Name}.simstate.h5"
    $jsonOut = "$BRIDGE_DIR/${Name}.json"
    $logOut = "$BRIDGE_DIR/${Name}.log"

    if (Test-Path $bridgeOut) {
        Write-Host "[chain] $(Get-Date) $Name already trained, skipping"
        return
    }

    # Read vocab from file (concept_pool_demo_shared expects comma-separated)
    $vocabContent = (Get-Content $VocabFile | Where-Object { $_ -and -not ($_ -match "^#") }) -join ","

    Write-Host "[chain] $(Get-Date) === Training $Name ($($vocabContent.Split(',').Count) concepts) ==="
    python -m research.runners.concept_pool_demo_shared `
        --seed 42 --n-concepts 32 --n-train-events 400 `
        --n-lang-input 8192 --n-shared-pool 1600 --slice-size 50 `
        --top-k 100 --topographic-factor 10.0 --off-target-factor 0.1 `
        --sparsity 0.03 `
        --vocab $vocabContent `
        --save-bridge $bridgeOut --out $jsonOut 2>&1 | Tee-Object -FilePath $logOut
}

# Reuse seed42 shared_pool_n32 bridge as bridgeA_nouns by saving its
# bridge file (the 32-word validated demo). Trained earlier; just copy.
# Actually we need to RE-TRAIN with the bridgeA vocab specifically since
# shared_pool_n32 used the first 32 words of ALL_60 (apple/river/dog/cat/
# ... go/come/stop/look ... big/small/hot/cold + tree/bird/sun/moon +
# walk/run/eat/sleep + red/blue/fast/slow + house/road/fire/water).
# That's MIXED categories. bridgeA_nouns has 32 nouns ONLY.

Train-G20-Bridge -Name "bridgeA_nouns" -VocabFile "$OUT_DIR/g20_bridgeA_nouns_vocab.txt"
Train-G20-Bridge -Name "bridgeB_verbs" -VocabFile "$OUT_DIR/g20_bridgeB_verbs_vocab.txt"
Train-G20-Bridge -Name "bridgeC_adj" -VocabFile "$OUT_DIR/g20_bridgeC_adj_vocab.txt"
Train-G20-Bridge -Name "bridgeD_spatial" -VocabFile "$OUT_DIR/g20_bridgeD_spatial_vocab.txt"
Train-G20-Bridge -Name "bridgeE_functional" -VocabFile "$OUT_DIR/g20_bridgeE_functional_vocab.txt"

Write-Host "[chain] $(Get-Date) DONE: 5 G.20 bridges trained (160-concept ensemble ready)"
