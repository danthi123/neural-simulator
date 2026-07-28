$ErrorActionPreference = "Continue"
$OUT_DIR = "research/findings/raw/g11_bg/g20_bridges"

function Train-G20-Seed {
    param([string]$Name, [string]$VocabFile, [int]$Seed)
    $bridgeOut = "$OUT_DIR/${Name}_seed${Seed}.simstate.h5"
    $jsonOut = "$OUT_DIR/${Name}_seed${Seed}.json"
    $logOut = "$OUT_DIR/${Name}_seed${Seed}.log"

    if (Test-Path $jsonOut) {
        Write-Host "[chain] $Name seed $Seed already done, skipping"
        return
    }

    $vocabContent = (Get-Content $VocabFile | Where-Object { $_ -and -not ($_ -match "^#") }) -join ","

    Write-Host "[chain] $(Get-Date) === $Name seed $Seed ==="
    python -m research.runners.concept_pool_demo_shared `
        --seed $Seed --n-concepts 32 --n-train-events 400 `
        --n-lang-input 8192 --n-shared-pool 1600 --slice-size 50 `
        --top-k 100 --topographic-factor 10.0 --off-target-factor 0.1 `
        --sparsity 0.03 `
        --vocab $vocabContent `
        --save-bridge $bridgeOut --out $jsonOut 2>&1 | Tee-Object -FilePath $logOut
}

# Seeds 43, 44 for each of the 5 vocabs (10 trains x ~18 min = ~3 hours)
foreach ($seed in 43, 44) {
    foreach ($name in "bridgeA_nouns", "bridgeB_verbs", "bridgeC_adj", "bridgeD_spatial", "bridgeE_functional") {
        Train-G20-Seed -Name $name -VocabFile "research/findings/raw/g11_bg/g20_${name}_vocab.txt" -Seed $seed
    }
}

Write-Host "[chain] $(Get-Date) DONE: 5 bridges x 2 seeds multi-seed validation"
