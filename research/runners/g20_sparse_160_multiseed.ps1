# Multi-seed hardening of the 160-concept sparse ensemble (the 100%
# tier). Seed 42 already validated end-to-end 2026-05-15. This trains
# + end-to-end-demos seeds 43,44,45,46 so the headline "160 ensemble
# integration" graduates from seed-42 to multi-seed (project mandates
# 6-seed-class rigor before a result is "validated").
#
# Per seed: 5 bridges x 32 sparse concepts (sparsity 0.02 -- valid at
# n_cues=32: n_active 164 < stride 8192/32=256), then the same scripted
# demo that passed at seed 42 (cross-bridge memory + N-word sentence +
# role queries). Writes per-seed dirs; does NOT touch the validated
# seed-42 bridges.
$ErrorActionPreference = "Continue"
$V = "research/findings/raw/g11_bg"
$names = @("bridgeA_nouns","bridgeB_verbs","bridgeC_adj","bridgeD_spatial","bridgeE_functional")

foreach ($seed in 43,44,45,46) {
    $BD = "$V/g20_sparse_bridges_s$seed"
    New-Item -ItemType Directory -Force -Path $BD | Out-Null
    Write-Host "[ms160] $(Get-Date) ===== SEED $seed ====="

    foreach ($name in $names) {
        $vocabFile = "$V/g20_${name}_vocab.txt"
        $jsonOut = "$BD/${name}_sparse.json"
        $bridgeOut = "$BD/${name}_sparse.simstate.h5"
        $logOut = "$BD/${name}_sparse.log"
        if (Test-Path $jsonOut) { Write-Host "[ms160] s$seed $name done, skip"; continue }
        $vc = (Get-Content $vocabFile | Where-Object { $_ -and -not ($_ -match "^#") }) -join ","
        $n = ($vc -split ',').Count
        Write-Host "[ms160] $(Get-Date) s$seed $name ($n) train"
        python -m research.runners.concept_pool_sparse_distributed `
            --seed $seed --n-concepts $n --n-train-events 400 `
            --n-lang-input 8192 --n-shared-pool 2000 --pattern-size 100 `
            --top-k 150 --sparsity 0.02 `
            --vocab $vc --save-bridge $bridgeOut --out $jsonOut 2>&1 |
            Tee-Object -FilePath $logOut
    }

    # End-to-end demo for this seed (pattern regen uses --seed $seed)
    $bridges = $names | ForEach-Object { "$BD/${_}_sparse.simstate.h5" }
    $vocabs = $names | ForEach-Object { "$V/g20_${_}_vocab.txt" }
    Write-Host "[ms160] $(Get-Date) s$seed end-to-end demo"
    python -m research.runners.g20_multibridge `
        --sparse --pattern-size 100 --n-shared-pool 2000 `
        --n-lang-input 8192 --sparsity 0.02 --seed $seed `
        --bridges $bridges --vocab-files $vocabs --names $names `
        --scripted "remember apple is big,what is apple,is apple big?,remember dog run fast,who run fast?,what did dog run?,quit" 2>&1 |
        Tee-Object -FilePath "$BD/g20_demo_s$seed.log"
}
Write-Host "[ms160] $(Get-Date) DONE: 160 ensemble multi-seed (43-46) trained+demoed"
