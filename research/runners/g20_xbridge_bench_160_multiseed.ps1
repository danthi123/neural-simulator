# Multi-seed the 160 cross-bridge benchmark using bridges that ALREADY
# EXIST from the multi-seed hardening (no retrain). Converts the
# "100% seed-42" cross-bridge claim into an honest multi-seed result.
#
# Anti-overclaim: seed 42 was a "clean" 160 seed (per-bridge 100%);
# seeds 43/46 were 96.9%/93.8% per-bridge. So cross-bridge retrieval
# at 43-46 may be lower than seed-42's 30/30. This measures it
# honestly. Bridges: research/findings/raw/g11_bg/g20_sparse_bridges_s{N}/
$ErrorActionPreference = "Continue"
$V = "research/findings/raw/g11_bg"
$names = @("bridgeA_nouns","bridgeB_verbs","bridgeC_adj","bridgeD_spatial","bridgeE_functional")

foreach ($seed in 43,44,45,46) {
    $BD = "$V/g20_sparse_bridges_s$seed"
    if (-not (Test-Path "$BD/bridgeE_functional_sparse.simstate.h5")) {
        Write-Host "[xb160ms] seed $seed bridges missing, skip"; continue
    }
    $bridges = $names | ForEach-Object { "$BD/${_}_sparse.simstate.h5" }
    $vocabs  = $names | ForEach-Object { "$V/g20_${_}_vocab.txt" }
    Write-Host "[xb160ms] $(Get-Date) === seed $seed xbridge benchmark ==="
    python -m research.runners.g20_xbridge_benchmark `
        --sparse --pattern-size 100 --n-shared-pool 2000 `
        --n-lang-input 8192 --sparsity 0.02 --seed $seed `
        --bridges $bridges --vocab-files $vocabs --names $names `
        --n-pairs 30 --encode-repeats 1 --exclude-idx 12 `
        --out "$V/g20_xbridge_bench_160_s$seed.json" 2>&1 |
        Tee-Object -FilePath "$V/g20_xbridge_bench_160_s$seed.log"
}
Write-Host "[xb160ms] $(Get-Date) DONE: 160 xbridge benchmark seeds 43-46"
