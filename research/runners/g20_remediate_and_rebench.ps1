# Definitive end-to-end test: does artifact-safe capture-quality
# remediation lift the CROSS-BRIDGE conversational metric (not just
# self-recall)? The capture-remediation finding was honestly bounded
# ("fixes self-recall n=1; cross-bridge impact UNMEASURED"). This
# measures it, controlled.
#
# Protocol (apples-to-apples, --exclude-idx -1 so idx-12 IS tested,
# since remediation's main effect is fixing idx-12-class under-recall):
#   1. Baseline xbridge benchmark on ORIGINAL 320 bridges (keep-all).
#   2. Remediate all 5 bridges (boosted re-capture of under-recallers),
#      save to a NEW dir (validated artifact preserved).
#   3. Remediated xbridge benchmark on the remediated ensemble.
#   4. Compare genuine cross-bridge rate: original vs remediated.
$ErrorActionPreference = "Continue"
$SRC = "research/findings/raw/g11_bg/g20_sparse_bridges_320"
$REM = "research/findings/raw/g11_bg/g20_sparse_bridges_320_remediated"
$V = "research/findings/raw/g11_bg"
New-Item -ItemType Directory -Force -Path $REM | Out-Null
$names = @("bridgeA_nouns","bridgeB_verbs","bridgeC_adj","bridgeD_spatial","bridgeE_functional")

# 1. Baseline (original bridges, keep-all incl idx-12)
$obr = $names | ForEach-Object { "$SRC/${_}_sparse64.simstate.h5" }
$vf  = $names | ForEach-Object { "$V/g20_${_}_vocab64.txt" }
Write-Host "[remed] $(Get-Date) baseline xbridge (original, keep-all)"
python -m research.runners.g20_xbridge_benchmark `
    --sparse --pattern-size 100 --n-shared-pool 2000 `
    --n-lang-input 8192 --sparsity 0.007 --seed 42 `
    --bridges $obr --vocab-files $vf --names $names `
    --n-pairs 30 --encode-repeats 1 --exclude-idx -1 `
    --out "$V/g20_xbridge_bench_320_keepall_baseline.json" 2>&1 |
    Tee-Object -FilePath "$V/g20_xbridge_bench_320_keepall_baseline.log"

# 2. Remediate all 5 bridges (save to NEW dir)
foreach ($n in $names) {
    Write-Host "[remed] $(Get-Date) remediating $n"
    python -m research.runners.g20_capture_remediation `
        --bridge "$SRC/${n}_sparse64.simstate.h5" `
        --vocab "$V/g20_${n}_vocab64.txt" `
        --seed 42 --n-concepts 64 --sparsity 0.007 `
        --boost-teacher-pA 400 --boost-steps 250 `
        --save-bridge "$REM/${n}_sparse64.simstate.h5" `
        --out "$REM/${n}_remediation.json" 2>&1 |
        Tee-Object -FilePath "$REM/${n}_remediation.log"
}

# 3. Remediated benchmark (same protocol)
$rbr = $names | ForEach-Object { "$REM/${_}_sparse64.simstate.h5" }
Write-Host "[remed] $(Get-Date) remediated xbridge (keep-all)"
python -m research.runners.g20_xbridge_benchmark `
    --sparse --pattern-size 100 --n-shared-pool 2000 `
    --n-lang-input 8192 --sparsity 0.007 --seed 42 `
    --bridges $rbr --vocab-files $vf --names $names `
    --n-pairs 30 --encode-repeats 1 --exclude-idx -1 `
    --out "$V/g20_xbridge_bench_320_keepall_remediated.json" 2>&1 |
    Tee-Object -FilePath "$V/g20_xbridge_bench_320_keepall_remediated.log"

Write-Host "[remed] $(Get-Date) DONE: remediate+rebench complete"
