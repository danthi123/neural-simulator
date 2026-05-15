$ErrorActionPreference = "Continue"
$OUT_DIR = "research/findings/raw/g11_bg/concept_pool_demo"
$SEED = 42
$SET3_LOG = "$OUT_DIR/seed${SEED}_set3.log"

Write-Host "[chain] $(Get-Date) Waiting on set3 VERDICT..."
$max_wait = 1800
$elapsed = 0
while ($elapsed -lt $max_wait) {
    $tail = Get-Content $SET3_LOG -Tail 20 -ErrorAction SilentlyContinue
    if ($tail | Where-Object { $_ -match "VERDICT" }) {
        Write-Host "[chain] set3 VERDICT detected"
        break
    }
    Start-Sleep -Seconds 30
    $elapsed += 30
}

Write-Host "[chain] $(Get-Date) Starting set4 training..."
python -m research.runners.concept_pool_demo_set4 `
    --seed 42 --n-train-events 200 --n-lang-input 2048 --n-per-pool 200 `
    --n-fs-per-pool 24 --weak-concept-dynamics --interleaved `
    --topographic-factor 3.0 --off-target-factor 0.3 `
    --enable-adjective --orthogonal-codes --sparsity 0.05 `
    --save-bridge "$OUT_DIR/seed${SEED}_set4.simstate.h5" `
    --out "$OUT_DIR/seed${SEED}_set4.json" `
    2>&1 | Tee-Object -FilePath "$OUT_DIR/seed${SEED}_set4.log"

Write-Host "[chain] $(Get-Date) Starting set5 training..."
python -m research.runners.concept_pool_demo_set5 `
    --seed 42 --n-train-events 200 --n-lang-input 2048 --n-per-pool 200 `
    --n-fs-per-pool 24 --weak-concept-dynamics --interleaved `
    --topographic-factor 3.0 --off-target-factor 0.3 `
    --enable-adjective --orthogonal-codes --sparsity 0.05 `
    --save-bridge "$OUT_DIR/seed${SEED}_set5.simstate.h5" `
    --out "$OUT_DIR/seed${SEED}_set5.json" `
    2>&1 | Tee-Object -FilePath "$OUT_DIR/seed${SEED}_set5.log"

Write-Host "[chain] $(Get-Date) DONE: set3 + set4 + set5 all trained -- 60-word vocab ready"
