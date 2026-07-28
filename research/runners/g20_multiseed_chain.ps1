$ErrorActionPreference = "Continue"
$OUT_DIR = "research/findings/raw/g11_bg"

# Wait for the current python process (60-concept smoke) to die.
Write-Host "[chain] $(Get-Date) Waiting on 60-concept smoke to complete..."
$maxWait = 6000  # 100 minutes max
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

# Multi-seed 32-concept validation (seeds 43, 44, 45)
foreach ($seed in 43, 44, 45) {
    $out = "$OUT_DIR/shared_pool_n32_seed${seed}.json"
    Write-Host "[chain] $(Get-Date) === seed $seed 32-concept ==="
    python -m research.runners.concept_pool_demo_shared `
        --seed $seed --n-concepts 32 --n-train-events 400 `
        --n-lang-input 8192 --n-shared-pool 1600 --slice-size 50 `
        --top-k 100 --topographic-factor 10.0 --off-target-factor 0.1 `
        --sparsity 0.03 --out $out
}

Write-Host "[chain] $(Get-Date) DONE: 32-concept multi-seed validation"
