# v17 28-word vocab multi-seed training: seeds 43, 44, 45, 46
# Each seed: ~44 min training + ~30s A->W readout
# Total: ~3.5 hours for 4 seeds

$ErrorActionPreference = "Stop"
Set-Location "E:\Documents\Projects\sim"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$OutDir = "research\findings\raw\g11_bg\concept_pool_demo"

function Run-V17-Seed {
    param([int]$Seed)
    Write-Host "[v17-MS] launching seed $Seed training (28-word vocab)..."
    python -m research.runners.concept_pool_demo_v2 `
        --seed $Seed `
        --n-train-events 200 `
        --n-lang-input 4096 `
        --n-per-pool 200 `
        --n-fs-per-pool 24 `
        --weak-concept-dynamics `
        --interleaved `
        --topographic-factor 3.0 `
        --off-target-factor 0.3 `
        --enable-adjective `
        --orthogonal-codes `
        --sparsity 0.03 `
        --save-bridge "$OutDir\seed${Seed}_v17.simstate.h5" `
        --out "$OutDir\seed${Seed}_v17.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_v17.log"

    Write-Host "[v17-MS] testing 96-cross compose for seed $Seed..."
    $allCues = "go,come,stop,look,walk,run,eat,sleep,apple,river,dog,cat,tree,bird,sun,moon,big,small,hot,cold,red,blue,fast,slow"
    $motors = "north,east,south,west"
    # Generate all 24x4 = 96 cross-pairs
    $pairs = @()
    foreach ($c in $allCues.Split(',')) {
        foreach ($m in $motors.Split(',')) {
            $pairs += "${c}:${m}"
        }
    }
    $pairStr = $pairs -join ","
    python -m research.runners.compose_engram_demo_v2 `
        --load-bridge "$OutDir\seed${Seed}_v17.simstate.h5" `
        --seed $Seed `
        --compose-pairs "$pairStr" `
        --encoding-steps 200 `
        --top-k 100 `
        --motor-teacher-pA 1500 `
        --recall-stim-pA 1500 `
        --recall-steps 100 `
        --n-lang-input 4096 `
        --sparsity 0.03 `
        --n-words-for-orthogonal 28 `
        --out "$OutDir\seed${Seed}_v17_96cross.json" 2>&1 | Tee-Object -FilePath "$OutDir\seed${Seed}_v17_96cross.log"

    $r = Get-Content "$OutDir\seed${Seed}_v17.json" -Raw | ConvertFrom-Json
    $c = Get-Content "$OutDir\seed${Seed}_v17_96cross.json" -Raw | ConvertFrom-Json
    Write-Host "[v17-MS] seed $Seed : P1=$($r.n_pass)/28, 96cross=$($c.n_pass)/96"
}

foreach ($seed in 43, 44, 45, 46) {
    Run-V17-Seed -Seed $seed
}

Write-Host ""
Write-Host "=== v17 multi-seed summary ==="
foreach ($s in 42, 43, 44, 45, 46) {
    $jp = "$OutDir\seed${s}_v17.json"
    $cp = "$OutDir\seed${s}_v17_96cross.json"
    if (Test-Path $jp -PathType Leaf) {
        $r = Get-Content $jp -Raw | ConvertFrom-Json
        $cstr = ""
        if (Test-Path $cp -PathType Leaf) {
            $c = Get-Content $cp -Raw | ConvertFrom-Json
            $cstr = "  96cross=$($c.n_pass)/96"
        }
        Write-Host "  seed $s : P1=$($r.n_pass)/28  $cstr"
    }
}
