# #6 SURPASS — log-polar SC retina grid-32 6-seed confirmation.
# For each seed: run host / sc_popvector(FIX1+log-polar) / sc_popvector_scr, then copy the 3
# per-arm JSONs to seed-tagged log-polar paths so a prior/next seed doesn't overwrite them.
# Run AFTER the seed-42 GO. GPU (SIM_BACKEND=cupy). grid-32 / 1800 / warmup-600 (the FAITHFUL verdict).
$ErrorActionPreference = "Stop"
$env:SIM_BACKEND = "cupy"
$seeds = @(42, 43, 44, 100, 101, 102)
$dir = "research/findings/raw/nav_gate_2a"
foreach ($s in $seeds) {
    Write-Host "===== #6 log-polar 6-seed: seed $s ====="
    python -m research.runners._nav_sc_popvector_readout_derisk `
        --seed $s --grid-size 32 --n-steps 1800 --warmup-steps 600 `
        --fix1 --log-polar --arms host,sc_popvector,sc_popvector_scr `
        --out "$dir/scpv_logpolar_summary_seed$s.json" 2>&1 | Tee-Object "$dir/scpv_logpolar_seed$s.log" | Select-Object -Last 5
    foreach ($arm in @("host", "sc_popvector", "sc_popvector_scr")) {
        $src = "$dir/scpv_${arm}_seed$s.json"
        $dst = "$dir/scpv_logpolar_${arm}_seed$s.json"
        if (Test-Path $src) { Copy-Item $src $dst -Force }
    }
}
Write-Host "===== aggregating ====="
python -m research.runners._shortcut6_logpolar_aggregate --seeds 42,43,44,100,101,102 `
    --out "$dir/scpv_logpolar_6seed_aggregate.json"
