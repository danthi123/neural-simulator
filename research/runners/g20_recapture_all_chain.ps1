$ErrorActionPreference = "Continue"
$BRIDGE_DIR = "research/findings/raw/g11_bg/g20_bridges"

foreach ($name in "bridgeA_nouns", "bridgeB_verbs", "bridgeC_adj", "bridgeD_spatial", "bridgeE_functional") {
    $orig = "$BRIDGE_DIR/${name}.simstate.h5"
    $new = "$BRIDGE_DIR/${name}_v2.simstate.h5"
    $resultIn = "$BRIDGE_DIR/${name}.json"
    $resultOut = "$BRIDGE_DIR/${name}_v2.json"

    if (-not (Test-Path $orig)) {
        Write-Host "[skip] $name bridge not found"
        continue
    }

    Write-Host "[recapture] $(Get-Date) $name with teacher-bias..."
    python -m research.runners.g20_recapture_engrams `
        --bridge $orig --result-json $resultIn --seed 42 `
        --method teacher-bias --teacher-pA 100.0 `
        --save-bridge $new --out $resultOut 2>&1 | Tee-Object -FilePath "$BRIDGE_DIR/${name}_recapture.log"
}

Write-Host "[done] $(Get-Date) all 5 bridges re-captured"
