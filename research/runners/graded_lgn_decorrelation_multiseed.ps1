# Multi-seed driver for the GRADED LGN decorrelation composition gate (2026-06-06).
# Heavy GPU runs SEQUENTIAL (one bridge at a time; two concurrent agent benchmarks OOM).
# Usage: pwsh research/runners/graded_lgn_decorrelation_multiseed.ps1 [-Signed] [-Gain 40] [-Epochs 8]
param(
    [int[]]$Seeds = @(42, 43, 44),
    [switch]$Signed,
    [double]$Gain = 40.0,
    [int]$Epochs = 8,
    [int]$K = 300,
    [int]$Window = 30,
    [int]$Settle = 10
)
$env:REALOBJ_CIFAR = "data/cifar10/cifar-10-batches-py/data_batch_1"
$env:SIM_BACKEND = "cupy"
$env:PYTHONIOENCODING = "utf-8"
$tag = if ($Signed) { "signed" } else { "rectified" }
foreach ($s in $Seeds) {
    $out = "research/findings/raw/_graded_lgn_${tag}_g${Gain}_e${Epochs}_s${s}.json"
    Write-Host "=== seed $s ($tag, gain=$Gain, epochs=$Epochs) -> $out ===" -ForegroundColor Cyan
    $signedFlag = if ($Signed) { "--signed" } else { "" }
    python -m research.runners.graded_lgn_decorrelation_compose --seeds $s --K $K `
        $signedFlag --gain $Gain --epochs $Epochs --window $Window --settle $Settle `
        --baseline --out $out
}
