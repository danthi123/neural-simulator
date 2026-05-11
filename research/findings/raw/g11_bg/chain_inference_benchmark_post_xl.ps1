# Auto-fire inference_benchmark across vocab tiers once XL frees the GPU.
#
# Triggered manually OR by a background watcher. Polls every 60s for:
#   1. The XL python process (PID 24012) to exit, AND
#   2. nvidia-smi VRAM usage to drop below 3 GB
# Then runs inference_benchmark for each vocab tier in the matrix,
# writing results to research/findings/raw/perf/inference_bench/.
#
# Per arch matrix (encoding-axis discovery 2026-05-10):
#   4-word   : n_lang=2048,  n_motor=500
#   8-word   : n_lang=4096,  n_motor=1000
#   12-word  : n_lang=4096,  n_motor=2000
#   16-word  : n_lang=4096,  n_motor=2000
#   24-word  : n_lang=8192,  n_motor=2000
#   32-word  : n_lang=8192,  n_motor=2000
#   48-word  : n_lang=8192,  n_motor=2000
#   64-word  : n_lang=8192,  n_motor=2000
#
# Per tier: --n-rounds 10 -> 40 :speak calls (10 rounds * 4 actions).
# Bridge build dominates wall clock; bench portion is seconds.
#
# Usage:
#   pwsh -File research/findings/raw/g11_bg/chain_inference_benchmark_post_xl.ps1

$ErrorActionPreference = "Continue"
Set-Location E:\Documents\Projects\sim
# Force UTF-8 for Python stdout to avoid Windows cp1252 encoding errors
# when subprocesses print Unicode (arrows, accented chars, etc.).
$env:PYTHONIOENCODING = "utf-8"

$XL_PID = 24012
$OUT_DIR = "research/findings/raw/perf/inference_bench"
$LOG = "$OUT_DIR/chain_inference_benchmark.log"
$WATCH_TIMEOUT_MIN = 300  # 5 hours max wait

if (-not (Test-Path $OUT_DIR)) { New-Item -ItemType Directory -Path $OUT_DIR -Force | Out-Null }

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts] $msg" | Tee-Object -FilePath $LOG -Append
}

Log "Chain started; watching for XL python.exe PID $XL_PID to exit + VRAM to drop."

# ─── Phase 1: wait for XL to finish + GPU to free ──────────────────────────
$startTime = Get-Date
while ($true) {
    $elapsedMin = ((Get-Date) - $startTime).TotalMinutes
    if ($elapsedMin -gt $WATCH_TIMEOUT_MIN) {
        Log "TIMEOUT after $WATCH_TIMEOUT_MIN minutes. Aborting."
        exit 2
    }

    $xlAlive = $false
    try {
        $proc = Get-Process -Id $XL_PID -ErrorAction Stop
        $xlAlive = $true
    } catch {
        $xlAlive = $false
    }

    $vramMB = 99999
    try {
        $vramLine = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null) | Select-Object -First 1
        $vramMB = [int]$vramLine
    } catch {
        # nvidia-smi failed; assume GPU busy
    }

    # Threshold raised 2026-05-11 01:43 EDT: baseline VRAM usage from
    # Windows compositor + browsers + Claude + Discord is ~5 GB even
    # with no Python on GPU. Use 7 GB as "free enough for inference
    # benchmark" — inference build uses ~10-12 GB at the biggest tier.
    if (-not $xlAlive -and $vramMB -lt 7000) {
        Log "GO: XL process exited; VRAM free ($vramMB MB used)."
        break
    }

    Log "wait: xl_alive=$xlAlive vram_used_mb=$vramMB elapsed_min=$([int]$elapsedMin)"
    Start-Sleep -Seconds 60
}

# ─── Phase 2: run inference_benchmark across vocab tiers ───────────────────
$tiers = @(
    @{ vocab = 4;  lang = 2048;  motor = 500;  motor_fs = 60 },
    @{ vocab = 8;  lang = 4096;  motor = 1000; motor_fs = 120 },
    @{ vocab = 12; lang = 4096;  motor = 2000; motor_fs = 240 },
    @{ vocab = 16; lang = 4096;  motor = 2000; motor_fs = 240 },
    @{ vocab = 24; lang = 8192;  motor = 2000; motor_fs = 240 },
    @{ vocab = 32; lang = 8192;  motor = 2000; motor_fs = 240 },
    @{ vocab = 48; lang = 8192;  motor = 2000; motor_fs = 240 },
    @{ vocab = 64; lang = 8192;  motor = 2000; motor_fs = 240 }
)

foreach ($t in $tiers) {
    $outFile = "$OUT_DIR/inference_bench_v$($t.vocab)w.json"
    if (Test-Path $outFile) {
        Log "SKIP $($t.vocab)-word: $outFile already exists."
        continue
    }
    Log "RUN $($t.vocab)-word: lang=$($t.lang) motor=$($t.motor) motor_fs=$($t.motor_fs)"
    $cmd = "python -m research.runners.inference_benchmark " +
           "--vocab-size $($t.vocab) " +
           "--n-lang-input $($t.lang) " +
           "--n-motor-per-action $($t.motor) " +
           "--n-motor-fs-per-action $($t.motor_fs) " +
           "--n-rounds 10 " +
           "--out $outFile"
    Log "CMD: $cmd"
    & cmd /c $cmd 2>&1 | Tee-Object -FilePath $LOG -Append
    if (Test-Path $outFile) {
        Log "OK $($t.vocab)-word -> $outFile"
    } else {
        Log "FAIL $($t.vocab)-word (no output file produced)"
    }
}

Log "Chain complete. Results in $OUT_DIR/"
