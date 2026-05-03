# Auto-followup: after the arch sweep on seed 42 finishes, parse the
# 5 variant JSONs, pick the highest-W->A variant, and IF it clears the
# threshold (>= 32% on seed 42, vs baseline 27%), launch 6-seed
# validation on that variant.
#
# Threshold rationale: baseline seed 42 = 27%. 6-seed mean = 28.5% +/- 2.1%.
# 32% is 1.7 std above mean — distinguishes signal from per-seed noise.
# Below that, the variant is within noise, and 6-seed validation
# wouldn't tell us anything new.

$outDir = "research/findings/raw/g11_bg"
$logFile = "$outDir/auto_followup_arch.log"
$pidFile = "$outDir/auto_followup_arch.orchestrator-pid"

$PID | Out-File -FilePath $pidFile -Encoding ASCII

"=== Auto-followup arch winner started $(Get-Date) ===" | Out-File -FilePath $logFile

# Pick winner via Python helper (more reliable than ps regex)
$winnerScript = @"
import json
from pathlib import Path

ROOT = Path('research/findings/raw/g11_bg')
variants = ['motor50', 'sparse005', 'lang512', 'motor50_sparse005', 'lang512_motor50']
results = {}
for v in variants:
    p = ROOT / f'text_eval_arch_{v}_seed42.json'
    if not p.exists():
        continue
    try:
        d = json.loads(p.read_text())
        wa = (d.get('word_to_action_eval') or {}).get('accuracy')
        if wa is not None:
            results[v] = wa
    except Exception:
        pass

if not results:
    print('NO_RESULTS')
else:
    winner = max(results.items(), key=lambda kv: kv[1])
    name, acc = winner
    print(f'WINNER:{name}:{acc:.4f}')
    for v, a in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f'  {v}: {100*a:.1f}%')
"@

$winnerOutput = & python -c $winnerScript 2>&1
"$winnerOutput" | Out-File -Append $logFile

$winnerLine = $winnerOutput | Where-Object { $_ -match '^WINNER:' } | Select-Object -First 1
if (-not $winnerLine) {
    "No winner found — exiting" | Out-File -Append $logFile
    if (Test-Path $pidFile) {
        Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
    }
    return
}

$parts = $winnerLine -split ':'
$winner = $parts[1]
$winnerAcc = [float]$parts[2]
"Selected winner: $winner ($([Math]::Round(100 * $winnerAcc, 1))%)" | Out-File -Append $logFile

if ($winnerAcc -lt 0.32) {
    "Winner accuracy $([Math]::Round(100 * $winnerAcc, 1))% below 32% threshold — skipping 6-seed validation" | Out-File -Append $logFile
    "(All variants within noise; manual review needed before committing GPU time)" | Out-File -Append $logFile
    if (Test-Path $pidFile) {
        Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
    }
    return
}

"" | Out-File -Append $logFile
"Winner clears 32% threshold — launching 6-seed validation" | Out-File -Append $logFile

# Build the variant args based on the name
$argsForVariant = @()
switch ($winner) {
    'motor50' {
        $argsForVariant = @('--n-motor-per-action', '50')
    }
    'sparse005' {
        $argsForVariant = @('--token-sparsity', '0.05')
    }
    'lang512' {
        $argsForVariant = @('--text-n-input-neurons', '512',
                             '--text-n-output-neurons', '512')
    }
    'motor50_sparse005' {
        $argsForVariant = @('--n-motor-per-action', '50',
                             '--token-sparsity', '0.05')
    }
    'lang512_motor50' {
        $argsForVariant = @('--text-n-input-neurons', '512',
                             '--text-n-output-neurons', '512',
                             '--n-motor-per-action', '50')
    }
    default {
        "Unknown winner: $winner — exiting" | Out-File -Append $logFile
        if (Test-Path $pidFile) {
            Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
        }
        return
    }
}

# Launch 6-seed validation. Skip seed 42 (already have data); start from 43.
$seeds = @(43, 44, 100, 101, 102)
"Validating $winner on seeds $($seeds -join ',') (seed 42 already done in arch sweep)" | Out-File -Append $logFile

foreach ($seed in $seeds) {
    "" | Out-File -Append $logFile
    "--- Starting $winner seed $seed at $(Get-Date) ---" | Out-File -Append $logFile
    $sLogFile = "$outDir/arch_$winner.seed$seed.log"
    $sErrFile = "$outDir/arch_$winner.seed$seed.log.err"
    $sPidFile = "$outDir/arch_$winner.seed$seed.pid"
    $sOutStats = "$outDir/text_eval_arch_$winner`_seed$seed.json"

    $procArgs = @(
        '-m', 'research.runners.text_train_curriculum',
        '--seed', "$seed",
        '--phase1-episodes', '0',
        '--phase2-episodes', '100',
        '--phase3-replays', '0',
        '--steps-per-episode', '30',
        '--stim-steps-per-step', '200',
        '--reset-steps', '100',
        '--out-stats', $sOutStats
    ) + $argsForVariant

    $proc = Start-Process -FilePath 'python.exe' -ArgumentList $procArgs `
        -RedirectStandardOutput $sLogFile -RedirectStandardError $sErrFile `
        -PassThru -NoNewWindow

    $proc.Id | Out-File -FilePath $sPidFile -Encoding ASCII
    "Seed $seed launched as PID $($proc.Id)" | Out-File -Append $logFile

    $proc.WaitForExit()
    "Seed $seed completed (exit $($proc.ExitCode)) at $(Get-Date)" | Out-File -Append $logFile

    if (Test-Path $sPidFile) {
        Move-Item -Path $sPidFile -Destination "$sPidFile.done" -Force
    }
}

"" | Out-File -Append $logFile
"=== AUTO-FOLLOWUP COMPLETE for winner=$winner at $(Get-Date) ===" | Out-File -Append $logFile

if (Test-Path $pidFile) {
    Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
}
