# Auto-followup: after the FUNDAMENTALS sweep finishes, parse the
# variant JSONs, pick the best variant, and IF it clears threshold
# launch 6-seed validation in parallel-3.
#
# Selection: prefer variants where TRUE labeled mapping is best of 24
# permutations (aligned = 1) AND true accuracy >= 32%. If no aligned
# winner, fall back to highest true accuracy >= 32%.
#
# Threshold rationale: baseline mean = 28.5% +/- 2.1%. 32% = 1.7 std
# above. Below 32% the variant is in noise; running 6 seeds wouldn't
# tell us anything new.

$outDir = "research/findings/raw/g11_bg"
$logFile = "$outDir/auto_followup_arch.log"
$pidFile = "$outDir/auto_followup_arch.orchestrator-pid"

$PID | Out-File -FilePath $pidFile -Encoding ASCII

"=== Auto-followup arch winner (aligned + parallel-3) started $(Get-Date) ===" | Out-File -FilePath $logFile

# Pick winner via Python helper. Returns lines like:
#   WINNER:<name>:<true_acc>:<aligned>
#   ALIGNED_VARIANTS:<comma-separated list of variants where aligned=1>
#   TOP3:<comma-separated triples "name:acc:aligned">
$winnerScript = @"
import json
import itertools
from pathlib import Path

ROOT = Path('research/findings/raw/g11_bg')
WORDS = ['north', 'east', 'south', 'west']
TRUE_MAP = {'north': 'N', 'east': 'E', 'south': 'S', 'west': 'W'}
ACTIONS = ('N', 'E', 'S', 'W')

# Fundamentals-sweep variant names
variants = [
    'heb_only', 'drive_5x', 'stdp_wmax_10',
    'heb_drive', 'heb_stdp', 'drive_stdp',
]

def acc_for(cm, mapping):
    correct = total = 0
    for word, row in cm.items():
        target = mapping[word]
        for a, count in row.items():
            count = int(count)
            total += count
            if a == target:
                correct += count
    return correct / max(total, 1)

results = []
for v in variants:
    p = ROOT / f'text_eval_arch_{v}_seed42.json'
    if not p.exists():
        continue
    try:
        d = json.loads(p.read_text())
        cm_raw = (d.get('word_to_action_eval') or {}).get('confusion_matrix') or {}
        if not cm_raw or len(cm_raw) != 4:
            continue
        cm = {w: {a: int(cm_raw.get(w, {}).get(a, 0)) for a in ACTIONS} for w in WORDS}
        true_acc = acc_for(cm, TRUE_MAP)
        best_acc = 0.0
        best_perm = None
        for perm in itertools.permutations(ACTIONS):
            mapping = dict(zip(WORDS, perm))
            a = acc_for(cm, mapping)
            if a > best_acc:
                best_acc = a
                best_perm = perm
        aligned = 1 if best_perm == ACTIONS else 0
        results.append((v, true_acc, best_acc, aligned))
    except Exception:
        pass

if not results:
    print('NO_RESULTS')
else:
    # Sort: aligned=1 first, then highest true acc
    results.sort(key=lambda r: (-r[3], -r[1]))
    name, true_acc, best_acc, aligned = results[0]
    print(f'WINNER:{name}:{true_acc:.4f}:{aligned}')
    aligned_list = [r[0] for r in results if r[3] == 1]
    print(f'ALIGNED_VARIANTS:{",".join(aligned_list)}')
    top3 = ';'.join(f'{r[0]}|{r[1]:.3f}|{r[2]:.3f}|{r[3]}' for r in results[:3])
    print(f'TOP3:{top3}')
    for v, t, b, al in sorted(results, key=lambda r: -r[1]):
        print(f'  {v}: true={100*t:.1f}% best={100*b:.1f}% aligned={al}')
"@

$winnerOutput = & python -c $winnerScript 2>&1
"$winnerOutput" | Out-File -Append $logFile

$winnerLine = $winnerOutput | Where-Object { $_ -match '^WINNER:' } | Select-Object -First 1
if (-not $winnerLine) {
    "No variant results yet — exiting" | Out-File -Append $logFile
    if (Test-Path $pidFile) {
        Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
    }
    return
}

$parts = $winnerLine -split ':'
$winner = $parts[1]
$winnerAcc = [float]$parts[2]
$winnerAligned = [int]$parts[3]
"Selected winner: $winner (true=$([Math]::Round(100 * $winnerAcc, 1))%, aligned=$winnerAligned)" | Out-File -Append $logFile

if ($winnerAcc -lt 0.32 -and $winnerAligned -eq 0) {
    "Winner accuracy $([Math]::Round(100 * $winnerAcc, 1))% < 32% threshold AND not aligned." | Out-File -Append $logFile
    "All variants within noise — manual review needed before committing GPU time" | Out-File -Append $logFile
    if (Test-Path $pidFile) {
        Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
    }
    return
}

if ($winnerAligned -eq 1) {
    "Winner is ALIGNED at seed 42 — running 6-seed validation to confirm real learning" | Out-File -Append $logFile
} else {
    "Winner clears 32% but not aligned — running 6-seed validation anyway" | Out-File -Append $logFile
}

# Map winner name back to CLI args
$argsForVariant = @()
switch ($winner) {
    'heb_only' {
        $argsForVariant = @('--enable-hebbian', '--hebbian-weight-decay', '1e-7')
    }
    'drive_5x' {
        $argsForVariant = @('--lang-input-drive-pA', '1000')
    }
    'stdp_wmax_10' {
        $argsForVariant = @('--stdp-w-max', '10')
    }
    'heb_drive' {
        $argsForVariant = @('--enable-hebbian', '--hebbian-weight-decay', '1e-7',
                             '--lang-input-drive-pA', '1000')
    }
    'heb_stdp' {
        $argsForVariant = @('--enable-hebbian', '--hebbian-weight-decay', '1e-7',
                             '--stdp-w-max', '10')
    }
    'drive_stdp' {
        $argsForVariant = @('--lang-input-drive-pA', '1000', '--stdp-w-max', '10')
    }
    default {
        "Unknown winner: $winner — exiting" | Out-File -Append $logFile
        if (Test-Path $pidFile) {
            Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
        }
        return
    }
}

# 6-seed validation in parallel-3 (seed 42 already done)
$seeds = @(43, 44, 100, 101, 102)
"Validating $winner on seeds $($seeds -join ',') (seed 42 already done)" | Out-File -Append $logFile
"Parallelism: 3 seeds at a time" | Out-File -Append $logFile

$parallelism = 3
$batchIndex = 0
for ($i = 0; $i -lt $seeds.Count; $i += $parallelism) {
    $batchIndex++
    $batchEnd = [Math]::Min($i + $parallelism - 1, $seeds.Count - 1)
    "" | Out-File -Append $logFile
    "--- Batch $batchIndex (seeds $($seeds[$i])..$($seeds[$batchEnd])) at $(Get-Date) ---" | Out-File -Append $logFile

    $procs = @()
    for ($j = $i; $j -le $batchEnd; $j++) {
        $seed = $seeds[$j]
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
        $procs += @{ Proc = $proc; PidFile = $sPidFile; Seed = $seed }
    }

    foreach ($info in $procs) {
        $info.Proc.WaitForExit()
        "Seed $($info.Seed) completed (exit $($info.Proc.ExitCode)) at $(Get-Date)" | Out-File -Append $logFile
        if (Test-Path $info.PidFile) {
            Move-Item -Path $info.PidFile -Destination "$($info.PidFile).done" -Force
        }
    }
}

"" | Out-File -Append $logFile
"=== AUTO-FOLLOWUP COMPLETE for winner=$winner at $(Get-Date) ===" | Out-File -Append $logFile

# Final aligned summary across 6 seeds
"" | Out-File -Append $logFile
"--- Final aligned summary at $(Get-Date) ---" | Out-File -Append $logFile
& python -m research.runners.permuted_label_check --pattern "text_eval_arch_$winner`_seed*.json" 2>&1 | Out-File -Append $logFile

if (Test-Path $pidFile) {
    Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
}
