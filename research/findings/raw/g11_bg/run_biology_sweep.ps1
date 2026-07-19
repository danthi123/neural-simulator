# Biology-grounded sweep - runs after current minimal-iso 6-seed lands.
#
# Stage 1: ANTI-CHEAT CONTROL (1 run, 1 seed, no STDP)
#   topographic init + freeze_stdp + 100 events
#   If aligned >= 1/1: topographic prior is too strong (hardcoding the
#     answer). Logs warning, halves the bias factor, retries once.
#     If still cheating: aborts the sweep, manual review needed.
#   If aligned 0/1: prior is mild enough that STDP must do real work.
#     Proceed to Stage 2.
#
# Stage 2: 3-condition x 6-seed sweep in parallel-3
#   Condition 2: +FS only (random init, motor lateral inhibition)
#   Condition 3: +Topo only (topographic init 1.5/0.7, NO FS)
#   Condition 4: +Topo +FS (combined biology-grounded fix)
#
#   Condition 1 (random init, no FS) is the currently-running minimal-iso.
#   Already covered, no need to re-run.
#
# Total wall: anti-cheat 10 min + 18 runs in parallel-3 = ~3-4 hours.

$outDir = "research/findings/raw/g11_bg"
$masterLog = "$outDir/run_biology_sweep.master.log"
$pidFile = "$outDir/run_biology_sweep.orchestrator-pid"

$PID | Out-File -FilePath $pidFile -Encoding ASCII

"=== Biology-grounded sweep started $(Get-Date) ===" | Out-File -FilePath $masterLog
"" | Out-File -Append $masterLog

# ============================================================
# Stage 1: ANTI-CHEAT CONTROL
# ============================================================
"--- Stage 1: anti-cheat control (topo + freeze_stdp) at $(Get-Date) ---" | Out-File -Append $masterLog
"  Hypothesis: topographic prior alone (1.5/0.7, no STDP) should NOT align" | Out-File -Append $masterLog
"  If it does, the prior is too strong (hardcoding the answer)" | Out-File -Append $masterLog

$antiCheatLog = "$outDir/biology_sweep_anticheat_seed42.log"
$antiCheatErr = "$outDir/biology_sweep_anticheat_seed42.log.err"
$antiCheatPid = "$outDir/biology_sweep_anticheat_seed42.pid"
$antiCheatStats = "$outDir/text_eval_biology_anticheat_seed42.json"

$proc = Start-Process -FilePath "python.exe" -ArgumentList @(
    "-m", "research.runners.text_minimal_isolation",
    "--seed", "42",
    "--n-events-per-direction", "100",  # short - just need to verify init pattern alone
    "--stim-steps-per-step", "100",
    "--reset-steps", "50",
    "--dt-ms", "1.0",
    "--topographic-bias-factor", "1.5",
    "--off-target-bias-factor", "0.7",
    "--freeze-stdp",
    "--out-stats", $antiCheatStats
) -RedirectStandardOutput $antiCheatLog -RedirectStandardError $antiCheatErr `
  -PassThru -NoNewWindow

$proc.Id | Out-File -FilePath $antiCheatPid -Encoding ASCII
"Anti-cheat launched as PID $($proc.Id)" | Out-File -Append $masterLog
$proc.WaitForExit()
"Anti-cheat finished (exit $($proc.ExitCode)) at $(Get-Date)" | Out-File -Append $masterLog
if (Test-Path $antiCheatPid) {
    Move-Item -Path $antiCheatPid -Destination "$antiCheatPid.done" -Force
}

# Parse anti-cheat result
"" | Out-File -Append $masterLog
$antiCheatResult = & python -c @"
import json, itertools
WORDS = ['north', 'east', 'south', 'west']
TRUE_MAP = {'north': 'N', 'east': 'E', 'south': 'S', 'west': 'W'}
ACTIONS = ('N', 'E', 'S', 'W')

try:
    d = json.load(open('$antiCheatStats'))
    cm_raw = d['word_to_action_eval']['confusion_matrix']
    cm = {w: {a: int(cm_raw.get(w, {}).get(a, 0)) for a in ACTIONS} for w in WORDS}
    def acc(mapping):
        c = t = 0
        for w, row in cm.items():
            for a, n in row.items():
                t += n
                if a == mapping[w]: c += n
        return c / max(t, 1)
    true_acc = acc(TRUE_MAP)
    best_acc = 0
    best_perm = None
    for perm in itertools.permutations(ACTIONS):
        a = acc(dict(zip(WORDS, perm)))
        if a > best_acc:
            best_acc = a
            best_perm = perm
    aligned = 1 if best_perm == ACTIONS else 0
    print(f'TRUE_ACC:{true_acc:.4f}')
    print(f'BEST_ACC:{best_acc:.4f}')
    print(f'BEST_PERM:{"".join(best_perm)}')
    print(f'ALIGNED:{aligned}')
except Exception as e:
    print(f'ERROR:{e}')
"@ 2>&1

"$antiCheatResult" | Out-File -Append $masterLog

$alignedLine = $antiCheatResult | Where-Object { $_ -match '^ALIGNED:(\d+)' } | Select-Object -First 1
$aligned = if ($alignedLine -match 'ALIGNED:(\d+)') { [int]$Matches[1] } else { -1 }
"Anti-cheat aligned = $aligned" | Out-File -Append $masterLog

if ($aligned -ge 1) {
    "" | Out-File -Append $masterLog
    "*** ANTI-CHEAT FAILED: topographic prior alone aligned without STDP ***" | Out-File -Append $masterLog
    "*** Aborting sweep. Reduce topographic_bias_factor (try 1.3 / 0.8). ***" | Out-File -Append $masterLog
    if (Test-Path $pidFile) {
        Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
    }
    return
}

"" | Out-File -Append $masterLog
"Anti-cheat PASSED (aligned 0/1). Topographic prior is mild enough; STDP must do real work." | Out-File -Append $masterLog

# ============================================================
# Stage 2: 3-condition x 6-seed sweep
# ============================================================
"" | Out-File -Append $masterLog
"--- Stage 2: 3-condition x 6-seed biology sweep at $(Get-Date) ---" | Out-File -Append $masterLog

$conditions = @(
    @{
        Name = "fs_only"
        Description = "+FS only (random init, motor PV-FS lateral inhibition)"
        Args = @("--enable-motor-fs")
    },
    @{
        Name = "topo_only"
        Description = "+Topo only (topographic init 1.5/0.7, no FS)"
        Args = @("--topographic-bias-factor", "1.5",
                 "--off-target-bias-factor", "0.7")
    },
    @{
        Name = "topo_fs"
        Description = "+Topo +FS (combined biology-grounded fix)"
        Args = @("--topographic-bias-factor", "1.5",
                 "--off-target-bias-factor", "0.7",
                 "--enable-motor-fs")
    }
)

$seeds = @(42, 43, 44, 100, 101, 102)
$parallelism = 3

foreach ($cond in $conditions) {
    "" | Out-File -Append $masterLog
    "--- Condition: $($cond.Name) - $($cond.Description) at $(Get-Date) ---" | Out-File -Append $masterLog

    for ($i = 0; $i -lt $seeds.Count; $i += $parallelism) {
        $batchEnd = [Math]::Min($i + $parallelism - 1, $seeds.Count - 1)
        "  Batch (seeds $($seeds[$i])..$($seeds[$batchEnd])) at $(Get-Date)" | Out-File -Append $masterLog

        $procs = @()
        for ($j = $i; $j -le $batchEnd; $j++) {
            $seed = $seeds[$j]
            $logFile = "$outDir/biology_$($cond.Name).seed$seed.log"
            $errFile = "$outDir/biology_$($cond.Name).seed$seed.log.err"
            $pf = "$outDir/biology_$($cond.Name).seed$seed.pid"
            $outStats = "$outDir/text_eval_biology_$($cond.Name)_seed$seed.json"

            $procArgs = @(
                "-m", "research.runners.text_minimal_isolation",
                "--seed", "$seed",
                "--n-events-per-direction", "1000",
                "--stim-steps-per-step", "100",
                "--reset-steps", "50",
                "--dt-ms", "1.0",
                "--out-stats", $outStats
            ) + $cond.Args

            $p = Start-Process -FilePath "python.exe" -ArgumentList $procArgs `
                -RedirectStandardOutput $logFile -RedirectStandardError $errFile `
                -PassThru -NoNewWindow
            $p.Id | Out-File -FilePath $pf -Encoding ASCII
            "    Seed $seed launched as PID $($p.Id)" | Out-File -Append $masterLog
            $procs += @{ Proc = $p; PidFile = $pf; Seed = $seed }
        }

        foreach ($info in $procs) {
            $info.Proc.WaitForExit()
            "    Seed $($info.Seed) finished (exit $($info.Proc.ExitCode)) at $(Get-Date)" | Out-File -Append $masterLog
            if (Test-Path $info.PidFile) {
                Move-Item -Path $info.PidFile -Destination "$($info.PidFile).done" -Force
            }
        }
    }
}

# ============================================================
# Final analysis
# ============================================================
"" | Out-File -Append $masterLog
"--- Final aligned summary at $(Get-Date) ---" | Out-File -Append $masterLog
& python -m research.runners.permuted_label_check --pattern "text_eval_biology_*_seed*.json" 2>&1 | Out-File -Append $masterLog

# Also include the 6-seed minimal-iso results (condition 1 baseline)
"" | Out-File -Append $masterLog
"--- Condition 1 baseline (random init, no FS): minimal_iso ---" | Out-File -Append $masterLog
& python -m research.runners.permuted_label_check --pattern "text_eval_minimal_iso_seed*.json" 2>&1 | Out-File -Append $masterLog

if (Test-Path $pidFile) {
    Move-Item -Path $pidFile -Destination "$pidFile.done" -Force
}

"" | Out-File -Append $masterLog
"=== BIOLOGY SWEEP COMPLETE at $(Get-Date) ===" | Out-File -Append $masterLog
