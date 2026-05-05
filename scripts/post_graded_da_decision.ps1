# Polls bio-three-factor-graded-da master log for COMPLETE, parses verdict,
# auto-fires the right next step:
#   >= 4/6 aligned -> graded-DA validation (fresh seeds 200s/300s)
#   0-3/6 aligned  -> stop, point at dendritic-learning design doc

$outDir = "research/findings/raw/g11_bg"
$findingsDir = "research/findings"
$logFile = "$outDir/post_graded_da_decision.log"
$gradedMaster = "$outDir/bio-three-factor-graded-da.master.log"

"=== Post-graded-DA orchestrator started $(Get-Date) ===" | Out-File -FilePath $logFile

while ($true) {
    if (Test-Path $gradedMaster) {
        $text = Get-Content $gradedMaster -Raw -ErrorAction SilentlyContinue
        if ($text -match "bio-three-factor-graded-da COMPLETE") {
            "Graded-DA done at $(Get-Date)" | Out-File -Append $logFile
            break
        }
    }
    Start-Sleep -Seconds 60
}

Start-Sleep -Seconds 30

# Aggregate
"" | Out-File -Append $logFile
"Aggregating bio_three_factor_graded_da..." | Out-File -Append $logFile
$gradedOutFile = "$findingsDir/2026-05-05-bio-three-factor-graded-da-results.md"
$gradedOutput = & "C:\python312\python.exe" -m research.result_aggregator `
    --config bio_three_factor_graded_da `
    --out $gradedOutFile 2>&1
"$gradedOutput" | Out-File -Append $logFile

# Parse aligned ratio
$alignedG = & "C:\python312\python.exe" -c @"
import json, itertools
WORDS = ['north', 'east', 'south', 'west']
ACTIONS = ('N','E','S','W')
def best_perm(cm):
    best, bp = 0, None
    for p in itertools.permutations(ACTIONS):
        m = dict(zip(WORDS, p))
        c = sum(cm[w][m[w]] for w in WORDS)
        t = sum(sum(cm[w].values()) for w in WORDS)
        a = c/t if t else 0
        if a > best: best, bp = a, p
    return best, bp
aligned = 0; total = 0
for s in [42, 43, 44, 100, 101, 102]:
    try:
        d = json.load(open(f'research/findings/raw/g11_bg/text_eval_3factor_tfg_with_topo_fs_seed{s}.json'))
        cm = d['word_to_action_eval']['confusion_matrix']
        _, bp = best_perm(cm)
        if bp == ('N','E','S','W'): aligned += 1
        total += 1
    except FileNotFoundError: pass
print(f'{aligned}/{total}')
"@
"Graded-DA tfg_with_topo_fs aligned: $alignedG" | Out-File -Append $logFile

$matched = $alignedG -match '(\d+)/(\d+)'
$alignedN = if ($matched) { [int]$Matches[1] } else { 0 }
$totalN = if ($matched) { [int]$Matches[2] } else { 0 }

if ($alignedN -ge 4) {
    "" | Out-File -Append $logFile
    "DECISION: graded-DA aligned $alignedN/$totalN -> success" | Out-File -Append $logFile
    "*** HEADLINE: scalar SIGN-ONLY DA was the bottleneck. ***" | Out-File -Append $logFile
    "*** Magnitude-graded DA (Schultz 1998) preserves biology AND works. ***" | Out-File -Append $logFile

    # Auto-launch validation
    "" | Out-File -Append $logFile
    "Auto-launching validation (fresh seeds 200s/300s)..." | Out-File -Append $logFile
    $valProc = Start-Process -FilePath "python.exe" `
        -ArgumentList "-u","-m","research.experiment_runner",
                      "experiments/bio_three_factor_graded_da_validation.yaml" `
        -RedirectStandardOutput "$outDir/bio_three_factor_graded_da_validation.stdout.log" `
        -RedirectStandardError "$outDir/bio_three_factor_graded_da_validation.stderr.log" `
        -WindowStyle Hidden -PassThru
    "Validation launched as PID $($valProc.Id)" | Out-File -Append $logFile
    $valProc.WaitForExit()
    "Validation finished (exit $($valProc.ExitCode))" | Out-File -Append $logFile

    $valOut = & "C:\python312\python.exe" -m research.result_aggregator `
        --config bio_three_factor_graded_da_validation `
        --out "$findingsDir/2026-05-05-bio-three-factor-graded-da-validation-results.md" 2>&1
    "$valOut" | Out-File -Append $logFile
} else {
    "" | Out-File -Append $logFile
    "DECISION: graded-DA aligned $alignedN/$totalN -> insufficient" | Out-File -Append $logFile
    "*** HEADLINE: even magnitude-graded DA insufficient. ***" | Out-File -Append $logFile
    "*** Global scalar feedback in any form cannot match gradient. ***" | Out-File -Append $logFile
    "*** ***" | Out-File -Append $logFile
    "*** Recommended next: APICAL-BASAL DENDRITIC LEARNING (Bono & Clopath 2017) ***" | Out-File -Append $logFile
    "*** Design doc: docs/plans/2026-05-05-dendritic-learning-design.md ***" | Out-File -Append $logFile
    "*** Estimated scope: 1.5-2 months focused engineering. ***" | Out-File -Append $logFile
    "*** ***" | Out-File -Append $logFile
    "*** STOPPING for manual research direction. ***" | Out-File -Append $logFile
}

"" | Out-File -Append $logFile
"=== Post-graded-DA orchestrator COMPLETE at $(Get-Date) ===" | Out-File -Append $logFile
