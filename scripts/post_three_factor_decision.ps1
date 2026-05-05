# Outcome-conditional orchestrator for what fires after bio_three_factor
# completes. Replaces the simpler post_three_factor_reporter.ps1 with a
# decision tree that auto-launches the right next experiment.
#
# Decision tree (keyed on tf_with_topo_fs aligned ratio):
#   >= 4/6 aligned: three-factor + biology fix WORKS at biological scale.
#     Auto-launch bio_three_factor_validation.yaml (fresh seeds 200/300s)
#     to confirm. Then write headline.
#
#   1-3/6 aligned: partial signal. Three-factor weaker than gradient
#     (gradient got 3/3 same condition). Stop for manual review —
#     ablations on eligibility-decay-tau or DA-scheme are warranted but
#     should be human-designed.
#
#   0/6 aligned: scalar-DA credit assignment insufficient. Headline =
#     "biology-plausible rules with global RPE alone cannot match
#     supervised gradient at this task." Stop for manual review;
#     next direction is apical-basal dendritic learning (Bono & Clopath
#     2017) or predictive coding (Rao & Ballard 1999) — neither
#     currently implemented.
#
# Replaces the simpler post_three_factor_reporter.ps1 (kept compatible:
# same poll-and-aggregate pattern, but adds decision logic).

$outDir = "research/findings/raw/g11_bg"
$findingsDir = "research/findings"
$logFile = "$outDir/post_three_factor_decision.log"
$tfMaster = "$outDir/bio-three-factor.master.log"

"=== Post-3factor decision orchestrator started $(Get-Date) ===" | Out-File -FilePath $logFile

# Step 1: poll for completion
"Polling for bio-three-factor COMPLETE..." | Out-File -Append $logFile
while ($true) {
    if (Test-Path $tfMaster) {
        $text = Get-Content $tfMaster -Raw -ErrorAction SilentlyContinue
        if ($text -match "bio-three-factor COMPLETE") {
            "Three-factor done at $(Get-Date)" | Out-File -Append $logFile
            break
        }
    }
    Start-Sleep -Seconds 60
}

# 30s grace for last seed's JSON to flush
Start-Sleep -Seconds 30

# Step 2: aggregate sweep — get the per-condition aligned ratios
"" | Out-File -Append $logFile
"Aggregating bio_three_factor at $(Get-Date)" | Out-File -Append $logFile
$tfOutFile = "$findingsDir/2026-05-05-bio-three-factor-results.md"
$tfOutput = & "C:\python312\python.exe" -m research.result_aggregator `
    --config bio_three_factor `
    --out $tfOutFile 2>&1
"$tfOutput" | Out-File -Append $logFile

# Step 3: parse the topo_fs aligned ratio specifically. The aggregator
# output table contains "B3 with topo + FS | 6 | true mean | best mean
# | excess | ALIGNED/N | ...". We use Python to extract this cleanly
# instead of brittle PowerShell regex.
"" | Out-File -Append $logFile
"Parsing tf_with_topo_fs aligned ratio..." | Out-File -Append $logFile
$alignedTopoFs = & "C:\python312\python.exe" -c @"
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
        d = json.load(open(f'research/findings/raw/g11_bg/text_eval_3factor_tf_with_topo_fs_seed{s}.json'))
        cm = d['word_to_action_eval']['confusion_matrix']
        _, bp = best_perm(cm)
        if bp == ('N','E','S','W'): aligned += 1
        total += 1
    except FileNotFoundError: pass
print(f'{aligned}/{total}')
"@
"tf_with_topo_fs aligned: $alignedTopoFs" | Out-File -Append $logFile

# Parse "X/Y" from output
$matched = $alignedTopoFs -match '(\d+)/(\d+)'
$alignedN = if ($matched) { [int]$Matches[1] } else { 0 }
$totalN = if ($matched) { [int]$Matches[2] } else { 0 }
"Parsed: aligned=$alignedN total=$totalN" | Out-File -Append $logFile

# Step 4: decision tree
if ($alignedN -ge 4) {
    # SUCCESS PATH
    "" | Out-File -Append $logFile
    "DECISION: tf_with_topo_fs aligned $alignedN/$totalN >= 4/6 -> success" | Out-File -Append $logFile
    "Three-factor + biology fix MATCHES gradient at biological scale." | Out-File -Append $logFile
    "Auto-launching bio_three_factor_validation.yaml (fresh seeds 200s/300s)" | Out-File -Append $logFile

    $valProc = Start-Process -FilePath "python.exe" `
        -ArgumentList "-u","-m","research.experiment_runner",
                      "experiments/bio_three_factor_validation.yaml" `
        -RedirectStandardOutput "$outDir/bio_three_factor_validation.stdout.log" `
        -RedirectStandardError "$outDir/bio_three_factor_validation.stderr.log" `
        -WindowStyle Hidden -PassThru
    "Validation sweep launched as PID $($valProc.Id)" | Out-File -Append $logFile
    $valProc.WaitForExit()
    "Validation finished (exit $($valProc.ExitCode)) at $(Get-Date)" | Out-File -Append $logFile

    # Aggregate validation (uses dedicated bio_three_factor_validation
    # config with the tfv_* file pattern + seeds 200s/300s).
    "" | Out-File -Append $logFile
    "Aggregating validation results..." | Out-File -Append $logFile
    $valOut = & "C:\python312\python.exe" -m research.result_aggregator `
        --config bio_three_factor_validation `
        --out "$findingsDir/2026-05-05-bio-three-factor-validation-results.md" 2>&1
    "$valOut" | Out-File -Append $logFile

} elseif ($alignedN -ge 1) {
    # PARTIAL PATH
    "" | Out-File -Append $logFile
    "DECISION: tf_with_topo_fs aligned $alignedN/$totalN -> partial" | Out-File -Append $logFile
    "Three-factor weaker than gradient (which got 3/3 same condition)." | Out-File -Append $logFile
    "STOPPING for manual review. Suggested next:" | Out-File -Append $logFile
    "  - Eligibility-decay-tau ablation (50ms vs 200ms)" | Out-File -Append $logFile
    "  - DA-scheme ablation (sign-only vs magnitude-graded)" | Out-File -Append $logFile

} else {
    # FAILURE PATH
    "" | Out-File -Append $logFile
    "DECISION: tf_with_topo_fs aligned 0/$totalN -> failure" | Out-File -Append $logFile
    "Scalar-DA credit assignment INSUFFICIENT for this task." | Out-File -Append $logFile
    "Headline finding: biology-plausible rules with global RPE alone" | Out-File -Append $logFile
    "  cannot match supervised gradient. Need richer biology:" | Out-File -Append $logFile
    "  - Apical-basal dendritic learning (Bono & Clopath 2017)" | Out-File -Append $logFile
    "  - Or predictive coding (Rao & Ballard 1999)" | Out-File -Append $logFile
    "Neither currently implemented; STOPPING for manual research direction." | Out-File -Append $logFile
}

"" | Out-File -Append $logFile
"=== Post-3factor decision orchestrator COMPLETE at $(Get-Date) ===" | Out-File -Append $logFile
