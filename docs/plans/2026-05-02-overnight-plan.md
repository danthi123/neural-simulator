# 2026-05-02 — Autonomous overnight session plan

**Started:** 2026-05-02 01:30 EDT (after user said "headed to bed")
**Goal:** push text I/O accuracy meaningfully above chance (>=40%) with biology-grounded fixes; no cheats.
**End condition:** user says stop OR no productive direction remains.

## State at session start

- Current run: 100-ep R3+R6 partial Tier 1 (PID 51184), seed=42
  - Started 00:43:30, training reached ep 100/100 = 35.3% correct moves at 01:30
  - Eval phase running now; JSON expected ~01:34
  - Expected outcome: ~30-32% on I->W and W->A (matching baseline)
- Prior baseline: 100-ep R3+R6 (combined.json) = 32.5% I->W / 30% W->A
  - **Critical insight from analyzer:** baseline p=0.18 is NOT statistically
    significant at n=40. Heavy "east" prediction bias (19/40 = 47.5%).
- Failed experiment: 300-ep with full Tier 1 = 20% / 20% (regressed, root cause:
  reset_steps 100->50 caused NMDA bleedover; reverted at cee3403)
- Failed experiment: heterogeneity+OU disable smoke = 2.4% (load-bearing, reverted)

## Infrastructure shipped this session

| Commit | Change |
|--------|--------|
| cee3403 | Revert T1.2 reset_steps=50; document regression |
| 7a7d9fd | Tier 2.6 heterogeneity+OU disable: NEGATIVE finding documented |
| d4410dd | Update speedup playbook with negative findings |
| d3f28f0 | text_eval_analyze.py: post-hoc analyzer with binomial p-value + decision-tree verdict |
| acd0a48 | Bump default eval n: 40->100, 10->25 (statistical power) + drive override CLI flags |
| dc0be53 | Interleave words in W->A eval: zero consecutive same-word trials, eliminates baseline contamination |
| 3e4e9e4 | Add --stim-steps and --reset-steps CLI flags |
| 7f50cf0 | Auto-checkpoint after training + text_reeval.py loads checkpoint for re-eval |
| d445996 | text_reeval_sweep.py: grid sweep over (drive_pA, n_reset_steps) |
| b211731 | Fix reeval to apply gabor weights before load_checkpoint |

All committed + pushed to GitHub origin and Gitea.

## Decision tree for current 100-ep result (when JSON lands)

```
result accuracy (max of I->W, W->A)?
  >= 40%: STRONG WIN. Validate at 6 seeds, then push higher.
  >= 35%: WIN (statistically significant at n=100). Validate at 6 seeds.
  30-35%: MATCHES baseline (which is itself not significant).
          Methodology fixes might or might not help -- run v2 to test.
  20-30%: PARTIAL REGRESSION. Also revert T1.1 (stim 200) for next run.
  <20%:   DEEP REGRESSION. Run baseline reproduction (no Tier 1).
```

## Step-by-step overnight plan

### Step 1: Aggregate current 100-ep result (~01:35-01:45)

```bash
python -m research.runners.text_eval_analyze \
    research/findings/raw/g11_bg/text_eval_R3_R6_100ep_partialT1.json \
    --baseline research/findings/raw/g11_bg/text_eval_R3_R6_combined.json
```

Write findings doc `2026-05-02-text-io-100ep-partialT1-results.md`.
Commit + push.

### Step 2: Launch v2 100-ep with all infra improvements (~01:45)

```bash
# Background launch via Start-Process (survives Claude restart)
$proc = Start-Process python -ArgumentList @(
    "-m", "research.runners.text_eval_embodied",
    "--n-episodes", "100",
    "--steps-per-episode", "30",
    "--seed", "42",
    "--out-stats", "research/findings/raw/g11_bg/text_eval_R3R6_100ep_v2_seed42.json"
) -RedirectStandardOutput "research/findings/raw/g11_bg/R3R6_100ep_v2_seed42.log" \
  -RedirectStandardError "research/findings/raw/g11_bg/R3R6_100ep_v2_seed42.log.err" \
  -WindowStyle Hidden -PassThru
$proc.Id | Out-File "research/findings/raw/g11_bg/R3R6_100ep_v2_seed42.pid" -Encoding ascii
```

This run will:
- Use new defaults: n-eval-image-word=100, n-eval-word-action=25 (4x more eval power)
- Use interleaved word ordering in W->A eval (cleaner readout)
- Auto-save checkpoint for downstream re-eval experiments
- Same training (deterministic with seed=42, same code path)

Expected: ~75 min training + ~5 min eval (2.5x more trials than baseline).
Completion: ~03:00.

### Step 3: Run reeval sweep on v2 checkpoint (~03:00-03:40)

```bash
python -m research.runners.text_reeval_sweep \
    research/findings/raw/g11_bg/text_eval_R3R6_100ep_v2_seed42.simstate.h5 \
    --output-dir research/findings/raw/g11_bg/sweep_v2_seed42/ \
    --drives 200 300 400 500 600 \
    --resets 100 200 400 \
    --include-legacy-block
```

15 (drive x reset) combos + 1 legacy = 16 evaluations. Each ~3 min = ~50 min.

This isolates eval-side effects: which (drive_pA, n_reset_steps) gives best
accuracy on the SAME trained network? Identifies whether the 30%-ish
ceiling is set by training, methodology, or both.

### Step 4: Decision based on sweep (~03:45-04:00)

If best sweep result is significantly higher (e.g., 45%+):
- The trained network has more capability than baseline eval revealed.
- Validate at 5 more seeds with best sweep config.
- Sequential, ~75min/seed * 5 = 6.25 hr -> done by 10:15 AM.

If best sweep result is similar to baseline eval (30-35%):
- Network's accuracy ceiling is real.
- Try architectural change (next experiment).
- Best candidate: stronger lang_input_drive during TRAINING (200->400 pA).
  Biology: louder utterance during embodied training.
  Same time as current (~75min training + ~5min eval).

### Step 5: Architectural followup if needed (~04:00-05:30)

```bash
# If sweep didn't help, retrain with stronger drive
$proc = Start-Process python -ArgumentList @(
    "-m", "research.runners.text_eval_embodied",
    "--n-episodes", "100",
    "--steps-per-episode", "30",
    "--seed", "42",
    "--lang-input-drive-pA", "400",
    "--lang-output-coactive-pA", "300",
    "--out-stats", "research/findings/raw/g11_bg/text_eval_R3R6_drive400_seed42.json"
) ...
```

### Step 6: 6-seed validation of best config (~05:30-12:00)

Sequential runs at seeds 43, 44, 100, 101, 102 with whichever config
performed best. Each ~80 min. Total ~6.5 hr.

User wakes up to: 1-6 results from overnight, depending on path taken.

## Backup directions if results don't improve

If accuracy stays at ~30% across all overnight experiments:
- Architectural review: maybe motor_X pools too small (10 neurons), language regions too small (256), etc.
- Training-time review: maybe reward signal too weak, more episodes needed
- Methodology review: maybe delta-from-baseline is fundamentally inadequate, need different decoding

These would NOT be implemented overnight -- documented for morning discussion.

## What I will NOT do overnight

- Modify load-bearing infrastructure (heterogeneity, OU process, NMDA settings)
- Disable safety-critical config (already proven to break network)
- Add architectural changes that haven't been internally debated
- Skip biology-grounded principles (per "no cheats/shortcuts" instruction)

## Live status

Updates appended below as each step completes.

### Step 1 status: PENDING (waiting for current 100-ep eval to finish)
