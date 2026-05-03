# Morning briefing — autonomous SWR investigation

**Generated:** 2026-05-03 ~05:45 EDT (autonomous)
**For:** waking-up user
**Status:** mid-investigation

---

## TL;DR

Three batches running tonight, queued in sequence:

1. ✅ **v2+SWR validation** (in progress — 4/6 seeds done as of 05:45)
2. ⏳ **H4 PFC bypass isolation** (auto-launches when batch 1 done)
3. ⏳ **H1 balanced replay** (auto-launches when H4 done)

By morning ETA ~13:00 EDT all three should be complete.
**Run `python -m research.runners.swr_decision`** to see the
recommended next step based on the data.

---

## What you'll find when you wake up

### Result files

`research/findings/raw/g11_bg/`:
* `text_eval_v2_swr500_seed*.json` — 6 v2+SWR seeds
* `text_eval_h4_isolation_seed*.json` — 6 H4 isolation seeds
* `text_eval_h1_balanced_seed*.json` — 6 H1 balanced seeds

### Auto-generated summary

`research/findings/2026-05-03-swr-multiseed-summary.md` — auto-updated
by `swr_aggregate.py` after each batch. Contains:
* Headline table (all 4 conditions × all 6 seeds)
* Per-direction W→A breakdown for each condition

### Hand-written analysis

* `2026-05-03-swr-multiseed-result.md` — narrative + hypotheses
* `2026-05-03-swr-mechanism-analysis.md` — per-direction analysis
* `2026-05-03-overnight-progress.md` — running notebook (this is for me)
* `docs/plans/2026-05-03-autonomous-overnight-plan.md` — strategy doc

### Webapp surfaces

`localhost:8765/#tab=language` shows:
* Per-direction W→A aggregate at top (now 32% N / 24% E / 21% S / 21%
  W across all 33 text I/O runs — a clear N-bias even at baseline)
* All v2+SWR + H4 + H1 runs are clickable for confusion-matrix view
* New filters auto-pick up the new files

`localhost:8765/#tab=brain` shows:
* The CURRENTLY in-flight run animated (Live mode) — likely either
  an H4 seed or an H1 seed depending on time
* Click "Load run..." to replay any past run

Active-runs badge (top-left of header) shows current count.

---

## Key data so far (LIVE — updated as H4 lands)

### Final results

| Condition | seeds | W→A mean | Note |
|---|---|---|---|
| v2 baseline | 6 | **28.5% ± 2.1** | reference |
| v2 + SWR (default) | **6** | **24.3% ± 2.4** | paired-t = -6.37 (highly significant regression) |
| v2 + SWR balanced (H1) | 0 | — | running |
| PFC isolation (H4) | 3 | **24.7% ± 6.7** | early signal: high variance, near-chance |

H4 per-seed: 42=30%, 43=27%, 44=17%, 100=?, 101=?, 102=?

### Surprise: H4 has MUCH higher variance than v2+SWR

Where v2+SWR is consistently regressed (24% across all 6 seeds, std 2.4),
H4 PFC bypass isolation is consistently random — 17%, 27%, 30% so far.
This isn't an "architecture limit" so much as "training procedure can't
reliably differentiate words via direct language→motor without cascade
support."

Implication: the **paired-stim training itself** isn't producing word-
discriminative weight changes. Each seed's initial cascade biases (which
direction's cortex pool fires most spontaneously) dominate the eval
outcome. STDP can't overcome the bias in 400 paired-stim events.

Per-seed confusion matrix highlights:
- Seed 42 H4: north and east have IDENTICAL count vectors (4,8,6,7) —
  motor pool can't distinguish. Probability of identical multinomial
  outcome ~1e-3.
- Seed 44 H4: east-bias dominates ALL 4 words (11/9/9/8 east-predictions
  for north/east/south/west) — the cascade's spontaneous E-firing wins.

### Architectural pivot — sweep on seed 42 (auto-launches after H1)

5 variants tested at seed 42 (no prior training, fresh runs):
- A: motor50 (--n-motor-per-action 50, larger readout pool)
- B: sparse005 (--token-sparsity 0.05, ~orthogonal codes)
- C: lang512 (256→512 language region size)
- D: A+B (motor50+sparse005)
- E: B+C (lang512+motor50)

Threshold for full 6-seed validation: ≥ 35% W→A on seed 42.

## Decision tree (architectural pivot)

H4 isolation looks like architecture-limit territory (n=3 mean 24.7%,
high variance 17-30%). The 6-seed completion will firm this up. Most
likely pivot: structural changes via the arch sweep.

```
H4 isolation result:
├── 80%+   → reverse curriculum (train bypass first, then cascade)
├── 50-79% → wait for H1; if H1 ≥ 27%, the bias was the issue
└── ~28%   → architectural change (bigger regions, soft readout)
```

`research/runners/swr_decision.py` applies this tree.

## What I'm doing autonomously tonight (UPDATED)

* **Arch sweep auto-launches after H1** (waiter PID 28684 in
  `wait_arch_sweep.orchestrator-pid`). 5 variants × seed 42 only,
  ~3-4 hours total. Goal: identify which structural change merits
  full 6-seed validation tomorrow.
* **--token-sparsity flag added** with TDD tests so we can test
  orthogonal codes. Wired through curriculum runner, H4 runner, and
  evaluate_word_to_action.

## What I won't have done (left for you)

* **6-seed validation of the arch sweep winner**: that takes ~3 hours
  per condition. Tomorrow's call. Once you see the 1-seed signal,
  pick a winner and I'll run full 6-seed.
* **Bridge instrumentation for real per-region viz**: deferred — too
  risky to modify mid-batch. Worth doing tomorrow.
* **Analysis of buffer composition** for seeds 101/102 — I added
  recording in commit a6e349f but only those seeds onward have it.
  Buffer counts will be in the JSON's `training_stats[1].buffer_per_direction`.

## Health monitoring tools

* `python -m research.runners.swr_status` — alive check + current state
* `python -m research.runners.swr_aggregate` — table of all results
* `python -m research.runners.swr_per_seed` — per-seed cross-condition
* `python -m research.runners.swr_decision` — recommendation

If anything looks wrong (orchestrator dead, weird results, GPU stuck),
the trail is in the various master logs:
* `run_swr_remaining.master.log` — first batch (DONE)
* `wait_h4_h1.log` — original waiter (DONE — chain launched)
* `run_h4_then_h1.super.log` — super-orchestrator (in progress)
* `run_h4.master.log` — H4 batch (in progress, 3/6)
* `run_h1.master.log` — H1 batch (queued)
* `wait_arch_sweep.log` — arch sweep waiter (polling for H1)
* `run_arch_sweep_seed42.master.log` — arch sweep (queued)

## Commits tonight

12+ commits since you went to sleep. All pushed to gitea + github.
The most recent: `8736d14`. Run `git log --oneline -20` to see them.

Notable:
* `dd354d7` H1 balanced replay flag
* `334899a` H4 PFC isolation runner + orchestrators
* `a6e349f` Phase 2 buffer composition + mechanism analysis
* `a1301df` Per-direction aggregate breakdown UI
* `3d087f7` Chained orchestrators (H4 → H1 auto)
* `fef2b32` swr_status.py + swr_decision.py for health/decisions

Sleep well. Or actually — by the time you read this, it's morning.
Good morning.
