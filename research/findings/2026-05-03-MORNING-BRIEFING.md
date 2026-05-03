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

## Key data so far (n=4 seeds, partial)

| Condition | seeds done | W→A mean | Note |
|---|---|---|---|
| v2 baseline | 6 (prior) | **28.5% ± 2.1** | reference |
| v2 + SWR (default) | 4 | **23.8% ± 2.9** | regression visible but not as tight as n=3 |
| v2 + SWR balanced (H1) | 0 | — | running tonight |
| PFC isolation (H4) | 0 | — | running tonight |

Seeds 42/43/44 all gave W→A 22-23%. Seed 100 gave 28% (no
regression). So the SWR W→A drop is **heterogeneous across seeds**
— most seeds regress, but not all.

This complicates the n=3 narrative ("consistent 6pp drop") but is
still substantively informative:
- The regression IS real for the majority of seeds
- Whatever causes it depends on stochastic per-seed dynamics
- Per-direction analysis showed prediction distributions shift
  unpredictably (each seed amplifies a DIFFERENT direction toward
  over-prediction)

## Decision tree (will run automatically when data is in)

```
H4 isolation result:
├── 80%+   → reverse curriculum (train bypass first, then cascade)
├── 50-79% → wait for H1; if H1 ≥ 27%, the bias was the issue
└── ~28%   → architectural change (bigger regions, soft readout)
```

`research/runners/swr_decision.py` applies this tree.

## What I won't have done (left for you)

* **Pivot launch**: I'll print the recommendation but won't auto-run
  the next architectural change. That's your call.
* **Bridge instrumentation for real per-region viz**: deferred — too
  risky to modify mid-batch. Worth doing tomorrow.
* **Analysis of buffer composition** for seeds 101/102 — I added
  recording in commit a6e349f but only those seeds onward have it.
  Buffer counts will be in the JSON's `training_stats[1].buffer_per_direction`.

## Health monitoring tools

* `python -m research.runners.swr_status` — alive check + current state
* `python -m research.runners.swr_aggregate` — table of all results
* `python -m research.runners.swr_decision` — recommendation

If anything looks wrong (orchestrator dead, weird results, GPU stuck),
the trail is in the various master logs:
* `run_swr_remaining.master.log` — first batch
* `wait_h4_h1.log` — waiter polling state
* `run_h4_then_h1.super.log` — super-orchestrator
* `run_h4.master.log` — H4 batch
* `run_h1.master.log` — H1 batch

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
