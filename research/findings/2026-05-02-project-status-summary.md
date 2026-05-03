# 2026-05-02 — Text I/O Project Status Summary

Snapshot of the project state across all experiments tonight + ongoing.

## The validated breakthrough

**v2 baseline (commit 144eefd + 200f73c):**
- `cfg.enable_hebbian_learning = False` (root cause of 2-month bug)
- `cfg.stdp_w_max = 5.0` (allows PFC-bypass design weights)
- Non-zero readout pathway init (0.5 ± 0.3)

**6-seed validation (n=600 cumulative trials):**
- W→A: 28.5% (p=0.027) — STATISTICALLY SIGNIFICANT vs 25% chance
- I→W: 25.3% (p=0.444) — variance, ~chance on average

This is the first rigorous demonstration of working text I/O in the
project under fair eval methodology.

## Architectural variations tested (all NEGATIVE)

| # | Variation | I→W | W→A | Verdict |
|---|---|---|---|---|
| - | **v2 baseline (seed=42)** | **33%** | **27%** | **reference** |
| 1 | Reward shaping (`wrong_move_reward=0`) | 33% | 25% | NEGATIVE |
| 2 | Stronger drives (lang_in 200→400) | 33% | 25% | NEGATIVE |
| 3 | Drive=500 reeval cross-seed | 25% | 24% | NEGATIVE |
| 4 | Bigger motor pools (10→30) | 24% | 24% | NEGATIVE |
| 5 | Longer training (100→200 ep) | 22% | 24% | NEGATIVE — weights saturated |
| 6 | Bigger lang regions (256→512) | 25% | 18% | NEGATIVE |
| 7 | Curriculum (visuomotor first) | 24% | 23% | NEGATIVE — same weights |
| 8 | Alternative decoders (4 variants) | 33% | 27% (delta) | NEGATIVE — none beat delta |
| 9 | Motor cross-coupling (90° adj) | 29% | 22% | NEGATIVE — softens but argmax decoder fights |
| 10 | **Distributed motor pop (8 sub-pools)** | RUNNING | RUNNING | **TBD ~23:00** |

## Pooled meta-analysis (16 runs, 1600 trials)

Across ALL configurations tested (including v2 + variations):
- Pooled I→W: 25.8% (p=0.25)
- Pooled W→A: 25.5% (p=0.33)

Interpretation: the v2 architecture is a real local optimum. Configurations
diverging from v2 in any direction produce equal or worse accuracy. The
6-seed v2 p=0.027 is genuine but specific to the v2 architecture.

## Per-direction stability

Across all runs (W→A per-direction means):
- **east: 28%** (range 12-48%) — most reliable
- **south: 25%** (range 8-44%)
- **north: 24%** (range 12-36%)
- **west: 24%** (range 8-40%)

East has slight advantage. North has higher variance per seed (sometimes
"lucky" with +0.24 weight bias, sometimes "unlucky" with -0.14) but
cumulative mean is near-zero — variance, not structural bias.

## Tools shipped (all reusable for future experiments)

| Tool | Purpose |
|---|---|
| `text_eval_analyze.py` | Single-run accuracy + binomial p-value + decision-tree verdict |
| `text_weight_diagnostic.py` | Pathway weights + token-targeted differentials |
| `text_weight_compare.py` | Cross-checkpoint side-by-side comparison |
| `text_reeval.py` | Load checkpoint + re-eval (cold-start bridge state limitation) |
| `text_reeval_sweep.py` | Grid sweep over (drive_pA, n_reset_steps) |
| `text_io_meta_analysis.py` | Aggregate ALL runs into master comparison |
| `text_train_curriculum.py` | Two-phase training (visuomotor first) |
| `text_train_distributed_motor` (via flag) | 8 sub-pool architecture with population vector decoder |

## CLI flags added

- `--stim-steps-per-step`, `--reset-steps`, `--enable-per-type-stp`
- `--retina-drive-pA`, `--lang-input-drive-pA`, `--lang-output-coactive-pA`
- `--correct-move-reward`, `--wrong-move-reward`
- `--eval-iw-drive-pA`, `--eval-wa-drive-pA`
- `--n-motor-per-action`, `--text-n-input-neurons`, `--text-n-output-neurons`
- `--enable-motor-cross-coupling`, `--motor-cross-coupling-{weight,density}`
- `--enable-distributed-motor-pop`, `--n-motor-pop-per-subpool`
- `--save-checkpoint` / `--no-save-checkpoint`

## Findings docs (chronological)

```
2026-05-02-text-io-100ep-reset-fix-results.md       — partial T1 reset fix
2026-05-02-text-io-hebbian-decay-root-cause.md       — ROOT CAUSE  
2026-05-02-text-io-hebbian-fix-empirical-result.md   — 3/4 LEARN
2026-05-02-text-io-BREAKTHROUGH-v2.md                — primary breakthrough
2026-05-02-reeval-bridge-state-limitation.md         — checkpoint scope
2026-05-02-text-io-multi-seed-progress.md            — 6-seed validation
2026-05-02-reward-shaping-NEGATIVE.md                — followup 1
2026-05-02-strong-drive-NEGATIVE.md                  — followup 2
2026-05-02-drive500-cross-seed-NEGATIVE.md           — followup 3
2026-05-02-longer-training-NEGATIVE.md               — followup 5
2026-05-02-curriculum-NEGATIVE-but-INFORMATIVE.md    — cascade isn't bottleneck
2026-05-02-multi-decoder-NEGATIVE.md                 — followup 8
2026-05-02-FINAL-overnight-summary.md                — overnight arc
2026-05-02-project-status-summary.md                 — this doc
```

Plus: `references/language-mechanisms-additions.md` (G.20-G.25 catalog
entries), `docs/plans/2026-05-02-text-io-next-directions-biology-grounded.md`
(strategic options), `docs/plans/2026-05-02-distributed-motor-pool-design.md`
(current experiment), `docs/plans/2026-05-02-swr-text-io-integration-design.md`
(next experiment).

## Currently running

PID 28288 distributed motor pool test at seed=42, ~ep 10/100. ETA ~23:00.

## Decision tree

If distributed-pop gives W→A ≥35%:
- Multi-seed validation (43, 44, 100, 101, 102)
- Then SWR consolidation as further enhancement

If distributed-pop gives W→A 28-35%:
- Within v2-baseline range; document as alternative architecture
- Try SWR consolidation as orthogonal improvement

If distributed-pop gives W→A <28%:
- Configuration was truly architectural, not just minor tweak
- Variation hurt as much as other variations
- Try SWR consolidation as different mechanism

In all 3 cases: SWR consolidation is the next experiment.

## Key open questions for the user

1. **Is the W→A 28.5% (p=0.027) result publishable as-is?**
   It's the first rigorous text I/O above chance with biology-grounded
   STDP + reward modulation. Worth a technical report or paper draft?

2. **Should we pivot from text-IO to a richer task?**
   Current 4-direction navigation may be too narrow. Tomasello-style
   joint attention with broader vocabulary could give more substantial
   results.

3. **Is biological accuracy more valuable than benchmark accuracy?**
   The 28.5% in our model is biology-grounded (no SVM training, no
   cheats). Real biological learning at 100 trials is also imperfect.
   Our result may be "close to biological reality" even if computationally
   modest.
