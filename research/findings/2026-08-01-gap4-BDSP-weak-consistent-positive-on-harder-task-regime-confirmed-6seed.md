---
type: finding
status: contributing
date: 2026-08-01
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/realspikes_bdsp/bdsp_hardtask_6seed_aggregate.json
---

# gap#4 crux: on a HARDER task the coincidence-gated BDSP gives a WEAK but CONSISTENT positive transport-free credit signal on real spikes — the regime hypothesis is confirmed (6-seed)

<!--derived-->
**One-line verdict:** the companion that tests the regime hypothesis the small-task BDSP finding named. On the small
sweet-spot task (k=9) a random-projection reservoir is near-optimal, so no rule (incl. the coincidence-gated BDSP)
reliably beat it (dcs>0 3/6, mean −0.022 — a tie). On a HARDER task (n_prop=4, n_super=48, **k=17**) where the
reservoir genuinely FAILS (frozen 0.131 vs oracle 0.796), the SAME coincidence-gated + sigmoid-baseline DFA BDSP,
trained + read on real spikes, gives a **consistent positive**: `deep_credit_share > 0` on **5/6** seeds, mean dcs
**+0.042** (BDSP held-out 0.159 vs frozen 0.131), with the directed error load-bearing on 4/6. This is the **FIRST
consistent positive transport-free credit signal on real spikes** — and it directly confirms the regime hypothesis:
when the reservoir isn't already near-optimal, directed credit adds. But it is **weak** — it closes only ~**4%** of
the frozen→oracle gap, still overfits (train 0.669 vs held-out 0.159), and is not a GO (beats-by-0.05-margin 2/6,
anti-cheats not clean on all 6). No `sim/` edit.

Artifact: `research/findings/raw/gap4/realspikes_bdsp/bdsp_hardtask_6seed_aggregate.json` (backend numpy/CPU). Runner:
`research/runners/_gap4_realspikes_bdsp_credit_derisk.py`.

## Result — 6 seeds, harder task (k=17), real-spikes read, eta 0.003

<!--derived-->
| metric | small task (k=9) | harder task (k=17) |
|---|---|---|
| dcs > 0 | 3/6 | **5/6** |
| mean deep_credit_share | −0.022 (tie) | **+0.042** (positive) |
| shuffle-DFA load-bearing | ambiguous | 4/6 |
| frozen vs oracle | 0.321 vs 0.975 (reservoir near-optimal) | 0.131 vs 0.796 (**reservoir fails**) |
| BDSP held-out vs oracle | 0.324 vs 0.975 | 0.159 vs 0.796 |
| fraction of frozen→oracle gap closed | ~0 | **~0.04** |

Per-seed dcs (harder task): +0.053 / −0.068 / +0.017 / +0.051 / +0.090 / +0.106.

## What this settles for the crux (the honest, refined conclusion)

<!--derived-->
The residual named across the gap#4 real-spikes arc — *"the regime, not the rule"* — is confirmed. On the small
task the reservoir was near-optimal, leaving directed credit nothing to add; on a harder task where the reservoir
fails, the coincidence-gated BDSP **consistently adds a small positive** transport-free credit (dcs>0 5/6). So
transport-free deep credit on real spikes is **not zero** — it is a real, consistent, but **weak** signal that
appears only when the task gives it room. The full arc, honestly stated: of four rules on the real-spikes movable
hidden, the unsupervised covariance rule *degrades* the codon, DFA/DFC overfit, and only the coincidence-gated
event-reading BDSP produces a positive — weakly, and only in the reservoir-fails regime. The credit closes ~4% of
the local-vs-oracle gap: a foothold on real spikes, not a solution.

## Next
<!--derived-->
The signal is real but weak; the levers to strengthen it (not exhausted): (a) reduce the overfit gap (train 0.669
vs held-out 0.159) — the credit fits train far better than held-out, so a regularizer / earlier stop / a p0/beta
BDSP sweep may convert more of the train-fit to held-out; (b) a still-harder / more-compositional task where the
directed signal has more generalizable structure to add; (c) a stronger post-event gate or a plateau-timing term
that raises per-seed reliability (load-bearing on 4/6 → aim for 6/6). The instruments (real-spikes read + the four
rule runners + the harder-task config) are all built and reusable. A weak-but-real foothold on the honest
substrate, with the strengthening levers named — no capability abandoned.
