---
type: finding
status: contributing
date: 2026-08-01
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/realspikes_bdsp/bdsp_eta003_6seed_aggregate.json
---

# gap#4 crux: the coincidence-gated BDSP rule TIES the frozen reservoir on real spikes — better than the covariance rule but no reliable transport-free credit — 6-seed NEGATIVE

<!--derived-->
**One-line verdict:** the strongest candidate for a real-spikes-matched rule (finding 2026-07-22's coincidence-gated
+ sigmoid-baseline BDSP: it reads BINARY EVENT co-firing — pre-spike × post-plateau-event — plus a class-aligned
DFA credit, and had beaten a reservoir at spiking sparsity on a different substrate). Ported onto this session's
validated real-spikes movable-plateau read and run against the exact frozen-reservoir gate the covariance rule
failed: it **ties** the reservoir. It beats frozen by the 0.05 margin on **2/6** seeds, `deep_credit_share > 0` on
**3/6** (per-seed +0.028/+0.122/−0.286/−0.129/−0.065/+0.200 — high variance), mean dcs **−0.022** (BDSP held-out
0.324 vs frozen 0.321). The directed error is not cleanly load-bearing (shuffle-DFA control fails on some seeds),
and an eta sweep finds only a **narrow** sweet spot at 0.003 (0.001/0.002 negative; 0.01+ overfit + DFA not
load-bearing). So it is **better than the covariance rule** — which *degraded below* the reservoir (dcs −0.063,
1/6) — but it does **not** establish a reliable transport-free credit signal on real spikes. `GO = False`. No
`sim/` edit.

Artifact: `research/findings/raw/gap4/realspikes_bdsp/bdsp_eta003_6seed_aggregate.json` (backend numpy/CPU). Runner:
`research/runners/_gap4_realspikes_bdsp_credit_derisk.py`.

## Result — 6 seeds, eta 0.003 (the swept sweet spot), real-spikes read

<!--derived-->
| seed | BDSP held-out | FROZEN held-out | deep_credit_share |
|---|---|---|---|
| 42 | 0.352 | 0.333 | +0.028 |
| 43 | 0.333 | 0.241 | +0.122 |
| 44 | 0.241 | 0.389 | −0.286 |
| 100 | 0.333 | 0.426 | −0.129 |
| 101 | 0.352 | 0.389 | −0.065 |
| 102 | 0.352 | 0.148 | +0.200 |

Mean BDSP 0.324 vs frozen 0.321 (a tie). `beats frozen by margin 2/6`, `dcs > 0 3/6`, mean dcs **−0.022**. Anti-
cheats not clean on all seeds (the shuffle-DFA-error control does not degrade below credit on every seed → the
per-sample directed routing is not reliably load-bearing). BDSP train ≈ 0.43–0.55 vs held-out ≈ 0.32 — a residual
train→held-out gap (overfitting), the same signature the scope flagged as the honest risk for a supervised rule.

## Why it beats the covariance rule but still ties

<!--derived-->
The covariance rule read GRADED co-firing (plateau margin × pre spike-count) UNSUPERVISED, and on the real-spikes
representation it sharpened columns onto conjunctions that carry no inheritance signal — degrading below the
reservoir. The coincidence-gated BDSP fixes two things the scope named: it reads the BINARY co-spike EVENT (pre AND
post both fire) — a sparser, cleaner target — and adds a class-aligned (DFA) bounded credit. Those get it back UP to
the reservoir (a tie, vs a degrade). But the directed signal is too weak / noisy at real-spikes sparsity on this
small task (k=8, ~coarse held-out) to reliably clear the reservoir: it wins on half the seeds and loses on the
other half, and the win is not cleanly attributable to the directed routing (shuffle-DFA control ambiguous).

## What this settles for the crux, and the honest residual

<!--derived-->
FOUR credit rules have now been tried on the movable plateau hidden read via REAL spikes: unsupervised covariance
(degrades, dcs −0.063), supervised DFA and DFC+Kolen-Pollack (overfit/null on the rate stand-in), and the
coincidence-gated event-reading BDSP (ties, dcs −0.022). **None reliably beats a frozen on-bridge reservoir.** The
local-credit-vs-oracle gap (oracle 0.975 vs local ~0.32 ≈ reservoir) is robust to the credit rule on this substrate
+ task. The BDSP's tie — the best of the four — suggests the residual is not purely the rule but the **regime**: at
real-spikes sparsity on this small sweet-spot task, a random-projection reservoir is already near-optimal and the
per-sample directed signal is too noisy to add reliably. Honest residual (NOT exhausted): a p0/beta sweep of the
BDSP credit was not run, and — the more likely lever — a **larger / richer task** where a reservoir cannot already
solve it and directed credit has generalizable structure to add (the covariance/DFA nulls were also partly
task-capped). That is the next lever; the four rate-and-real-spikes rules at THIS task/op-point are closed. A mapped
boundary with the next lever named, on the honest real-spikes substrate — no capability abandoned.
