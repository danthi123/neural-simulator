---
type: finding
status: live
date: 2026-08-01
mechanism: deep-credit-on-spikes
supersedes_claim: "the K=8 residual is CLOSED ... RESIDUAL now only the 6-seed bar (deep-credit-on-spikes current_status, 2026-08-01 morning)"
artifacts:
  - research/findings/raw/gap4_6seed_shuffleDFA/gap4_6seed_shuffleDFA_audit.json
---

# gap#4: the 6-seed bar RAN — forward-learning + √K trend are REAL, but the deep-credit control (shuffle-DFA) LEAKS on the majority of seeds → the deep-CREDIT *attribution* is NOT closed

**One-line verdict:** the deep-credit-on-spikes closure named "the 6-seed bar" as its *only* remaining residual.
That bar has now been run (noise-OFF, depth-2, at K=8 and K=16, seeds 42/43/44/100/101/102), and the runner
returns **SIGNAL=False on 11 of 12 runs** — because the **shuffle-DFA credit control leaks** on 4/6 seeds at
each K. The forward learning and the √K `inherit` trend are genuinely real; what is *not* established at 6
seeds is that **correct credit routing** — rather than the population reservoir's own expressivity — is what
drives the result. This is a silent-failure rule #1 correction: the banked √K "closure" read the `inherit`
field past the run's own negative verdict.

Artifact (aggregate, all 12 runs + per-seed): `research/findings/raw/gap4_6seed_shuffleDFA/gap4_6seed_shuffleDFA_audit.json`.

## The 6-seed × K table (GO gate: shuffle-DFA ≤ chance+0.10 = 0.433; chance 0.333)

| K | SIGNAL=GO | shuffle-DFA leaks (>0.433) | mean inherit | mean permuted | mean shuffle-DFA |
|---|---|---|---|---|---|
| 8  | **1/6** | **4/6** | 0.685 | 0.247 (clean) | 0.438 |
| 16 | **0/6** | **4/6** | 0.852 | 0.247 (clean) | 0.494 |

<!--derived from the per_seed arrays in the cited artifact-->
Per-seed shuffle-DFA — K=16: 0.593/0.593/0.519/0.556/0.333/0.370; K=8: 0.296/0.444/0.593/0.481/0.370/0.444.
`inherit` rises with K (0.685→0.852) and the **permuted-label control is clean at both K** (0.247), so the
teacher signal genuinely matters and the net genuinely learns the forward map. The one failing control is
shuffle-DFA — scrambling the DFA credit route across the batch, forward unchanged — and it stays well above
chance on 8 of 12 runs.

## What this means (and does not)

- **REAL:** e-prop trains the forward task on the production Izhikevich bridge; the teacher signal is
  load-bearing (permuted → chance); `inherit` climbs monotonically with the population factor K.
- **NOT ESTABLISHED at 6 seeds:** that the climb is *deep credit* (correct routing) rather than **reservoir
  expressivity that grows with K**. At 1–2 hidden layers with K-fold population coding, the spiking reservoir +
  trained readout carry enough of the task that mis-routing the credit only partially degrades it (shuffle-DFA
  0.33–0.59, not chance 0.333). The √K `inherit` gain and the √K *reservoir-capacity* gain are confounded on
  this task.
- **The banked claim to retract:** "the K=8 residual is CLOSED and the population lever surpasses the reference
  ceiling; RESIDUAL now only the 6-seed bar." The 6-seed bar was the residual, it ran, and it does not clear
  the deep-credit control. Corrected in `research/biology/deep-credit-on-spikes.md` current_status and the
  master roadmap §7 gap#4 banner.

## The real residual (a method, not "more seeds")

A control the reservoir **cannot defeat**: a task/operating-point where credit routing is genuinely
load-bearing, measured against a **frozen-hidden reservoir** baseline (train only the readout on a fixed random
hidden net). If e-prop with correct credit clears the frozen-hidden reservoir *and* shuffle-DFA collapses to
chance, the deep-credit attribution is earned. Until then gap#4 has a demonstrated **forward-learning-on-spikes
+ population-capacity** result, not a clean **deep-credit** one. Same conclusion the arc-A shallow atom reached
independently (its shuffle-DFA was reported-not-gated for exactly this reason) — the two arcs converge on the
depth-2, reservoir-controlled testbed as the next de-risk.
