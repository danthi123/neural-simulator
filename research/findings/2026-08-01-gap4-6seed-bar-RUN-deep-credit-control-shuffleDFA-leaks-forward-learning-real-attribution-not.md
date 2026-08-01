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
That bar has now been run (noise-OFF, depth-2, K=8 and K=16, seeds 42/43/44/100/101/102) and **SIGNAL=False on
11 of 12 runs** — for two independent reasons, the second decisive: (1) the **shuffle-DFA credit control leaks**
on 4/6 seeds at each K; and (2) **a frozen random hidden reservoir is as good as full e-prop** — the runner's own
`reservoir_control` (added 2026-07-16, `reservoir_control_run=True` on every run) reports **`deep_credit_share`
mean 0.066 at K=8 and 0.005 at K=16**, *negative* on 3/6 seeds at each K. At K=16, e-prop 0.852 vs frozen-hidden
0.852 — training the hidden feedforward pathways adds **nothing**. So the √K `inherit` curve is a
**reservoir-capacity** curve; the closure read the `eprop_inherit` field (0.85) and never read the
`deep_credit_share` field (0.005) the same runner computed. This is silent-failure rule #1, and rule #7 (the
control existed and the answer was in the artifact all along). Forward learning + teacher-contingency (permuted
clean) are still real.

Artifact (aggregate, all 12 runs + per-seed): `research/findings/raw/gap4_6seed_shuffleDFA/gap4_6seed_shuffleDFA_audit.json`.

## The 6-seed × K table (GO gate: shuffle-DFA ≤ chance+0.10 = 0.433; chance 0.333)

| K | SIGNAL=GO | mean e-prop | mean frozen-hidden reservoir | **mean deep_credit_share** | deep_share negative | shuffle-DFA leaks (>0.433) |
|---|---|---|---|---|---|---|
| 8  | **1/6** | 0.685 | 0.679 | **0.066** | 3/6 | 4/6 |
| 16 | **0/6** | 0.852 | 0.852 | **0.005** | 3/6 | 4/6 |

Permuted control clean at both K (mean 0.247 → chance). `deep_credit_share = (e-prop − frozen-hidden) / (oracle −
frozen-hidden)`: at K=16 the numerator is ~0 (0.852 − 0.852), so training the hidden layers via e-prop adds
essentially nothing over freezing them at random init.

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

## Where this sits in the record (a RE-CONFIRMATION, not a discovery — cite, don't re-derive)

<!--derived: all numbers in this section are quoted from the cited findings, not this finding's own artifact-->>

This finding re-confirms a lesson the project already had and then drifted from:
`2026-07-16-deep-credit-GO-is-80pct-RESERVOIR-the-frozen-hidden-control-was-never-run.md` (the deep-credit GO was
mostly reservoir; the frozen-hidden control belongs in the gate). It adds the 6-seed `deep_credit_share`
measurement with the control now actually run.

The broader record makes the on-bridge open question precise. **Deep credit BEATS a frozen reservoir on a proper
depth-required task — 6-seed, repeatedly — but only at RATE/numpy:** XOR-over-pool best-credit 0.694 vs reservoir
0.117 (`2026-07-24-gap4-learned-selfpredicting-microcircuit-CPUrate-GO.md`, the roadmap #4 run, already RUN);
MNIST depth-4 FA 0.928 vs 0.102 (`2026-07-22-gap4-credit-BEATS-reservoir-on-MNIST-cleanxor-was-the-wrong-instrument.md`);
faithful BDSP +0.16→+0.30 at sparsity (`2026-07-23-gap4-faithful-bdsp-credit-beats-reservoir-6seed-GO.md`);
data-efficiency +0.24→+0.28 (`2026-07-24-gap4-deconfounded-credit-is-DATA-EFFICIENCY-6seed.md`). **On the
production Izhikevich BRIDGE deep credit has NEVER beaten the reservoir**, and the binding obstacle is FORWARD
REPRESENTABILITY, not the credit rule: at depth-2 the reservoir carries it (this finding); at depth≥3 the
spiking forward COLLAPSES — even the weight-transport ceiling cannot fit its own training set
(`2026-07-24-gap4-surpass-POWERED-NO-GO-tonic-pinned-frozen-representation-root-cause.md`: φ′-vanishing credit
~1600× at E≈0.04 + tonic-pinned frozen hidden code; `2026-07-31-gap4-the-crux-was-never-askable-...md`). Cite
these before any on-bridge depth run to avoid re-deriving "spiking doesn't train at depth."

## The real residual (a mechanism, not "build the control" — the control already exists)

The frozen-hidden reservoir control is **already in the runner** (`reservoir_control=True` by default,
`_onbridge_eprop_port_derisk.py:512`) and **already ran** on every one of these 12 runs — it is not something to
build. What it reports is that at this operating point (depth-2, noise-off, clean tonic drive, ≤80 epochs)
**deep credit adds ~0 over a fixed random reservoir**. So the residual is not a missing control but a missing
*signal*: an operating-point or mechanism where training the hidden feedforward pathways actually contributes.
Candidates, all already named in the record: the **learned instructive signal** (arc B / §2.8, replaces the
fixed-random DFA feedback that FA-degrades with depth); the **φ′-vanishing-credit fix** (the 2026-07-24 root
cause: credit shrinks ~1600× over depth at E≈0.04); a **representable forward** (the 2026-07-25 coincidence-
plateau expander, GO but never combined with the credit runner). Until one of those moves `deep_credit_share`
off ~0, gap#4 has a **forward-learning-on-spikes + population-capacity** result, not a **deep-credit** one. The
arc-A shallow atom reached the same place independently — the two converge on making the hidden-layer credit
*matter*, not on adding a control.
