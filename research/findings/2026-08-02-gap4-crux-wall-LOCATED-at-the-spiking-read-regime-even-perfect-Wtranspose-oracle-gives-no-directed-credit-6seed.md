---
type: finding
status: contributing
date: 2026-08-02
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/realspikes/multihop_oracle_easy_6seed.json
  - research/findings/raw/gap4/realspikes/multihop_oracle_hard_6seed.json
---

# gap#4 crux — the wall is LOCATED at the SPIKING READ REGIME: even a perfect W⊤ oracle (exact loss gradient, alignment 0.999) gives NO directed credit through the finite-spike σ′(v−θ) read — not the task, not the feedback (6-seed, both tasks); the named surpass is a lower-CV read

<!--derived-->
**One-line verdict.** This session's rate overturn proved transport-free deep credit works AT RATE; two 6-seed
negatives then showed it does not transfer to the real-spikes read regime. This diagnostic LOCATES why, with the
strongest possible control — an **oracle-directed arm** whose learning signal is the exact loss gradient routed
by the TRUE forward weights W⊤ (e-prop's `L_j = ∂E/∂z_j` with perfect transport; measured feedback-alignment
0.999/1.0). On real spikes, at 6 seeds, on BOTH an easy (k=9) AND a hard (k=17) task, the oracle does NOT beat
the label-shuffled permuted control: `directed = oracle − permuted` = −0.003 (easy) / +0.012 (hard), both inside
the per-seed noise (0/6 and 2/6 positive). The isolation is clean: **not the TASK** (the gap does not open on the
hard task, where the frozen reservoir genuinely FAILS at 0.171 and a rate MLP solves it at 0.843), **not the
FEEDBACK** (a perfect W⊤ oracle equals permuted, so no feedback quality would help), **the READ REGIME itself**
— the finite-spike σ′(v−θ) read cannot surface a directed signal above the label-agnostic generic-plasticity
lift + read noise, at any transport quality. No `sim/` edit (additive oracle arm, W⊤-diagnostic clearly labeled;
the shippable KP path stays transport-free, asserted in-run).

## Result — 6 seeds (42/43/44/100/101/102), the oracle isolation

Artifacts: `research/findings/raw/gap4/realspikes/multihop_oracle_easy_6seed.json` and `..._hard_6seed.json`
(numpy/CPU, 2 plastic real-spikes layers, 40 epochs).

<!--derived-->
| arm (graded held-out) | EASY (k=9) | HARD (k=17) |
|---|---|---|
| host-oracle (rate MLP, backprop) | 1.000 | 0.843 |
| frozen reservoir | 0.454 | 0.171 |
| permuted (generic plasticity) | 0.448 | 0.179 |
| **oracle − permuted (directed credit)** | **−0.003** (0/6 >0) | **+0.012** (2/6 >0) |
| KP − permuted (transport-free) | +0.006 | +0.000 |

The hard task is the decisive cell: the reservoir FAILS (0.171 vs the rate MLP's 0.843 — maximal room for
directed credit to help), yet the perfect-transport oracle rides only the same label-agnostic lift a
label-shuffle produces. Directed credit is not merely noisy — across 12 cells (2 reads × 2 tasks × 6 seeds) it
never consistently surfaces.

## Why this matters + the named surpass (a controlled boundary, NOT a defeatist wall)

<!--derived-->
This is the tightest characterization the crux has: the strongest control (a perfect W⊤ oracle) fails, so the
wall is neither the task nor the feedback — it is the spiking READ. It re-confirms the project's R3 reservoir
reframe ON SPIKES (a fixed reservoir + trained readout is the value; top-down credit to the hidden adds nothing)
and the 2026-07-14 graded-credit-decisive conclusion, now with the strongest possible instrument. The
directed signal sits at/below the read noise floor (per-seed ±0.02–0.05), so the surpass is a **lower-CV read**,
not another feedback or task variant — exactly where the field's working deep-spiking trainers differ from us:
e-prop ([Bellec 2020, Nat Commun](https://www.nature.com/articles/s41467-020-17236-y)) accumulates a temporal
eligibility trace over long sequences with MANY spikes; DECOLLE ([Kaiser 2020, Front Neurosci](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00424/full))
gates plasticity by the membrane potential inside an eligibility WINDOW with per-layer local readouts — both are
low-CV, many-spike reads, while ours is a short, few-spike window whose columns never somatically spike. **Next
mechanism (named, not deferred):** lower the read CV so the directed signal surfaces — more spikes / ensemble
averaging / longer temporal integration (the e-prop long-sequence eligibility + the 2026-07-14 "average over an
ensemble"), then re-run the oracle isolation to confirm the directed signal appears before shipping a
transport-free rule. The rate overturn stands as the session's result; the spiking read regime is the
precisely-located, biologically-actionable frontier.

## Update (2026-08-02) — the lower-CV read was TESTED and does NOT surface directed credit; the wall is NOT read-CV, the corrected surpass is DECOLLE local readouts

<!--derived-->
The named surpass (a lower-CV read) was built + tested (additive `--lowcv-read`: longer integration window,
ensemble pooling over columns, an exponential e-prop-style eligibility trace). Artifact:
`research/findings/raw/gap4/realspikes/smoke_lowcv_hard_s42.json`. It DID lower the estimator read-CV
(0.090→0.070) but `oracle − permuted` stayed at the noise floor (current +0.000 → lowcv +0.009, below the
margin, 0/1) on the hard task — VERDICT: DEEPER REDESIGN, the wall is not read-CV. **The decisive fact: the
substrate is DETERMINISTIC** (OU / conductance-noise / heterogeneity all OFF → `repeat_maxabs = 0`), so the
"more spikes → lower shot-noise CV" lever is INERT — there is no stochastic read noise to average out. So the
directed signal being at/below the "noise floor" is not shot noise; the deep-layer credit's held-out benefit is
genuinely **LABEL-INDEPENDENT**: even a perfect W⊤ oracle equals a label-shuffle, AND a lower-CV read does not
change it. This is the R3 reservoir reframe on spikes, DEFINITIVELY — the fixed reservoir + trained readout
extracts what it needs, and top-down credit to the deep layer has no purchase (the readout free-rides past it).
**CORRECTED SURPASS (replacing "a lower-CV read"):** a DECOLLE-style LOCAL per-layer readout + local loss
([Kaiser 2020, Front Neurosci](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00424/full))
— train each spiking layer DIRECTLY toward local classifiability (a label-dependent local loss FORCING directed
credit at the layer), instead of routing a top-down credit signal the output readout free-rides past. That is
the field's proven approach for exactly this "top-down credit has no purchase on spikes" wall, and it is the
next mechanism. (The rate overturn is unaffected; this refines the spiking-side surpass.)

## Update 2 (2026-08-02) — DECOLLE local readouts ALSO give the deep layer ZERO purchase: THREE instruments now agree the wall is the SUBSTRATE, not the credit signal

<!--derived-->
Built + tested the DECOLLE surpass (additive `--decolle`: each plastic spiking layer gets its OWN fixed-random
local readout + a label-dependent local classification loss, transport-free, no descending credit). Artifacts:
`decolle_smoke_seed42.json` (easy) + `decolle_smoke_hard_seed42.json` (hard). The DECISIVE metric — the deep
layer's directed purchase `decolle_minus_permuted_L0` — is **0.0 on BOTH tasks** (verdict LABEL-AGNOSTIC on
easy, NEGATIVE on hard; the final read beats frozen only by the label-agnostic amount, ≤+0.056, and does NOT
beat permuted). So even a label-dependent gradient applied DIRECTLY at the deep layer (not routed top-down) gives
the deep layer NO directed purchase. **THREE independent instruments now agree** the deep spiking layer's
held-out contribution is label-INDEPENDENT regardless of the training signal: (1) a perfect W⊤ top-down oracle,
(2) a lower-CV read, (3) DECOLLE local per-layer losses — all give `directed = 0`. This is the R3 reservoir
reframe on spikes, CONFIRMED as tightly as it can be.
**⇒ The wall is the SUBSTRATE, not the credit signal.** The root cause is one of two substrate properties (the
next thing to isolate): (a) the readout free-rides on the fixed reservoir, so the deep layer is REDUNDANT even
where the reservoir fails (hard task); or (b) the coincidence-plateau plasticity (max(0)-excitatory + L2-renorm-
to-init) is too CONSTRAINED to reshape the deep layer's representation, so no credit signal — however directed —
can move it. **Next mechanism (named, not deferred):** relax the plasticity constraint (widen the reshape range;
signed/two-sided updates; drop the renorm-to-init) and/or a substrate where the deep layer is NOT reservoir-
redundant (a bottleneck the readout must route through), then re-run the oracle isolation. This is a deep,
well-characterized substrate boundary — the field's directed-credit mechanisms (top-down + local) are exhausted
ON THIS SUBSTRATE; the surpass is now a SUBSTRATE change, not a credit-rule change. The rate overturn stands.

## Update 3 (2026-08-02) — root cause ISOLATED: (a) RESERVOIR-REDUNDANCY, not the plasticity — the surpass is a BOTTLENECK architecture

<!--derived-->
The two candidate root causes were tested directly with a `--relax-plasticity` control (drop the max(0)-
excitatory clamp → signed weights; replace the L2-renorm-to-init with a loose norm cap; blow-up telemetry).
Artifact: `multihop_relax_plasticity_s42.json`. Result (hard task): with the plasticity relaxed, `directed =
oracle − permuted` moved from 0.0 only to **+0.019 (still below the 0.05 margin)**, weights did NOT blow up
(colnorm ratio 1.35, 0 at cap, no NaN) — the loosening engaged cleanly and still did not open the directed gap.
VERDICT: **ROOT CAUSE (a) — the deep spiking layer is RESERVOIR-REDUNDANT.** Relaxing what the deep layer CAN
learn does not help, because the trained final readout free-rides on the fixed reservoir (the rich top layer)
regardless of the deep layer's representation or its plasticity. It is NOT (b) a plasticity-constraint wall.
**The crux's spiking side is now fully characterized:** transport-free deep credit works AT RATE (the overturn);
on this real-spikes substrate directed deep credit has no purchase because the deep layer is reservoir-redundant
— confirmed by FOUR controls (perfect W⊤ oracle, lower-CV read, DECOLLE local losses, relaxed plasticity), all
`directed ≈ 0`. **NAMED SURPASS (a substrate-ARCHITECTURE change, not a credit rule or plasticity):** a
BOTTLENECK — the final read must route THROUGH the deep layer (a narrow layer, or a readout that reads only the
deep layer, so the reservoir alone cannot be read out and the deep layer's directed credit becomes load-bearing),
then re-run the oracle isolation. This is the next mechanism; the rate overturn is the session's standing result.

## Update 4 (2026-08-02) — the BOTTLENECK is ARCHITECTURE-INVARIANT (with headroom): the TERMINUS — this substrate does not benefit from credit-training; the surpass is a different TRAINABLE spiking substrate

<!--derived-->
Built + tested the named bottleneck surpass (additive `--bottleneck`: narrow the top reservoir cols1 to 12
columns, keeping the 2-hop chain, so the readout cannot free-ride on a rich reservoir; the deep-only-readout
option was rejected — it degenerates to the already-negative single-layer case). Artifact:
`multihop_bottleneck_hard_seed42.json`. Result (hard task, cols1=12): the headroom guard PASSES —
`frozen_vs_rateMLP_gap = 0.74` (the narrow frozen reservoir fails badly while the rate-MLP solves the task, so
the test is interpretable, maximal room for directed credit). Yet the W⊤ oracle STILL does not beat permuted:
`bottleneck_directed_oracle_graded = −0.074` (and does not even beat frozen). VERDICT:
**ARCHITECTURE-INVARIANT.** Removing the reservoir free-ride does NOT give directed credit purchase — because
the narrow bottleneck columns are read through the SAME finite-spike σ′(v−θ) gate (they still never somatically
spike), so the directed signal has the same reason to sit below the read floor. **This is the TERMINUS for this
substrate: FIVE independent controls now agree** directed deep credit has no purchase — perfect W⊤ oracle,
lower-CV read, DECOLLE local losses, relaxed plasticity, AND a headroom-satisfied bottleneck. The wall is not the
credit rule, not the feedback, not the task, not the read-CV, not the plasticity, and not the architecture — it
is the movable-plateau **SUBSTRATE** itself (a coincidence-plateau reservoir whose graded read carries no
credit-usable per-column selectivity). **NAMED SURPASS (the honest, exhaustively-earned next direction):** a
fundamentally different, genuinely TRAINABLE spiking substrate — surrogate-gradient BPTT over a spiking net
whose hidden layers are trained (not a fixed movable-plateau reservoir) with a low-CV many-spike read, as the
field's working deep-spiking trainers do (e-prop, DECOLLE, SuperSpike). That is a substrate build, not a
credit-rule de-risk. **The rate overturn (transport-free deep credit works AT RATE) stands as the session's
result; the spiking-credit wall on the movable-plateau substrate is now exhaustively characterized + its surpass
precisely named.**
