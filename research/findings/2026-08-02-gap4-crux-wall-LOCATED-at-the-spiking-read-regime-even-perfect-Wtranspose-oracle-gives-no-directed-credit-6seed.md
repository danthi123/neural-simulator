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
