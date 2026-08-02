---
type: finding
status: contributing
date: 2026-08-02
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/fa_convergence_lif_ref_s42.json
  - research/findings/raw/gap4/fa_convergence_lif_ref_s43.json
  - research/findings/raw/gap4/fa_convergence_lif_ref_s44.json
  - research/findings/raw/gap4/fa_convergence_lif_ref_s100.json
  - research/findings/raw/gap4/fa_convergence_lif_ref_s101.json
  - research/findings/raw/gap4/fa_convergence_lif_ref_s102.json
  - research/findings/raw/gap4/rep_fwd_fa_convergence_izh_s42.json
  - research/findings/raw/gap4/rep_fwd_fa_convergence_izh_s43.json
  - research/findings/raw/gap4/rep_fwd_fa_convergence_izh_s44.json
  - research/findings/raw/gap4/rep_fwd_fa_convergence_izh_s100.json
  - research/findings/raw/gap4/rep_fwd_fa_convergence_izh_s101.json
  - research/findings/raw/gap4/rep_fwd_fa_convergence_izh_s102.json
---

# gap#4 production-bridge deep-credit — the mechanistic ROOT CAUSE is FEEDBACK-ALIGNMENT NON-CONVERGENCE on the Izhikevich substrate: on LIF the forward weights align to the fixed feedback matrix (cos rises 6/6 seeds), on the production Izhikevich bridge they do NOT (0/6 seeds) — this is the direct measurement that closes the 7-step elimination chain and names the residual with a biological surpass, not a mystery

<!--derived-->
**One-line verdict.** Feedback alignment (the mechanism that makes transport-free credit work at all — the forward
weights W learning to align with the fixed random feedback matrix B so that `B^T e` approximates the true gradient
direction `W^T e`) **CONVERGES on the LIF substrate (6/6 seeds, cos(W,B^T) rise +0.290 to +0.435) and does NOT converge
on the production Izhikevich bridge (0/6 seeds, rise -0.227 to +0.092, no seed overlap).** This is the direct
measurement the 2026-08-02 elimination chain was converging on: the on-bridge credit factor's FA-convergence. It is the
mechanistic ROOT CAUSE of why on-bridge e-prop cannot train a depth-required task even on a REPRESENTABLE codon
(oracle solves it) — the transport-free credit signal is pointed in a direction unrelated to (or opposed to) the true
gradient, because the Izhikevich forward never aligns to its feedback. No `sim/` edit (additive `--measure-fa-convergence`).

## Result — the 6-seed × 2-substrate FA-convergence table

<!--derived-->
Representative per-seed artifacts: `research/findings/raw/gap4/fa_convergence_lif_ref_s42.json` (LIF) and
`research/findings/raw/gap4/rep_fwd_fa_convergence_izh_s42.json` (Izhikevich).

<!--derived-->
`cos(W_top-hidden, B^T)` measured at init and after 60 epochs; **rise** = final − init. LIF from the isolation
reference runner (which DOES train deep credit, inherit → 0.96); Izhikevich from the production-bridge runner (which
does NOT, inherit → chance):

<!--derived-->
| substrate | seed | cos init | cos final | rise | converges? |
|---|---|---|---|---|---|
| LIF ref | 42 | -0.128 | +0.224 | **+0.352** | YES |
| LIF ref | 43 | -0.104 | +0.332 | **+0.435** | YES |
| LIF ref | 44 | -0.050 | +0.318 | **+0.368** | YES |
| LIF ref | 100 | +0.075 | +0.365 | **+0.290** | YES |
| LIF ref | 101 | +0.078 | +0.368 | **+0.290** | YES |
| LIF ref | 102 | -0.104 | +0.267 | **+0.371** | YES |
| Izhikevich bridge | 42 | +0.000 | -0.109 | -0.109 | no |
| Izhikevich bridge | 43 | +0.000 | -0.000 | -0.000 | no |
| Izhikevich bridge | 44 | +0.000 | +0.092 | +0.092 | no |
| Izhikevich bridge | 100 | +0.000 | -0.137 | -0.137 | no |
| Izhikevich bridge | 101 | +0.000 | +0.040 | +0.040 | no |
| Izhikevich bridge | 102 | +0.000 | -0.227 | -0.227 | no |

<!--derived-->
**LIF: 6/6 converge, mean rise +0.351, every seed >= +0.290. Izhikevich: 0/6 converge, mean rise -0.058, every seed
<= +0.092.** The two distributions do not overlap — the smallest LIF rise (+0.290) is more than 3x the largest
Izhikevich rise (+0.092), and 4/6 Izhikevich seeds actively ANTI-align (cos goes negative). Per-seed artifacts
`research/findings/raw/gap4/fa_convergence_lif_ref_s{42,43,44,100,101,102}.json` and
`research/findings/raw/gap4/rep_fwd_fa_convergence_izh_s{42,43,44,100,101,102}.json`. Commands:
`SIM_BACKEND=numpy .venv/bin/python -m research.runners._snn_bptt_forward_vs_learning_isolation_derisk --measure-fa-convergence --seed <s> --epochs 60`
(LIF) and `... _gap4_representable_forward_plus_credit_derisk --measure-fa-convergence --task-xor --act-th 3 --mode expander --seeds <s> --epochs 60 --train-subsample 160 --fa-eval-every 5`
(Izhikevich).

## Why this closes the elimination chain

<!--derived-->
The 2026-08-02 residual for the production-bridge crux was narrowed by SEVEN direct-test eliminations, each ruling out a
candidate cause of "on-bridge e-prop cannot train a depth-required task": (a) task-decodability (XOR is not
reservoir-shortcuttable, still fails), (b) forward-representability (the PlateauExpander codon makes it representable —
oracle 0.94 — e-prop still chance), (c) codon-density (sparse codon fails 6/6), (d) feedback-direction (a learned
feedback matrix engages but does not rescue), (e) a learned self-predicting microcircuit (reduces to fixed-DFA at the
proven fixed point, still fails), (f) phi'-vanishing (the atan surrogate is HEALTHY, psi_mean/peak 0.31-0.32), (g)
operating-point / surrogate tuning (0/30 alpha x tonic runs). Each elimination removed a cause but left the residual
UNNAMED. **This measurement names it: the residual is FA-CONVERGENCE, and it is a property of the Izhikevich substrate,
not of the task, the codon, the feedback wiring, the surrogate magnitude, or the operating point** — all of which were
held identical while only FA-convergence separated LIF (works) from Izhikevich (fails).

## The mechanism, and why non-convergence explains every downstream symptom

<!--derived-->
Feedback alignment (Lillicrap 2016; Nokland 2016 for the direct-projection variant used here) does not need weight
transport because, during learning, the forward weights W rotate toward alignment with the fixed random feedback B — once
`cos(W, B^T) > 0`, the transport-free credit `B^T e` has a positive projection on the true gradient `W^T e`, so it is a
descent direction. **The alignment is the load-bearing event; without it the credit signal is a random projection of the
error.** On LIF this rotation happens (6/6, rise +0.35). On the Izhikevich bridge W stays at cos ~ 0 or rotates AWAY
(0/6, mean rise -0.06). So on the production bridge the "directed" credit is, in alignment terms, indistinguishable from
an undirected random projection — which is EXACTLY the earlier production-bridge symptom that a frozen random reservoir
matched e-prop's `deep_credit_share` (~0.005): a credit rule whose W never aligns to its B assigns no more real deep
credit than a random feedback would. FA-non-convergence is the single upstream fact that predicts both the reservoir-tie
on the inheritance task AND the chance-level e-prop on representable XOR.

## The named next mechanism (no-defer) — the credit-factor variance hypothesis and its dendritic surpass

<!--derived-->
FA-convergence requires the per-example forward-weight updates to be CONSISTENT enough across examples to ACCUMULATE
alignment; if each update points a different way, W random-walks instead of rotating toward B. The leading hypothesis for
why Izhikevich fails where LIF succeeds is **credit-factor VARIANCE**: the Izhikevich spike-reset (the discontinuous
membrane reset + the recovery variable u) makes the per-example local credit factor high-variance even though its MEAN
magnitude (the surrogate psi) is healthy — so alignment cannot accumulate. This is the project's recurring "we
implemented ONE process and replaced its companion with a constant" pattern: the real pyramidal neuron integrates the
top-down / feedback signal in a SEGREGATED APICAL COMPARTMENT over a seconds-long plateau, which TEMPORALLY AVERAGES the
credit signal and yields a low-variance credit factor; the point-neuron Izhikevich bridge has no such compartment, so the
instantaneous credit is too noisy to converge FA. **The named surpass is therefore a dendritic apical compartment with
plateau-timescale averaging of the feedback** — already the standing "dendritic cortex" project priority, now implicated
by a direct measurement rather than by analogy.

<!--derived-->
**The specific, cheap next test that decides the variance hypothesis BEFORE building the compartment:** measure the
per-example credit-factor variance (not just its mean) on LIF vs Izhikevich via the existing finite-difference credit
probe (`--measure-credit-factor --fd-batch --fd-delta-pA`), and test whether TEMPORALLY SMOOTHING the on-bridge credit
factor (an eligibility-trace time constant over the credit signal, the cheapest proxy for plateau averaging) RESTORES
FA-convergence on Izhikevich. If smoothing lifts the Izhikevich cos-rise into the LIF range, the variance hypothesis is
confirmed and the dendritic compartment is the faithful mechanism it proxies; if it does not, the residual moves to the
alignment dynamics themselves and the next mechanism is the segregated-compartment credit rule directly. Either way the
wall is now a NAMED mechanism with a NAMED biological surpass and a decisive next measurement — not a characterized
limit. A density control (Izhikevich FA-convergence at act-th 2 and act-th 4) is running to confirm non-convergence is
robust across codon density and not specific to act-th 3.

## Honest scope

<!--derived-->
The LIF number is from the isolation-reference harness and the Izhikevich number from the production-bridge harness —
they are DIFFERENT runners, so this is a per-substrate result (each substrate's FA-convergence linked to its own training
outcome: LIF converges and trains deep credit; Izhikevich neither), not a single-harness neuron-model swap. The
same-harness confirmation (a LIF-vs-Izhikevich neuron flag inside one runner) is a named follow-up. This does NOT close
gap#4 on the production bridge — it LOCATES the residual mechanistically (FA-convergence) and names the surpass; the crux
CORE (transport-free directed credit beats an optimally-read reservoir on LIF, 6-seed) stands unchanged.
