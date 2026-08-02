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
**⚠️ REFUTED SAME-CYCLE by direct measurement — see "Update 1" below. The hypothesis in this section (credit-factor
variance / reset jitter → dendritic plateau averaging) is WRONG: the per-example credit is measured to be CONSISTENT
(low within-seed variance), not noisy. The measured FA-convergence result above STANDS; only this speculative cause is
retracted. The section is kept as the honest hypothesis-that-was-tested; the corrected residual is in Update 1.**

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

## Update 1 (2026-08-02, same-cycle) — the VARIANCE hypothesis is REFUTED; the credit is consistent-but-MISALIGNED, and W anti-rotates

<!--derived-->
Ran the credit-factor probe (`--measure-credit-factor`, 6 seeds, fd_batch 64, same act-th-3 XOR expander codon at the
FA-convergence operating point; artifact `research/findings/raw/gap4/cf_variance_izh_s42.json`) which reports the
per-example alignment of the on-bridge DFA credit factor with the finite-difference backprop oracle, its MEAN and its STD
across examples. The variance hypothesis predicted a HIGH per-example std (noisy credit). **The measurement REFUTES it:
the within-seed cos(credit, oracle) STD is TINY (0.002-0.047 across the 6 seeds), SNR |mean|/std = 0.4-40 — the
per-example credit direction is CONSISTENT, not noisy.** What is variable is the ACROSS-SEED alignment MEAN
(-0.509, -0.169, -0.066, +0.017, +0.052, +0.158; mean -0.086, sign-random) — i.e. per seed the credit points a
well-defined but MIS-aligned (often anti-aligned) direction, set by the random feedback B relative to the init forward W,
and it does not improve. So temporal averaging / a dendritic plateau (which reduces per-example jitter) has NO jitter to
average away — the named surpass in the section above does not address the measured defect.

<!--derived-->
Two further reads triangulate the corrected residual. **(a) The surrogate psi is EXONERATED:** the credit WITHOUT psi
(`cos_lsig_vs_oracle` = delta@B alone) is ALSO misaligned (|cos_lsig| mean 0.117 ~= |cos_bridge| 0.162 across seeds;
`surrogate_degrades_alignment` true on only 2/6) — so the misalignment lives in the DFA signal delta@B itself, not in the
Izhikevich membrane surrogate. **(b) W MOVES but the WRONG way:** the headline FA-convergence table shows 4/6 Izhikevich
seeds go cos(W,B) NEGATIVE over training (anti-align), so it is not that W fails to move (weak learning) — the FA update
actively ANTI-ROTATES W on Izhikevich, the opposite of the LIF rotation. **Corrected residual: the FA weight-update DIRECTION
is mis-directed on the Izhikevich forward — a structural property of its credit dynamics, not noise (refuted here), not the
surrogate (exonerated here), not weak learning (W moves).** An interventional settle-steps sweep (more temporal integration
per example) is running as the direct confirmation that averaging does not help; a corrected next mechanism is a LEARNED
feedback (Kolen-Pollack / weight-mirror rotates B toward W instead of relying on W rotating to a fixed B) re-tested at THIS
operating point, and/or a two-compartment dendritic credit with a different FA fixed-point structure. This Update is the
workflow working as intended: a hypothesis was named, DIRECTLY MEASURED, and refuted same-cycle before it could propagate.

## Update 2 (2026-08-02, same-cycle) — the two corrected next-mechanism tests BOTH come back negative: averaging does not help, and LEARNED feedback (KP) does not restore convergence either — the failure is agnostic to feedback type

<!--derived-->
**Interventional averaging test (settle-steps sweep).** Re-ran the Izhikevich FA-convergence at more temporal
integration per example (`--settle-steps` 100 vs the default 30; artifacts `research/findings/raw/gap4/fa_conv_izh_settle100_s42.json`).
More averaging does NOT restore convergence at ANY dose: settle=100 rises +0.097 / -0.015 / +0.079 (mean +0.054, 0/3) and
settle=300 (10x the baseline) rises -0.105 / +0.125 / +0.109 (mean +0.043, 0/3), both vs the settle=30 baseline (mean
-0.058, 0/6) — 0/12 converge across settle in {30, 100, 300}. This is the direct interventional confirmation of Update 1:
since the per-example credit was measured CONSISTENT (not noisy), adding averaging has nothing to remove and does not
help — doubly refuting the variance hypothesis.

<!--derived-->
**Learned-feedback test (Kolen-Pollack), at the MATCHED headline operating point.** A 10-epoch smoke initially showed a
positive cos rise (+0.237) and I nearly banked "learned feedback fixes the alignment" — but VERIFYING at 3 seeds and the
FULL 60-epoch / 160-subsample operating point that matches the headline REFUTED it: KP FA-convergence is 0/3 (peaks -0.298,
+0.167, +0.094; finals -0.106, -0.004, +0.092), and inherit stays at chance (0.451-0.476). The smoke's positive was a
transient at a smaller op-point that does not survive the matched config — caught only because a surprising single-seed
smoke was re-run at 3 seeds + full epochs before being claimed. **FIRMED to 6 seeds: KP FA-convergence is 0/6** (rises
-0.139, +0.040, -0.232, -0.106, -0.004, +0.092; artifacts `research/findings/raw/gap4/kp_faconv_izh_s42.json`), exactly
matching the fixed-B headline (0/6). So **learned feedback does NOT restore FA-convergence on Izhikevich either** — the
alignment failure is AGNOSTIC to whether the feedback matrix is fixed-random (headline) or learned toward W (KP). Both
feedback-alignment routes fail on the Izhikevich forward.

<!--derived-->
**What this leaves, and the unification.** Two of the three named surpasses are now closed by measurement (plateau
averaging: refuted; learned feedback: refuted), leaving the **two-compartment dendritic credit** — a fundamentally
DIFFERENT credit computation (apical-basal segregation, not delta@B feedback-alignment) with a different fixed-point
structure — as the standing candidate. And the result UNIFIES with the project's long-standing reservoir reframe: the
Izhikevich forward's credit-dynamics do not support feedback-alignment credit regardless of feedback type, and separately
its representational ceiling caps a trained readout below the oracle — so on this substrate the accuracy value is a trained
read-out over a fixed/reservoir hidden, not feedback-alignment credit-training of the hidden. gap#4's production-bridge
residual is now characterized to the mechanism (FA-alignment fails on the Izhikevich forward, both feedback types, not
noise/surrogate/averaging), with the one remaining biological surpass (dendritic two-compartment credit) named — a
deprioritized parallel frontier per the 2026-07-10 steer, with the crux CORE (LIF) standing.
