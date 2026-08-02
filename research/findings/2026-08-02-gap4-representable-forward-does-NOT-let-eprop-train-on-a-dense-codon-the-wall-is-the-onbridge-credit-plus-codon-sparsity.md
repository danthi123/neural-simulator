---
type: finding
status: contributing
date: 2026-08-02
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/rep_fwd_credit_xor_smoke_s42.json
  - research/findings/raw/gap4/rep_fwd_credit_xor_smoke_onbits_s42.json
---

# gap#4 — a REPRESENTABLE forward (PlateauExpander) does NOT by itself let on-bridge e-prop train XOR: the oracle solves the codon but e-prop stays at chance on the DENSE codon (all encodings tried) — REVISING "the wall is the forward": the wall is the on-bridge e-prop credit's weight-finding on a dense codon, and a genuinely SPARSE representable codon is the untested residual

<!--derived-->
**One-line verdict.** The roadmap's named highest-value lever: credit ON TOP OF a REPRESENTABLE forward (the
`PlateauExpander`), to close the production-bridge deep-credit residual. Ran it on XOR (`--task-xor`): the PlateauExpander
codon MAKES XOR representable — a backprop oracle on the codon reaches 0.994 (literal encoding) / 0.877 (onbits) — but the
on-bridge e-prop **still does not train** (eprop 0.50/0.48 ≈ chance 0.55/0.55; trains_the_task=False; deep_credit_share
degenerate/nan because eprop ≈ frozen ≈ chance). The codon is DENSE (`codon_sparsity` = 0.499 = ~50% columns active) under
BOTH input encodings — the encoding lever does not sparsify it. **This REVISES this session's earlier "the wall is the
Izhikevich FORWARD, not the credit rule" claim**: even a REPRESENTABLE forward (oracle solves it) does not let on-bridge
e-prop train XOR, so the wall is the on-bridge e-prop CREDIT's weight-finding on a DENSE codon — not forward-representability
alone. No `sim/` edit (additive `--task-xor` on `_gap4_representable_forward_plus_credit_derisk.py`).

## Result — 1-seed smokes, XOR, PlateauExpander (representable) forward + e-prop credit

<!--derived-->
| encoding | oracle (on codon) | eprop_inherit | frozen_hidden | trains_the_task | codon_sparsity |
|---|---|---|---|---|---|
| literal (default) | 0.994 | 0.501 | 0.487 | False | 0.499 |
| onbits | 0.877 | 0.479 | 0.454 | False | 0.499 |

chance ~0.55. Artifact e.g. `research/findings/raw/gap4/rep_fwd_credit_xor_smoke_s42.json`. Command:
`SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap4_representable_forward_plus_credit_derisk --task-xor --mode expander`.
The two encodings differ in whether the codon fully linearizes XOR (oracle 0.994 vs 0.877) but give the SAME dense codon
(0.499) and the SAME e-prop failure — so the input-encoding lever is not the sparsity lever.

## The decisive read + what it revises

<!--derived-->
The oracle column is load-bearing: a backprop oracle ON THE CODON solves XOR (0.88-0.99), so the codon IS a
representable forward (the PlateauExpander did its job — it makes the level-2 XOR computation representable). Yet the
on-bridge e-prop, on that same representable codon, sits at chance. Two things are therefore established: (1) the
production-bridge wall is NOT merely that the raw Izhikevich forward can't represent XOR — a representable forward exists
and the oracle uses it; (2) but a representable forward is NOT SUFFICIENT for on-bridge e-prop to train — the on-bridge
e-prop credit cannot find the weights the oracle finds, ON A DENSE codon (0.499). **⇒ the wall is the on-bridge e-prop
credit's weight-finding on a dense representable code, which supersedes the earlier "the wall is the forward" framing
(that framing was correct that the LIF forward works and the raw Izhikevich forward doesn't, but INCOMPLETE — a
representable Izhikevich forward still fails).**

## Honest scope + the named residual (the sparsity lever is not yet exposed)

<!--derived-->
**The confound / residual:** the codon is DENSE (0.499). The PlateauExpander probe's own GO condition depends on codon
sparsity, and dense codes are known to block local surrogate-credit training (the agent flagged this a-priori). The
input-ENCODING lever (literal/onbits) does NOT change the codon sparsity — sparsity is set by the coincidence threshold
`ACT_TH` (=2) and `SAMP` (=3) in `_gap4_plateau_expander_probe.py`, which are NOT CLI-exposed. So the clean test — does
a genuinely SPARSE representable codon let on-bridge e-prop train — is NOT YET RUN; it needs a code change to expose
`ACT_TH` (raise it -> sparser codon) and confirm the codon stays representable (oracle high) at the sparser setting.
**NEXT (no-defer, the clean next-session build):** expose `ACT_TH`/`SAMP` as flags, sweep to a sparse-BUT-representable
codon (oracle high, sparsity < ~0.15), and re-read deep_credit_share. If e-prop then trains -> the wall was the dense
code (fixable) and the representable-forward lever works; if it still fails on a sparse representable codon -> the wall
is the on-bridge e-prop credit rule ITSELF on the Izhikevich substrate (the deepest residual), pointing to the
learned-instructive-signal / operating-point levers the roadmap tracks. The crux CORE (LIF/rate) is untouched; this
precisely narrows the PRODUCTION-bridge residual from "the forward" to "on-bridge e-prop weight-finding on a dense
representable code", with the sparse-codon test as the decisive next step.

## Update (2026-08-02) — the sparse-codon test RAN and RESOLVES the fork: the wall is the on-bridge e-prop CREDIT RULE ITSELF, not the codon density

<!--derived-->
Exposed the codon sparsity lever (`--act-th`, sets the probe's `ACT_TH` before PlateauExpander construction; additive,
default-preserving) and ran act_th=3 (a column fires only if all 3 sampled features are active), **6 seeds**. Artifact
e.g. `research/findings/raw/gap4/rep_fwd_xor_actth3_s42.json`. Result (6/6): the codon is now SPARSE (codon_sparsity
mean ~0.11, down from 0.499) AND STILL REPRESENTABLE (oracle mean 0.940 solves XOR on the sparse codon, all 6) — but
on-bridge e-prop STILL does not train (eprop mean 0.484 = chance 0.524; trains_the_task 0/6). ⇒ **sparsifying the
representable codon did NOT help, 6/6.** So the production-bridge deep-credit residual is NOT the dense code, NOT forward-representability, NOT
task-decodability — it is the **on-bridge e-prop CREDIT RULE's weight-finding on the Izhikevich substrate itself**: the
local biological credit rule cannot find the weights the backprop oracle finds, even on a sparse, representable code.
This is the deepest, precisely-located residual, and it matches the roadmap's own standing gap#4 diagnosis (the learned
self-predicting microcircuit / the learned instructive signal §2.8 "the true crux" / operating-point / phi'-vanishing).

<!--derived-->
**The clean summary of the whole production-bridge arc (this session):** the transport-free credit RULE works at RATE
and on the LIF net (DFA depth-robust, beats reservoir on XOR); on the production IZHIKEVICH bridge it does NOT train XOR
regardless of the forward — raw forward (deep_share ~0), dense representable codon (oracle 0.99, eprop chance), OR sparse
representable codon (oracle 0.95, sparsity 0.11, eprop chance, 6/6). The single remaining wall is the on-bridge e-prop
credit rule's weight-finding on the Izhikevich substrate.

## Update 2 (2026-08-02) — LEARNED FEEDBACK (KP) also does NOT rescue it: verified to ENGAGE but leaves e-prop at chance — so the wall is NOT feedback DIRECTION, it is the surrogate/eligibility HIDDEN-WEIGHT-FINDING on Izhikevich

<!--derived-->
Tested the roadmap's named fix — LEARN the DFA feedback via Kolen-Pollack (`--learned-feedback`: B_direct updated to
track W^T in direction, transport-free) — on the sparse representable codon (act_th=3), the config where fixed DFA gave
chance. Artifact: `research/findings/raw/gap4/rep_fwd_credit_xor_kp_smoke_s42.json`. Result: e-prop STILL at chance
(eprop 0.451 = frozen 0.532-ish ~ chance; trains_the_task=False), IDENTICAL across a kp-lr sweep (0.1 / 0.5 / 2.0 all
give eprop 0.451). **VERIFIED that KP is NOT a no-op** (silent-failure discipline: "inert" is a hypothesis, checked):
the KP arm's `eprop_ff_weight_moved` differs from fixed-DFA (1124844.7 vs 1124874.3), and its `permuted`/`shuffle_dfa`
controls differ — so B_direct genuinely moves and the credit route changes; the forward weights move DIFFERENTLY. Yet
the held-out classification is EXACTLY the same (both 162/359 correct = chance). ⇒ learned feedback ENGAGES but does NOT
rescue: the on-bridge e-prop stays at chance regardless of feedback DIRECTION or rate.

<!--derived-->
**⇒ THE DEFINITIVE, ELIMINATIVE CONCLUSION of the production-bridge arc.** The on-bridge deep-credit wall is NOT: task
decodability (XOR ~0), forward representability (representable codon fails), codon density (sparse representable codon
fails 6/6), OR feedback direction/alignment (learned KP feedback engages but does not rescue). Every candidate that a
FEEDBACK-ROUTING or FORWARD-REPRESENTATION fix could address is eliminated. What remains is the on-bridge e-prop's
LOCAL credit factor itself — the surrogate/eligibility HIDDEN-WEIGHT-FINDING on the Izhikevich substrate: the local rule
cannot move the hidden weights toward the solution the backprop oracle finds, no matter how the error is routed to them.
**NEXT (roadmap's §2.8 "the true crux"): a LEARNED SELF-PREDICTING MICROCIRCUIT (Sacramento Eq.9) that shapes the LOCAL
credit factor** (the LIF-rate version was already 6-seed GO, 2026-07-24; the on-bridge port is the open build) — OR a
fundamentally stronger on-bridge surrogate/eligibility / an operating-point fix (phi'-vanishing). NOT more forward,
codon, or feedback-routing tuning — those are now exhaustively eliminated. The crux CORE (LIF/rate) is untouched.

## Update 3 (2026-08-02) — the LEARNED SELF-PREDICTING MICROCIRCUIT (§2.8 "the true crux") ALSO does not rescue it; the fixed-point analysis PROVES the wall is the LOCAL CREDIT FACTOR (surrogate/eligibility), not any error-routing

<!--derived-->
Built + ran the roadmap's named §2.8 fix — the on-bridge Sacramento self-predicting microcircuit (`--microcircuit`,
`MicrocircuitEpropNet`: a plastic interneuron `W_PI` learns Eq.9 to predict/cancel the top-down; the hidden local credit
is the apical residual `src_pred@W_PI − onehot@B_direct`), on the sparse representable codon (act_th=3), seed 42, with a
wpi_lr sweep + a frozen control. Artifacts `rep_fwd_credit_xor_micro_*.json`. **It does NOT rescue, and the mechanism is
FULLY diagnosed:** wpi_lr=1.0 -> selfpred_cos 0.999 (interneuron fully learns W_PI==B_direct) -> eprop 0.451 = EXACTLY
fixed-DFA (below chance 0.549); wpi_lr=0.2 (partial, cos 0.45) -> eprop 0.546 ~ chance (transient); wpi-frozen (cos ~0,
no cancellation) -> eprop 0.554 ~ chance. So the microcircuit's ENTIRE behaviour is a trajectory between random-credit
(~chance) and its fixed point (= fixed-DFA, below chance); it NEVER exceeds chance, NEVER trains. At convergence it IS
fixed-DFA, which the finding already showed fails.

<!--derived-->
**⇒ THE FINAL, PROVEN ELIMINATION.** Three distinct error-routing / credit-shaping mechanisms — FIXED-random DFA,
LEARNED KP feedback (B->W^T), and the LEARNED self-predicting MICROCIRCUIT (interneuron-cancelled apical error) — ALL
leave on-bridge e-prop at chance on a sparse representable codon the oracle solves at 0.94. The microcircuit's
fixed-point (cos 0.999 -> exactly fixed-DFA) PROVES the point: no matter how you route or shape the error signal, the
on-bridge e-prop cannot move the hidden weights toward the solution. **The wall is definitively the LOCAL CREDIT FACTOR
itself — the sigma'(v-theta) membrane surrogate x eligibility on the Izhikevich post-reset membrane** (the roadmap's
phi'-vanishing / operating-point diagnosis, now the SOLE surviving residual after every error-routing fix is
eliminated). NEXT is NOT another feedback/instructive-signal (proven inert here) — it is a fundamentally stronger LOCAL
credit factor (a better on-bridge surrogate that does not vanish on the Izhikevich membrane, or an operating-point that
keeps the surrogate informative) OR an honest substrate-level limit of the point-neuron Izhikevich surrogate for
credit. The crux CORE (LIF/rate) stands; the production-bridge residual is now isolated to a single, precisely-named
mechanism.

## Update 4 (2026-08-02) — the phi'-VANISHING hypothesis is REFUTED (surrogate is HEALTHY); no operating-point/surrogate tuning helps (0/30); the residual is NOT surrogate-collapse

<!--derived-->
A PARALLEL fix-sweep + a direct credit-factor DIAGNOSTIC (21 runs across the local box + the 3 pool nodes, plus
`--measure-credit-factor` reading the on-bridge credit factor Lsig x psi against a finite-difference backprop oracle)
refutes the phi'-vanishing hypothesis. On the sparse representable codon: the atan membrane surrogate is HEALTHY, not
vanishing — psi_mean/peak 0.31-0.32, dynamic-range 0.94, at BOTH init AND trained state (the FD oracle is validated at
init: FD-vs-readout-gradient cos +0.916). And a 30-run sweep of the surrogate sharpness (--alpha-surr 0.05..2.0) x the
operating-point tonic drive (--tonic-h/o-pA) trains 0/30 (all at chance). So the residual is NOT the surrogate
collapsing and NOT operating-point-tunable — correcting the phi'-vanishing framing.

<!--derived-->
**What the diagnostic shows (honest, with its caveat):** the credit factor's alignment with the backprop oracle is ~0
at init (cos +0.05, EXPECTED for FA — the forward has not yet aligned to the fixed feedback) and reads weakly negative
at trained (cos -0.27) — BUT the FD oracle degrades at the trained state (validation drops to +0.235), so the trained
alignment number is not clean; the reliable read is init (surrogate healthy). So the OPEN, sharpened question is FA
CONVERGENCE: does the transport-free credit ALIGN over training on the Izhikevich substrate (it does on LIF), measured
with a trained-state oracle that stays validated? The robust conclusion so far: the wall is NOT surrogate-collapse,
NOT operating-point, NOT error-routing (3 mechanisms) — it is that the transport-free local credit does not become
usefully aligned on Izhikevich despite a healthy surrogate. NEXT: a clean trained-state FA-convergence measurement
(fix the FD-oracle degradation), + the pending population (pool_k) sweep (does sqrt(K) cleaning align the credit); if
neither, this is an honest substrate-level limit of transport-free credit on the point-neuron Izhikevich rule. Runner
gains `--measure-credit-factor` / `--alpha-surr` sweep support (additive). Crux CORE (LIF/rate) stands.