# gap#5 residual: plateau termination is INTRINSIC in the biology, and the engine already has the machinery

**Date:** 2026-07-30 · **Status:** research-gate finding, NOT yet a result. No build, no `sim/` edit, no GO claimed.
**Trigger:** research gate conditions (d) *new mechanism class* + (e) *`sim/`-edit-to-overcome*, fired before
designing the multi-compartment change that gap#5's last ~33% of field quality was said to require.

## The residual, as it stood

BTSP place-field formation is GO (order read 0.969 single-trial; differentiation 11.0-11.2 of 12 tiling fields;
join 6/6 against raw AND magnitude-matched nulls). Field QUALITY tops out at `circ_resultant` 0.588 = 67% of the
σ=5 oracle (0.8719). Three cheap routes were already proven structurally incapable, each because it acted on the
wrong array: negative-weight lateral inhibition (engine E/I is trait-based via `cp_traits`, so a negative weight
subtracts from `g_e` rather than driving `g_i`); a subunit-per-block trick (`cp_v_apical` is per-NEURON,
`cp.full_like(cp_membrane_potential_v)` at `sim/bridge.py:7159`, so there is no per-branch state to address); and
GABA_B as a terminator (UNDEFINED, not negative — `I_gabab` is computed from `cp_membrane_potential_v` at
`sim/bridge.py:7330` and never touches the apical compartment, so the probe measured the wrong compartment).

The standing conclusion was that closing the residual needs a genuine multi-subunit apical rewrite. **Two
primary-source reads say that framing is wrong, and much more expensive than necessary.**

## Finding 1 — in the biology, the plateau terminates INTRINSICALLY, not by inhibition

Kandel 6e, Figure 10-15 (Ch. 10, *Propagated Signaling*), read in full rather than cited from a rerank snippet.
On the thalamic burst, the text is explicit that the plateau ends by its own voltage-dependent inactivation:
*"The strong depolarization during the burst causes the HCN channels to close and inactivates the Ca2+ channels,
allowing hyperpolarization to develop between bursts of firing."*

Two intrinsic conductances carry it. Low-threshold Ca²⁺ generates the plateau itself — *"Depolarizing inward
current through these Ca2+ channels (ICa) generates a plateau potential of about 20 mV"* — and must first recover
from inactivation via hyperpolarization. A-type K⁺ opposes the approach to threshold: *"a transient outward K+
current, IK,A, that briefly slows the approach of Vm to threshold. These channels typically are inactivated at
the resting potential (−55 mV), but steady hyperpolarization removes the inactivation."*

Inhibition is named as a MODULATOR of this intrinsic machinery, not the terminator: *"steady hyperpolarization,
such as might be produced by inhibitory synaptic input to a neuron, can profoundly affect the spike train
pattern."* It sets the inactivation state the intrinsic currents then operate from.

⇒ The three failed routes all reached for **synaptic inhibition** as the sharpening mechanism. The biology puts
the terminator **inside the compartment**. That is a reframe, not a tuning fix.

## Finding 2 — the engine already has both conductances, and an existing precedent for adding them apically

Read from our own code before theorizing (the standing lesson). All four extended HH currents already exist as
fused kernels in `sim/kernels.py`: `fused_hh_m_current_update` (M-current, :120), `fused_hh_CaT_current_update`
(T-type Ca²⁺, :139), `fused_hh_h_current_update` (Ih/HCN, :157), `fused_hh_NaP_current_update` (:171). They are
invoked at `sim/bridge.py:7492` and `:7507` — **on the somatic voltage only.**

Decisively, `cp_v_apical` already has its own integration loop with its own voltage-gated term
(`sim/bridge.py:7185-7198`). The apical update is `_dv = -(v_apical - E_r) + R*I_coincidence + g_c*(v_soma -
v_apical)`, and an **optional Kir (inward-rectifier) branch** is already layered on top of it as a gated additive
term: `_gkir = kir_g / (1 + exp((v_apical - vh)/kk))`, then `_dv = _dv + _gkir*(E_k - v_apical)`.

⇒ There is already an in-file, in-compartment precedent for exactly the edit shape required: one config-gated
additive `_dv` term on the apical voltage. An intrinsic plateau terminator follows that precedent using kernels
that already exist. This is a small additive default-off change, **not** a multi-subunit reshape of
`cp_v_apical`, and not the months-scale dendritic rewrite.

## The mechanism this implies (hypothesis, with its kill criterion)

Under Bittner-Magee BTSP, field width is set by the eligibility window, and the plateau sets that window. If the
plateau is never terminated it stays wide, so the learned field stays wide — which is what a `circ_resultant` of
0.588 against a σ=5 oracle looks like. Adding intrinsic termination should narrow the field toward the oracle.

**KILL CRITERION (stated before the run, so it can fail):** if field `circ_resultant` does NOT improve as plateau
duration is shortened, then width is not set by plateau duration and this whole reframe is refuted — the residual
lives in the eligibility kernel, the input bump width, or the operating point instead. That refutation is a real
deliverable and would redirect the arc, so the probe is worth running either way.

## Honest scope — what this is NOT

This is a research-gate finding: two source reads and a code trace. **Nothing has been built, measured, or
validated.** No claim is made that the residual is closed. The mechanism above is a hypothesis with a stated kill
criterion, awaiting the adversarial round now running (four skeptics chartered on: already-possible-without-a-
`sim/`-edit, wrong-knob, instrument/metric defect, and project-law compliance). The instrument objection is live
and serious — this project voided three separate results this month to metric defects (a permutation-invariant
score, a rounded-then-differenced delta, and `peaks=1.00` that meant UNIFORM rather than sharp), so the σ=5 oracle
being the right target is itself an open question.

Only after that round survives does a probe get built, with the usual anti-cheat controls (lesion, permuted, and
a magnitude-matched null).
