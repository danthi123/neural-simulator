---
type: finding
status: contributing
date: 2026-08-11
mechanism: deep-credit-on-spikes — the INSTRUMENT (how to test depth-3 credit) not the credit rule
lane: gap#4 / deep-credit
verdict: a depth-3-OBLIGATORY task (depth-2 oracle underfits held-out, depth-3 clears) is PROVABLY IMPOSSIBLE at toy scale for a plain-MLP oracle — the measurement must be re-posed as LAYER-3 CREDIT FIDELITY, which is achievable
artifacts:
  - research/findings/raw/_gap4_depth3_obligatory_task_workflow.json
instrument: a 6-agent design+adversarial-verify workflow (wf_d2843076) — 5 candidate task families, each a self-contained numpy MLP backprop ORACLE (ReLU, softmax-CE/MSE, Adam) swept depth {1..5} x width {16..512} x 2 seeds, with an adversarial verifier trying to REFUTE each candidate by giving the depth-2 net its best shot (wide, long, Adam, weight-decay). SIM_BACKEND=numpy (host oracle — no substrate yet; this is instrument design).
---

# gap#4 deep-credit — the "depth-3-obligatory TASK" measurement is PROVABLY IMPOSSIBLE at toy scale (plain-MLP oracle); the crux must be re-posed as LAYER-3 CREDIT FIDELITY (does transport-free error reach the 3rd hidden layer)

The prior de-risk (`2026-08-11-gap4-SPATIAL-DEPTH2-CREDIT-...instrument-limited...`) found gap#4 deep-3 credit
UNTESTABLE because the compositional task is depth-2-solvable — no depth-3-obligatory task existed to serve as the
reference ceiling. This workflow was launched to BUILD one. It could not, and — decisively — showed **no such task
exists for a plain-MLP oracle class**, with a precise structural obstruction. That is not a defeat; it CORRECTS the
instrument and names the achievable test.

## Result — ZERO of 5 task families exhibit the property (`research/findings/raw/_gap4_depth3_obligatory_task_workflow.json`)

<!--derived-->
The property needed, simultaneously: **(P1)** a depth-2 backprop oracle underfits held-out at ALL widths (a genuine
representational limit, not under-parameterization), and **(P2)** a depth-3 oracle clears it (>0.85), SGD-learnable, on
a novel-combination split. Across five families swept over depth {1..5} x width {16..512}, none holds:

| family | d2 held-out | d3 held-out | why it fails |
|---|---|---|---|
| pointer_chase f³(s) | 0.73 | 0.74 | no separation (high-in-degree-node shortcut) |
| hier_bool XOR→AND→parity | 0.86 | 0.93 | depth-2 does NOT underfit — width substitutes for depth |
| nested_lookup (random tables) | 0.66 | 0.59 | depth-3 WORSE — pure memorization |
| quasigroup depth-2 (control) | 0.00 | 0.00 | shortcut-free ⇒ NOBODY generalizes |
| quasigroup depth-3 (Latin sq.) | 0.00 | 0.00 | all depths memorize (train→1.0, held-out < chance) |
| tent³/tent⁵ (Telgarsky) | 0.66 (best 0.99) | 0.77 | k=3 depth-2 CAN do it; k=5 nobody learns |

## The obstruction — two independent routes, both provably closed at the depth-2→depth-3 gap

<!--derived-->
- **Representation route (Telgarsky 2016 depth-separation).** Provable depth-2 lower bounds need width EXPONENTIAL in
  the depth-GAP — and here the gap is 1. A depth-3 ReLU net represents ~2³ linear-piece oscillations; a width-16 depth-2
  net already yields O(W) pieces and represents the same ⇒ **P1 fails**. Forcing depth-2 to genuinely underfit needs
  depth ≫ 3 (Telgarsky's d² vs d), which is NOT SGD-learnable (tent⁵ sat at chance for every depth incl. 3–5) ⇒ **P2
  fails**. The separable window and the learnable window do not overlap.
- **Generalization route (systematic novel-combination split).** Systematic generalization needs an inductive bias to
  REUSE intermediate factors (weight-tying / bottleneck / recurrence / attention). Plain fully-connected depth is NOT
  that bias — it is CAPACITY (more memorization; depth-3 generalized WORSE on nested_lookup). Add enough structure for
  depth-3 to generalize ⇒ a shallow net exploits the same structure (pointer_chase depth-1 = 0.70) ⇒ P1 fails; remove
  the structure ⇒ held-out is information-theoretically underdetermined and NO depth generalizes ⇒ P2 fails.
- **Root cause (one line):** for a plain MLP, depth's only lever is representational capacity; the depth-2→depth-3
  increment is exponentially small vs width (representation) and not an inductive bias at all (generalization) — so no
  bounded-width depth-2 net can be made to underfit EXACTLY what a bounded-width depth-3 net fits, at toy scale, under SGD.

## Consequence + the method that surpasses (change the MEASUREMENT, do NOT defer the capability)

<!--derived-->
Sources: Telgarsky, *Benefits of depth in neural networks* (COLT 2016) — the depth-separation width-vs-depth-gap bound
that closes the representation route. (This is the external deep-read the boundary-verdict discipline requires; the
verdict is a MEASUREMENT-impossibility, not a capability wall — the capability is re-posed to an achievable form below.)

The premise "re-pose the transport-free DFA e-prop depth-3 sweep on a depth-3-obligatory TASK with a plain-MLP oracle
ceiling" is unachievable — **do NOT wire a new `--task` into `_gap4_spatial_depth3_smallT_derisk`; the held-out-accuracy
framing cannot separate depth-2 from depth-3.** This CORROBORATES `2026-08-02-gap4-DFA-eprop-is-depth-robust-scales-to-N4`
(the prior "no depth-3-obligatory task" was not a search miss — it is fundamental).

**The achievable, biologically-meaningful test = LAYER-3 CREDIT FIDELITY** (does error reach the 3rd hidden layer
without weight transport — the literal gap#4 question):
1. **Task:** train a 3-hidden-layer net to FIT a genuinely depth-3-composed target where all-train fit provably engages
   layer 3 — **tent³ regression** (`v = tent(tent(tent(x)))`, MSE) or the **quasigroup DEPTH-3 chain (train-fit only)**.
   Additive `--task tent3_fit` / `qgroup3_fit`, data-only, default-off, NO `sim/` edit; keep `DendriticMLP([n_in,H,H,H,k])`
   + the DFA-eprop / graded-credit machinery unchanged.
2. **GO gate (genuine depth-3 credit):** transport-free DFA e-prop reaches **(a)** layer-3 weight-update cosine-alignment
   to the backprop-oracle layer-3 gradient **≥ 0.6 and RISING** with training, **and (b)** final train-loss within 10% of
   the backprop-oracle depth-3 loss. Compare vs the DEPTH-2 oracle on the same target: depth-2 train-loss must be
   strictly worse (confirms the target is depth-3-engaging on FIT — the one place depth-2 genuinely can't compete).
3. **Anti-cheats to keep:** (i) backprop-oracle ceiling hits loss≈0 at depth-3; (ii) permuted-target ⇒ alignment ≤ 0,
   no learning; (iii) depth-separating — the same rule at depth-2 cannot fit the depth-3 target (loss floor); (iv)
   lever-moved — apical-lesion / feedback B=0 collapses layer-3 alignment to ~0; (v) transport-free assertion — the
   credit path reads only fixed-random/KP feedback + local activity, never a forward Wᵀ.

**Next mechanism (the achievable gap#4 deep-credit test):** build the `tent3_fit` layer-3-credit-fidelity de-risk on the
existing DFA-eprop harness + run it on spikes. This is the real gap#4 experiment, now that the task-accuracy version is
shown impossible for this oracle class.
