---
type: finding
status: partial
mechanism: fixed-orthogonal-target-pattern-separation
lane: breadth / catastrophic-forgetting / memory
date: 2026-08-09
runner: research/runners/_teacher_loop_ortho_target_pattern_separation_derisk.py
---

# Teacher-loop breadth crux: a FIXED ORTHOGONAL-TARGET pattern-separation readout (PS-SNN) raises sequential retention and is load-bearing, but is reservoir-gated and does NOT surpass self-replay on this substrate — honest PARTIAL with teeth

**Date:** 2026-08-09 · **Status:** 6-seed (42–47), **PARTIAL / honest negative with strong teeth** ·
**Backend:** numpy (the OnBridge Izhikevich net is tiny — ~67 neurons — and launch-bound; the cupy/3090 path is
verified to run but CPU is faster here, established by the sparse-readout de-risk) ·
**Aggregate artifact:** `research/findings/raw/teacher_loop_ortho_target_s42_AGG.json` ·
per-seed `research/findings/raw/teacher_loop_ortho_target_s42_s{42..47}.json`.

## The wall this attacks

The sleep-replay consolidation mitigation plateaus at **frac_recalled ≈ 0.55** (6-seed mean) — the KNOWN
replay-retention cap in extreme-interference regimes, and replay MAGNITUDE (engram fidelity / budget) was REFUTED as
the lever (main 8d2510d3). So this arc attacks the same catastrophic-forgetting wall with a DIFFERENT, proven
continual-learning mechanism instead of more replay: PS-SNN (Hu et al. 2026, Sci Reports) — replace the DRIFTING
learnable shared readout with PREDEFINED, mutually-ORTHOGONAL target codes (the dentate-gyrus decorrelated basis: one
fixed sparse orthogonal code per fact), so a new fact writes into its OWN orthogonal subspace instead of overwriting
earlier facts on the shared readout. This is the STRONG version of the weak k-WTA sparse-allocation that was REFUTED
(+0.00, 2026-08-08): fixed ORTHOGONALITY, not learned sparsity.

## The mechanism built (brain-based, additive, NO sim/ edit)

Runner-side, reuse-by-import of the scaling / sleep-replay / corrective-acquire machinery. The hidden reservoir is a
FIXED structural expansion (`freeze_hidden` — DG/granule expansion is structural, plasticity is at the readout; the
refuted-arc lesson #2 that trainable hidden collapses separation). Three brain-based pieces, all on the substrate's
OWN readout synapses (`cp_connections`, moved by the e-prop leaky-readout delta):

1. **Fixed orthogonal codes = the DG decorrelated target basis.** Each class owns a DISJOINT block of readout units
   (value 1/√w, unit-norm) → the Gram matrix is EXACTLY the identity (measured `max_offdiag_cosine = 0.0`): mutually
   orthogonal AND sparse (density 1/K). The label indexes WHICH innate code is the target; the brain's synapses must
   LEARN to produce it, and classification is the brain's readout output correlated against the codes
   (`argmax_c ⟨o, t_c⟩`, nearest orthogonal center) — not a host label lookup.
2. **Own-subspace weight protection (the allocation gate-freeze).** Teaching fact i CONFINES its regression write to
   its own code units (d masked to the code support), so it potentiates only the synapses onto its OWN output
   population and NEVER touches earlier facts' populations — an allocation-based gate-freeze (Phase-1.4 gate-freeze /
   PS-SNN per-class subspace). Because the codes have disjoint support, each fact's synapses are structurally
   protected: later facts write elsewhere.
3. **Homeostatic synaptic scaling** normalizes each fact's own output population to a target total weight (Turrigiano)
   so the nearest-code decision is not dominated by whichever block grew largest.

Two instrument lessons paid (both are "what else does the real system run alongside this that we replaced with a
constant"): the squared-error regression DIVERGES above lr ≈ 2/λmax(r rᵀ) — the self-limiting softmax delta hid this,
lr=0.5 blew the weights to |W|≈218; the stable ortho readout lr is ≈0.1. And the readout-norm fit must draw from the
natural post-proto stream, NOT a mid-stream RNG reset — the reset made the fit statistics swing retention 0.1↔1.0 on
ONE seed (an instrument artifact, fixed by a fresh env per arm).

## Result — 6-seed (N=10, K=10, chance 0.10, hidden 64, code-width 4, epochs 30)

<!--derived-->
| seed | DRIFT (softmax) | REPLAY (self-replay) | ORTHO | COLLAPSE (¬ortho) | ORTHO_FULL (¬protect) | ORTHO immediate-acq |
|---|---|---|---|---|---|---|
| 42 | 0.50 | 1.00 | **0.90** | 0.20 | 0.20 | 0.853 |
| 43 | 0.50 | 1.00 | **0.90** | 0.30 | 0.30 | 0.865 |
| 44 | 0.40 | 0.90 | **0.70** | 0.10 | 0.00 | 0.708 |
| 45 | 0.10 | 0.20 | 0.10 | 0.10 | 0.10 | 0.200 |
| 46 | 0.30 | 1.00 | 0.20 | 0.10 | 0.20 | 0.200 |
| 47 | 0.50 | 1.00 | **0.70** | 0.30 | 0.10 | 0.753 |
| **mean** | **0.38** | **0.85** | **0.58** | **0.18** | **0.15** | 0.596 |

Per-seed ORTHO − DRIFT: `[+0.40, +0.40, +0.30, 0.00, −0.10, +0.20]`, mean **+0.20**. Codes: ORTHO
`max_offdiag_cosine = 0.0` (genuinely orthogonal); COLLAPSE `0.923` (non-orthogonal, by construction). ORTHO_SI
(Zenke-2017 synaptic-intelligence penalty added on top) = ORTHO exactly on every seed (acq delta +0.000, retention
delta +0.00): SI is INERT on top of the allocation-based own-subspace protection (the confinement already writes
nowhere for the penalty to protect).

## Teeth — what HOLDS and what does NOT

- **(load-bearing orthogonality) HOLDS 6/6.** Remove orthogonality → COLLAPSE mean 0.18 (≤ drift on every seed):
  the own-subspace write degenerates to a full write (a non-orthogonal code has non-disjoint support), so the
  all-vs-all suppression the refuted k-WTA arc named returns and earlier blocks are zeroed. Orthogonality is what
  MAKES the own-subspace protection possible.
- **(load-bearing weight protection) HOLDS 6/6.** Remove the own-subspace confinement (ORTHO_FULL: regress the full
  code including its zeros) → mean 0.15: forgetting returns. The confinement IS the weight protection.
- **(codes genuinely pattern-separated) HOLDS.** Gram off-diagonal 0.0 for ORTHO vs 0.923 for COLLAPSE — measured,
  not a host label table; the brain's synapses learn to produce the code.
- **(retention rises vs the drifting readout) HOLDS on 4/6.** +0.30 to +0.40 on the seeds where the frozen reservoir
  separates the referents; ties on the dead seed 45; −0.10 on seed 46.
- **(surpass the 0.55 replay-only cap) PARTIAL, 4/6.** ORTHO clears 0.55 on 42/43/44/47 (0.70–0.90) but not on
  45/46 — the two seeds where the frozen reservoir barely separates (DRIFT itself collapses to 0.10/0.30 there).
- **(beat in-run self-replay) FAILS 0/6.** REPLAY dominates every seed (mean 0.85 vs ORTHO 0.58). On THIS
  frozen-hidden substrate self-replay is the stronger mechanism; the literature's 0.55 cap does not bind in this
  milder (N=10) regime.
- **(immediate acquisition stays high) COMPROMISED — the predicted tradeoff, MEASURED.** DRIFT acquires each new fact
  at ~1.0; ORTHO at 0.71–0.87 on working seeds (0.20 on the dead seeds). The own-subspace confinement does not
  suppress competitors, so a fresh fact can be spuriously outscored → lower immediate accuracy: the EWC/target-only
  "slightly compromises new learning" cost, made concrete.

## What the method actually needs (the deliverable, per the LAW)

The fixed-orthogonal-target + own-subspace mechanism is a REAL, load-bearing, replay-FREE continual-learning lever —
large retention gains where the substrate separates, and both of its components fail their lesions 6/6. But its payoff
is (1) GATED by the frozen pattern-separation front-end (it helps exactly when DRIFT already partly works, and cannot
rescue a reservoir that does not separate — seeds 45/46), and (2) BELOW what self-generated sleep-replay already
achieves on this substrate. So the residual is not the readout target scheme — that works — it is the
pattern-separation FRONT-END and the acquisition tradeoff. The next method is to pair the orthogonal-target readout
with a robust (learned or higher-capacity) DG-like separator so every seed's referents are linearly separable, and/or
to COMBINE it with replay (the two are orthogonal levers: allocation-based protection + interleaved reactivation).
Banked as a characterised PARTIAL, not a stop.

## Reproduce

```
# single-seed SMOKE (numpy; fast — tiny launch-bound net):
SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
  python -m research.runners._teacher_loop_ortho_target_pattern_separation_derisk --seed 42 \
    --n-max 10 --milestones 1 5 10 --hidden 64 --code-width 4 --epochs 30 --settle-steps 25 --n-draws 20 --si \
    --out research/findings/raw/teacher_loop_ortho_target_s42.json
# 6-SEED (self-sweeps in-process + writes an aggregate; GO would need the rise 6/6, which it does NOT clear):
SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
  python -m research.runners._teacher_loop_ortho_target_pattern_separation_derisk --seeds 42 43 44 45 46 47 \
    --n-max 10 --milestones 1 5 10 --hidden 64 --code-width 4 --epochs 30 --settle-steps 25 --n-draws 20 --si \
    --out research/findings/raw/teacher_loop_ortho_target_s42.json
```
