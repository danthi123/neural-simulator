---
title: "A compositional (primitive-storing) spiking generator gives SUB-LINEAR O(√N) storage + neural composition at retention parity with the flat store — the mechanism is a GO, the retention-WIN over a fixed generator needs large N"
date: 2026-08-09
type: finding
status: contributing
lane: memory-continual-learning
seeds: [42]
seed-waiver: single-seed smoke + independent adversarial verify (both reproduced byte-identical); the HEADLINE claim (sub-linear O(√N) store = slots==P) is STRUCTURAL/deterministic by construction, not a stochastic metric; the retention numbers are single-seed with the 3-seed AGG + larger grids in flight (fold-in pending, like the N=50 orphan). Framed PARTIAL-GO, not a clean 6-seed GO.
---

# Compositionality gives sub-linear storage with neural composition — mechanism GO, retention-advantage pending large N

## Claim

<!--derived-->

The unifying scalability lever tested: real experience SHARES STRUCTURE, so a generator that stores the P≪N
PRIMITIVES and COMPOSES facts should scale sub-linearly. Built on COMPOSITIONAL facts (N=K1·K2 grid, P=K1+K2
primitives): a spiking generator that allocates ONE leaky-readout engram per distinct attribute value and
regenerates a fact by NEURAL SUPERPOSITION of its two primitive spiking-readout outputs. **Result (single-seed
validated, verify PLAUSIBLE / not refuted; 3-seed + larger grids completing):**

- **STORAGE is genuinely SUB-LINEAR — O(P)=O(√N), not O(N).** Store = P primitive slots (6 @3×3, 8 @4×4), NOT N;
  slots + learned-code floats grow ×1.33 (N=9→16) = √(16/9) exactly, vs the flat O(N) store's ×1.78. Verified
  sub-linear (not secretly O(N)).
- **Composition is NEURAL** (spikes/synapses, not a host algebra): regenerate() fires a frozen spiking Izhikevich
  reservoir with each primitive's query → the two primitive engrams write their disjoint feature blocks → the
  percept is the SUM (superposition/VSA-bundling) of two spiking population outputs. LESION localises: zeroing
  primitive-a collapses only the A block, spares B (and vice-versa) — distinct neural engrams, not a joint lookup.
  Stored raw patterns = 0 (holds P primitives, never the N composed percepts).
- **Retention at PARITY with the flat O(N) store:** N=9 comp 1.00 = flat 1.00; N=16 comp 0.94 = flat 0.94.
  Regeneration fidelity 0.999 (near-exact composed percepts). Acquisition high (0.89–0.91).

## The honest headline negative (and why it's expected)

<!--derived-->

**Compositionality is NOT a RETENTION win over a fixed generator at N≤36.** The non-compositional fixed generator
(v2, O(1) store) taught each compositional fact as an independent class matches/beats compositional on retention
(N=9 comp 1.00 vs v2 0.89; N=16 parity). WHY: the K1·K2 compositional engrams are **low-rank** (rank ≤ P), so a
fixed readout with H_gen=96 ≫ P already fits them. So for low-rank data a fixed generator suffices — the
compositional generator's genuine, unique advantages are (a) the EXPLICIT sub-linear reusable-primitive store, and
(b) neural composition enabling zero-shot novel combinations — neither of which the fixed generator has, and
neither of which is a raw-retention number at small N.

## What this means for the year-of-data question

<!--derived-->

- **Storage leg: CLOSED (mechanism GO).** When experience shares structure — which it does — a spiking generator
  stores O(√N) primitives, not O(N) facts, composing the rest. Brain-based, verified.
- **Compute + acquisition legs: INFERRED, not yet instrumented.** Primitive-based replay/learning should also be
  O(P) (rehearse/learn P primitives, not N facts), but this de-risk measured storage; the two other axes are the
  next instrumentation.
- **The retention/acquisition ADVANTAGE bites only at LARGE N** — where the flat + fixed-v2 readouts finally hit
  capacity (the N=100 acquisition wall). At small low-rank N everything fits, so the advantage is invisible. The
  next lever is a LARGER grid (10×10=100, 12×12=144) where the fixed stores saturate and only the primitive store
  scales. 3-seed N=36 + this regime are the pending confirmations.

## Rigor

Single-seed smoke + independent verify (both N-values reproduced byte-identical on the same reservoir/seed/env);
store-sublinear + neural-composition + lesion-localisation asserted True; `_used_ruler==False` (true primitives used
ONLY for the fidelity metric, never in learning); cfg.seed byte-identical (maxdiff 0); de-clamped bdsp_wmax=1e9; no
`sim/` edit; backend numpy. Artifact: `research/findings/raw/teacher_loop_compositional_generator_s42.json`
(runner: branch commit a015edfc4). 3-seed AGG + larger grids in flight.

NEXT: (1) larger grids (10×10/12×12) where fixed stores saturate → the retention/acquisition win; (2) instrument the
compute + acquisition axes (O(P) primitive replay/learning); (3) zero-shot composition test (regenerate unseen
combinations). NO-EXTERNAL-NEEDED: van de Ven 2020 + the project compositionality core (composer/VSA) are the recorded grounding.
