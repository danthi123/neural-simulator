# One-brain INTEGRATION program — the sequenced path from de-risked pieces to one integrated brain (2026-09-02)

**A PLAN, not a finding.** Synthesizes a 3-axis design workflow (organ-merge sequencing · full cross-edge wiring plan ·
verification+production strategy). Goal: genuine one-brain INTEGRATION — the organs INFLUENCING each other's
computation through LEARNED spiking cross-synapses, on ONE substrate, ON BY DEFAULT, scaffolds retired — NOT mere
co-residence (migration). Every step carries the four-stage bar: **de-risked → WIRED → on-by-default →
scaffold-retired** (not "done" until all four; a flip is not a landing until its host path is retired). Functional
read-outs only; no phenomenal claim.

## Current connectome (verify-first ground truth — the DESIGN finding's "5 built" was optimistic)

- **LIVE learned cross-edges (default-ON): TWO** — d6-WM→comprehension (the template, per-turn DA-credit) + curiosity→d6-WM.
- **WIRED but default-OFF:** surprise→episodic-encode, surprise→source_provenance, self_schema→source_provenance (R4).
- **DE-RISKED (2-seed SMOKE-GO, 6-seed cupy QUEUED):** C1 surprise→world-model, C2 surprise→metacog, C3 arousal→surprise.
- **Substrate:** the single-pool merge (`BRAIN_ONEBRAIN_SINGLE_POOL`) is WIRED default-OFF, 6-seed soak queued. Today's
  4 core organs co-reside as pool#1(surprise+world-model) + pool#2(metacog+pragmatic); surprise↔metacog edges CANNOT be
  produced until the single pool lands.

## The CRUX — ⛔ CORRECTED 2026-09-02: the "FP-determinism scale ceiling" was a FALSE premise; the 7-organ merge is ALREADY 7/7 GO

The Axis-1 design named an FP-determinism ceiling at N≈4968 (slow-NMDA-recurrent matvec summation-order variance) as
the merge crux. A verify-first Phase-2 pass REFUTED it: on numpy the matvecs were already co-residence-invariant
(hardening moved `read_maxerr` by exactly 0 — commit `06ce99c76`, 2026-08-27), and the REAL 7-organ wall was a
co-residence-DEPENDENT nmda_slow WIRING/RNG seam — **already CLOSED**: `dedup_synapse_masks` (`b22286162`, closes d6
organ-read) + `per_region_inhibitory_seed` (`cb8bc175b`/`07de22d6e`, closes prospective_memory's residual) → **7/7
organs GO** (board #180). So there is **NO scale/FP prerequisite** — the merge waves proceed directly, gated only by the
migration pattern + the Phase-1 harness. The one genuine FP gap that remained (the opt-in megakernel-v1 GPU fast-path's
cuSPARSE csrmv) is now closed too (`86b8e6384`, additive/default-OFF) but is a determinism-contract completion, NOT a
blocker. **Net: the program is LESS blocked than designed — the old "Phase 2 / Wave-3 scale gate" is DONE, not a gate;
Wave 3 (d6 + prospective-memory) can proceed on the same migration pattern as Waves 1-2.**

## The seam taxonomy (the merge's silent killers — a MergeConflict is NOT raised; the union accepts a default and the faculty dies quietly)

hebbian (→ per-synapse gain-0 freeze) · param-het (→ name-keyed per-region mask) · `hebbian_max_weight` (global 45 vs
attractor design-weights 400 — verify frozen edges survive unclipped) · **homeostasis** (the world-model killer → per-region
ON mask) · read-isolation (standardize ALL merged organs on full-snapshot restore, not per-neuron) · OU+neuromod (global
subsystem) · **FP-determinism** (the N≈4968 ceiling above) · structural (d3 multi-bridge, d5 own-pool CA3).

## The sequenced program

### Phase 0 — HARVEST (in flight, 0 new work)
The single-pool 6-seed cupy soak + the 3 Group-C cupy `run_gate` verifies are QUEUED on the GPU. Each Group-C edge stays
PARTIAL until its 6/6 cupy GO; a NO-GO is a real finding. Controller harvests.

### Phase 1 — INFRASTRUCTURE (READY NOW, CPU, no GPU, no dependency — the highest-value immediate build)
1. **Reusable flip-verify harness** `onebrain_flip_verify_harness.py` — generalize `_xedge_flip_production_verify.py`'s
   ARM A (byte-identical-off) / ARM B (visible-on-real-traffic through the REAL `webapp.server.brain_chat`, lesion-attributable,
   `n_hollow=0`) / ARM C (no-regression). De-risk: reproduce the banked d6→comprehension verdict byte-for-byte.
2. **The shipped-faculty REGRESSION BATTERY** — *this instrument does not exist*: ARM C today checks only ONE faculty's
   fixed items. Build a cross-faculty battery that asserts a flip does not break the OTHER ~29 default-on faculties. This is
   the single most load-bearing missing instrument — every merge + flip below depends on it.
3. **Fix the OFF-arm-staleness bug + its gate** (the 2026-08-27 audit's deferred `tools/gates/` check): OFF arms must set
   `="0"` explicitly, never `os.environ.pop()` (which silently becomes ON-vs-ON once a default flips). Closing it here
   protects every later step.

### Phase 2 — ✅ DONE / NOT A GATE (see the corrected CRUX above)
The FP-determinism "scale gate" was a false premise: the 7-organ merge is already 7/7 GO (nmda_slow wiring/RNG seam
closed via `dedup_synapse_masks` + `per_region_inhibitory_seed`). The remaining opt-in megakernel-v1 determinism gap is
also closed (`86b8e6384`). No scale prerequisite remains — Wave 3 proceeds on the same migration pattern as Waves 1-2.

### Phase 3 — THE MERGE WAVES (organ merges + cross-edge flips/wire-ins, each gated by Phase-1's harness)
- **Wave 1** (no new seam, small, default-ON endpoints): merge **comprehension + source_provenance** onto the single pool
  → moves the shipped **surprise→source_provenance** edge onto the one substrate.
- **Wave 2** (param-het wrinkles): merge **self_schema + curiosity + causal_whatif** — resolve the self_schema/metacog
  `workspace` NAME COLLISION + the 400>45 attractor-weight survival; lands the GNW workspace on the pool.
- **Wave 3** (post-Phase-2 scale): merge **d6-multiref-WM + prospective-memory** (largest, 1720 neurons; pushes N→~4968)
  → completes the d6→comprehension template on the single pool.
- **Interleave the cross-edges** as their endpoints co-reside: FLIP Group B (already wired) with the harness; WIRE-IN
  Group C (C3 arousal→surprise can attach to pool#1 before the full merge; C1/C2 after single-pool). Each: 6-seed cupy
  GO → default-ON flip → host-path retire.

### Phase 4 — NEXT-TIER cross-edges (design + de-risk) & HOST-COUPLING conversions
Group D (D1 pragmatic↔metacog — CHEAPEST, endpoints already co-resident; D2 self_schema→metacog; D3 world-model→comprehension
reciprocal biased-competition; D4 metacog→repair conflict-monitoring; D5 surprise→memory). Group E — convert host formulas
to spiking cross-edges: E1 affect→tone (#84), E2 confidence→forthcomingness (#94), E3 swap→focus (#85).

### Phase 5 — INTEGRATION VERIFICATION + PRODUCTION-DEFAULT
- **Combined-integration test:** with N cross-edges live at once, do they compose without regressing each other + the
  roster (the wave-flip no-regression pattern, now on the Phase-1 battery).
- **Does integration IMPROVE conversation?** A/B the integrated-vs-migrated brain on the `_conversation_turing_test`
  honesty-first battery; anti-hollow bar — the improvement must VANISH when the cross-edges are lesioned.
- **Sequence to production-default + scaffold-retirement:** flip `BRAIN_ONEBRAIN_SINGLE_POOL` + each edge default-ON
  (each gated on its soak + the combined battery), then retire `MergedSubstrate`/`MergedSubstrate2` + the converted host
  couplings. Only then does a row read `scaffold_retired=YES`.

## The immediately-actionable next builds (GPU-free)
1. **Phase-1 harness + shipped-faculty battery + OFF-arm gate** — IN PROGRESS (branch `research/onebrain-flip-verify-harness`);
   the highest-value build, it unblocks the gating for every merge/flip below. (Phase-2 FP-determinism is DONE — see CRUX.)
2. **Then Wave 1** (merge comprehension + source_provenance onto the single pool) — the first merge, on the migration
   pattern, gated by the Phase-1 battery; no scale prerequisite (the FP "crux" was refuted). This is the true next step.
