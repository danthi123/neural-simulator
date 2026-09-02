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

### Phase 1 — INFRASTRUCTURE (BUILT + DE-RISKED 2026-09-02, CPU/numpy — see finding 2026-09-02-onebrain-flip-verify-harness-and-regression-battery-BUILT-derisk-GO)
1. **Reusable flip-verify harness** `research/runners/onebrain_flip_verify_harness.py` — DONE. Generalizes
   `_xedge_flip_production_verify.py`'s ARM A (byte-identical-off) / ARM B (visible-on-real-traffic through the REAL
   `webapp.server.brain_chat`, lesion-attributable, `n_hollow=0`) / ARM C (no-regression) into one `EdgeSpec`-parameterized
   entry. DE-RISK GO: the generalized aggregate reproduces the banked d6→comprehension verdict BYTE-FOR-BYTE on all 3 banked
   cupy artifacts (GO and NO-GO) vs the reference `_aggregate` (`--derisk`, no brain builds needed).
2. **The shipped-faculty REGRESSION BATTERY** `research/runners/onebrain_regression_battery.py` — DONE. Cross-faculty
   no-regression instrument: given a flag flipped ON-vs-OFF, runs a representative deterministic probe per default-on faculty
   through the real `brain_chat` (fresh per-arm builds at one seed) and asserts each still DECIDES identically (categorical
   decision variables only; continuous noise excluded). 38 faculties registered (16 exercised by the default probe set, 22
   thin/trigger-gated — the honest residual). Verified: synthetic identical→all-pass, deliberately-broken probe→caught
   (only the target faculty regressed); real no-op flip→all exercised pass. The harness ARM C now calls it.
3. **OFF-arm-staleness gate** `tools/gates/flip_offarm_staleness.py` (CLASS OS, BLOCKS) — DONE (the 2026-08-27 audit's
   deferred check). Flags a non-`*_LESION* `BRAIN_ flag popped as an OFF arm while its reader default resolves ON; selftest
   fails-in-the-failing-direction. Building it immediately caught a 7th live instance (`_wkv_mouth_open_ended_wiring_verify.py`,
   stale since the 2026-08-30 `BRAIN_OPEN_ENDED_WKV_MOUTH` flip), now fixed. The same explicit-`="0"` discipline is baked into
   the harness + battery OFF arms.

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

## The immediately-actionable next builds (GPU-free, ready now)
1. ~~**Phase-1 harness + shipped-faculty battery + OFF-arm gate**~~ — DONE 2026-09-02 (all three built + de-risked; the gating
   layer every merge/flip below depends on now exists). Follow-on (mechanical): lift the 22 thin battery probes to driving
   ones (a mismatch turn for surprise, a scalar turn for pragmatic, a 2-turn intention for prospective-memory, a visual
   percept for vision-identity, a between-turn tick for self-initiation).
2. **Wave 1** (merge comprehension + source_provenance onto the single pool) — the first merge, on the migration pattern,
   gated by the now-existing Phase-1 battery; moves the shipped surprise→source_provenance edge onto the one substrate.
   NO scale prerequisite: Phase-2 FP-determinism was refuted (the 7-organ merge is already 7/7 GO — see the corrected CRUX).
   This is the true next step.
