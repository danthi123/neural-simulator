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

## The CRUX (both design cruxes converge here)

**An FP-determinism / scale ceiling at total-N ≈ 4968.** The non-flagged conductance matvec paths (esp. the
slow-NMDA-recurrent `_nr_mat.T @ prev_firing` in `sim/bridge.py`) carry floating-point summation-order variance that a
long spiking read amplifies into a 1-spike divergence past ~4968 neurons — which is why the strict 7-organ byte-identity
batch was NO-GO (an FP artifact, NOT an organ defect). **One `sim/` edit** — generalize the existing
`deterministic_transpose_matvec` to ALL conductance paths — is the load-bearing prerequisite for merging the full organ
set. This is the highest-leverage single build in the program.

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

### Phase 2 — THE FP-DETERMINISM `sim/` EDIT (the crux; unblocks scale)
Generalize deterministic matvec to all conductance paths (additive/guarded, `tests/test_determinism.py`-pinned). Gate:
the previously-NO-GO strict full-batch byte-identity now passes past N≈4968.

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
1. **Phase-1 harness + shipped-faculty battery + OFF-arm gate** — the highest-value ready build; unblocks all gating below.
2. **Phase-2 FP-determinism `sim/` edit** — the scale crux.
Both are CPU/design-buildable while the GPU finishes the queued verdicts.
