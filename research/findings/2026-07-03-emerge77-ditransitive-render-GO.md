# EMERGE-77 — the DITRANSITIVE renders on spikes: the EMERGE-74 N_SLOT_POOLS=6 capacity boundary SURPASSED (GO, 6-seed)

**Date:** 2026-07-03
**Verdict:** **GO** — "the dog gives the cat a bone" (the 7-slot ditransitive EMERGE-74 mined every seed but could not render because `N_SLOT_POOLS=6`) now renders **EXACT on real spikes** every seed. The FrameCQ slot-pool count is made a **per-instance, default-preserving parameter** (default 6 = byte-identical; 8 for the ditransitive producer) — the bounded scale lever EMERGE-74 named — plus a **2-stage per-pool bias-calibrated read** (the read-out lever the boundary predicted). All **7** named constructions render exact on the 8-pool substrate (EMERGE-74 rendered only 6); the capacity boundary is surpassed.
**Runner:** `research/runners/_emerge77_ditransitive_render_derisk.py`
**CI:** `tests/test_emerge77_ditransitive_render.py` (13 tests, CPU/numpy, offline — all pass)
**Raw:** `research/findings/raw/_emerge77_ditransitive_render.json`
**Boundary surpassed:** `2026-07-03-emerge74-transitive-ditransitive-GO.md` (the DITRANSITIVE capacity boundary: 7 slots > N_SLOT_POOLS=6; mined every seed, render capacity-gated; "fix = a bounded SCALE lever, N_SLOT_POOLS 6 → 8, after which the ditransitive renders with ZERO further mechanism").
**Reuse-by-import; NO `sim/` edit; the gate-first moat untouched. The ONLY edits are the additive, default-preserving `n_slot_pools` threading in the `research/runners` EMERGE-59/72 helpers (byte-identity verified).**

---

## What EMERGE-74 named (the boundary) and what EMERGE-77 does

EMERGE-74 **mined** the ditransitive's full 7-role signature (`det subj verb:3sg det iobj det obj` — Goldberg's ditransitive argument-structure construction "X causes Y to receive Z") + routed it to `C_DITRANS` every seed, but could **not render** it: 7 slots > `N_SLOT_POOLS=6` (`_emerge59_spiking_broca_frame_slots_derisk.py:118`). EMERGE-74 correctly named this a **spiking-substrate CAPACITY wall** (NOT a data/label wall — the mine found it) and the exact fix: bump the pool count 6 → 8.

EMERGE-77 surpasses it. **Because MANY EMERGE runners import `N_SLOT_POOLS` / `build_slot_bridge` / `slot_pool_rates`, bumping the module constant would cascade and break byte-identity.** So the pool count is made a **per-instance parameter** threaded additively:

- `build_slot_bridge(seed, n_slot_pools=N_SLOT_POOLS)` — region size `n_slot_pools * N_PER`.
- `slot_pool_rates(bridge, slot_idx, drive, n_slot_pools=N_SLOT_POOLS)` — the reshape matches the instance's region.
- `FrameSlotCQ.__init__(..., n_slot_pools=None)` — `None → N_SLOT_POOLS`; a per-instance `self.primacy_pA` re-spaced over `n_slot_pools` ranks; the prim init widened. `emit`/`emit_order_indices` use `self.primacy_pA` + `self.n_slot_pools`.
- `RegistryProducer.__init__` (EMERGE-72) — the per-construction prim init uses `self.n_slot_pools` (so a 7-slot construction's prim vector is wide enough).

**All default to the module `N_SLOT_POOLS=6`, so the shipped path is BYTE-IDENTICAL** (verified: the default `FrameSlotCQ` prim init is bit-identical to the pre-edit `standard_normal(6)`; `self.primacy_pA is PRIMACY_pA`; the EMERGE-59..76 CI passes — 100 tests, below). EMERGE-77 then instantiates a **`DitransRegistryProducer` at `n_slot_pools=8`** and renders the whole EMERGE-74 inventory on real spikes.

## Result (6 seeds: 42/43/44/100/101/102, CPU/numpy)

Every seed is identical:

| metric | value (all 6 seeds) |
|---|---|
| constructions registered (mined) | **7** |
| constructions rendered EXACT on spikes | **7** (the 5 EMERGE-72 + C_TRANS + **C_DITRANS**) — render **1.000** |
| DITRANSITIVE mined + rendered exact | **True / True** every seed |
| DITRANSITIVE position-independent (emit-pos 1/3/5) | **True** every seed |
| RAW (uncalibrated) read orders the ditransitive | **False** on 3/6 (seeds 42/43/102) — the 2-stage read is LOAD-BEARING |
| PERMUTED-CORPUS render | **0.000** (n_registered 0.0) |
| CROSS-CONSTRUCTION render | **0.000** |
| NO-CORPUS registered | **0** |
| gate-first moat calls on abstain | **0** |
| default-6 FrameSlotCQ path byte-identical | **True** |

**Live transcript (spikes, gate-first moat) — the ditransitive now renders:**
```
you> what does the wolf chase?          broca> the wolf chases the ball          [TRANS;   producer INVOKED]
you> what does the wolf give the cub?   broca> the wolf gives the cub a bone     [DITRANS; producer INVOKED]   <-- was capacity-gated in EMERGE-74
you> can a zzz fly?                      broca> I don't know.                     [MOAT;    producer NOT invoked]
```

## The ONE tuned variable — the read-out limit the boundary predicted (honest)

EMERGE-59's `PRIMACY_pA` range (1800..300 pA) was tuned so **6** ranks separate cleanly in RATE below the f-I saturation shoulder. Re-spaced over **8** ranks, the top three currents (1800/1585/1371 pA) sit in the ~0.42-rate **saturation band** where the fixed per-pool f-I heterogeneity (`cp_izh_vr`/`cp_izh_b` bias, per-pool std ~0.02) **FLIPS the two top adjacent ranks in the RAW rate read** — verified: the raw 8-rank read fails to order the ditransitive on **3/6 seeds** (42, 43, 102).

The single principled fix is a **2-STAGE READ** — the exact lever EMERGE-74's boundary named ("more sim steps / wider primacy / a 2-stage read"): a **per-pool BIAS CALIBRATION** — measure each pool's rate at a common reference current (a per-unit homeostatic normalization; Turrigiano), subtract it from the primacy read. This equalizes the population's f-I so the rate code is unbiased, and recovers the correct order on **all 6 seeds**. Key checks:
- **RAW is the causal control:** with `calibrate=False` the 8-rank read fails on 3/6 seeds → the 2-stage read is **load-bearing, not decorative**.
- **Diagnosis, not a metric hack:** the f-I curve is monotone-linear 200→1800 pA (rate 0.059→0.374, ~0.019/100pA); the flips are a fixed per-pool bias (std ~0.02) swamping the tightly-packed top-rank gaps, exactly removed by the bias subtraction.
- **Read-side only:** the calibration touches the rate read, NOT the moat, and is runner-local (no `sim/` edit).

I did **not** widen the primacy range (a single lower-top sweep confirmed lowering the top alone doesn't help — the top ranks still tie; the 2-stage read is the clean single-variable fix). One variable changed per rung.

## Position-independence (the 7-slot frame is the hardest for the EMERGE-61 adaptation tail)

The ditransitive is the LONGEST frame (7 slots) → the most spike-frequency-adaptation accumulates across pools, the case the EMERGE-61 inter-utterance wash-out (`_reset_substrate`) is designed for. Verified: the ditransitive renders **IDENTICALLY at emit-position 1 / 3 / 5** (0/2/4 prior productions) every seed — the wash-out holds at 8 pools.

## Anti-cheats (all collapse)

- **PERMUTED-CORPUS** — shuffling each exemplar's word order before mining → 0 registered (render 0.000). The render is corpus-order-driven.
- **CROSS-CONSTRUCTION** — rendering construction A's fact through a DIFFERENT construction B's mined structure is wrong (0.000; Dominey-Hinaut form-specificity — the ditransitive rendered through the transitive ≠ the ditransitive).
- **NO-CORPUS** — empty stream → 0 registered.
- **RAW-READ (2-stage causal)** — the uncalibrated read fails on 3/6 seeds → the 2-stage bias calibration is load-bearing.

## The moat (gate-first, untouched)

The no-confab moat holds by construction: on ABSTAIN the producer is NEVER invoked (0 productions on abstains, 6 seeds). The moat's positive-control ANSWER runs through the 8-pool calibrated `emit` (EMERGE-77 overrides `RegistryProducer.emit` so the moat path uses the instance pool count + the calibrated read — the base `emit` hard-called `slot_pool_rates` at 6 pools, which shape-mismatched an 8-pool bridge). Not weakened.

## Regression — the default-6 path is BYTE-IDENTICAL

- EMERGE-59/61/63 CI: **25 passed**.
- EMERGE-72/74/75/76 CI: **41 passed**.
- EMERGE-60/65/66/73 CI: **34 passed**.
- ⇒ **100 tests green**; the default `FrameSlotCQ` prim init is bit-identical; `self.primacy_pA is PRIMACY_pA` at the default pool count. The `n_slot_pools` threading did NOT change the shipped path.

## Honest scope / what this is NOT

- SURPASSES the ditransitive **capacity** boundary via the bounded pool-count scale lever (6 → 8, per-instance) + the 2-stage read. The producer now renders **all 7** named constructions on spikes INCLUDING the ditransitive (the biggest post-verbal-argument core construction — a recipient + a theme after the verb). NOT open prose (R4, the separate deferred wall; the from-scratch spiking LM is ~4 orders too small).
- The **A→W SPELL stays the token surface** for THIS de-risk; the fully-spiking A→W of the ditransitive's new content nouns (iobj/theme) is the batched EMERGE-75-style follow-on (its own spiking validation is `concept_speak_demo`).
- The corpus mining is offline syllabus prep (BRAIN-BASED-ONLY compliant, like rendering a retinal image the neural retina reads); the STRUCTURE is rendered on REAL spikes.
- The named EMERGE-74 residual is unchanged: the ditransitive's DISTINCTIVE part (the IOBJ — a second post-verbal content noun) is attested only by the ditransitive itself (a shared-vs-distinctive residual), while the shared SVO backbone generalizes. EMERGE-77 renders it; it does not change that hold-out map.

## Sources
- Goldberg, *Constructions* — the ditransitive argument-structure construction ("X causes Y to receive Z").
- Hinaut & Dominey (PLoS ONE 2013; Brain & Language 2015) — production = SELECTING the construction; the reservoir generalizes to NEW constructions from closed-class order/position.
- Grossberg 1978 / Bullock-Rhodes 2003 — competitive queuing; the primacy gradient → the rate-coded emission order; the tie-break-stability read risk (exactly the tightly-packed 8-rank case).
- Turrigiano — homeostatic per-unit gain/threshold normalization (the 2-stage per-pool bias calibration: equalize the population's f-I so the rate code is unbiased).
- Project precedents: `_emerge74_transitive_ditransitive_derisk.py` (the mined ditransitive inventory + the named boundary), `_emerge59_spiking_broca_frame_slots_derisk.py:118,126` (`N_SLOT_POOLS`, `PRIMACY_pA`), `_emerge61_spiking_broca_order_robustness_derisk.py` (the inter-utterance wash-out), `_emerge72_construction_registry_derisk.py` (the signature-keyed registry + `RegistryProducer`), catalog G.12 (Broca grammatical encoding).
