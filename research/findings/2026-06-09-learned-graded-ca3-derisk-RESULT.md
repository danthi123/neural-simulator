# Learned-graded CA3 autoassociator de-risk (the protected `nmda_slow` recurrent, actually tested) — **NEGATIVE (0/3)**, but the harness bug is FIXED and the wall is now mapped precisely with the recurrent ABLATION proving the recurrent is not load-bearing

**Date:** 2026-06-09
**Type:** runner-side debug + decisive CuPy de-risk of the protected `exc_receptor="nmda_slow"` edit (committed `069d3023`, byte-reviewed). **NO `sim/` edits** — `git status --short sim/` byte-empty before AND after (verified; the protected edit is the only `sim/` change and it was left untouched).
**Backend:** `SIM_BACKEND=cupy` (production backend; numpy DISQUALIFIED for striatal/near-threshold work). Deterministic regime (OU / conductance-noise / global-homeostasis / heterogeneity / STP OFF), hard-asserted each run.
**Owner directive:** biologize everything, no banking, brain-based-only. **An honest negative IS the scientific deliverable.**
**Supersedes / corrects:** the prior `_learned_graded_ca3_3seed.json` 0/3 was a **harness bug, not a result** (silent-CA3 artifact — every metric exactly 0.0). This re-test fixes the bug and runs the *actual* test.
**Builds on:** `2026-06-09-C1-trisynaptic-ca1-place-code.md` (the fire-vs-grade wall at the CA3 recurrent), `2026-06-09-place-code-selforg-stage1-derisk.md` (the validated Stage-1 single-hop place code reused as the fix), `2026-06-09-learned-graded-ca3-design.md` (the design).

---

## TL;DR — VERDICT: **NEGATIVE (0/3 seeds, both operating points).** The harness bug is FIXED (CA3 now fires); the `nmda_slow` recurrent was genuinely exercised; and the fire-vs-grade wall is irreducible — **DISTINCT and HIGH-RATE (fires the MSN) are on opposite sides of a sharp boundary with no overlap, and the recurrent-ablation anti-cheat proves the learned recurrent is NOT the load-bearing element of any passing gate.**

**The harness bug (controller-diagnosed, confirmed):** the original 0/3 had every metric exactly `0.0` because the harness drove CA3 ONLY through the trisynaptic feedforward `landmark_sensors → ec → dg → ca3`, which does **not conduct** at probe scale (the EC fire-vs-select tension + DG-FFI kills the sparse mossy fan-in — the C1 finding). **CA3 never fired → the `nmda_slow` recurrent had nothing to store → the protected edit was never actually tested.**

**The fix (preferred route — bypass the broken multi-hop):** a **FIXED direct `landmark_sensors → ca3` AMPA detonator** (the validated Stage-1 single-hop competitive place mechanism), so the AMPA feedforward reliably fires a sparse distinct CA3 ensemble per location; the `ca3 → ca3` recurrent stays routed `exc_receptor="nmda_slow"` (the protected edit) for the graded sustain + autoassociative storage. **An INSTRUMENTATION GUARD now measures CA3 firing DURING storage and hard-asserts it > 0** (a silent-CA3 run is caught immediately). The `--no-direct-ca3` control reproduces the original bug (the guard then fires the AssertionError, CA3 = 0.000 spk/step).

**CA3 now fires during storage: 0.18–0.22 spk/step (≈0.4–0.6 Hz pop) at the distinct point; 3.75–3.84 spk/step at the dense point** — the silent-CA3 artifact is gone, so this is the *real* test of the protected edit.

### The two operating points are perfect mirror images (3 seeds 42/43/44, CuPy):

| Gate | **DISTINCT point** (intensity 450, w 20) | **DENSE point** (intensity 900, w 40) |
|---|---|---|
| CA3 fires during storage (the guard) | ✅ 0.18–0.22 spk/step | ✅ 3.75–3.84 spk/step |
| **G1 DISTINCT** (CA3 diff-cos < 0.30) | ✅ **3/3** (0.135 / 0.260 / 0.188) | ❌ **0/3** (0.72 / 0.70 / 0.68 — position-blind) |
| **G2 GRADED** (active-cell ~10–40 Hz, not 0/not ceiling) | ✅ 3/3 (act 10.2–10.4 Hz; pop 0.5) | ✅ 3/3 (act 12–14 Hz; pop 7.5) |
| **G3 STABLE** (same-loc cos > 0.70) | ✅ 3/3 (0.94–0.98) | ✅ 3/3 (0.74–0.87) |
| **G4 HIGH-RATE** (CA1→MSN-D1 ≥ 5 Hz / ~420 pA) | ❌ **0/3** (CA1 **0.00 spk/step** → MSN 0 Hz, 0 pA) | ✅ **3/3** (MSN **21–28 Hz**, **372–453 pA**) |
| **G5 SENSOR-DRIVEN** (ablate sensors → collapse) | ✅ 3/3 (cos 0.000) | ❌ **0/3** (cos 0.63–0.80 — autonomous reverberation) |
| **G6 COMPLETION** (drop 1/3 landmarks → recall > 0.7) | ❌ 0/3 (0.37–0.46 — cue-fragile) | ✅ 3/3 (0.74–0.80) |
| **ALL_PASS** | 0/3 | 0/3 |

**The load-bearing observation:** `G1 DISTINCT` (needs CA3 sparsity ≤ ~5%) and `G4 HIGH-RATE` (needs CA3 dense enough — ≥ ~29% — to fire CA1) are separated by a **sharp boundary with NO overlap** (boundary sweep below). The `nmda_slow` recurrent NARROWS but does not close the gap.

---

## The decisive recurrent-ablation anti-cheat (the contract: "does removing the recurrent collapse the graded sustain?")

This is the test that makes the negative honest — and it returns a **stronger negative than expected: the learned recurrent is not the source of any passing gate.**

| Condition | G2 GRADED (active Hz) | G4 (MSN Hz / drive) | G6 COMPLETION | G5 (ablation cos) |
|---|---|---|---|---|
| **DISTINCT, recurrent ON** | 10.2–10.4 | 0 Hz / 0 pA | 0.37–0.46 | 0.000 |
| **DISTINCT, recurrent ABLATED** | 10.2–10.3 (**unchanged**) | 0 Hz / 0 pA | 0.38–0.44 (**unchanged**) | 0.000 |
| **DENSE, recurrent ON** | 12–14 | 21–28 Hz / 372–453 pA | 0.74–0.80 | 0.63–0.80 |
| **DENSE, recurrent ABLATED** | 17 (**up**) | 25–37 Hz / 552–616 pA (**up**) | 0.80–0.82 (**unchanged**) | 0.000 (**now passes G5**) |

- **At the DISTINCT point, zeroing the `ca3→ca3` recurrent changes NOTHING** (G1 0.135→0.138, G2 10.2→10.2, G6 0.41→0.41). The distinctness, stability, and (weak) completion are ALL from the FIXED feedforward — **the recurrent is silent there** (too few cells co-fire during storage to grow a basin: the recurrent weight grows only ~0.01→0.24 and contributes ~0% of the recall rate). The graded sustain at the distinct point is **NOT a learned recurrent attractor.**
- **At the DENSE point, zeroing the recurrent does NOT collapse G4/G6 — it IMPROVES them** (MSN 21→30 Hz, drive 372→552 pA; completion unchanged) and **fixes G5** (the autonomous reverberation disappears, cos 0.63→0.000). So the high-rate + completion at the dense point are **the strong FIXED feedforward, not the learned recurrent**; the recurrent's only net effect there is to add the **position-blind self-sustaining reverberation that FAILS G5** (exactly the C1 "global basin" pathology).

**Conclusion of the anti-cheat:** the `nmda_slow` recurrent never delivers the "distinct + graded + MSN-firing" combination it was designed for. Where CA3 is distinct, the recurrent can't ignite; where it ignites, it collapses distinctness and self-sustains without sensors. The recurrent is **not the load-bearing element of any gate that passes.**

---

## The decisive boundary sweep (single seed 42, full storage, G1 vs G4 measured together)

Sweeping the feedforward drive maps the bifurcation directly — and shows there is **no point where CA3 is both distinct AND drives the MSN:**

```
 intensity  w   d  | CA3 diff-cos  CA3 sparsity  MSN max  | G1(distinct<0.30)  G4(MSN>=5Hz)
   450     20  0.3 |   0.135          4.7%        0.0 Hz  |       PASS              FAIL
   500     22  .35 |   0.432         19.3%        0.0 Hz  |       FAIL              FAIL
   550     24  .35 |   0.689         29.0%       14.8 Hz  |       FAIL              PASS
   600     26  0.4 |   0.942         44.2%       22.2 Hz  |       FAIL              PASS
   700     30  0.4 |   0.857         49.0%       25.0 Hz  |       FAIL              PASS
   800     34  .45 |   0.747         54.6%       15.0 Hz  |       FAIL              PASS
```

The crossover sits between **4.7% sparsity (distinct, MSN-silent)** and **29% sparsity (MSN fires, position-blind)**. And critically — at the distinct point, **CA1 fires 0.00 spk/step even with a near-fully-dense, very strong Schaffer projection** (`ca3→ca1` weight 120, density 0.9) and an arbitrarily strong `ca1→msn` (up to 500): a sparse-distinct CA3 ensemble (~19 cells each firing ~once per 100 steps) simply produces synaptic transients too brief/sparse to summate CA1 above threshold. **You cannot fire a downstream cell from a presynaptic population that itself fires at <0.2 spk/step.** This is the irreducible point-neuron rate-coding wall (the Stage-2 / C1 boundary), now confirmed at the CA3→CA1→MSN stage with the protected edit fully exercised.

### Why the recurrent can't bridge it (the bootstrap + overlap problem)

- **Sparse-distinct ⇒ recurrent can't store.** At ~5% sparsity (0.18 spk/step), almost no CA3 cells are co-active in the same timestep → STDP on `ca3→ca3` has nearly nothing to potentiate → the basin never forms (weight stays ~0.24) → no amplification.
- **Push drive up so cells co-fire ⇒ basins overlap.** A stronger storage ensemble = more cells = more overlap across the 6 locations in a 400-cell pool (6 × ~70 cells ≫ 400) → the recurrent recruits shared cells → distinctness collapses (diff-cos → 0.5–0.9). With strong-store/sparse-recall the recurrent contribution rises to ~70%, but distinctness still degrades to 0.5–0.6.
- **The Izhikevich/point-neuron recurrent is bistable**, not gradedly settable: stronger inhibitory feedback (the de Almeida E%-max k-WTA) flips the system between weak-distinct and a 200–300 spk/step saturated runaway, with no stable sparse-graded middle (matching the C1 diagnosis). The `nmda_slow` conductance (Wang graded attractor, with Mg²⁺ self-limiting) shifts the recurrent contribution up but does not create a sparse-AND-distinct-AND-high-rate fixed point.

---

## What was built (runner-side ONLY; `sim/` byte-empty)

`research/runners/learned_graded_ca3_derisk.py` — the existing harness, with the DIRECT-CA3 fix + the instrumentation guard added:
1. **`--direct-ca3` (default ON):** a FIXED (`plastic=False`) `landmark_sensors → ca3` AMPA detonator (weight 20, density 0.3, jitter 0.6 at intensity 450 → CA3 ~5% sparse, diff-cos ~0.13 — the Stage-1 mechanism). FIXED so the global `stdp_w_max` can stay LOW (the recurrent's runaway ceiling, Step A) without the soft-bound collapsing this strong AMPA weight (the documented `stdp_w_max` gotcha), AND so the recurrent is the ONLY learned structure → the cleanest possible recurrent-ablation anti-cheat. The `ec/dg/dg_pv_basket` regions remain wired but CA3 no longer DEPENDS on the silent multi-hop.
2. **The instrumentation guard:** `_store_recurrent` now accumulates CA3 spikes across the store-phase clamp windows and returns the CA3 spk/step DURING storage; the caller prints it ALWAYS (`*** CA3-FIRES-DURING-STORAGE GUARD: X spk/step ***`) and HARD-ASSERTS it `> 0` on the non-ablated run. A silent-CA3 run is caught immediately, not after the fact.
3. New CLI: `--no-direct-ca3` (reproduce the bug), `--lm-to-ca3-{weight,density,jitter}`, `--ca3-to-ca1-weight` (CA1 must fire enough to drive the MSN), defaults retuned to the distinct operating point. Summary JSON now carries `gate_pass_counts` + `verdict` + per-seed `ca3_storage_spk_per_step`/`ca3_storage_pop_hz`.

**Anti-cheat ledger (all behave consistently with the negative):**
- **NO host teacher:** the only `cp_external_input_current` write targets `landmark_sensors`; the direct afferent is the brain's own `landmark_sensors→ca3` detonator (the body sensing the world). CA3/CA1/MSN never receive external current. Asserted by construction. ✅
- **GENUINELY-learned recurrent test:** the recurrent-ablation anti-cheat (above) — and it returns the honest stronger negative (the recurrent is not load-bearing). ✅
- **CuPy regime:** `backend=="cupy"` + deterministic knobs OFF, hard-asserted. ✅
- **The silent-CA3 instrumentation guard** is the new structural anti-cheat against ever false-passing on all-zeros again. ✅

---

## Honest bottom line + recommendation

The **harness bug is fixed** (CA3 fires a sparse distinct ensemble per location, confirmed by the storage guard), and the **protected `nmda_slow` edit was genuinely exercised** for the first time. The result is a clean, multiply-confirmed, BRAIN-BASED-ONLY **NEGATIVE**: the `nmda_slow` recurrent does NOT produce a CA1 place code that is simultaneously DISTINCT and HIGH-RATE-enough-to-fire-an-MSN-D1. The fire-vs-grade wall (C1) is irreducible on this point-neuron substrate, and the recurrent-ablation anti-cheat proves the recurrent is not the source of any passing gate (silent at the distinct point; only adds the position-blind reverberation at the dense point).

**This maps the substrate boundary precisely: the blocker is NOT the recurrent's graded dynamics (the `nmda_slow` edit works — it amplifies up to ~70% at the dense point) — it is the point-neuron RATE-CODING wall.** A sparse-distinct CA3 ensemble fires too few spikes/step to drive a downstream cell, and any drive level that fires the downstream cell makes CA3 dense/position-blind. The `nmda_slow` recurrent cannot bridge this because (a) at the sparse point too few cells co-fire to STORE a basin (the bootstrap problem), and (b) basins dense enough to store/ignite necessarily overlap in a finite pool.

**Recommendation — do NOT wire this CA3 stage into the N9 place-grading critic; the place-grading re-read is NOT unblocked by this route.** The remaining faithful levers are the ones C1 already named, and they are genuine `sim/`-level changes (byte-review), not runner tunes:
- **A sharper read-out (C4):** an FS-WTA critic that can win on the *sparse-distinct* CA1/CA3 (diff-cos 0.13) at LOW rate, removing the need for CA3 to supply a high rate it can't give distinctly. C1 and Stage-2 both point here; this is the cheapest next move and it keeps the place code in the distinct regime.
- **BTSP single-trial plateau potentiation (C2)** to carve distinct CA3 basins in one traversal (bypasses the STDP-from-sparse-firing bootstrap) — a protected `sim/` kernel addition.
- **A conductance-based, adaptation-equipped CA3** that can hold a graded sparse ~10–40 Hz attractor (the genuine Marr autoassociator the Izhikevich point neuron can't) — a real `sim/` dynamics change.

The harness + instrumentation guard ship as the re-runnable test bed for whichever is tried next.

---

### Artifacts
- Runner: `research/runners/learned_graded_ca3_derisk.py` (DIRECT-CA3 fix + storage instrumentation guard + `--no-direct-ca3` control; gates G1–G6 + recurrent-ablation anti-cheat + regime/position-leak; CuPy-only).
- CuPy de-risk JSONs (3 seeds 42/43/44): `research/findings/raw/_learned_graded_ca3_directfix_3seed.json` (DISTINCT, recurrent ON — the headline), `_learned_graded_ca3_directfix_ablate_3seed.json` (DISTINCT, recurrent ablated), `_learned_graded_ca3_densepoint_3seed.json` (DENSE, recurrent ON), `_learned_graded_ca3_densepoint_ablate_3seed.json` (DENSE, recurrent ablated).
- `git status --short sim/` byte-empty (verified before AND after; the protected `069d3023` edit untouched).
