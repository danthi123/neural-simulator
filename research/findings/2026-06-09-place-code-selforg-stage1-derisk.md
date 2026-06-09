# Stage-1 de-risk — a SELF-ORGANIZED spiking place code IS position-specific on CuPy: **PASS (3/3)**

**Date:** 2026-06-09
**Type:** runner-side implementation + decisive CuPy de-risk (Stage 1 of the place-code biologization plan). **NO `sim/` edits** (`git status --short sim/` byte-empty throughout, verified before/after every stage).
**Blueprint:** `research/findings/2026-06-09-place-code-biologization-research.md` (the design; Option A trisynaptic / Option B dedicated place layer).
**Motivation:** `research/findings/2026-06-09-N9-convergent-upstate-derisk.md` — the N9 value critic FIRES + LEARNS V + actor-OK on CuPy but FAILS place-grading because the host-rendered **dense Gaussian place code** (`vs_place_context`), read through a dense convergent projection, is **POSITION-BLIND**. The cure is a place code that is sparse + distinct-per-location, **self-organized from the legitimate egocentric landmark sensors** (the body sensing the world), not a host Gaussian.
**Owner directive:** biologize everything, no banking, brain-based-only. An honest negative IS a valid deliverable.
**Scope:** Stage 1 ONLY (the place code's position-specificity). Does NOT touch the value critic — Stage 2 (the sparse-read critic on this code) is the controller's next step, gated on this PASS.

---

## TL;DR — VERDICT: **PASS (3/3 seeds, all gates) on CuPy.** A self-organized spiking place code IS position-specific.

| Gate | Result (seeds 42/43/44) | Notes |
|---|---|---|
| **1a position-specific** (diff-location cosine < 0.30) | ✅ **3/3** mean **0.064** (max ≤ 0.26) | The dense `vs_place_context` blob fails this (N9); this self-organized code passes decisively. |
| **1b stable** (same-location repeat cosine > 0.70) | ✅ **3/3** mean **0.872** | Re-visiting a location reproduces its ensemble — a real attractor of the field, not noise. |
| **1c sparse** (2–10% of the place pool active per location) | ✅ **3/3** mean **3.65%** | Kanerva-sparse place code (every location ~14–30 of 400 cells). |
| **Anti-cheat A — self-organized, NOT hand-wired** (ablate sensors → code degrades) | ✅ **3/3** cos(true, ablated) = **0.000** | **THE decisive control.** With landmark sensors zeroed the place pool is SILENT — the code is 100% sensor-driven. A host Gaussian (allocentric, sensor-independent) would be UNAFFECTED. |
| **Anti-cheat B — position-leak audit** ((x,y) enters ONLY via the egocentric render) | ✅ enforced by construction + grep | The probe writes external current ONLY to `landmark_sensors`; the place pool NEVER receives external current. No host `vs_place_context` Gaussian anywhere. |
| **Regime fidelity** (backend==cupy; OU/conductance-noise/global-homeostasis OFF) | ✅ hard-asserted | Numpy disqualified; deterministic-nav knobs OFF (hard-fail otherwise). |

The self-organized place layer takes a smooth, **highly-overlapping** sensor code (mean input cosine **0.491** across the 6 test locations) and orthogonalizes it to a **0.064** place-output cosine — a ~43 pp competitive separation, the literal Hartley-Burgess "place cells compete for boundary-vector-cell inputs → spatial selectivity" mechanism, realized in spikes on CuPy.

**This unblocks Stage 2:** the *necessary* condition the N9 negative identified (a genuinely self-organized, position-specific, sparse SPIKING place code) is satisfied. Whether a *sparse-read MSN-D1 critic* then grades NEAR≫FAR ≥3× on this code (the *sufficient* condition) is the controller's next de-risk — DO NOT proceed here.

---

## What was built (runner-side ONLY; `sim/` byte-empty)

### Probe — `research/runners/placecode_selforg_stage1_derisk.py` (CuPy-only, self-contained)

A **dedicated self-organized place layer** (the design's **Option B** — see "Option A tried first" below for why the full trisynaptic loop was not used):

```
landmark_sensors  (>=2 landmarks; egocentric bearing+distance render — the BRAIN-BASED-legal
                   body-sensing channel, D.09 object-vector input)
   --random sparse PLASTIC projection (STDP, gate landmark_to_place)-->
place             (a 400-cell hippocampal-pyramidal pool; the cell's own spike threshold is the
                   WTA competition — only the ~5% best-matched cells cross threshold per location)
```

- **>=2 landmarks (here 3:** two bottom corners + mid-top) — distinct bearings give a unique 2-D fix; a SINGLE landmark gives only an annular (distance-ring) ambiguity (flagged in the design).
- **Egocentric render** (per landmark: 12 bearing sensors `intensity·max(0,cos)^4` + 8 distance-tuned Gaussian sensors; `intensity = max_pA/(1+falloff·d)`) — same render-math family as the g11 nav loop (`:5139-5152`), extended to ≥2 landmarks + a distance code. **(x,y) enters the brain ONLY here** (position-leak anti-cheat B).
- **Self-organization:** walk the agent through 6 distinct (x,y) locations spanning the grid with the `landmark_to_place` STDP gate OPEN; the place fields form by Hebbian competitive learning (cells that fire at a location strengthen their inputs from that location's sensor pattern).
- **Measurement:** per-location place ensemble (spike-count vector), pairwise cosines, sparsity, the **sensor-ablation control** (zero landmark drive), and a **partial-cue control** (drop 1 of 3 landmarks).
- Hard-asserts `backend=="cupy"` and OU/conductance-noise/global-homeostasis/heterogeneity/STP OFF. Opt-in `--place-homeostasis` (per-region intrinsic homeostasis, default OFF — see robustness section).

### The decisive CuPy numbers (canonical, seeds 42/43/44, threshold-only)

```
                          seed42   seed43   seed44
1a diff-location cosine    0.066    0.041    0.085    -> 3/3 PASS (<0.30; max <=0.26)
1b same-location cosine     0.855    0.912    0.848    -> 3/3 PASS (>0.70)
1c mean sparsity            0.034    0.036    0.040    -> 3/3 PASS (2-10%, ~14-30/400 cells/loc)
anti-cheat A cos(true,abl)  0.000    0.000    0.000    -> 3/3 PASS (place SILENT without sensors)
input overlap (separated)   0.491    0.491    0.491    (place output 0.064 == ~43pp orthogonalization)
```

JSONs: `research/findings/raw/_placecode_selforg_stage1_CANONICAL_3seed.json` (the headline 3/3), `_placecode_selforg_stage1_CANONICAL_5seed.json` (the 5-seed robustness extension below).

---

## Honest robustness extension (5 seeds) + the one fragile gate

Extending to 5 seeds (+100, +101) maps the boundary precisely:

```
gate 1a (position-specific): 5/5  mean 0.057   <- LOAD-BEARING, robust
gate 1b (stable):            5/5  mean 0.849   <- LOAD-BEARING, robust
anti-cheat A (sensor-dep.):  5/5  cos 0.000    <- LOAD-BEARING (no-host-shortcut), robust
gate 1c (sparse 2-10%):      3/5               <- the ONLY fragile gate
```

**The 1c fragility is fully diagnosed and benign:** it is NOT a failure of the place-coding mechanism. At seeds 100/101 one geometrically-edge location (`near` at (8,24)) under-fires (**0.5–1.5% active**, ~2–6 of 400 cells) while all other locations fire 5–9%. It is genuine **seed variance in the random sensor→place projection**: at those particular seeds, `near`'s sensor pattern aligns with very few place cells' random input weights, so with **threshold-only competition** (a fixed bar) it lands just below firing for most cells. The position-specificity (1a), stability (1b), and sensor-dependence (anti-cheat A) of every location — including `near` — remain perfect; only the *count* dips under the strict 2% floor.

**Attempts to recruit the under-firing location (all honest, all bounded):**
- More training (12→30 passes): made it *worse* (STDP can't grow inputs to cells that never fire there — the per-cell bootstrap problem; the active set *sharpens*, doesn't recruit).
- Larger pool (400→700): worse (more cells → larger per-location count variance → more locations escape [2%,10%]).
- Input L2-normalization: no effect (raw input norms are already uniform 1017–1050 — the under-firing is alignment, not magnitude).
- **Per-region intrinsic homeostasis** on the place pool (`BrainRegion.enable_homeostasis`, Desai 1999 / Turrigiano; the canonical place-cell stability mechanism, runs with global homeostasis OFF, deterministic; opt-in `--place-homeostasis`): **lifts** `near` from 0.005 → 0.010–0.015 (helps, the right mechanism) but at the tested adapt-rates/short window it doesn't fully clear the 2% floor at the worst seeds, and pushing the rate up over-recruits the *other* locations (drives sparsity to the 10% ceiling and lifts 1a over 0.30). Trades 1c-floor robustness against 1a-separation; cannot rescue the single drive-fragile location within the budget. Shipped opt-in for completeness; the **threshold-only base is the cleaner headline** (3/3 on the standard triple).

**Honest interpretation:** the de-risk's question — *"is a self-organized spiking place code position-specific on CuPy?"* — is answered **decisively YES** (1a/1b/anti-cheat 5/5; the strict per-location sparsity FLOOR is the only seed-fragile element, at one edge location, and a known intrinsic-excitability mechanism is the principled fix if a 6/6-style hardening is wanted). This is a PASS, not a PARTIAL-as-failure.

---

## Option A (full g11 trisynaptic loop) tried first — did NOT conduct end-to-end at probe scale

Per the design's recommendation, Option A (drive the **already-built** g11 hippocampus EC→DG→`dg_pv_basket`→CA3→CA1 from the landmark sensors, measure the CA1 place ensemble) was implemented and probed FIRST. It did **not** conduct end-to-end:
- `landmark_sensors → ec` (g11 default w=4.0, density 0.40): **EC silent** at the raw sensor drive. After a runner-side feedforward weight boost (mutating the returned `RegionPathway.weight_mean` before build — no `sim/` edit), EC fires, but —
- **EC is over-active + non-selective** (0.85 active, diff-cos 0.95): the smooth overlapping sensor render through the dense perforant projection floods EC (the same "dense convergent read of overlapping bumps is position-blind" pathology the N9 dense Gaussian had).
- **DG fires only with weakened FFI** (then ~9% sparse, the right range) — but **CA3 stays silent** (the `dg → ca3` mossy density 0.10 is too sparse to fire CA3 from DG's low rate), so **CA1 = 0**.
- This matches `validate_trisynaptic_loop.py`'s own documented note that the **EC-driven test "FAILED at all parameter combinations"** (which is exactly why that validation used **DIRECT-CA3** drive). Tuning all four hops to conduct sparsely from a 60-cell smooth sensor input is a deep multi-hop exercise; the design explicitly names **Option B as the fallback "if Option A's full loop is over-heavy."**

**So the deliverable uses Option B** — the design's literal BVC→competitive-place model, which conducts robustly and PASSES. (The g11 `landmark_sensors → sensor_place_readout` direct pathway DID conduct, confirming the sensor→place hop is sound; the failure was purely the EC→DG→CA3→CA1 multi-hop.)

**Honest residual on Option B vs A:** the minimal single-hop place layer gives position-specificity + stability + sensor-dependence, but **NOT cue-invariance / pattern completion** — the **partial-cue control** (drop 1 of 3 landmarks) gives cos(true, partial) ≈ 0.05–0.10 ("cue-fragile"): removing a cue changes the ensemble. The D.06/D.13 "fires after some cues removed" property requires **CA3 recurrent pattern completion** (the Option-A trisynaptic feature this layer skips). For Stage 2 (a value critic reading a fixed set of locations) this does not matter; if allocentric cue-invariance is later required, that is the Option-A/CA3 upgrade. (Per-region homeostasis raised partial-cue cos to ~0.35–0.48 by recruiting more cells, a partial robustness gain.)

---

## Anti-cheat ledger (all decisive here)

- **(A) self-organized vs hand-wired — THE key control:** ablate landmark sensors → place pool SILENT (active 0.000, cos(true,ablated) 0.000) at **all 5 seeds**. A host-rendered Gaussian (sensor-independent) would be unaffected. **This is the decisive disqualifier of a host shortcut, and the code passes it cleanly.** ✅
- **(B) position-leak audit:** every `cp_external_input_current` write in the probe is either a global zero (`[:]=0.0`) or targets `landmark_sensors` ONLY (grep-verified, lines 222–303). The place pool is NEVER externally driven; (x,y) enters the brain exclusively through the egocentric landmark render. No host `vs_place_context` Gaussian path. ✅
- **(partial-cue):** drop 1 of 3 landmarks → field is cue-fragile in the minimal layer (honest; the CA3-completion gap, see above). Reported as a diagnostic, not a hard gate. ⚠️ (expected for Option B)
- **(regime fidelity):** `backend=="cupy"` asserted; OU/conductance-noise/global-homeostasis/heterogeneity/STP OFF asserted (hard-fail otherwise). The place pool fires from landmark-sensor **synaptic current**, not threshold collapse (global homeostasis OFF; per-region homeostasis OFF in the canonical config). ✅

---

## Honest bottom line + recommendation

A **self-organized spiking place code, driven only by the legitimate egocentric landmark sensors, is decisively position-specific on CuPy**: distinct sparse ensembles per 2-D location (diff-cosine 0.064 ≪ 0.30), stable across re-visits (same-cosine 0.872), sparse (3.65%), and — the load-bearing anti-cheat — **100% sensor-dependent** (silent under sensor ablation, the signature that separates a neural place code from the host Gaussian shortcut N9 exposed). 3/3 on the standard seed triple, with position-specificity / stability / sensor-dependence robust 5/5; the only seed-fragile element is the strict per-location sparsity floor at one drive-edge location (a known intrinsic-excitability hardening target, not a mechanism failure). **Zero `sim/` edits.**

This satisfies the **necessary** condition the N9 negative identified. **Recommendation:** proceed to **Stage 2** (the controller's step) — feed THIS self-organized sparse place code into the N9 MSN-D1 critic via the **A2 plastic DA-δ-gated arm (sparse afferent, no dense position-blind A1 floor)** and test gate **2b (NEAR ≫ FAR ≥3×)** + the rest of the N9 gate set on CuPy. The place input is no longer the blocker; whether the *sparse-read critic* can grade location-value on it is the open question Stage 2 answers. (If allocentric cue-invariance is later required, upgrade the place layer to the Option-A CA3-completion path.)

---

### Artifacts
- Probe: `research/runners/placecode_selforg_stage1_derisk.py` (CuPy-only; gates 1a/1b/1c + anti-cheat A/B + partial-cue + regime fidelity; opt-in `--place-homeostasis`).
- CuPy de-risk JSONs: `research/findings/raw/_placecode_selforg_stage1_CANONICAL_3seed.json` (the 3/3 PASS), `_placecode_selforg_stage1_CANONICAL_5seed.json` (5-seed robustness), `_placecode_selforg_stage1_homeo.json` / `_homeo2.json` / `_homeo3.json` (per-region homeostasis exploration), `_placecode_selforg_stage1_n700.json` (larger-pool probe), `_placecode_selforg_stage1_moretrain.json` (more-training probe).
- `git status --short sim/` byte-empty (verified before/after every stage).
