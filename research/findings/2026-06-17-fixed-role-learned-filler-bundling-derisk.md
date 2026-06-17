# Fixed self-inverse role + LEARNED filler codes — does it recover BUNDLED facts? (the dendritic-build gate A/B)

**Date:** 2026-06-17
**Status:** **DONE — 6-seed (42, 43, 44, 100, 101, 102).** Headline question resolves **GO** (the fixed self-inverse role + LEARNED filler recovers bundled facts where the learned LINEAR inverse cannot), with one honest caveat (the additive baseline ran hotter than its prior). **Build call: DEFER** the weeks-scale on-bridge binding build — the learned-filler version (0.603) is well below the production composer's fixed-algebra ceiling (0.993), so it is not yet the obvious unlocker; cheap localization first.
**Scope:** cheap-first numpy A/B, CPU, **NO `sim/` edit, NO GPU**. The gate before the project's deepest (dendritic) build, per `2026-06-17-dendritic-multiplicative-binding-scoping.md` §4 ("#1 cheap-first A/B").
**Runner:** `research/runners/_phaseB_fixed_role_learned_filler_bundling_derisk.py`
**Raw:** `research/findings/raw/_phaseB_fixed_role_learned_filler_bundling.json`

## The question (the one change this A/B isolates)

The conversational bind's open frontier is "step 3": replace the production composer's **fixed exact-inverse VSA algebra** (a principled idealization) with a cortex that **learns** to bind role-filler facts. The capability map (`2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`) placed it precisely: codes + **single-attribute** binding are learnable on the spiking substrate (real-LIF 0.833 = 100% of numpy); but **multi-attribute BUNDLING** (a fact = a superposition of 3 role-filler bindings, unbind one role to recover its filler) is **NEGATIVE for any naive learned bind**:

| Bind | Single-attribute | 3-way bundle (a fact) |
|---|---|---|
| **Fixed ±1 / FHRR algebra** (self-inverse role) | 1.000 | **0.989** |
| **Learned additive** (point-neuron) | 0.806 → real-LIF 0.833 | **0.193** |
| **Learned multiplicative + learned LINEAR inverse** | 0.083 (broken) | **0.056** (broken) |
| chance (1/F, F=16) | 0.062 | 0.062 |

**The confound this A/B resolves:** the learned-multiplicative arm tried to **LEARN a linear map `W_Rinv`** as the superposition-unbind inverse — but a linear map provably cannot be the reciprocal/self-inverse a superposition-unbind needs (the role-dependent `1/u` multiplication a summing soma lacks). It collapsed *even single-attribute* (0.056) — broken, **not** "multiplication doesn't help."

**The one change (the genuinely-untested middle):** use a **FIXED self-inverse role** (the ±1 hypervector the production composer already uses — its own inverse under the elementwise product) for the bind/unbind, but **LEARN the filler codes** (`W_F`) and the cleanup read-out (`W_O`), trained bundle-aware on the SAME data. This is the un-explored point between "everything learned-linear" (0.056, broken) and "everything fixed ±1" (0.989, the composer's known operating point).

> **Does a fixed self-inverse role + LEARNED filler codes recover bundled multi-attribute facts where the learned-linear inverse could not — AND still generalize systematically to held-out (role, filler) combinations?**

## Method — a clean 4-arm A/B on IDENTICAL data

All four arms run on the **same** cached 320 stream codes (`_phaseB_stream_codes_320_seed42.npy`, the brain's own learned-from-conversation codes), the **same** leakage-free systematicity splits (`make_systematicity_splits`, R=4, F=16, 3 splits/seed), the **same** SVO bundled-fact eval (roles 0,1,2 = agent/verb/object; `N_EVAL_FACTS=40`), the **same** bundle-aware training budget (`N_FACT_STEPS=24000`), and the **same** cleanup codebook (nearest original filler by cosine). The only thing that changes is the bind/unbind op:

| # | Arm | bind / unbind | trained? | prior |
|---|---|---|---|---|
| **1** | **FIXED-ROLE + LEARNED-FILLER** (NEW) | `bind = role_pm1 ⊙ (filler @ W_F)`; `unbind = (bundle ⊙ role_pm1) @ W_O` | `W_F`, `W_O` learned bundle-aware | — |
| 2 | learned-linear (`MultFHRRBinder`) | `bind = (role@W_R) ⊙ (filler@W_F)`; `unbind = (bundle ⊙ role@W_Rinv) @ W_O` | all learned (linear inverse) | 0.056 |
| 3 | additive (`OnOffRateBinder`) | ON/OFF opponency additive bind, linear unbind | learned | 0.193 |
| 4 | fixed ±1 FHRR | ±1 self-inverse role, projected fillers | none (fixed both sides) | 0.989 |

`role_pm1[r] ∈ {±1}^D_h` is a fixed random projection of the role code, binarized (exactly the fixed-FHRR control's construction; its own inverse under `⊙`, NOT trained). Arms 2/3/4 are re-run **verbatim** from their existing harnesses so the A/B lands directly against the established 0.056 / 0.193 / 0.989.

### Anti-cheats (all on identical data, 6 seeds, fractional ≥5/6 bar)
1. **ADDITIVE + LEARNED-LINEAR MUST FALL SHORT** (the headline A/B): on the SAME data they must stay NEGATIVE (<~0.25) while arm 1 clears 0.40 — proving any GO is the fixed-role product supplying superposition the linear/additive bind lacks, not an easier dataset.
2. **FIXED-±1 POSITIVE CONTROL CARRIES** (~0.989): the harness DETECTS working bundling (a NEGATIVE would be real, not a broken harness).
3. **HELD-OUT SYSTEMATICITY** (leakage-free): the GO bar is on **held-out** combos (Fodor-Pylyshyn generalization), not memorization; single-binding held-out reported alongside.
4. **PERMUTED-ROLE**: bind/train with the true role hypervectors, but **unbind** each query role with a *different* role's ±1 vector (a derangement). This genuinely breaks the bind↔unbind correspondence (permuting consistently on BOTH sides self-cancels, since ±1 is its own inverse for any vector — so the unbind-only permutation is the correct control). Recall must collapse to ~chance.
5. **LESION**: replace the unbind product `bundle ⊙ role_pm1` with a plain SUM (drop the multiplicative self-inverse). The bundling lift must collapse to ~the additive value — proving the lift RIDES the multiplicative op.
6. **MOAT (a PLUS, not a hard gate per owner 2026-06-17)**: bind+unbind an ABSENT filler (random unit code) → confidence (max cosine to the codebook) should be clearly below a known filler's. Reported as the known-vs-novel separation gap.

## Results

### 6-seed A/B (seeds 42, 43, 44, 100, 101, 102) — BUNDLED held-out-combo recall

| Arm | bundled held-out (6-seed mean) | per-seed | prior (3-seed) |
|---|---|---|---|
| **FIXED-ROLE + LEARNED-FILLER** (NEW) | **0.603** | [0.502, 0.394, 0.667, 0.606, 0.727, 0.724] | — |
| learned-linear (`MultFHRRBinder`) | **0.069** | [0.167, 0.000, 0.000, 0.125, 0.120, 0.000] | 0.056 |
| additive (`OnOffRateBinder`) | **0.238** | [0.111, 0.167, 0.448, 0.262, 0.188, 0.252] | 0.193 |
| fixed ±1 FHRR (ceiling) | **0.993** | [1.000, 1.000, 0.983, 1.000, 0.975, 1.000] | 0.989 |
| chance (1/F, F=16) | 0.062 | — | 0.062 |

**FR+LF systematicity:** single-binding held-out **0.806** ≈ train-combo **0.788** (held-out generalizes as well as training combinations — Fodor-Pylyshyn, not memorization).

**Controls (6-seed mean):** permuted-role **0.015** (collapses to ~0) · lesion(replace the multiplicative self-inverse product with a plain sum) **0.057** (collapses to ~additive/chance) · moat known **0.850** vs novel **0.501** (gap **+0.349**, all 6 `moat_ok=true`).

**Pass counts:**
- `n_pass` (FR+LF held-out ≥ 0.40): **5/6** (only seed 43 at 0.394 just misses) — clears the ≥5/6 bar.
- FR+LF **>** additive, per-seed: **6/6** (relative separation unanimous, min margin +0.219 at seed 44).
- FR+LF **>** learned-linear, per-seed: **6/6** (the confound the A/B exists to resolve — decisively broken at 0.069 ≈ its 0.056 prior, a ~9× separation from FR+LF).
- `n_beats_baselines` (FR+LF ≥ 0.40 **AND** both baselines absolute < 0.25): **2/6** — dropped by the **additive** arm, which ran hotter than its 0.193 prior (3/6 seeds creep over the absolute 0.25 line; seed 44 = 0.448 is an outlier). The learned-linear arm stays < 0.25 unanimously (6/6).

### Reading it (honest)
- **The headline question resolves GO.** A fixed self-inverse role + LEARNED filler recovers bundled superposition (0.603, 5/6 ≥ 0.40) where a learned LINEAR inverse cannot (0.069). FR+LF beats BOTH baselines at every seed; permuted-role and lesion both collapse (the lift rides the multiplicative self-inverse product); the fixed-±1 positive control carries at 0.993 (the harness genuinely detects working bundling); held-out systematicity holds; the moat separation holds. The confound the A/B was built to resolve — "a learned linear map can't be the reciprocal a superposition-unbind needs" — is confirmed and the genuinely-untested middle (fixed role, learned filler) clears the bar.
- **The one honest caveat:** the additive (ON/OFF opponency) baseline did not stay cleanly below the *absolute* 0.25 line (mean 0.238; 3/6 over). This is expected in kind — additive opponency carries SOME superposition (its 0.193 prior is already 3× chance), it just carries less than the fixed-role product — but it means the strict pre-registered "both baselines absolute < 0.25" is met at only 2/6 seeds. The RELATIVE claim (FR+LF > additive) is unanimous 6/6; the absolute-line blemish is on the baseline, not on FR+LF.

## Verdict

**GO on the scientific question; DEFER the build.**

The mechanism is real and the load-bearing confound is resolved: a fixed self-inverse role with LEARNED filler codes recovers bundled multi-attribute facts (0.603, 5/6 ≥ 0.40, systematicity + moat intact, all controls collapse) where a learned LINEAR inverse provably cannot (0.069). The single honest caveat is on the *additive* baseline's absolute level (mean 0.238, over the 0.25 line at 3/6), not on the FR+LF capability — FR+LF beats additive at every seed.

**But this does NOT trigger the weeks-scale on-bridge build, and the reason is the load-bearing framing in §"What a GO means":** a fixed self-inverse role is *exactly what the production composer already does*, and the fully-fixed algebra bundles at **0.993** here (0.989 prior). The LEARNED-filler version lands at **0.603** — a real lever over the learned-linear/additive baselines, but **~0.39 below the fixed-algebra ceiling**. By the doc's own pre-registered BOUNDARY clause, *"the LEARNED fillers cost accuracy vs the fully-fixed algebra; localize (more capacity / a multiplicative cleanup) before committing the build."* Spending weeks to realize a 0.603 learned-filler bind on the spiking substrate is not justified while the production composer already bundles the same facts at 0.993 with fixed codes.

**Why this is the disciplined call, not a punt:**
- The strategic value of a *learned* binder is **generalization across similar/correlated concepts** (replacing the idealized exact-inverse algebra that demands decorrelated codes) — a DIFFERENT axis from this bundling A/B, and one already carried by the separate PPMI-cortex + cross-modal-Hebbian generalization arc (CLAUDE.md, CYCLE 88+). This A/B tested learned FILLERS through a fixed role; it did not test, and does not need, a months-scale dendritic rewrite.
- Per the owner's standing guidance ("build the dendritic/substrate addition when it becomes the obvious unlocker of things we desire"), a 0.603-vs-0.993 lever is **not yet the obvious unlocker.** The honest next move is the cheap localization the doc itself prescribes.

**Recommended next de-risk (cheap-first, before any build):** a capacity + cleanup sweep on the SAME harness — (a) bind-space dimension `D_h` sweep (64 → 128 → 256) and (b) a multiplicative (vs nearest-cosine) cleanup read-out — to test whether the learned-filler bundling lifts 0.603 → ~0.9 cheaply. If it reaches parity with the fixed algebra, the on-bridge build becomes justified (route the learned fillers through the already-built, guarded `fused_coincidence_plateau` self-inverse primitive). If it plateaus well below 0.9, the fixed FHRR algebra stays the load-bearing bundler and the learned frontier remains the generalization axis, not the bundling axis. Either outcome is a clean, citable result.

## What a GO / NEGATIVE means for committing the dendritic on-bridge build (the precise framing)

- **GO** ⇒ **lifts the LEARNED-CODES boundary**: a fixed self-inverse role + **LEARNED** filler codes recovers bundled superposition where a learned *linear* inverse (and additive) could not. **HONEST FRAMING (load-bearing):** a fixed self-inverse role is *what the production composer already does at 0.989* — so a GO proves the **LEARNED-FILLER** version holds, it is **NOT** "multiplication-from-scratch is new." It justifies the **weeks-scale** (not months-scale) on-bridge spiking realization: route the LEARNED filler codes through the already-built, guarded `fused_coincidence_plateau` self-inverse bind primitive (additive wiring + a binding synapse-mask, the D2-Phase-1-scale protected edit — NOT a new `NeuronModel`). The composer's exact-inverse algebra would then bind **learned** codes through a genuine branch-local product.
- **NEGATIVE** ⇒ the wall is **deeper than the linear-inverse confound**: even the fixed-product bind can't carry LEARNED fillers in superposition. The fixed FHRR algebra (fixed on **both** sides) stays load-bearing for bundling, the learned-bind frontier is **closed** for multi-attribute facts, and this does **NOT** justify the dendritic on-bridge build for this op. (A clean, citable boundary.)
- **BOUNDARY** ⇒ the lever is real (beats learned-linear/additive) but seed-fragile or well below the 0.989 ceiling — the LEARNED fillers cost accuracy vs the fully-fixed algebra; localize (more capacity / a multiplicative cleanup) before committing the build.

## Reproduce

```bash
SIM_BACKEND=numpy python -u -m research.runners._phaseB_fixed_role_learned_filler_bundling_derisk \
    --seeds 42,43,44,100,101,102
```

Reuse-by-import (the systematicity protocol + the three reference binders `MultFHRRBinder` / `OnOffRateBinder` / fixed-FHRR control); cached 320 stream codes; CPU; no GPU; no `sim/` edit.
