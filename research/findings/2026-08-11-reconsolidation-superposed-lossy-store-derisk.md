---
type: finding
status: contributing
date: 2026-08-11
mechanism: reconsolidation
---

# Reconsolidation on a SUPERPOSED (shared-synapse) store — the lossy / learned-binder path (WALL: reconsolidation)

**Date:** 2026-08-11
**Status:** de-risk PROBE, **1-seed smoke passes the GO bar** at the clean operating point (6-seed sweep is the
confirmation step, command below — NOT yet a multi-seed GO). Reuse-by-import, numpy/CPU, **NO `sim/` edit**.
**Runner:** `research/runners/_reconsolidation_superposed_lossy_derisk.py`
**Raw:** `research/findings/raw/_reconsolidation_superposed_lossy.json`
**Prior art (do not re-derive):** `2026-06-17-reconsolidation-update-derisk-GO.md` (Option A, composer KB, 6/6),
`2026-06-18-emergent-reconsolidation-in-loop-derisk.md` (isolation to K=24, block-major), the Closure-5 on-substrate
PE (`2ed48ce65`), and the scope `2026-06-17-reconsolidation-conversational-memory-scoping.md`.

## The un-tried lever (why this is not a re-derivation)

Reconsolidation is banked GO three ways, but **every prior GO used a BLOCK-MAJOR / list store** — each fact in its
own composite / its own `(1+D)` trigger→readout block. Update-specificity ("correct one fact, leave the rest intact")
was therefore **STRUCTURAL**: the facts are PHYSICALLY SEPARATE, so a rewrite cannot touch a neighbour by
construction (the emergent-isolation finding says so verbatim: *"the store is block-major … so isolation is
STRUCTURAL"*). They got the brain-like update-specificity for free by **NOT superposing**.

The wall's genuinely un-tried residual — the owner-OK'd **"moat-not-hard LOSSY path"** (learned binders / PPMI /
distributed VSA), named as the NEGATIVE branch of the scoping doc: *"in-place fact-correction needs either a
cleaner-code representation (the PPMI/learned-cortex arc) or the synaptic tier directly"* — is a **SUPERPOSED store
where all facts SHARE one distributed trace** (the biologically-real, capacity-efficient regime: real synapses are
shared, not one-slot-per-memory). Here isolation is **not free** — it must survive cross-talk. This probe builds
exactly that store and asks whether PE-gated reconsolidation still holds on it.

## Mechanism (faithful FHRR superposition, on the resonate substrate)

Reuse `RFPhasorComposer`'s codebook + its `_bind` (which runs through `_resonate`). A fact `(a,v,p)` is a
KEY→VALUE association: `key = bind(code[a], code[v])`, `kv = bind(key, code[p])`, and **all facts superpose into ONE
complex bundle** `M = Σ_f kv_f` (a magnitude-carrying Plate-HRR / Gayler-MAP holographic memory — lossy, distributed,
the "learned binder" the wall names). Recover: `rec = conj(key) * M`; matched-filter cleanup over the patient
codebook. Reconsolidation is the **delta-rule** (error-gated Hebbian) analogue of forget+re-store on the
superposition: reactivate → familiarity gate (moat) → `PE = 1 − phase-cos(rec, code[new])`; if `PE ≥ PE_LABILE`,
`M += kv(a,v,new) − kv(a,v,p_est)` where `p_est = cleanup(rec)` is the **reactivated estimate** of the stale filler
(you can only weaken what reactivation surfaces); else re-stabilize (no write). Both gates are calibrated from the
data (same-vs-different PE midpoint; stored-vs-random familiarity midpoint), frozen before the probes.

Four arms on the same corrective utterances: **RECONSOLIDATE** (gated forget+store), **NAIVE-ACCUMULATE**
(`M += kv(new)` always, no forget — the "append" analogue on a shared trace), **OVERWRITE-ALWAYS** (forget+store
regardless of PE — ablate the gate), **DO-NOTHING**.

## Smoke result (seed 42; D=256, K_base=6; D_stress=64, K_stress=18; 5 re-statements)

Numbers in this section are rounded reads of the cited raw `_reconsolidation_superposed_lossy.json`.
<!--derived-->

| check @ clean base cell (D=256, K=6) | result |
|---|---|
| baseline recall (superposed store recovers all 6 before any correction) | 1.000 |
| **REWRITE** — target recovers the corrected patient | ✓ |
| **ISOLATION** — every other fact still recovers its patient (with permuted control) | ✓ (collateral 0) |
| **C1 restabilize** — same-patient re-statement (PE 0.734 < gate 0.830) does NOT write, store intact | ✓ |
| **C2 moat** — never-stored cue ABSTAINS (familiarity below the calibrated abstain gate) | ✓ |
| **C3 lesion** (force PE=0 → no-write) + **permuted** (wrong cue → target intact) | ✓ / ✓ |
| four arms separated: NAIVE-ACCUMULATE others-recall 0.400 vs RECONSOLIDATE 1.000 | ✓ |
| capacity (baseline recall / rewrite / isolation) at K = 6 / 10 / 14 | 1.00 / 1.00 / 1.00 |

**Headline (1-seed smoke passes the GO bar):** PE-gated reconsolidation **works on a superposed / shared-synapse
store** — in-place correction, isolation (with the permuted control), restabilize, moat, lesion all clean at adequate
SNR. This closes the wall's un-tried lossy/distributed regime that the prior GOs sidestepped with structural
isolation. correction PE 0.992 vs re-statement PE 0.734, gate 0.830 (note: the lossy store LIFTS the same-fact PE
off 0 — the block-major list read 0.48 — so the same-vs-different margin is narrower on a shared trace; still
separated at K=6).

## Two pre-registered treatment/control claims — reported honestly whichever way they land

**Claim 1 — does the PE GATE protect neighbours?** (OVERWRITE-ALWAYS vs RECONSOLIDATE). **REFUTED / null.** At the
clean cell both arms do 0 neighbour-damage (`attributable_to` → UNDEFINED: both ~0); at the stress cell the collateral
is 100% present in the control (intrinsic cross-talk, not the writing). On this HRR key-value store the forget+store
perturbation is **well-localized to the corrected key**, so ungated writing does NOT damage neighbours. The PE gate
is NOT the neighbour-protector here — its value is target-restabilize + the moat + bounded write traffic
(5 writes → 0 under re-statements).

**Claim 2 — does the FORGET step protect neighbours?** (NAIVE-ACCUMULATE vs RECONSOLIDATE). **CONFIRMED, 100%
attributable.** NAIVE-ACCUMULATE degrades others-recall to 0.400 (vs 1.000); the forget step removes ALL of it
(`attributable_to` → 100% of the effect, 0% in the control). On a distributed store, accumulating without forgetting
grows the magnitude on one key **unboundedly**, raising the cross-talk floor for EVERY read — the weight-level
manifestation of "contradictory duplicates." **The load-bearing isolator on a shared trace is the FORGET (delta-rule)
step, not the PE gate** — a mechanism the block-major list never needed (a list just holds two entries). This is the
real mechanistic content the superposed regime exposes.

## Boundary characterized — the lossy path has an SNR knee

At D=64/K=18 the store is already broken **before** any correction: baseline recall 0.61, reconsolidation rewrite
FAILS (rewrite=False, isolation=False). Below the SNR floor (a function of the D/K ratio) the single-shot delta-rule
forget/store cannot hold. The moat and restabilize still hold there (they read the familiarity/PE, which stays
separable), but the capability itself needs more than a one-shot write.

## The next mechanism (per THE LAW — the boundary is an undiscovered mechanism, not a stop)

To carry reconsolidation on the lossy store BELOW the SNR knee, the un-tried levers, in order: (1) **iterative
delta-rule cleanup** (Widrow-Hoff / pseudo-inverse re-store that actively drives cross-talk down instead of a
single outer-product write — the Anderson-Kohonen error-correcting associative memory); (2) a **learned decorrelating
code** (the PPMI / learned-binder arc the wall names — whiten the cue codes so superposition cross-talk drops); (3)
the **synaptic-literal tier** (Option B: engram `stimulate_tag` reactivation + `plasticity_window_gate` PE-driven
window + `BridgeMemory.forget`/`consolidate`), which the block-major GO already de-risks the gate logic for.

## Honest scope

The superposition is held as a magnitude-carrying complex bundle; the phase-only RF readout floor (`_store_substrate`
/ the RF magnitude floor) is the follow-on fidelity rung — this probe de-risks the **distributed-store
reconsolidation LOGIC**, not yet its phase-only spiking realization. The bind ops run on the resonate substrate
(`_bind` → `_resonate`); the bundle/unbind/delta-rule are the runner's distributed-memory algebra over the substrate's
own FHRR codes. Nader 2000; Osan-Tort-Amaral 2011 (mismatch-gated attractor update); Sevenster 2013 (PE necessity);
catalog J.27 / J.34; Plate HRR; Anderson-Kohonen error-correcting associative memory.

## Reproduce

```bash
# 1-seed smoke (this doc)
SIM_BACKEND=numpy python -u -m research.runners._reconsolidation_superposed_lossy_derisk --seeds 42

# 6-seed confirmation (the GO-licensing sweep; pass a fresh --out to keep a separate 6-seed artifact)
SIM_BACKEND=numpy python -u -m research.runners._reconsolidation_superposed_lossy_derisk \
    --seeds 42,43,44,100,101,102
```
