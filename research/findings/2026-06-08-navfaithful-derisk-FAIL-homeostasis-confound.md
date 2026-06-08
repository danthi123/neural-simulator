# Deterministic-nav-faithful de-risk — FAIL (4th probe-vs-deployment gap caught: the homeostasis confound)

**Date:** 2026-06-08
**Type:** deterministic-nav-faithful cheap-first de-risk (CPU, no `sim/` edits). The methodology paid off — it caught a confound BEFORE a nav build.
**Predecessors:** `2026-06-08-striatal-value-critic-firing-research.md` (proposed the dense-afferent fix), `2026-06-08-nav-placecritic-calibration-NEGATIVE.md` (the roadblock).

## Verdict: **FAIL** — and the de-risk did its job

The recommended fix (1A: a dedicated dense `vs_place_context` afferent feeding only the critic) does **not** carry the value subtraction under the *full* deterministic-nav regime. The de-risk was designed to replicate deployment exactly and to fail gracefully — it did both, catching the **fourth "probe ≠ deployment" gap** in this arc, this one hiding *inside the diagnostic that motivated the fix*.

## The de-risk result (gates 1-4, multi-seed 42/43/44, OU+conductance-noise+homeostasis OFF, dense dedicated afferent)

| Gate | Result | Detail |
|---|---|---|
| (1) V-learned-spatial | **FAIL 0/3** | V(near) = 0.00 Hz — the critic never fires |
| (2) state-specific RPE | **FAIL 0/3** | gap ≡ 1.00 at every lead (no V → no GABA_B subtraction) |
| (3) location-selective LTP | **FAIL 0/3** | w_near 0.199→0.199 (no growth — critic never fires → no eligibility) |
| (4) actor-not-perturbed | **PASS 3/3** | dedicated afferent does NOT leak onto the actor (Layer 3 clean) |
| anti-cheats (a-e) | **PASS** | population code (Jaccard 0.0); GABA_B lesion; GABA_A-direct fails; deterministic-regime asserted; actor untouched |

## The root cause — the 4th gap (mechanistically pinned)

The afferent diagnostic (`_strio_critic_afferent_diag.py`) that "proved" the dense afferent fires the MSN-D1 at 22-49 Hz with OU off had **`enable_homeostasis=True` silently** — it set `enable_ou_process=False` but never touched homeostasis, and `CoreSimConfig` defaults it True. **Nav sets `enable_homeostasis=False`** (`g11_bg_runner.py:3340`). Decisive isolation (identical wiring, flip homeostasis only):

| 80 dense cells @1500 pA, w=6 | striosome firing |
|---|---|
| homeostasis **ON** (the diagnostic's hidden default) | **59.71 Hz** |
| homeostasis **OFF** (deterministic nav) | **0.00 Hz** |

**Mechanism:** with homeostasis on, the bridge uses `cp_neuron_firing_thresholds` (not the fixed `cp_izh_vpeak`) as the spike threshold (`bridge.py:5562`), and `fused_homeostasis_update` LOWERS the under-active MSN's threshold toward its target rate — so the firing came from *threshold-homeostasis*, not convergent excitation. With homeostasis off, the MSN-D1's KIR2-clamped ~55 mV rest-to-threshold gap (`vr=−80, vt=−25`) is unreachable through the afferent at any tested weight/density/σ. So ALL of the afferent diagnostic's 20-265 Hz numbers were homeostasis-on artifacts; the true homeostasis-off firing is ~0.

## Forensic confirmation — the fix is pinned to ONE flag

Re-flipping homeostasis ON on the *same* de-risk wiring (NOT part of the verdict): V(near) 0.78→1.72 Hz (near/far V ratio **2.06**, V-learned-spatial **True**); w_near 0.199→0.825 vs w_far 0.203 (**4.06×, LTP True** — directly refutes the calibration-NEGATIVE's "LTD" claim). So the *entire chain* (fire → learn → place-graded → GABA_B subtract) works the instant the critic can fire — the only missing piece is a way to fire the MSN critic under `enable_homeostasis=False`.

## What this means (the honest, pinned negative)

The deterministic-nav constraint (specifically **homeostasis-off**) and the MSN critic's ability to fire from a place afferent are in genuine tension. The dense-afferent fix as scoped does NOT clear it. The faithful fix needs the critic to reach a firing range under the deterministic regime — and that is a **protected `sim/` decision**, not a runner-side calibration:

- **Per-region homeostasis / up-state on the critic only** (research option 1C). Biologically defensible: intrinsic homeostatic plasticity / excitability homeostasis is a real mechanism (Desai 1999; Turrigiano) that lets an MSN operate in a firing range, and it is deterministic (an EMA update, no randomness). The forensic test shows it WORKS (fire + learn + place-graded). **Fidelity nuance:** it fires the critic via threshold-homeostasis, not the textbook *convergent-excitation* up-state (B.02) — a real but different mechanism. Surface: a small protected edit (a per-region homeostasis mask, mirroring the per-region NMDA mask `cp_nmda_neuron_mask`), byte-reviewed.
- **Relax determinism** (run homeostasis/OU on globally + more seeds for significance). Cost: loses the strict seed-to-seed reproducibility the nav eval relies on.
- **Bank** the mechanistically-pinned negative + move to the next step-1 nav item. The negative is now decisively pinned — a clean BRAIN-BASED-ONLY deliverable (it maps exactly what the substrate can/can't do: an MSN value critic can't fire from a place afferent under homeostasis-off without an added up-state mechanism). The GABA_B/GIRK substrate win + excellent nav stay banked.

## Methodology note (the real win)

The deterministic-faithful de-risk **caught the 4th gap before a nav build** — exactly its purpose. The recurring lesson is now sharp and actionable: *every* future de-risk in this codebase must explicitly match the deployment's `enable_ou_process` / `enable_conductance_noise` / `enable_homeostasis` / `enable_parameter_heterogeneity` flags, because the default-on background mechanisms (OU, homeostasis) silently rescue results that the deterministic deployment won't. This finding's de-risk hard-asserts all four — a template for the rest of the project.

## Status

`sim/` byte-empty; no protected edits. New (uncommitted): `research/runners/snc_stageb_critic_probe_navfaithful.py` (the deterministic-faithful probe — a reusable template), `research/findings/raw/_navfaithful_derisk_{gabab,gaba_a}.json`. The 6-seed nav A/B was NOT run (the firing gate failed). Decision (owner steer): per-region homeostasis/up-state protected edit vs determinism relaxation vs bank.
