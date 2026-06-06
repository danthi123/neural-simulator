# Option-1 on-bridge SPIKING realization of the local whitening rule = BOUNDARY (the LEARNING is stable+bounded on the bridge — the −λM works — but the shared-FS spiking lateral does NOT realize the pairwise whitening, so composition stays at the RAW floor) — 2026-06-06

**Status:** BOUNDARY, gated on COMPOSITION (the agent benchmark), multi-seed (42/43/44), with the no-lateral
baseline + IT-pool/lateral guards that catch false positives. The engineering follow-on to the validated
rate-model result (`2026-06-06-option1-local-learning-whitening-VALIDATED-6seed.md`). NO `sim/` edits.

## The one-line result

The regularized local whitening rule — `ΔM ∝ ⟨y_i y_j⟩ − I − λM` — was realized on the SPIKING bridge
(IT pool driven by the fixed projection of CIFAR-real-object V1 codes; the anti-Hebbian lateral as a plastic
it→fs→it inhibitory loop; homeostasis as the identity diagonal; **`hebbian_weight_decay` as the −λM**). The
−λM did exactly what the task asked — it made the spiking lateral **STABLE and BOUNDED** (the prior
de-grounding attempt's lateral was unstable). **But the on-bridge spiking learned whitening does NOT compose:
26/39 = 66.7% on all 3 seeds = the RAW floor, == the no-lateral baseline.** The shared-FS spiking lateral
implements global gain control, not the pairwise decorrelation M_ij the composing whitening needs.

## The decisive table (K=300, CIFAR real-object grounding, 320 concepts, seeds 42/43/44)

`research/runners/onbridge_spiking_whitening_compose.py` → `_onbridge_spiking_whitening_K300_3seed.json`.

| condition | composition | reading |
|---|---|---|
| RAW grounded (no whitening) | **26/39 = 66.7%** | floor control ✓ (matches the rate model's 66.7%) |
| CONCEPT-whiten (N×N gram; not realizable) | **39/39 = 100%** | target control ✓ (matches the rate model's 100%) |
| rate-model LEARNED (−λM, the validated result) | **39/39 = 100%** | what we are trying to reproduce on the bridge |
| **on-bridge SPIKING learned whitening (−λM)** | **26/39 = 66.7%, all 3 seeds** | **BOUNDARY — == RAW floor** |
| on-bridge NO-lateral baseline (lateral disabled) | 26/39 = 66.7%, all 3 seeds | the learned lateral adds NOTHING |

The controls bracket exactly as the rate model — the harness is valid. The result is unanimous across 3 seeds.

## Guards (every run; the false-positive catchers) — all GREEN, so the BOUNDARY is GENUINE not degenerate

| seed | mean IT active /300 | min active | n_silent /320 | lateral_norm | lateral_max (cap 4) | drive_coh → code_coh |
|---|---|---|---|---|---|---|
| 42 | 85.9 | 64 | **0** | 29.8 | 2.85 (bounded) | 0.486 → 0.328 |
| 43 | 88.7 | 73 | **0** | 37.4 | 3.24 (bounded) | 0.482 → 0.341 |
| 44 | 86.2 | 65 | **0** | 32.8 | 3.10 (bounded) | 0.468 → 0.324 |

- **IT pool is HEALTHY, not silent, not collapsed** (86–89 of 300 active per concept, 0 silent, 0 blown up).
  This is NOT the degenerate-IT false positive the prior bare attempt risked. The codes are real and alive.
- **The lateral LEARNED and is BOUNDED** (norm ~30, max < cap). The −λM regularizer worked: the lateral
  settled at a stable fixed point (the crux the prior de-grounding de-risk lacked — its anti-Hebbian FS
  lateral over-suppressed to silence). **So the LEARNING-instability worry is resolved; the −λM stabilizes it.**
- **The composition is at the floor anyway**, and the no-lateral baseline proves the (small) coherence drop
  0.486→0.33 is the spiking THRESHOLD nonlinearity, not the learned lateral (baseline code_coh ~0.34, same).

## Why it is a BOUNDARY — the mechanism (calibrated against the rate model's composing code)

The composing solution is a GENTLE partial whitening, NOT maximal decorrelation. The rate-model's K=300 codes:

| rate-model code | code_coh mean | code_coh max | composes? |
|---|---|---|---|
| RAW feats | 0.249 | 0.968 | 66.7% |
| LEARNED (−λM, the composing solution) | **0.043** | **0.576** | **100%** |
| DIM-analytic (full C^−1/2 over-whiten) | 0.191 | 0.929 | 66.7% (over-whitens) |
| on-bridge SPIKING (this work) | **0.33** | **0.90** | **66.7%** |

The composing target pulls coherence to **mean 0.043 / max 0.576** — a SPECIFIC, gentle, full-rank
re-coordinatization. The spiking codes sit at **0.33 / 0.90** — actually slightly WORSE than RAW (the random
projection + rectification add structure), and the FS lateral cannot pull them toward the 0.04/0.58 corner.

The reason is structural (the **Mikulasch-Priesemann** limit, the same citable wall the validated arc opened
with): the rate-model M is a **full K×K pairwise** matrix that subtracts the specific correlation between each
pair of dims. The spiking it→fs→it loop with a SHARED FS pool fires to the **SUM** of IT activity → it
delivers **global gain control / competitive normalization**, not targeted pairwise decorrelation. Giving the
FS pool full rank (`n_fs = K = 300`) did not help — at small scale the learned lateral moved coherence by
only ~0.006 whether `n_fs = K` or `n_fs = 2K`. The anti-Hebbian-on-co-firing + shared-inhibition primitive is
the wrong shape for pairwise whitening, independent of its rank or its (now-bounded) magnitude.

## Honest scope of the −λM win (do NOT overclaim, do NOT under-claim)

- **WIN (real):** the −λM weight-decay (the bridge's `hebbian_weight_decay`, gated to the lateral) does
  exactly what the rate model said — it gives the spiking anti-Hebbian lateral a STABLE, BOUNDED fixed point.
  The prior de-grounding attempt's lateral had no fixed point and over-suppressed; this one is well-behaved
  (norm ~30, max < cap, 0 silent concepts across 320×3). The *learnability + stability* of the rule on the
  bridge is **confirmed**.
- **BOUNDARY (real):** a stable bounded SHARED-FS lateral is NOT a pairwise whitening. The composition does
  not lift above the raw floor (66.7%, 3/3 seeds). The numpy rate model's M is full-rank pairwise; the
  spiking primitive available on the bridge (FS lateral inhibition) is rank-/shape-limited to global gain
  control. This is **not** a tuning failure (guards green, lateral learned, bounded, IT alive); it is the
  representational mismatch between the rule's M and the spiking lateral primitive.

## What broke, precisely (so a future build does not repeat it)

1. **A real bridge gotcha, fixed:** the global Hebbian block unconditionally CLIPS every `cp_connections`
   weight to `[hebbian_min, hebbian_max]` each step (the CLAUDE.md soft-bound `w_max` gotcha, in Hebbian
   form) — and a `plastic=False` pathway is NOT protected (`cp_plasticity_rate_gain` inits to 1.0 everywhere;
   only gain=0 protects it). A large fixed projection weight placed in `cp_connections` is therefore collapsed
   to `hebbian_max` within one epoch → silent IT (the first smoke's degenerate false positive, caught by the
   guard). **Fix:** put the fixed projection P in numpy and DRIVE IT directly (more faithful anyway — the
   rate-model's `x` is the recurrence input, applied as drive; M is the only learned weight), leaving only the
   small lateral in `cp_connections` where the clip bounds it as intended.
2. **The λ/η balance is the lateral's fixed point:** too much decay (λ=0.001 at this co-firing rate) → the
   lateral decays to 0 (no learning); λ=0.0002 + η=0.05 → a bounded non-zero lateral (norm ~30). The −λM is a
   genuine knob on the lateral magnitude (λ=0.0005→norm 0.7, λ=0.0001→norm 55) — but no setting makes a
   shared-FS lateral decorrelate pairwise.

## Where this sits in the project's decorrelation arc (converges with prior on-bridge findings)

This is the COMPOSITION-gated, gentle-regime complement to
`2026-05-31-foldiak-learned-decorrelation-BOUNDARY...md` (which found on-bridge learned anti-Hebbian
decorrelation hits a separation-vs-reliability frontier, gated on coherence/reliability). The 2026-06-06
rate-model insight reframed the target — near-orthogonality is the WRONG goal (it over-whitens, → 66.7%); the
composing target is a GENTLE partial whitening (coh ~0.04/0.58). This work tested the spiking lateral in that
correct gentle regime, gated on composition, and it STILL does not compose — because the shared-FS lateral
realizes global normalization, not pairwise whitening. Both findings converge: a single on-bridge spiking
lateral stage (shared inhibitory pool) cannot realize the whitening the composition needs, in either the
maximal-decorrelation or the gentle-partial regime.

## Net for option 1

- **Algorithm level (rate/numpy):** RESOLVED — a regularized local rule learns a composing whitening, 100%,
  6/6 (the prior finding). Unchanged.
- **On-bridge SPIKING realization:** the LEARNING is **stable + bounded** on the bridge (the −λM works, the
  crux), but a **shared-FS spiking lateral does NOT realize the pairwise whitening** → composition stays at
  the RAW floor (66.7%, 3/3 seeds, gated on composition, guards green). The honest limit is the
  **representational mismatch** between the rate-model's full pairwise M and the bridge's shared-inhibitory-
  pool lateral primitive (the Mikulasch-Priesemann wall). A faithful on-bridge pairwise lateral would need a
  different primitive (per-pair recurrent inhibition / a full-rank structured inhibitory layer), or the
  whitening stays upstream (graded retina/LGN) — both are bigger builds, not this drop-in.

## Artifacts

- Runner: `research/runners/onbridge_spiking_whitening_compose.py` (`--signed` ON/OFF variant; `--baseline`
  no-lateral attribution; full guards + controls; composition-gated verdict).
- `research/findings/raw/_onbridge_spiking_whitening_K300_3seed.json` (the decisive 3-seed rectified result).
- `research/findings/raw/_onbridge_spiking_whitening_K300_signed_3seed.json` (the ON/OFF sign-preserving
  variant — localizing sign-loss vs lateral; see the SIGNED section below once filled).
- Reuse-by-import: `unified_agent_realobject_grounded.build_realobject_features` + `run_seed`,
  `unified_agent_visual_grounded._decorrelate`, `unified_agent_benchmark`, `_visual_grounding_probe._v1_matrix`.
