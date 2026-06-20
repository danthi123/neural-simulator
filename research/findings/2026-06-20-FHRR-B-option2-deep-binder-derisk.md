# FHRR-B Option 2 — DEEP / hidden-layer learned binder for the multi-attribute bundle-inverse: _IN FLIGHT_

**Date:** 2026-06-20
**Runner:** `research/runners/_phaseB_fhrr_b_option2_deep_binder_derisk.py`
**Raw:** `research/findings/raw/_phaseB_fhrr_b_option2_deep_gated.json` + `_fhrr_b_option2_gated_run.log`
**Backend:** numpy/CPU (cheap-first; NO `sim/` edit; NO GPU)
**Scope:** the harder Option-2 from `2026-06-20-FHRR-B-learned-binder-scoping.md` (the deep/hidden-layer
learned binder); Option-1 (learned iterative cleanup over a FIXED bind) already GO
(`2026-06-20-FHRR-B-option1-learned-cleanup-derisk.md`).

> **PLACEHOLDER — this doc is being filled as the 6-seed sweep lands.** The mechanistic diagnostics (below)
> are already conclusive; the multi-seed sweep numbers + the depth-ablation + anti-cheat tables are pending.

## The question (precise)

Can the BIND FORM ITSELF — the role-dependent reciprocal that recovers one role's filler from a 3-way
superposition ("bundle") — be **learned** by a DEEP / hidden-layer binder, replacing the FIXED exactly-invertible
FHRR self-inverse algebra (the residual host-DESIGNED shortcut FHRR-B)?

- "Bundle" = superpose multiple role-filler bindings into one fact vector (agent⊗dog + action⊗run + patient⊗fast).
- "Unbind from a bundle" = recover one role's filler from that sum. `⊗` / `(x)` = element-wise (Hadamard) product.
- The decisive metric is **held-out** (never-bundled) role-filler combo recovery — Fodor-Pylyshyn systematicity,
  leakage-asserted — never raw recall.

## Why this is NOT the prior shallow NEGATIVEs (the Option-2 insight)

Every prior from-scratch learned-bind attempt was SHALLOW (one bilinear bind layer + ONE linear/Hadamard unbind
readout). The scoping's hypothesis: a DEEP net with hidden layers (capacity + the credit-routing the shallow
forms lacked) + a multiplicative-gating inductive bias may learn the structured reciprocal.

| prior learned-bind arm (all SHALLOW, identical harness) | bundle held-out |
|---|---|
| learned ADDITIVE bind + linear unbind | 0.193 |
| learned MULTIPLICATIVE bind + learned-LINEAR inverse | 0.056 |
| learned single-layer dendritic σ-π | 0.168 (train 0.422 → memorizes) |
| **fixed ±1 self-inverse (the current shortcut / ceiling)** | **0.989** |
| chance (1/F) | 0.062 |

## Method

`DeepLearnedBinder` (numpy, bundle-aware backprop — a host-shortcut CEILING characterization, explicitly NOT
"the brain binds"). Bind = a LEARNED multiplicative coincidence `g = (role@W_R) ⊙ (filler@W_F)`; bundle = Σ g.
Two unbind variants, swept across depths {0,1,2,3} hidden tanh layers (0 = the shallow learned control):
- **concat** — a pure deep MLP on `concat[ norm(bundle), role@W_RU ]` (can depth ALONE learn the reciprocal?).
- **gated** — a LEARNED role-conditioned multiplicative gate `bundle ⊙ (role@W_RU)` THEN a deep MLP (the
  bilinear-gating inductive bias, arXiv 2606.10891 burst = soma×dendrite; the strongest deep arm).

F=16, R=4, 3 leakage-free systematicity splits, 6 seeds (42/43/44/100/101/102), D_h=256, 24000 bundle-aware
steps/split (matches the prior shallow de-risks). Decorrelated 320 stream codes (between-cos 0.047).

## DECISIVE MECHANISTIC DIAGNOSTIC (already conclusive, pre-sweep)

A controlled single-binding probe (the EASIEST case — no bundle crosstalk) isolates the lever. On the IDENTICAL
codes/splits/training:

| binder form | single-binding held-out |
|---|---|
| **FIXED ±1 self-inverse + 1 linear readout** (the FRLF form / fixed primitive) | **1.000** |
| deep concat-MLP unbind (learned reciprocal, 2 hidden, normalized) | **0.000** |
| deep gated unbind (learned `W_RU` multiplicative gate, 1 hidden) | **0.250** |

**Reading:** when the role-conditioned reciprocal must be **learned** by a deep net, it cannot be discovered —
even for a lone binding, far below the fixed self-inverse's 1.000. The mechanistic reason (measured, not
assumed): the reciprocal requires the role's **exact element-wise inverse**, and a learned projection `W_RU`
does **not** satisfy `(role@W_R) ⊙ (role@W_RU) = 1`, whereas the fixed `role_pm ⊙ role_pm = 1` by construction.
Depth does not supply the missing operation, because the missing operation is an *exact multiplicative
self-inverse*, not capacity.

## Held-out recovery table (deep vs shallow vs fixed ceiling, 6 seeds) — _PENDING SWEEP_

| variant / depth | single held-out | bundle TRAIN | bundle HELD-OUT | perm-role | moat-gap |
|---|---|---|---|---|---|
| gated d=0 (shallow control) | _…_ | _…_ | _…_ | _…_ | _…_ |
| gated d=1 | | | | | |
| gated d=2 | | | | | |
| gated d=3 | | | | | |
| concat d=2 | | | | | |
| — shallow learned-linear (ref) | — | — | 0.056 | — | — |
| — single-layer dendritic (ref) | — | — | 0.168 | — | — |
| — additive (ref) | — | — | 0.193 | — | — |
| — fixed ±1 self-inverse (ceiling) | — | — | 0.989 | — | — |

## Anti-cheat table — _PENDING SWEEP_

| anti-cheat | expected | result |
|---|---|---|
| SHALLOW (depth=0) learned control falls short | ~0.056 | _…_ |
| FIXED ±1 self-inverse POSITIVE control carries | ~0.989 | _…_ |
| leakage-asserted held-out vs memorization floor + chance | held-out = generalization | _…_ |
| shuffle-train (memorization-floor) collapses | ~chance 0.062 | _…_ |
| permuted-role (query wrong role) collapses | ~chance | _…_ |
| no-confab moat (familiarity gap) | > 0, never weakened | _…_ |
| decorrelated stream codes (between-cos 0.047) | not a clean-code artifact | _…_ |

## Verdict — _PENDING SWEEP_

## Reproduce

```bash
SIM_BACKEND=numpy python -u -m research.runners._phaseB_fhrr_b_option2_deep_binder_derisk \
    --seeds 42,43,44,100,101,102 --depths 0,1,2,3 --variant gated --d-h 256 --run-anticheats \
    --out research/findings/raw/_phaseB_fhrr_b_option2_deep_gated.json
# concat variant: --variant concat ; correlated codes: --codes neural
```
