# Learned filler + fixed self-inverse bind BUNDLES + generalizes — the learned-bind corner closes (2026-06-18, CYCLE 196)

## Headline

The one untested cell of the learned-bind capability map is resolved **GO**: a **FIXED self-inverse
role** (the exact inverse, no learning) combined with a **LEARNED filler embedding** (trained
bundle-aware) **bundles a 3-way superposition AND generalizes to held-out (role, filler) combos**.
A lesion that drops the multiplicative self-inverse collapses it to chance, confirming the product
is load-bearing. This **closes the learned-bind frontier** on a clean, leakage-controlled,
anti-cheated signal: the production design — **learned codes flowing through a fixed
biology-grounded coincidence/multiplicative binding primitive** — is the validated, principled
resting point. The prior bundled NEGATIVEs were about *learning the bind*, not about the codes.

## The capability map, now complete on one harness

All on the identical systematicity harness (stream codes, F=16, leakage-asserted held-out combo
splits, 3 seeds; `cortex_learned_binder_systematicity_probe` protocol):

| binding scheme | bundled held-out-combo | verdict |
|----------------|------------------------|---------|
| chance | 0.062 | — |
| LEARNED additive bind (no inverse) | 0.193 | NEGATIVE (CYCLE 102) |
| LEARNED multiplicative, **learned-LINEAR** inverse | 0.056 | NEGATIVE (CYCLE 103) — a linear map can't be a reciprocal |
| **FIXED self-inverse role + LEARNED filler** | **0.639** | **GO (CYCLE 196)** — single-binding 0.833, 92% of train-combo |
| fixed self-inverse role + fixed-random filler (positive control) | 0.989 | GO (the production-style bind) |
| LESION: fixed role + learned filler, **inverse dropped → sum** | 0.082 ≈ chance | collapses (anti-cheat) |

Per-seed (fixed-role + learned-filler): bundled held-out 0.516 / 0.697 / 0.705; single held-out
0.750 / 0.833 / 0.917.

## What this resolves

The earlier multi-attribute-bundling NEGATIVE (CYCLE 102–103) was **confounded**: it tried to
*learn* the bind/unbind (additive, or a learned linear inverse), and a learned linear map provably
cannot implement the role-dependent reciprocal a superposition unbind needs. Swapping ONLY the
binding op to a **fixed self-inverse role** (the exact inverse, free) while keeping the filler
embedding LEARNED isolates the question: is a *learned filler representation* compatible with the
fixed bind, or does bundle-aware learning collapse the near-orthogonality bundling needs?

Answer: **fully compatible.** Learning W_F (and the readout W_O) bundle-aware preserves bundling
(0.639, 92% of train-combo) AND generalizes single bindings (0.833). So the bind must be a **fixed
self-inverse multiplicative primitive**; the **fillers may be learned** — exactly the production
composer's architecture (the FHRR resonate-and-fire coincidence bind on the learned/grounded stream
codes, V=320 GO). This de-risk validates that design choice on a measured, anti-cheated signal.

## Anti-cheat controls (all passed)

- **Identical harness** to the additive (0.193) + learned-linear (0.056) NEGATIVE arms — same
  corpus, splits, seeds — so the contrast is apples-to-apples (mirroring the established
  0.989-vs-0.193 contrast).
- **Held-out is a leakage-asserted combo split** (`make_systematicity_splits`) vs the memorization
  floor — generalization, not lookup.
- **LESION** (replace the ±1 self-inverse unbind with a plain sum, dropping the multiplicative
  inverse) collapses bundled held-out to 0.082 ≈ chance — the multiplicative self-inverse is the
  load-bearing lever, not the readout.

## Honest scope

This **validates + closes** the learned-bind corner; it does NOT claim a new from-scratch
capability. The fixed multiplicative self-inverse bind was always known to bundle (0.989 control);
the new, decisive bit is that a **learned filler embedding is compatible** with it (and the learned
binds are confirmed to fail for the *bind-learning* reason, not a code reason). The genuinely deep
remaining frontier is a *different* problem — apical-basal credit assignment on the two-compartment
dendritic substrate — not the bind, which is settled. The production composer's bind is the right
resting point; "learn the codes, fix the coincidence bind" is the validated principle.

## Reproduce

```bash
SIM_BACKEND=numpy python -u -m research.runners._phaseB_fixed_role_learned_filler_bundled_derisk
```
Runner: `research/runners/_phaseB_fixed_role_learned_filler_bundled_derisk.py`. Scoping:
`2026-06-18-step3-dendritic-learned-bind-frontier-scoping.md`.
