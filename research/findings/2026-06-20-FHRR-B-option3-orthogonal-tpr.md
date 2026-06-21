# FHRR-B Option 3 — orthogonal per-attribute role tags (tensor-product representation): _IN FLIGHT (decisive structural result already in hand)_

**Date:** 2026-06-20
**Runner:** `research/runners/_phaseB_fhrr_b_option3_orthogonal_tpr_derisk.py`
**Raw:** `research/findings/raw/_phaseB_fhrr_b_option3_grid.json` (+ `_grid_run.log`) and
`_phaseB_fhrr_b_option3_anticheat.json` (+ `_anticheat_run.log`)
**Backend:** numpy/CPU (cheap-first; NO `sim/` edit; NO GPU)
**Scope:** Option-3 from `2026-06-20-FHRR-B-learned-binder-scoping.md` — the last learnable-representation path before
the structural-primitive question. Option-1 (learned cleanup over a FIXED bind) is GO; Option-2 (deep/hidden-layer
learned binder) is NEGATIVE (depth cannot discover the exact multiplicative reciprocal).

> **PLACEHOLDER — being filled as the 6-seed grid lands.** The DECISIVE structural result (the fixed-read-out
> `proj` arm) is already conclusive (below); the multi-seed held-out table + the attribute-count/role-dim ablation
> + the anti-cheat battery are pending the two background sweeps. Each result is committed as it lands.

## The question (precise — verify against the scoping + Option-2 doc)

The fixed FHRR/±1 bind BUNDLES a fact = a SUPERPOSITION of role⊗filler bindings; recovering one role's filler
needs the exact reciprocal — Option-2 proved that reciprocal is NOT learnable even by a deep net (the missing
operation is an *exact element-wise self-inverse*, not capacity). **Option-3 SIDESTEPS the reciprocal:** if each
attribute uses a DISTINCT, ORTHOGONAL role subspace (a tensor-product structure), the bindings are SEPARABLE — no
superposition crosstalk within a role's subspace — so recovery is a fixed orthogonal projection onto that
subspace + a cleanup, with no exact inverse anywhere. The decisive metric: HELD-OUT (never-bundled) role-filler
recovery — Fodor-Pylyshyn systematicity, leakage-asserted — NOT raw recall.

**Definitions (plain).** *Bundle* = superpose multiple role-filler bindings into one fact vector
(agent⊗dog + action⊗run + patient⊗fast). *Unbind from a bundle* = recover one role's filler from that sum. `⊗` /
`(x)` = element-wise (Hadamard) product. *Tensor-product representation (TPR, Smolensky 1990)* = bind a role and a
filler by their outer product so that DISTINCT roles occupy DISJOINT / ORTHOGONAL subspaces of the fact vector;
recovering one role's filler is then a fixed orthogonal PROJECTION onto that role's subspace, not a multiplicative
reciprocal.

## Honest framing (stated up front, per the task)

Orthogonal-TPR role tags are themselves a **FIXED structural choice** (a fixed block assignment / an outer-product
structure), NOT a learned bind. So Option-3 tests whether a **DIFFERENT fixed structure** makes multi-attribute
recovery LEARNABLE + GENERALIZING (the fillers' projection + the decomposition cleanup are learned; the
orthogonal role-subspace structure is fixed), NOT whether the BIND ITSELF is learned. Two outcomes:
- **If the LEARNED read-out GOes** → FHRR-B's multi-attribute capability closes via learned codes + a separable
  structure + a learned read-out.
- **If the LEARNED read-out also fails to generalize**, then the evidence converges (Option-1 cleanup GO +
  Option-2 deep-bind NEGATIVE + Option-3 learned-read-out NEGATIVE) on a genuine finding: **the role-filler BIND
  (and the read-out that inverts it) is a FIXED STRUCTURAL neural primitive** (binding-by-coincidence / dendritic
  multiplication / a fixed projection), NOT a learnable host op — a finding to hand the controller, NOT a "closed
  boundary."

## The mechanism

Each of the A roles gets a DISJOINT block of width `d_role` in the fact vector (the orthogonal role basis = a
one-hot-over-blocks role tag). A filler is projected by a **LEARNED shared map** `w = filler @ W_F` (∈ R^{d_role})
and PLACED into role-r's block; the fact `T = concat_r [block_r]`; the bundle = the sum. Because the blocks are
disjoint, block r holds ONLY role-r's filler projection — **ZERO crosstalk** (the whole point of orthogonal
subspaces). Recovery is scored two ways on the SAME bundle:

- **(full) the LEARNED read-out** — a learned MLP decodes the read block back to the full D_in=300-dim filler
  space → nearest filler. (Tests whether a *learned* read-out generalizes.)
- **(proj) the FIXED read-out** — read block r (= `filler @ W_F` exactly, no crosstalk) and pick the nearest
  filler IN BLOCK SPACE (compare to each filler's OWN `W_F` projection). NO learned decode. (Tests whether the
  separable structure ALONE solves the bundle-inverse.)

`W_F` + the learned cleanup are SHARED across blocks (one decoder applied per block) — that sharing is what forces
systematicity. All trained bundle-aware by backprop (a host-shortcut CEILING characterization, per the scoping —
a PASS = "a spiking read-out of this separable form CAN be systematic"; explicitly NOT "the brain binds").

## DECISIVE STRUCTURAL RESULT (already conclusive, MULTI-SEED)

On the IDENTICAL codes/splits, scored on HELD-OUT (never-bundled) bundle bindings. The first complete 6-seed
cell (A=2, d_role=32) confirms the structural split unanimously: the FIXED read-out hits **1.000 on 6/6 seeds**;
the LEARNED decode hits **0/6 seeds ≥0.90** (mean 0.236).

| read-out on the orthogonal-TPR fact | held-out (bundle) | permuted-role (wrong block) |
|---|---|---|
| **(proj) FIXED nearest-block read-out — NO learning** | **1.000 (6/6 seeds ≥0.90)** [A=2 d_role=32] | **0.00–0.02** ≈ chance |
| (full) LEARNED MLP decode → 300-dim filler space | 0.236 mean, 0/6 ≥0.90 [A=2 d_role=32]; 0.488 best single seed [A=3 d_role=128] | ~chance |
| — fixed ±1 self-inverse FHRR (the prior ceiling) | 0.989 | — |
| — shallow learned-additive bind (the prior NEGATIVE) | 0.193 | — |
| chance (1/F) | 0.0625 | — |

**Reading.** The orthogonal-TPR structure **DOES dissolve the bundle-inverse** — but ONLY with a **FIXED**
read-out: a fixed nearest-block projection recovers held-out bundle bindings PERFECTLY (1.000), and the permuted-
role control (read the wrong block) collapses to chance, confirming the orthogonal separation is genuinely
load-bearing. When the read-out is forced to be **LEARNED** (decode back to full filler space), it does NOT
generalize (0.488 at full training; single-binding — the easiest case, no other bindings — is only 0.444, so the
ceiling is the learned read-out's systematicity, not the superposition). This is the converged structural finding
in its sharpest form: **a separable structure + a FIXED read-out solves multi-attribute recovery, but that
read-out is a fixed projection, not a learned op.**

## Held-out recovery — 6-seed grid (CLEAN decorrelated stream codes, between-cos 0.047) — _PENDING SWEEP_

| A | d_role | held-out (LEARNED) per-seed | learned mean | held-out (PROJ/fixed) mean | perm-role | moat-gap |
|---|---|---|---|---|---|---|
| 2 | 32 | _…_ | _…_ | _…_ | _…_ | _…_ |
| 2 | 64 | | | | | |
| 2 | 128 | | | | | |
| 3 | 32 | | | | | |
| 3 | 64 | | | | | |
| 3 | 128 | | | | | |
| 4 | 32 | | | | | |
| 4 | 64 | | | | | |
| 4 | 128 | | | | | |
| — fixed ±1 self-inverse (ceiling) | | — | — | 0.989 | — | — |
| — shallow additive (ref NEGATIVE) | | — | 0.193 | — | — | — |
| — Option-1 learned cleanup (GO ref) | | — | 1.000 | — | — | — |

## Anti-cheat battery — _PENDING SWEEP_

| anti-cheat | expected | result |
|---|---|---|
| shallow learned-ADDITIVE control falls short (live re-run) | ~0.193 | _…_ |
| FIXED ±1 self-inverse POSITIVE control carries | ~0.989 | _…_ |
| leakage-asserted held-out vs memorization floor + chance | held-out = generalization | _…_ |
| shuffle-train (memorization-floor) collapses | ~chance 0.0625 | _…_ |
| permuted-role (read the wrong block) collapses | ~chance | 0.000 (proj) / _…_ (learned) |
| LESION (scramble the learned cleanup) collapses | ~chance | _…_ |
| no-confab moat (familiarity gap) | > 0, never weakened | +0.20 (smoke) / _…_ |
| decorrelated stream codes (between-cos 0.047) | not a clean-code artifact | confirmed (between-cos 0.047) |

## Verdict — _PENDING SWEEP (leaning: LEARNED-read-out NEGATIVE → converged structural-primitive finding)_

## Reproduce

```bash
# Full grid (3 attrs x 3 role-dims x 6 seeds):
SIM_BACKEND=numpy python -u -m research.runners._phaseB_fhrr_b_option3_orthogonal_tpr_derisk \
    --seeds 42,43,44,100,101,102 --attrs 2,3,4 --d-role 32,64,128 --n-hidden 1 \
    --out research/findings/raw/_phaseB_fhrr_b_option3_grid.json
# Anti-cheat battery on a representative cell:
SIM_BACKEND=numpy python -u -m research.runners._phaseB_fhrr_b_option3_orthogonal_tpr_derisk \
    --seeds 42,43,44,100,101,102 --attrs 3 --d-role 64 --run-anticheats \
    --out research/findings/raw/_phaseB_fhrr_b_option3_anticheat.json
# correlated codes: --codes neural
```
