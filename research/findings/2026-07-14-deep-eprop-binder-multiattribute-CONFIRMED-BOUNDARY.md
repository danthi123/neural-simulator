# Learned cortical binder (ROADMAP #2) — a DEEP e-prop-trained binder MEMORIZES multi-attribute bundling but does NOT GENERALIZE it: depth is NOT the lever, the wall is SYSTEMATICITY → the dendritic-multiplication substrate (D2) is the honest next lever (3-seed CONFIRMED-BOUNDARY)

**Date:** 2026-07-14
**Runner:** `research/runners/_deep_eprop_binder_bundling_derisk.py` (subagent-built, controller-verified: fit-gate + positive-control + BPTT-ceiling all present) · raw `research/findings/raw/_deep_eprop_binder_bundling.json`. numpy CPU; NO `sim/` edit.
**Status:** CONFIRMED-BOUNDARY — an honest negative that SHARPENS the 2026-06-16 binding boundary and names the exact next lever.

## Why this ran (the SURPASS test of a comfortable boundary)
The conversational-fact bind (a fact = a 3-way superposition of role-filler binds, agent+verb+object) had a mapped boundary (`2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`): single-attribute binding is learnable (0.833 on-bridge) but 3-way BUNDLING was NEGATIVE for a SHALLOW/LINEAR learned bind (additive 0.193), and the verdict was "multiplication is a DENDRITIC operation, not point-neuron." But that only ever tested a shallow/LINEAR unbind. Tonight's validated deep-credit rule (e-prop trains DEEP nonlinear spiking nets end-to-end) is the untested lever: a deep nonlinear net can approximate multiplication (universal approximation), so it might crack the bundling the linear unbind couldn't. This tests it — the SURPASS discipline on a comfortable "it's a point-neuron limit" verdict.

## Result (R=4 roles, F=16 fillers, chance 0.0625; 3 seeds)
| | train-combo bundling | held-out (novel role-filler combos) |
|---|---|---|
| shallow/additive (2026-06-16) | 0.193 | ~0.035 |
| **DEEP e-prop unbind (2-hidden nonlinear)** | **0.903** | **0.002** |
| BPTT ceiling (best-possible credit, same net) | (fits) | **0.001** |
| FIXED ±1 algebra (positive control) | 0.99 | **0.993** |
| chance | | 0.062 |

- **FIT GATE passed** (deep e-prop memorize-small train acc **1.000**) → the e-prop wiring works, so the negative is REAL not a bug.
- **Depth DOES help compute the multiplication for SEEN pairs:** train-combo bundling **0.903 ≫ the additive 0.193** — a deep nonlinear net CAN represent the role-dependent inverse (multiplication) that a shallow linear unbind provably cannot. So my hypothesis (depth cracks the linearity wall) was PARTLY right.
- **BUT it does NOT GENERALIZE:** held-out (novel role-filler combos never trained) collapses to **0.002 ≈ shallow 0.035 ≈ chance 0.062** — and the **BPTT ceiling ALSO fails to generalize (0.001)**, so this is NOT a credit-rule limitation (the best-possible credit fails too). The **FIXED ±1 algebra generalizes for free (0.993 held)**.

## ⇒ honest verdict — the wall is SYSTEMATICITY, and it is STRUCTURAL (not depth)
**Depth is NOT the lever.** A deep nonlinear point-neuron net buys a MEMORIZED per-pair multiplication (great on seen pairs), NOT the fixed algebra's BUILT-IN systematicity (generalizing the bind/unbind to ANY novel role-filler combination for free). This SHARPENS the 2026-06-16 boundary: the wall is not "can't multiply" (a deep net CAN, for seen pairs) but **"can't SYSTEMATICALLY generalize the multiplication to novel combinations"** — and even BPTT can't, so more/better credit is not the answer. The fixed self-inverse algebra's systematicity is a STRUCTURAL property (a role's own inverse works for any filler), not something a from-scratch learned point-neuron mapping recovers.

**The named next lever (not point-neuron depth):** the DENDRITIC-multiplication substrate — the two-compartment D2 neuron already on the bridge realizes multiplication STRUCTURALLY (Mikulasch-Priesemann: the analog/multiplicative interactions live in dendrites, not point-neuron sums). A binder built on dendritic multiplication (fixed self-inverse structure realized by dendritic conjunction + learned filler codes) is the honest path — which is, in the limit, exactly what the production composer already does (learned codes through a fixed coincidence primitive). So: the LEARNED representations (codes, single-attribute bind) are done; the bundling SYSTEMATICITY is a structural/dendritic primitive, not a from-scratch point-neuron learning problem — confirmed now for DEEP nets, not just shallow.

## NEXT
The honest lever is the dendritic (two-compartment) substrate for the binding multiplication — test whether a dendritic-conjunction binder gives systematic (generalizing) bundling where the point-neuron deep net memorizes-but-doesn't-generalize. That is a `sim/`-substrate arc (the D2 two-compartment neuron), fair game per the master directive (biology, not a cheat), and precisely named by this boundary.
