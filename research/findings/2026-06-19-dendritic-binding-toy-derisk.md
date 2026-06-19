# Learned dendritic-multiplication two-attribute binding — cheap-first toy de-risk: honest NEGATIVE (memorizes, does NOT generalize) (2026-06-19)

**The question (the dendrite's OTHER named job — multiplicative binding):** does a LEARNED dendritic
supralinear (sigma-pi / NMDA-plateau) conjunction binder achieve invertible TWO-attribute binding that
**generalizes** to held-out attribute pairs, where the point / learned-linear baselines hit the documented
K=5 two-attribute boundary? (The prior dendrite test — apical-basal CREDIT ASSIGNMENT for navigation — was
NEGATIVE because a single-layer actor has nothing to route; binding is a different, single-OP multiplicative
computation, so that failure mode does not apply. This tests the genuinely-different binding application.)

**Method (cheap-first, CPU/numpy, NO `sim/` edit):** extended the rigorous existing harness
`_phaseB_multiplicative_bind_bundled_derisk.py` (leakage-free systematicity splits, memorization-floor,
held-out-combo recall, the documented baselines) — swapping its learned-LINEAR unbind for a learned
**dendritic supralinear (sigma-pi/plateau) conjunction** bind+unbind, matched-filter argmax cleanup, cached
320 codes, R=4/F=16, 3 leakage-free splits, 3 seeds. Runner: `research/runners/_phaseB_dendritic_bind_derisk.py`.

## Result (3-seed mean; the DECISIVE metric is held-out generalization, not raw recall)

| arm | single-attr held | two-attr bundle TRAIN | two-attr bundle HELD-OUT |
|---|---|---|---|
| **dendritic sigma-pi (test)** | 0.500 | 0.422 | **0.168**  (train→held gap **+0.254**) |
| lesion = point/additive (must fail) | 0.250 | 0.085 | 0.032 |
| permuted (must fail) | — | — | 0.165 |
| memorization-floor (must ≈ chance) | — | — | 0.000 |
| chance (1/F) | — | — | 0.062 |
| **FHRR fixed ±1 primitive (the production reference / ceiling)** | — | — | **0.261** |

**Verdict: NEGATIVE.** The learned dendritic multiplication **memorizes** the training pairs (bundle-train
0.422) but **does NOT generalize** — held-out-combo recall is **0.168**, far below the 0.40 GO bar, with a
large +0.254 train→held gap (the signature of memorization, the exact confound the held-out split exists to
catch). It is even **below the production fixed FHRR primitive (0.261)** on the same two-attribute held-out
test — so the learned dendrite is not an improvement; it is worse than the fixed primitive it would replace.

**Controls valid (the test is fair):** the lesion (plateau→identity) collapses the dendritic arm to the
point/additive failure (0.032) — so the supralinearity IS load-bearing for the train *fit*; the
memorization-floor scores 0.000 and permuted 0.165 — so the held-out metric is discriminating, not leaky.
The dendrite genuinely fits (memorizes) via its multiplication, but that does not buy generalization.

## Interpretation (honest)

The two-attribute held-out generalization is a REAL wall that NEITHER the production fixed FHRR primitive
(0.261, at the documented K=5 boundary) NOR a learned dendritic multiplication (0.168) lifts. ⇒ **the binding
wall is NOT (only) the missing dendritic multiplication.** The dendrite's native analog op (multiplication)
lets it *memorize* two-attribute bindings, but generalizable two-attribute composition is a deeper problem
(more codes/capacity, a structurally different representation like the F=3 resonator, or a different learning
target — not a dendritic nonlinearity). **Production keeps the fixed ±1 / FHRR primitive** (single-attribute
GO; two-attribute = the standing K=5 boundary).

## ⇒ The dendrite is now THOROUGHLY assessed — both named jobs tested cheap-first, both NEGATIVE

- **(a) learnable multi-attribute BINDING via dendritic multiplication** — NEGATIVE (this doc: memorizes,
  doesn't generalize; worse than the fixed primitive).
- **(b) apical-basal CREDIT ASSIGNMENT** — NEGATIVE (`2026-06-19-dendrite-credit-assignment-toy-stage1.md`:
  single-layer actor, nothing to route).

Two afternoons of CPU/numpy cheap-first tests assessed both dendrite applications and **saved the months-scale
dendritic-substrate build on both premises** — the value of cheap-first de-risk + the BRAIN-BASED-ONLY
honest-negative standard. The dendritic substrate stays in the toolkit but is NOT the unlocker for the
current walls (nav credit-assignment OR conversational two-attribute binding). The forward path is the
conversational-scaling primary via other levers (more codes/capacity, the F=3 resonator for two-attribute,
the WTA biased-competition for multi-referent), not the dendrite.

## Reproduce
```bash
SIM_BACKEND=numpy python -m research.runners._phaseB_dendritic_bind_derisk --seeds 42,43,44
```
Raw: `research/findings/raw/_phaseB_dendritic_bind.json` (gitignored per project convention; all numbers above).
