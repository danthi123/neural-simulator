# EMERGE-7 (rung-3 Fork-2) — the held-out next-symbol test was grokking-hard (mis-designed, NOT a mechanism verdict), BUT the within-distribution result is the 3rd independent confirmation that a naive LOCAL recurrent credit rule DEGRADES a random reservoir → the cheap Fork-2 is a false economy; the real rung-3 lever is Predictive Alignment (Fork-1). The two forks have MERGED.

**2026-07-02 (autonomous; full cores, gaming done).** Runner `research/runners/_emerge7_nextsymbol_context_derisk.py`; result `research/findings/raw/_emerge7_nextsymbol_context.json`; probes `research/findings/raw/_fork2_predesign_*.py` + `_fork2_modadd_gradient_oracle.py`. Reuse-by-import (confirmed Task-A credit); NO `sim/` edit; CPU `SIM_BACKEND=numpy`; multi-seed 42/43/44.

## Why this ran
Rung-3a re-localized the wall to autonomous-generation stability (credit quality was solved). The research gate + verify+design workflow recommended **Fork-2**: reframe rung-3 to discrete high-order **next-symbol prediction** (teacher-forced-by-construction → the free-run wall is structurally absent), as the cheap-first, communication-apt move. My controller pre-design scratch de-risk (`2026-07-02-fork2-predesign-scratch-derisk-reservoir-is-the-bar.md`) had already flagged the load-bearing risk: a **fixed random reservoir + trained readout is a high bar**, and naive local recurrent credit underperformed it. EMERGE-7 built the airtight de-risk (combinatorial systematic-routing task `g=(p+m) mod n_suffix` with held-out `(p,m)` cells; divergent-position-only scoring; order-`middle_len` Markov floor + fixed-reservoir lesion + clean-readout + oracle controls; the confirmed target-based `(s*−h)` credit reused verbatim, `s*` = a fixed-random next-symbol embedding — fully local, no transport, no BPTT, no free-run; `used_transpose` False all arms/seeds).

## Results (N=200, 800 epochs, lr 0.3, κ 0.9, 3 held-splits, seeds 42/43/44; divergent-position accuracy)

| arm | train-div | held-div |
|---|---|---|
| seqB_lesion (**FIXED RESERVOIR** + trained readout) | **1.000** | 0.000 |
| seqB_microcircuit (forward eligibility, trains recurrent) | 0.648 | 0.000 |
| seqB_eprop (e-prop eligibility, trains recurrent) | 0.398 | 0.056 |
| seqB_hebbian / seqB_wrong | 0.259 | 0.222 |
| seqB_null (recurrent frozen, readout frozen path) | 1.000 | 0.000 |
| seqB_untrained | 0.000 | 0.000 |
| **modadd gradient oracle** (factor-access backprop MLP) | **1.000** | **0.000** |
| systematic-rule oracle (knows g=(p+m)) | — | 1.000 |

chance = 0.250; 12 train cells / 4 held cells; V=13.

## Finding A — the held-out test is grokking-hard (mis-designed as a mechanism test)
A 2-layer backprop MLP given the factors `(p,m)` **directly** (no context-carrying required) fits train at 1.000 but generalizes the held-out diagonal at **0.000** across 3 seeds (below chance). So the held-out `(p,m)→g` split is **not gradient-learnable at this scale by ANY method** — a small-modular-addition non-generalization (grokking) artifact: removing the diagonal leaves a systematic hole the model can memorize without learning the cyclic-group structure. ⇒ **every held-div number in EMERGE-7 is uninformative about the local credit rule.** The runner now auto-flags this (`held_learnable` gate on the gradient oracle → `TASK-MISDESIGNED` verdict). The combinatorial-generalization design over-reached: to defeat the reservoir it demanded systematic-rule extrapolation, which is grokking-hard.

## Finding B — the robust, build-informative within-distribution result (3rd independent confirmation)
On the TRAINED cells (within-distribution high-order context — carry the cue across the shared middle to the branch):
- A **fixed random reservoir + trained readout MEMORIZES the high-order-context routings perfectly (train 1.000).** So carrying high-order context on seen sequences does NOT require recurrent credit — the random reservoir + a local readout suffices.
- **Training the recurrent weights with either confirmed local credit rule DEGRADES this fit** — forward eligibility 0.648, e-prop 0.398, both well below the fixed reservoir's 1.000. The local recurrent credit **hurts** the useful reservoir dynamics rather than improving them.

This is the **third independent confirmation** of the same pattern: (1) rung-3a (naive target-based recurrent credit — one-step map fine, autonomous recall dead); (2) the pre-design scratch RFLO probe (naive local recurrent credit underperforms a fixed reservoir under noise); (3) EMERGE-7 (local recurrent credit degrades a reservoir's within-distribution memorization). **A naive local recurrent credit rule does not beat — and typically degrades — a random recurrent reservoir + trained readout on these small rate-recurrent tasks.** The useful recurrent dynamics come *free* from the random init; the value of TRAINING the recurrent weights with a naive local rule is not demonstrated.

## Conclusion — the cheap Fork-2 is a false economy; Fork-1 and Fork-2 MERGE onto Predictive Alignment
The research gate's Fork-2-first recommendation has now been empirically tested (as the pre-design scratch predicted): **"our local rule predicts next symbols" is not a meaningful rung-3 milestone** — a reservoir + readout already does it on seen data, for free, and naive local recurrent training degrades it. For recurrent credit to be *load-bearing*, a rule must **beat the reservoir in a regime the reservoir fails** — the natural one is **noise-robustness** (the pre-design scratch showed a chaotic reservoir degrades under state noise: 0.90→0.66), which is exactly what the careful chaos-taming rules address. So:
- **The real rung-3 lever is Predictive Alignment (Asabuki-Clopath 2025, Nat Commun 16:6784, fully-local + online + spiking-LIF)** — the scoped Fork-1 mechanism — tested on **noise-robustness** (does a PA-trained recurrent net stay robust where a fixed reservoir degrades?), not on grokking-hard held-out generalization.
- **Fork-1 and Fork-2 have merged:** use a teacher-forced next-symbol / trajectory task (no free-run) as the harness, but the mechanism under test is PA (the rule designed to shape recurrent dynamics stably), judged against the fixed-reservoir baseline under noise. This is cheaper and more rigorous than either fork alone.

**NEXT (EMERGE-8): build Predictive Alignment as the recurrent-training rule** (the research gate is already done — PA is the scoped, citation-verified mechanism; no new gate needed, this composes an already-researched local rule). GO = a PA-trained recurrent net beats the fixed-reservoir baseline under noise (multi-seed, anti-cheats). Do NOT start the `sim/` rung-4 port until a rung-3 GO.

## Honest scope / caveats
- The held-out combinatorial design was a mis-step (grokking-hard); the honest deliverable is Finding B (within-distribution) + the disambiguation (the gradient oracle proved the held-out test unlearnable). This is the cheap-first de-risk doing its job — catching a mis-designed metric before a `sim/` build, not after.
- Finding B rests on a naive local rule; a *careful* rule (PA) may genuinely beat the reservoir — that is precisely the untested EMERGE-8 lever, not foreclosed here.
- All arms fully local (`used_transpose` False every seed); the gradient/systematic oracles are clearly-labeled NON-shipped upper bounds for disambiguation only.

## Artifacts
`research/runners/_emerge7_nextsymbol_context_derisk.py` (+ `make_seqB_task`, `SeqBNet` target-based credit, `markov_divergent_acc`, `modadd_gradient_oracle`, TASK-MISDESIGNED verdict branch), `research/findings/raw/_emerge7_nextsymbol_context.json`, `research/findings/raw/_fork2_modadd_gradient_oracle.py` + `_fork2_predesign_*.py`. Prior: `2026-07-02-fork2-predesign-scratch-derisk-reservoir-is-the-bar.md`, `2026-07-02-rung3-generation-stability-mechanisms-scoping.md`, `2026-07-02-emerge6b-rung3a-eprop-eligibility-relocalizes-wall-to-generation-stability.md`.
