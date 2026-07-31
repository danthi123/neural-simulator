---
type: plan
status: live
date: 2026-07-07
---

# D2 build spec — depth-stability of biologically-plausible deep credit on the spiking substrate (depth-3, plain-FA baseline)

> The next rung after D1 (`2026-07-07-D1-microcircuit-noise-robust-deep-credit-clears-bar-on-spikes.md`). D1 established: the microcircuit interneuron-cancellation rule is BUILT on the substrate (additive/default-off, on-bridge cancellation validated), its clean-error credit clears the depth-2 bar (0.96) and is BATCH-ROBUST where raw Burstprop is batch-fragile; and (correction) raw Burstprop is NOT hard-noise-limited (0.92 @ best batch) — the depth-2 "FA wall" was budget-dependent. **D2 probes the GENUINE feedback-alignment depth wall: does the microcircuit hold at depth-3 where FA depth-instability + the 2025-neuromorphic "depth HURTS" result predict degradation?** Research gate: `2026-07-07-deep-lever-research-gate-spiking-deep-credit.md` (D2). Expect boundary-then-surpass, NOT a clean GO (deep×spiking is where the field's depth-instability lives).

## The question D2 answers (the honest open one D1 deferred)
D1 showed clean-error credit clears depth-2 — but depth-2 is NOT where FA breaks (EMERGE-1's depth-2 "wall" was budget/batch-dependent, per the D1-microcircuit control). The genuine feedback-alignment depth wall is at depth-3+ (per the field: Lillicrap FA degrades with depth; Bartunov 2018; 2025 neuromorphic burstprop depth HURT 97.2→95.9%). So D2 = **depth-3** (`10 → H → H → H → 2`), a Boolean function that provably needs 3 nonlinear layers, with a **plain-FA baseline arm at matched depth/batch**. The load-bearing contrast: does the microcircuit (clean-error credit + M2.6 somatic FF + interneuron cancellation) generalize at depth-3 where **plain-FA degrades toward the memorization floor**? If yes → the microcircuit surpasses the FA depth wall (the deep lever's core claim). If it also degrades → the honest boundary + the surpass search (µPC/EqProp depth-stability ideas: per-layer normalization / the interneuron's homeostatic role / feedback learning).

## The task (depth-3, reuse the EMERGE-1 generator, deepen it)
A depth-3 Boolean composition: e.g. a THRESHOLD over (XORs of XORs) of the 10 input bits — a function whose minimal circuit is 3 nonlinear layers (single/2-layer nets provably can't generalize → held-out at floor). Reuse the EMERGE-1 `make_task` structure, add one composition level (`make_task_d3`). Same held-out split discipline (the tested tuples never in train). VERIFY the oracle (fenced backprop) clears ≥0.80 at depth-3 first (else the task/width/lr is wrong, INCONCLUSIVE — the D1 lr=0.5-destabilizes-oracle lesson).

## Arms (all at matched depth-3, matched BEST batch per the D1 control — batch 32; report the batch sweep too)
| Arm | Credit rule | Expected (hypothesis) |
|---|---|---|
| **oracle** | fenced backprop (weight transport) | ceiling ≥0.80 (validity gate) |
| **plain-FA** | clean error `e_k=φ'·(Yᵀ@e_{k+1})`, fixed-random Y, NO interneuron | the FA depth-wall baseline — degrades at depth-3 (the hypothesis under test) |
| **microcircuit** | plain-FA credit + M2.6 somatic FF + interneuron cancellation (self-pred `W_PI=−Y`) | the test: holds at depth-3? (surpass) or degrades (boundary) |
| **burstprop** | noisy `b=B−Pbar·E` per-unit | batch-fragile (D1); at depth-3 the noise compounds per layer |
| **single-layer / apical-lesion** | no deep credit | memorization floor |

NB: in the NUMPY reference the microcircuit credit == plain-FA credit (the interneuron `W_PI` loop is corroboration-only, inert on the weights) — so at the RATE level microcircuit≈plain-FA. **The microcircuit's depth-3 DIFFERENCE is ON THE SUBSTRATE** (the interneuron physically produces the clean error via cancellation, where a point-neuron spiking layer cannot carry a clean continuous error without it, and where burst-noise compounds over depth). ⇒ D2's decisive arm is the **on-bridge depth-3 spiking net**, not the numpy reference. The numpy plain-FA-vs-oracle arm establishes whether depth-3 has a genuine FA wall AT ALL (if plain-FA also clears depth-3 in numpy, the wall is deeper still and D2 escalates to depth-4).

## Machinery (REUSE — D2 is additive over D1, likely NO new `sim/` edit)
- The `sim/` `enable_bdsp` + `enable_bdsp_microcircuit` path already handles per-layer apical + burst + cancellation — a 3rd layer is just another region slice + another fixed-random apical `RegionPathway`. Confirm the guarded block loops over layers generically (no hard-coded 2-layer assumption).
- Runner: extend `_gnw_d1_spiking_bdsp_derisk.py` `deep = [N_BITS, H, H, H, 2]` + a `--depth {2,3}` flag; add the plain-FA arm (a `FANet` = `BDSPNet` with the clean-error credit + no burst/interneuron, OR `MicrocircuitBDSPNet` with the interneuron loop asserted-inert = the same weights, relabeled "plain-FA" for the RATE arm). The `--rule microcircuit` on-bridge Stage-B at depth-3 is the decisive run.
- Per-layer alignment metric: cos(the FA/microcircuit update, the oracle-backprop update) PER LAYER — the direct depth-stability readout (does layer-1's credit stay oracle-aligned as depth grows, or does alignment decay with distance from the output?).

## Ladder (cheap-first, single-variable, gate each rung)
1. **Rate reference, numpy (hours):** oracle validity at depth-3; plain-FA-vs-oracle (is there a genuine depth-3 FA wall?); microcircuit-vs-burstprop batch sweep at depth-3; per-layer alignment. GATE: oracle ≥0.80; report whether plain-FA degrades vs depth-2.
2. **On-bridge, GPU multi-seed (decisive):** the depth-3 two-compartment spiking microcircuit net on ONE bridge; the interneuron cancellation carrying credit through 3 spiking layers; per-layer on-bridge alignment. GATE (pre-registered): held-out ≥ 0.75 AND > plain-FA + 0.10 AND per-layer-1 alignment does not collapse; anti-cheats hold.

## Pre-registered anti-cheats (same 7 as D1)
fixed-vs-learned feedback (no transport) · permuted-error (→chance) · wrong-sign apical (anti-learns) · apical-lesion (→floor) · no-teaching null (hidden-drift ~0) · oracle-ceiling (≥0.80 else INCONCLUSIVE) · memorization-floor (single-layer). PLUS the D1 lesson: **like-for-like batch** (compare all rules at the SAME batch; report the sweep — never a batch-mismatched headline).

## Honest priors (per the research gate — expect boundary-then-surpass)
- 2025 neuromorphic burstprop: depth HURT (97.2→95.9%) → the microcircuit degrading at depth-3 is a REAL possible outcome, and an HONEST NEGATIVE that localizes the deep lever's residual to depth-stability (not a failure — the map).
- The surpass-if-boundary: µPC/EqProp depth-stability IDEAS (per-layer normalization; the interneuron's homeostatic gain-control role; feedback-weight learning à la Guerguiev 2017) — BORROW the depth-stability insight, do NOT adopt the settling/non-spiking machinery.
- The multi-order SCALE wall stands behind ANY depth-3-on-a-toy GO (3-4 orders below language-scale). D2 is a mechanism de-risk, not a scale claim. NO expensive training.

## Sources
Research gate `2026-07-07-deep-lever-research-gate-spiking-deep-credit.md`; D1 `2026-07-07-D1-microcircuit-*.md`; Lillicrap et al. 2016 (feedback alignment); Bartunov et al. 2018 (FA depth scaling); Sacramento et al. 2018 + Urbanczik-Senn 2014 (microcircuit); Guerguiev-Lillicrap-Richards 2017 eLife (feedback learning); µPC arXiv 2505.13124; Stuck-Naud 2024 (spiking burstprop depth). Runner: `_gnw_d1_spiking_bdsp_derisk.py`, `_gnw_d1_microcircuit_control_probe.py`.
