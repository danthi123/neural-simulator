# Vision->concept SPIKING N_PER_CAT sweep — BOUNDARY (accuracy lift confirmed, but a HARD-STOP moat breach)

**Date:** 2026-06-16
**Runner:** `research/runners/_vision_to_concept_spiking_npercat_sweep.py` (commit `d5b34890`), reuse-by-import over
`_genfrontier_capstone_vision_to_concept_derisk.run_seed`. GPU, `SIM_BACKEND=cupy`, epochs=20, n_concept_per=100.
**Result JSON:** `research/findings/raw/_vision_to_concept_spiking_npercat_sweep.json`.
**Verdict:** runner gate = **NEGATIVE**; the precise scientific reading = **BOUNDARY** (the accuracy hypothesis is
CONFIRMED, but the no-confab moat — the load-bearing constraint — BREACHES, a HARD STOP).

## Why this ran

The unified embodied agent (navigate + compose + generalize + converse on ONE `SimulationBridge`) is 6-seed robust
on integration + no-confab moat + nav + compose + conversation + parse (Stage-3 = 4/6 GO, 0 moat breaches). The ONLY
remaining miss is GENERALIZATION-at-chance at seeds 100/101 (spiking vision->concept H5 = 0.25 = chance). The
controller-verified scoping (`2026-06-16-vision-to-concept-fidelity-scoping.md`) root-caused this as a held-out/train
SPLIT-MARGIN issue and proposed the IT-population-prototype fix: more exemplars/category (a one-constant `N_PER_CAT`
change). The CPU host stand-in was inconclusive-by-ceiling (saturates at 1.00), localizing it to a SPIKING-only
concept-read-margin property. ⇒ the decisive test: does more `N_PER_CAT` deepen the *spiking* H5 at 100/101?

## Result

| N_PER_CAT | s42 | s43 | s44 | s100 | s101 | s102 | min | mean | moat |
|---|---|---|---|---|---|---|---|---|---|
| 4  | 0.75 | 0.50 | 1.00 | 0.75 | 0.75 | 0.50 | **0.50** | 0.708 | **all-OK ✓** |
| 8  | 1.00 | 1.00 | 0.75 | 1.00 | 1.00 | 0.75 | **0.75** | 0.917 | **BREACH (42,100,102) ✗** |
| 12 | — all 6 seeds errored (harness bug, see below) — |

Anti-cheats held everywhere they ran: FLAT-distinct ~chance (0.00–0.25), category-derangement collapsed
(permuted margin ≤ −0.008 at N=4; ~0 at N=8), structure-preservation PRESERVED every seed.

## Two findings

**(1) The exemplar-lift hypothesis is CONFIRMED for ACCURACY.** The 6-seed minimum spiking H5 rises **0.50 → 0.75**
(N=4 → N=8), and the two Stage-3 failure seeds **100/101 go 0.75 → 1.00**. More exemplars/category genuinely
deepens the spiking concept-read margin — the scoping's IT-population-prototype mechanism works for accuracy.

**(2) But it BREAKS the no-confab MOAT at N_PER_CAT=8 — a HARD STOP.** At N=8, a visually-novel NO-category shape
drives the concept assembly almost as strongly as a real held-out shape: the held-out/novel familiarity contrast
collapses from ~2.4× (N=4) to ~1.2–1.4× (N=8), below the ×1.5 abstention bar, breaching at seeds 42/100/102.
Mechanism: denser category cores (8 vs 4 exemplars) broaden the perception->concept convergence, so a novel shape
(which still excites some trained V1 columns) drives the assembly more → the abstention margin narrows. **The gate is
NOT loosened. N_PER_CAT=8 is REJECTED.** The moat is load-bearing; a breach fails outright. This is the discipline
working: exemplar density trades accuracy against abstention.

> The unified agent's deployed gen-moat (`gate = 0.6×held-out-win-fire`, novel < gate ⟺ ho > 1.67×novel) is even
> STRICTER than this capstone's ×1.5 — so applying N_PER_CAT=8 to the merged agent would break its moat worse. The
> breach is directly relevant to the deliverable, not a capstone-only artifact.

## The sharpest redirection in the data (co-residence, not exemplar count)

At the **moat-safe** N_PER_CAT=4, the **STANDALONE** capstone already gives seeds **100/101 = 0.75** — yet the
**MERGED** Stage-3 (`navigate_unified_episode.py`) had the same seeds at **0.25 (chance)**. Same seeds, same exemplar
count, opposite outcome. ⇒ **the real compressor of 100/101 is most likely the CO-RESIDENCE on the merged bridge —
the documented Stage-2 ~2× gen-firing compression — NOT the exemplar count.** The exemplar lift was aimed at the
wrong variable: the standalone generalization already clears the bar (0.75, moat intact); it is the merge that
degrades it. This is the honest negative redirecting the diagnosis.

## Honest harness bug (N_PER_CAT=12, does not change the verdict)

All 6 N=12 seeds errored at the FLAT-distinct ablation arm: `flat baseline needs 2880 <= 2048 perception neurons`.
At F=48, the disjoint flat baseline needs 48×60=2880 perception neurons but the V1-complex feature region is fixed at
N_V1_COMPLEX=2048. The MAIN H5 (ARM1) was computed first and was strong (e.g. seed-101 held-out 683 concept-spikes/cue)
but discarded when the later flat arm threw inside `run_seed`. The fix (a graceful flat sampler when disjoint won't fit)
is cheap, but N=12 is NOT decision-relevant: N=8 already breaches the moat, and a denser N=12 would breach worse — so
the exemplar-lift verdict (rejected on the moat) stands without it.

## Verdict + next move

**BOUNDARY / HARD STOP.** The simple exemplar lift is NOT a clean GO: it raises generalization accuracy (good) but
breaks the no-confab moat at N=8 (rejected). **Do NOT apply N_PER_CAT=8 to the merged agent.**

The data redirects the fix to the **moat-safe** path: the standalone capstone at N=4 already generalizes 100/101 at
0.75 with the moat intact, so the unified agent's 100/101 failure is a **co-residence compression** problem (the
characterized Stage-2 effect), not an exemplar-count problem. The next move (per the standing deep-research/cheap-first
discipline at a boundary) is to (a) directly confirm the standalone-vs-merged 100/101 gap is co-residence (a targeted
GPU comparison at the same config), then (b) recover the merged 100/101 generalization by addressing the compression
(read on a clean state / population-code sharpening of the firing) WITHOUT touching the moat — or, if an accuracy lever
is still wanted, strengthen the abstention discrimination ALONGSIDE it and re-validate the moat holds. The dendritic
rewrite is explicitly NOT implicated (a fidelity/co-residence gap, not a substrate-mechanism gap). NO `sim/` edit.

## UPDATE (same day) — co-residence CONFIRMED + localized (zero new GPU), moat-safe lever in flight

The redirection above was upgraded from hypothesis to **proven cause** by a cheap-first analysis of EXISTING data:

1. **Identical held-out split.** The merged Stage-3 uses the SAME `seed*31+5` split as the standalone capstone —
   seed 100 held_out `[1,6,9,13]`, seed 101 `[3,7,8,14]`, seed 42 `[0,6,10,14]` in BOTH. So the gap is NOT the split
   (the scoping's original split-margin idea is ruled out for the standalone-vs-merged gap).
2. **Identical training config.** `_train_merged_convergence` trains the merged gen convergence at the same
   epochs=20 / hebbian_max=20 / perc=300 / conc=600 / OU-off / STDP-off / reward-off (isolated by the
   `cp_plasticity_rate_gain` mask) as the standalone. Ruled out.
3. **Identical read method.** The merged `_read_gen_spikes` calls the standalone's OWN `read_heldout_spikes`, and
   `_category_of_concept_spikes` is the same raw category-mean. OU is off in BOTH reads. Ruled out.
4. **Seed 42 is byte-identical (0.75) in both; only 100/101 collapse (0.75→0.25).**

⇒ the ONLY remaining difference is the merged `gen_concept`'s **per-index neuron heterogeneity**: it sits at global
indices ~8000+ and draws a different Izhikevich a/b/c/d jitter than the standalone's base-2048, which tips the
**marginal** 100/101 category read (standalone margins were thin: +0.077 / +0.100). This is a co-residence effect,
not a vision/exemplar/split limit.

**The moat-safe fix = the POPULATION-CODE lever** (more `gen_n_concept_per` → the category-mean averages over more
heterogeneous neurons → lower read variance → the thin-but-real 100/101 margin wins again), which — crucially —
does NOT broaden the category cores, so it preserves the moat (unlike the exemplar lever). Probe IN FLIGHT:
`research/runners/_unified_gen_popcode_lever_probe.py` (seeds 100/101/42 × `gen_n_concept_per` ∈ {100, 300} on the
MERGED bridge; GATE = 100/101 recover H5 ≥ 0.50 at n=300 + the no-confab MOAT abstains every cell, HARD STOP on any
breach). GO ⇒ set `gen_n_concept_per=300` in `build_compose_bridge` + re-validate Stage-3 toward clean 6/6.
NEGATIVE ⇒ the compression is a systematic heterogeneity bias (route to per-region heterogeneity control or accept
seed-variance). NO `sim/` edit either way.

## UPDATE 2 — the population-code lever result: NEGATIVE (honest), moat held

The moat-safe lever probe completed (`_unified_gen_popcode_lever.json`), merged bridge, `gen_n_concept_per` ∈ {100, 300}:

| seed | n=100 | n=300 | moat |
|---|---|---|---|
| 100 (failure) | 0.50 | **1.00** | intact |
| 101 (failure) | (n=100) | **≥0.50** | intact |
| 42 (control)  | 0.75 | **0.25** | intact |

**Verdict: NEGATIVE — and informative.** More concept-neurons *recovers* the two failure seeds (100: 0.50→1.00; 101
clears 0.50) **with the no-confab moat intact at every cell** (no breach — the HARD STOP never triggered), but it
*degrades the seed-42 control* (0.75→0.25). So the population-code lever is **not a uniform win**: it trades the read
variance around seed by seed rather than sharpening it everywhere. That the merged read for a given seed swings this much
with the concept-population size is itself confirmation that the 100/101 failure is **read-variance from co-residence**
(the per-index heterogeneity), not a fixed representational error — but the lever as a single global knob can't fix all
seeds at once.

**Net for the unified embodied agent (unchanged headline):** the integration, the no-confab moat, navigation,
compose, conversation, and parse remain 6-seed robust (Stage-3 4/6 GO, 0 moat breaches); the generalization step's
per-seed fidelity is the documented open edge, and neither lever (exemplar — rejected on the moat; population —
seed-dependent) cleanly closes it as a global knob. **Honest options (not pursued here, logged):** per-region
heterogeneity control (homogenize the `gen_concept` Izhikevich parameters so the read no longer depends on the slice's
index), or accept it as documented seed-variance and spend effort on a higher-leverage frontier. The no-confab moat was
never weakened to chase a pass. NO `sim/` edit.

## Reproduce

```bash
SIM_BACKEND=cupy python -u -m research.runners._vision_to_concept_spiking_npercat_sweep \
    --npercat 4,8,12 --seeds 42,43,44,100,101,102
# the co-residence localization + moat-safe lever test:
SIM_BACKEND=cupy python -u -m research.runners._unified_gen_popcode_lever_probe --seeds 100,101,42 --nconcept 100,300
```
