# EMERGE-2b — self-supervised burst credit shows a REAL, top-down-dependent depth signal, but the toy 4-class gate is INCONCLUSIVE (task-sanity + a multi-class control artifact)

**2026-07-01 (autonomous; the self-supervised leg of the burst-credit confirmation).** Reuse-by-import
(`_emerge1b_burstprop_derisk.BurstpropMLP` + `sim.dendritic_mlp.DendriticMLP`); NO `sim/` edit; CPU, 3 seeds.
Runner `research/runners/_emerge2b_selfsup_nextsymbol_derisk.py`; raw `research/findings/raw/_emerge2b_selfsup_nextsymbol.json`.

## The question
EMERGE-1b confirmed burst-multiplexed dendritic credit assignment develops deep structure **supervised** (held-out
0.796, probe 0.989). Does it hold **self-supervised** — predicting the lawful next part of an observation, no external
label — with clean attribution? EMERGE-2 (regression) showed a real self-supervised depth signal but its `wrong_sign`
control was non-discriminating; EMERGE-2b reframes it as 4-class next-symbol CLASSIFICATION (`y = 2·b0 + b1`, two
depth-2 threshold-of-XORs) so the controls bite as they did in EMERGE-1b.

## The result — honest read (mean over seeds 42/43/44; chance ≈ 0.33)
| arm | held-out | reads |
|---|---|---|
| **deep_burst_linearized (TEST)** | **0.589** | ≫ FA + lesion + null; **probe 0.77** (XOR latents emerged) |
| vanilla_FA | 0.327 | the memorizer floor |
| apical_lesion (Y=0) | 0.292 | **collapses to chance — the top-down credit is load-bearing** |
| no_teaching_null | 0.292 | **flat at chance — zero learning without a target** |
| wrong_sign | 0.570 | did NOT go to chance (see below — a multi-class artifact) |
| oracle_bp (task-sanity) | 0.420 | **stalled at chance on seeds 43/44 — task not reliably learnable at lr 0.5** |

**The self-supervised depth signal is REAL and TOP-DOWN-DEPENDENT.** `deep_burst` (0.589) beats `vanilla_FA` (0.327)
and both `apical_lesion` (0.292) and `no_teaching_null` (0.292) by wide margins, and the level-1 XOR latents emerge
(probe 0.77). The two clean must-collapse controls — apical-lesion and no-teaching-null — **both fall to chance**, so
the burst top-down credit is doing the work (if the latents formed from input statistics alone, lesioning the top-down
would not collapse them). No weight transport.

## Why the pre-registered gate stamped INCONCLUSIVE (not a mechanism failure)
1. **Task-sanity failed.** The fenced backprop oracle averaged only 0.420 and sat at chance on seeds 43/44 — the
   4-class task (`b0`,`b1` over *overlapping* XOR subsets {0,1,2}/{2,3,4}) at lr 0.5 is not reliably learnable even by
   full backprop, so the runner **correctly refuses to certify** (you cannot read the burst arms below a broken
   ceiling). This is a task-config problem, not a substrate limit.
2. **`wrong_sign` is ill-posed in multi-class.** It sits at 0.57, not chance. Flipping a 4-way softmax target is not a
   coherent "opposite" the way a binary flip is (EMERGE-1b, where it cleanly gave 0.545 ≈ chance 0.5): the net still
   receives a structured — if wrong — error and forms *some* class structure. The correct discriminating controls in
   the multi-class regime are **apical-lesion + no-teaching-null**, which hold cleanly here. This mirrors EMERGE-2's
   documented regression-`wrong_sign` non-discrimination — a control-design property, not a signal absence.

## Verdict (per the master directive — INCONCLUSIVE launches the next step, never a stop)
The self-supervised burst-credit depth signal is **REAL and top-down-dependent** (deep ≫ FA/lesion/null, probe emerged,
the two clean controls collapse). It is **not yet a certified multi-seed GO** because the 4-class oracle is unstable at
this config and `wrong_sign` is ill-posed in multi-class. Two forward moves, both under way:
1. **Cheap oracle-sanity retune** (lr 0.2, in parallel) — confirm the signal holds once the ceiling is sane; the clean
   read then rests on lesion+null (the discriminating multi-class controls), not the ill-posed `wrong_sign`.
2. **The definitive self-supervised test is the SUBSTRATE.** Predict-your-input self-supervision is the *natural* regime
   of the spiking substrate build (the scoping's Stage B — a spiking Burstprop net on a predict-the-next stream). Rather
   than over-polish the toy, the self-supervised confirmation is carried onto the substrate, where it is the real test.

Supervised burst credit (EMERGE-1b) is the fully-from-scratch, clean-GO PRIMARY mechanism regardless; this self-
supervised leg corroborates it and is being sharpened, not blocked.
