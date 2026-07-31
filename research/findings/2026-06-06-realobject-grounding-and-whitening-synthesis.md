---
type: finding
status: superseded
superseded_by:
  - research/findings/2026-06-06-whitening-computation-spikes-CAN-compute-it.md
  - research/findings/2026-06-06-option1-local-learning-whitening-VALIDATED-6seed.md
date: 2026-06-06
mechanism: whitening
---

# Real-object grounding works (100%) + the whitening boundary is the LOCAL COMPUTATION, not representation — 2026-06-06

> **⚠️ CORRECTION (same day):** the "boundary is the local COMPUTATION of whitening" conclusion below is SUPERSEDED by
> `2026-06-06-whitening-computation-spikes-CAN-compute-it.md`. The computation de-risk shows rate-coded spiking CAN
> compute whitening (with the analytic lateral inhibition + stable leaky dynamics; the membrane averages the rate-noise).
> My first computation attempt's "wall" was an unstable-solver bug, not the spiking. The boundary is NARROWER: LOCAL
> LEARNING of the lateral inhibition. Option 1 is RE-OPENED with a concrete path (the Pehlevan-Chklovskii stable rule).
> The Track A real-object grounding result (100%) below is unaffected and stands.

Owner steered: "start on 2 (real-object grounding) + background deep-research for 1 (the decorrelation blocker)." Both
tracks landed together and converge to a clean, honest, biology-translatable picture.

## Track A (option 2) — deep-semantic grounding in REAL objects: 100%
`research/runners/unified_agent_realobject_grounded.py`: the 200 NOUN codes grounded in REAL object images through the
real V1 Gabor bank (verbs/adjs stay word-grounded — the abstract-concept limit); full unified-agent benchmark, 3 seeds.

| grounding | raw coh (mean/max) | RAW composition | ZCA-decorrelated composition |
|---|---|---|---|
| synthetic stimuli (2026-06-04) | engineered-separable | 66.7% | 100% |
| real handwritten digits (8×8↑32) | 0.322 / 0.988 | 65.0% | (not run to ZCA) |
| **CIFAR-10 natural object photos (32×32)** | 0.249 / **0.968** | **66.7%** | **100% (117/117, 3 seeds)** |

CIFAR ZCA per-category: flat 24/24, **1-attribute 18/18, 2-attribute 15/15**, clause-d1 15/15, clause-d2 9/9, who
18/18, abstain 18/18 — **every category 100%.** The natural-image redundancy is GENUINE (max coherence 0.968 = real
near-duplicate object pairs, not the separability synthetic stimuli engineer away), and the ventral-decorrelation
(ZCA) fully handles it. **Deep-semantic grounding in real objects is validated** — real objects → V1 → decorrelation →
composer composes at full parity with the constructed/synthetic baselines. (CIFAR downloaded with owner authorization,
trusted canonical source, gitignored.)

## Track B (option 1) + the decisive de-risk — the boundary is the LOCAL COMPUTATION of whitening, NOT representation
The deep-research findings (`2026-06-06-decorrelation-blocker-deep-research.md`): the worst-pair limit is a **citable
point-neuron limit** (Mikulasch-Priesemann PNAS 2021: a point neuron + single global inhibitory pool cannot whiten;
dendritic compartmentalization is required); the numpy ZCA is **biologically faithful** (whitening's
variance-equalization is a graded/analog pre-spike op, the retina/LGN stage); the 3 failed local attempts were
WRONG-TARGET (sparsify ≠ variance-equalize); recommended decisive test = the fixed Ω=ΓᵀΓ balanced-net.

**Cheap-first de-risk (`_A_whitening_ratecode_derisk.py`) — SURPRISING POSITIVE:** can rate-coded spiking even HOLD a
whitened code? Whitened codes are SIGNED, rate codes are non-negative → carry as ON/OFF, recover by the ON−OFF
subtraction (the project's opponency operation). Prediction (per the opponency wall): re-correlation. RESULT (3 seeds):
**spikes HOLD the whitening at every integration window** — RAW coh max 0.96 → analytic-whitened 0.032 → rate-coded
ON/OFF estimate max coh **0.13 at window=5, 0.06 at window=20, 0.036 at window=2000** (never re-correlates toward raw).
**Why the opponency wall does NOT apply here:** opponency is a SMALL signed difference of a LARGE common mode (low SNR);
whitened codes are variance-EQUALIZED, so the ON/OFF magnitudes are comparable and the subtraction has good SNR.

## The synthesis (both tracks)
1. **The deep-grounding pipeline is fully validated + biologically grounded:** real objects → V1 Gabor → graded
   whitening → composer = 100%; and the graded-whitening stage is biologically FAITHFUL (the retina/LGN efficient-coding
   stage), not a cheat (Track B). Option 2 is essentially DONE for the groundable (visual-noun) subset.
2. **The whitening boundary is precisely relocated:** NOT in grounding (works), NOT in representing whitened codes
   (spikes hold them — the de-risk), but in the **LOCAL on-bridge COMPUTATION of whitening from correlated input** — a
   point-neuron limit (dendritic compartments needed; no local learning rule reaches it). The whitening must be
   computed by an upstream graded stage (faithful) or handed-in analytic wiring (Ω=ΓᵀΓ = ZCA-as-wiring).
3. **Honest net:** the worst-pair limit stands as a citable point-neuron boundary, but it is now bounded ABOVE and
   BELOW — spikes hold the solution (de-risk), the grounding+decorrelation pipeline composes at 100% (Track A), and the
   graded whitening that bridges them is biology-faithful. The conversational composition path is grounded end-to-end.

## Remaining confirmation (next cycle, clean fresh context)
The full fixed **Ω=ΓᵀΓ COMPUTATION test**: does the analytic lateral inhibition COMPUTE the whitening in spikes from
RAW correlated input (the opponency-relevant step the representation de-risk did NOT test — subtracting the LARGE
correlated common mode, where SNR is low), vs. the learned local rules that bottom out? Since the representation holds,
this isolates the computation. Construction note (subtle): the project's ZCA decorrelates CONCEPTS (the N×N gram
`_decorrelate`), but concepts are sequential on the substrate — the on-bridge analogue decorrelates the IT-pool DIMS;
the concept↔dim whitening mapping must be pinned before the spiking build (re-read the research doc §1 + §Pehlevan).

## Artifacts
`research/runners/unified_agent_realobject_grounded.py`, `research/findings/raw/_realobject_cifar.json` (100% ZCA),
`research/findings/raw/_A_whitening_ratecode_derisk.py` (representation holds), `_download_cifar.py`. NO sim/ edits.
