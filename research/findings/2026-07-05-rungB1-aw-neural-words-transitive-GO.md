# RUNG B-1 / A→W words — the transitive producer speaks EVERY WORD on spikes (not just the order) — **GO** (6-seed)

**Date:** 2026-07-05
**Runner:** `research/runners/_rungB1_aw_neural_words_transitive_derisk.py`
**Test:** `tests/test_rungB1_aw_neural_words_transitive.py`
**Raw:** `research/findings/raw/_rungB1_aw_neural_words_transitive.json`

## Why

The one-brain-substrate capstone (EMERGE-90..95) renders the transitive answer "the dog chases the ball" with the
spiking frame-slot emission ORDER, but the word SURFACES were the host-token spell (`spell=str`). This rung makes the
WORDS spiking too — closing the "order-is-spiking, words-are-host-token" caveat for the transitive capstone.

## The mechanism

Retrain the EMERGE-67 A→W read-out (BRIDGE-A) on a 16-word TRANSITIVE content vocab (4 subjects + 8 objects + 4 verb-3sg
surfaces, rebound onto the 16 validated concept pools; a NEW cache `bridges/rungB1_aw_trans/` so the EMERGE-frame cache
is untouched), reuse the existing EMERGE-68 function BRIDGE-F for the determiner "the", and pass the combined neural
spell as the producer's `spell=`. Each content slot (subject / verb-3sg / object) is decoded from `language_output`
SPIKES on BRIDGE-A; the determiner from BRIDGE-F. Reuse-by-import (EMERGE-67/68 + EMERGE-72/74/77 producer); NO `sim/`
edit; GPU/cupy, trained ONCE + cached.

## The de-risk — **GO** (6 seeds; cupy)

| gate | value (6-seed) | bar |
|---|---|---|
| **all-word spike render** — every word decoded from spikes == the ground-truth transitive | **1.000** | ≥ 0.90 |
| **content-lesion** — zero the concept-pool → language_output pathway → the content decode collapses | **0.000** | ≤ 0.30 |
| per-word isolated decode | **16/16 + "the"** | all correct |

**The result:** the transitive producer now speaks the answer with **both the order AND every word on spikes** —
render 1.000 across all 6 seeds, and the content-lesion collapses the render to 0.000 (the words are genuinely decoded
from `language_output` spikes, not a host lookup; a host str-lookup would be unaffected by the lesion).

## A genuine side-finding — a cross-backend producer-order near-tie, isolated + fixed

The FIRST 6-seed run was 4/6 (seeds 100/102 rendered 0.000). Diagnosis (systematic, not papered over): the per-word
decode is clean 16/16 in isolation, and the mis-renders were **order swaps, not word errors** (`the finds dog the fish`
instead of `the dog finds ...`; `cat the` instead of `the cat`). The host-token spell (`spell=str`) reproduces the SAME
0.000 on cupy seed 102 but renders **1.000 on numpy** — so the A→W spell is exonerated: this is a **backend-sensitive
producer-order near-tie** (EMERGE-59..77's C_TRANS order, validated on numpy, has per-pool f-I heterogeneity flip
adjacent slots on 2/6 seeds on cupy). The fix is the already-built **EMERGE-77 2-stage per-pool bias calibration**
(`DitransRegistryProducer(calibrate=True)` at `n_slot_pools=6`): subtract each pool's reference-current rate so the
order follows the primacy, not the heterogeneity. With it, the cupy order is 1.000 on 100/102 (was 0.000) — the
calibration is load-bearing (the raw read is the causal control that fails). This is a real, newly-isolated finding: the
2-stage calibrated read should be the default for the registry producer on cupy.

## Honest scope

- Validated on the 16-word transitive vocab (`train_word_to_pool` caps a bridge at 16 words = 4 kinds × 4; the full
  capstone transitive vocab exceeds that — the EMERGE-75 overflow boundary — so a reduced ≤16-word vocab is used, which
  is the clean regime). Broader vocab = the EMERGE-75 multi-bridge overflow follow-on.
- The A→W read-out is GPU/cupy (the validated scale), trained once + cached; the capstone reasoner (reservoir + composer)
  can co-execute on cupy (EMERGE-70/71's one-process result), so a fully-spiking-words transitive turn is now reachable.
- Reuse-by-import; NO `sim/` edit.

## Files
- `research/runners/_rungB1_aw_neural_words_transitive_derisk.py` — the transitive A→W spell + the calibrated producer +
  the de-risk (+ `--train`, `--diagnose`, `--host-order-check`).
- `tests/test_rungB1_aw_neural_words_transitive.py` — 3 CPU tests (structure; the GPU render is skip-if-no-cache).
- `research/findings/raw/_rungB1_aw_neural_words_transitive.json` — the 6-seed all-word spike render.
