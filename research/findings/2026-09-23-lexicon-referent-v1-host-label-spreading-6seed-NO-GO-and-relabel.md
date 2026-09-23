---
type: finding
status: live
lane: language (lexicon)
date: 2026-09-23
mechanism: lexicon v1 — corpus frame-graph label-spreading referent (noun-category) score, host-computed, spike-RELAYED
---

# Lexicon v1 (learned referent detector): 6-seed verdict NO-GO on organ recall; the "spiking WTA" was a relay (2026-09-23)

## What was measured

`research/runners/_lexicon_learned_referent_derisk.py` (pre-registered in d97029564, amended as A1 in 416cc562b
**before** the 6-seed artifacts were opened) ran on the pool at revision fc8a0e10a for seeds 42 43 44 100 101 102.
Artifacts: `research/findings/raw/_lexicon_learned_referent/lexicon_referent_s42_43.json`,
`research/findings/raw/_lexicon_learned_referent/lexicon_referent_s44_100.json`,
`research/findings/raw/_lexicon_learned_referent/lexicon_referent_s101_102.json`, scored into
`research/findings/raw/_lexicon_learned_referent/verdict.json` by the pre-registered scorer.

**Verdict: NO-GO.** One evidence gate fails, exactly as pre-registered:

| gate (kind) | value | threshold | pass |
|---|---|---|---|
| G1 learned held-out balanced acc, mean (evidence) | 0.9294 | >= 0.80 | yes |
| G1 min over seeds (evidence) | 0.9267 | >= 0.75 | yes |
| G2 shuffled-graph mean (evidence) | 0.4942 | <= 0.60 | yes |
| G3 learned - max(shuffled 0.4942, freq-only 0.5316, label-permuted 0.5104) (evidence) | 0.39785 | >= 0.15 | yes |
| **G7 organ learned recovers both, mean (evidence)** | **0.7778** | >= 0.80 | **no** |
| G7 false-positive in-scope (evidence) | 0 | <= 0.20 | yes |
| G4 spiking agreement / spiking bacc (integrity smoke) | 1 / 0.9263 | | yes |
| G5 lesion abstain (integrity smoke) | 1 | | yes |
| G6 determinism (integrity smoke) | true, **partial** (graph not rebuilt) | | yes |
| G7 hand baseline / lesion in-scope (integrity smokes) | 0 / 0 | | yes |

G3 is reported as pre-registered: it is a MEAN over seeds and it passes. The single-seed-42 value the builder
worried about is real in the artifact (label-permuted 0.8050, a per-seed gap of 0.12), and the other five seeds'
single permuted controls ranged 0.3337-0.5869, which is why one fixed shuffle is a noisy control.

G7 recover per seed (42, 43, 44, 100, 101, 102): 0.83, 0.92, 0.67, 0.67, 0.92, 0.67 (12 trials each). A trial
needs BOTH held-out nouns admitted, so recover is roughly recall squared; at ~0.93 balanced accuracy some pairs lose
one noun.

## Report-only null (amendment A1(e), added after pre-registration so NOT a gate)

`--v1-null 1000` (`research/findings/raw/_lexicon_learned_referent/v1_null_6seed.json`): 1000 permutations of the
24 cross-validation seed labels per seed, label-spreading re-run for each. The learned score sits at the
99.9-100.0th percentile on all six seeds: null median 0.4908-0.5055, null 99th percentile 0.8599-0.8916, null max
0.9077-0.9367, against learned 0.9267-0.9332. The host label-spreading score is carried by the corpus, not by a
lucky labelling; one permutation on seed 43 (null max 0.9367) did beat the learned score.

## What the review established (why v1 makes NO spiking claim)

The "spiking two-pool WTA" was a RELAY: the pools of `_gap3_spiking_feature_compat_derisk._build` are uncoupled
(internal density 0, one weight-0 pathway), and `classify()` drove ONE pool with a fixed current chosen by
`np.sign(host label-spread score)`. So G4 (spike/offline agreement) and G5 (lesion abstain) could not fail, and the
G7 hand and lesion arms passed by definition. They are relabelled integrity smokes (amendment A1(b)); the v1
mechanism is "host-computed category, spike-relayed". Per docs/TERMS.md this is not "fully spiking" and not a
spiking decision. The v1 seed curriculum is 38 hand nouns, not the "40" first written.

## What replaced it

v2 (`research/runners/lexicon_spiking_frame_category.py`, pre-registered in 4b0113e6c, de-risk
`_lexicon_spiking_referent_derisk.py`): the decision is made by a coupled two-pool WTA (reciprocal FSI inhibition)
whose graded drive arrives through Hebbian (Oja) frame->category synapses learned on the bridge. Host label-spreading
is gone from that path. Its 6-seed run is staged; no v2 verdict exists at the time of writing, and none of the three
v1 artifacts above is evidence about v2.
