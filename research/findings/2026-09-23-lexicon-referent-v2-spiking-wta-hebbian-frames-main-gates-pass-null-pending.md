---
type: finding
status: partial
lane: language (lexicon)
date: 2026-09-23
mechanism: lexicon v2 — referent (noun-category) decision by a coupled two-pool spiking WTA (reciprocal FSI inhibition) with graded drive through Hebbian (Oja) frame->category synapses, teacher-supervised seed curriculum, host read-out of the winning pool
---

# Lexicon v2: the referent decision moves onto a coupled spiking WTA with Hebbian-learned synapses — main gates pass on 6 seeds; the permutation null is still running (2026-09-23)

**Verdict: INCOMPLETE.** Every pre-registered gate measured so far passes on all 6 seeds; S2 (the >=1000-permutation
null) is queued on the pool and has not run. No GO is claimed until S2 lands.

## Mechanism (what changed from v1)

v1 computed the category with host label-spreading and relayed its sign into one of two uncoupled pools (banked
NO-GO: `research/findings/2026-09-23-lexicon-referent-v1-host-label-spreading-6seed-NO-GO-and-relabel.md`).
v2 (`research/runners/lexicon_spiking_frame_category.py`) has no host category score:

- **Environment (host):** a word is presented as 32 of its real corpus occurrences; each occurrence drives the
  frame afferents of its heard neighbours (one afferent per offset x context token, offsets -2..+2, top 100 tokens).
- **Learned edge:** FR -> CN (referent) and FR -> CX (non-referent), all-to-all, updated by Oja's rule from the
  pre/post spike rates the bridge produced. Oja's decay is the normaliser; there is no clamp.
- **Competition:** CN and CX each drive their own fast-spiking interneuron pool, which inhibits the other pool.
- **Curriculum:** a teacher current on the labelled pool during training (12+12 hand words in cross-validation,
  38+37 at deployment). This is "teacher-driven" learning in docs/TERMS.md terms, not self-organized.
- **Read-out:** the host reads which pool fired more ("spiking with a host read-out", not "fully spiking").

Pre-registration: `research/runners/_lexicon_spiking_referent_derisk.py` docstring, commit 4b0113e6c. It was written
before any evaluation seed ran; parameters were calibrated on dev seed 7 only, and the dev numbers that were seen
are listed in that docstring.

## Results (seeds 42, 43, 44, 100, 101, 102)

Artifacts: `research/findings/raw/_lexicon_spiking_referent/main_s42.json`,
`research/findings/raw/_lexicon_spiking_referent/main_s43.json`,
`research/findings/raw/_lexicon_spiking_referent/main_s44.json`,
`research/findings/raw/_lexicon_spiking_referent/main_s100.json`,
`research/findings/raw/_lexicon_spiking_referent/main_s101.json`,
`research/findings/raw/_lexicon_spiking_referent/main_s102.json`.

| gate (evidence) | per seed 42 / 43 / 44 / 100 / 101 / 102 | threshold | pass |
|---|---|---|---|
| S1 held-out balanced acc (abstain = wrong, 400 words) | 0.787603 / 0.824436 / 0.756554 / 0.766506 / 0.813262 / 0.79394 | mean >= 0.75, min >= 0.70 | yes |
| S3a learned-edge lesion balanced acc | 0.331551 / 0.136466 / 0.242712 / 0.102377 / 0.118416 / 0.161921 | mean <= 0.55, drop >= 0.20 | yes |
| S3b lesion, decided words only | 0.496 / 0.446829 / 0.496151 / 0.387934 / 0.566445 / 0.525424 | mean <= 0.60 | yes |
| S4 loser/winner rate, intact -> competition lesioned | 0.12785->0.410517 / 0.155644->0.47166 / 0.133183->0.392614 / 0.138142->0.440955 / 0.101312->0.366979 / 0.152764->0.476001 | rise >= 0.25 every seed | yes |
| S5 Spearman(synaptic drive margin, spike-rate margin) | 0.982476 / 0.97527 / 0.980328 / 0.982595 / 0.973063 / 0.974539 | mean >= 0.50 | yes |
| S7 organ: learned recovers both / learned-edge lesion / false positive | 0.75,0,0.3333 / 0.8333,0,0 / 0.6667,0,0 / 0.6667,0,0.16667 / 0.8333,0,0.08333 / 0.8333,0,0 | >= 0.60 / <= 0.20 / <= 0.20 (means) | yes |
| S8 "the wolf watches the owl" -> ["wolf","owl"], organ held turn in scope, n_referents 2 | 6 of 6 seeds | >= 5 of 6 | yes |
| S2 1000-permutation null | not run yet | >= 99th percentile on >= 5 of 6 | pending |

Integrity smokes (pass by construction, not evidence): full rebuild from the corpus reproduces weights and
decisions on all 6 seeds; afferent-zero lesion abstains on every word; hand-table organ baseline in-scope 0.

Report-only (not gated): under the learned-edge lesion the capability turn drops "owl" on 6 of 6 seeds (5 extract
["wolf"]; seed 101 extracts ["wolf","watches"]), so the held turn goes out of scope on 5 of 6.

## What the numbers say

- **The learned synapses carry the category.** Restoring them to their pre-learning values drops balanced accuracy
  from 0.76-0.82 to 0.10-0.33, and the decisions the lesioned circuit still makes are at chance (0.39-0.57).
  In the organ, recall of two held-out nouns goes from 0.6667-0.8333 to 0 on every seed.
- **The pools really compete.** Removing only the cross-inhibition leaves accuracy nearly unchanged (0.74-0.80)
  but roughly triples the loser pool's relative rate. Lateral inhibition makes the decision categorical; the
  accuracy comes from the learned drive. The rise clears the 0.25 threshold narrowly on seeds 44 and 101.
- **The WTA outcome follows the graded synaptic input** (Spearman 0.97-0.98), not a binary sign.
- **Cost of moving the computation onto synapses:** v2 is less accurate than v1's host label-spreading (v1 about
  0.93 against v2 0.76-0.82). v2 learns a prototype from 24 labelled words, while label-spreading also propagates
  through unlabelled words. An unsupervised competitive phase was tried on dev seed 7 and did not help (dev scratch run, not banked: about 0.79 vs 0.80).
  It is not part of the pre-registered mechanism.

## The capability target is met at the organ, NOT in the full brain

At the organ level, the learned detector lets the D6 buffer bind "the wolf watches the owl" as two referents on 6
of 6 seeds, and it rejects "watches". But the full-brain load-bearing battery still reads `wm-binding-advanced`
NOT-EXERCISED with `BRAIN_LEARNED_REFERENT_LEXICON=1`
(`research/findings/raw/_lexicon_spiking_referent/lb_wm_binding_advanced_learned_lexicon.json`, seed 42, repeats 2,
deterministic). The arm artifact (`research/findings/raw/_lexicon_spiking_referent/intact_b_hold_held.json`) shows
why: the `held` turn exits through the comprehension-abstain / gate-B role-repair clarification ("which of them is
the 'watch' done to...?") before `webapp/server.py` reaches the multiref maintain block. So the turn carries no
`multiref` key, whatever the lexicon does. The build's diagnosis (the hand noun list alone) was incomplete; this is
logged in `research/FAILURE_LOG.md`. **Next rung:** fold the multiref maintain step into the abstain/repair return
path. There is a precedent: the open-ended-turn maintain-fold at `webapp/server.py` ~4994. Add it as a default-off
additive, then re-run `load_bearing_fraction --only wm-binding-advanced` with the flag, intact vs `BRAIN_MULTIREF_LESION`
and `BRAIN_LEARNED_REFERENT_LESION`.

## Staged

S2 nulls: 6 pool jobs (`--part null --seeds <s>`, revision b9d034940). Each is 1000 permutations on 41-replica
circuits, about 70 minutes per seed once dispatched. `tools/pool_sync.sh` pulls the results into
`research/findings/raw/_lexicon_spiking_referent/`. Then run
`python -m research.runners._lexicon_spiking_referent_derisk --score research/findings/raw/_lexicon_spiking_referent`.
