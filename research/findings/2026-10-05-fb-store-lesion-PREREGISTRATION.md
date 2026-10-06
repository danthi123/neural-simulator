---
type: preregistration
status: preregistration
date: 2026-10-05
packet: fb-store-lesion
runner: research/runners/load_bearing_fraction.py
scorer: tools/fb_store_score.py
---
# fb-store-lesion: does the spiking belief store carry the false-belief reply? (PRE-REGISTRATION, 2026-10-05)

Written by cloud Claude session 7 before any gate-seed run. Context: the false-belief chat is on by default (merged
2026-10-05), and its registered row `tom-false-belief` is reply-level load-bearing on six seeds. But that row's
lesion forces the host witnessing gate open: it overrides a host decision and never cuts the spiking store (the review
section of `research/findings/2026-10-05-iw2-false-belief-chat-default-on-flip-validated-PROVISIONAL.md`). This
measurement asks the question that lesion cannot: is the store itself necessary for the reply?

## 1. The cut and the reply change it can produce

- `BRAIN_FALSE_BELIEF_STORE_LESION=1` builds the belief store with its assembly self-recurrence weight at 0
  (`build_tom_bridge(..., belief_attractor_weight=0.0)`). The witnessing gate, the host write, and the world and self
  stores are untouched. Unset, the build is byte-identical (`tests/test_fb_store_lesion.py`).
- An empty or tied store now abstains (`webapp/false_belief_chat.py` `belief_store_read_ok`); before this, the
  host argmax returned slot 0, which on this probe is the stale location, so an emptied store could never change the
  reply. Default replies are unchanged wherever the store names one location.
- Row `tom-false-belief-store` (`research/runners/lbf_rows/fb_store_lesion.py`), opt-in via
  `LB_FB_STORE_LESION_PROBE=1`, probe turn `tom_fb` (the classic unwitnessed move), reply fields only.

## 2. Dev observation that motivates the prediction (not evidence)

On dev seeds 7, 8 and 9, through the production handler, the cut lowered the belief store's late-window rate by
roughly an order of magnitude, yet the read still named the stale location on all three, and the true-belief story
still named the true one. The belief reaching the read is the host-gated write current and its residual a few
hundred steps earlier; no simulated time passes between narration and question, so nothing yet needs the attractor.

## 3. Runs

Pool, numpy, gate seeds 42/43/44/100/101/102, one line each from `research/packets/fb-store-lesion/jobs_template.txt`
(`<PIN>` = the main revision that carries this file, `<PIN7>` its first seven characters). First one dev-seed-7
shard of the same line with `--seed 7` and its own tag, to confirm the row is exercised and the lesion arm records
`store_lesioned` (checklist item 4). Corpus step first: `bash tools/before_you_build.sh "belief store attractor
lesion false belief persistence"`. One re-run per not-DEFINED cell, logged; then score once:
`python tools/fb_store_score.py --tag fbstore-<PIN7> --pin <PIN> --out "$OUT"`, where OUT is `verdict.json` in the
directory `_fb_store/` under the raw findings tree (not yet written).

## 4. Reading (fixed now)

A cell is DEFINED per the scorer's docstring (null clean, lesion reproduced, intact answers the stale location through
the organ, lesion engaged). UNDEFINED never counts.

| scorer verdict | reading |
|---|---|
| STORE-NOT-LOAD-BEARING (0 of 6 DEFINED seeds change the reply) | The store's attractor is not necessary on the immediate-question probe; the reply is carried by the host-gated write. Every load-bearing-fraction report lists `tom-false-belief` as a host-input lesion. Next method: a probe with simulated time between narration and question (the companion process this probe replaced with zero), on T18/T21. |
| STORE-LOAD-BEARING (6 of 6) | The store carries the reply. Claude rules on lifting the row into the headline set. |
| MIXED | Per-seed report, no generalization; Claude rules. |
| UNDEFINED | Fix the cause (one fix round); never read as either word above. |

## 5. Prediction (an expectation, not a bar)

STORE-NOT-LOAD-BEARING, from section 2.

## 6. Seeds, cost, files

Six gate lines, about the cost of one `tom-false-belief` R cell each (mem_gb 8 declared, peak about 11 GB per arm on
the iw2 runs). Files: the row module, the scorer and its tests, the job template, this prereg.
