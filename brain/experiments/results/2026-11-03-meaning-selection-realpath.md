# 2026-11-03 Meaning-based selection through the real hippocampus path (S_all vs S0), refit bar, no duplicate memory

Follow-up to `2026-11-03-meaning-selection.md` (skeptic: not fit to merge). That probe recomputed similarities from
Gram matrices and never called `KeyedHippocampus.recall` (no strength weighting, no bar, no margin, no stored fp16
delta); its bar/margin were S0's; `S_all` stored a second fp32 copy of the whole memory per episode. Capability:
memory selection (`brain/PREREQUISITES.md`); record: `brain/research/2026-11-02-record-brief-consolidation-and-selection.md`
part B; `tools/before_you_build.sh` ran (11-03) and surfaced only the 11-03 writeup itself.

## Changes

- `Hippocampus.recall` split into `similarities()` -> `scores()` (strength-weighted) -> `_compete()` (bar, margin, use
  boost). Pure refactor: class constants (FAMILIARITY 0.12, MARGIN 0.02, GAIN 4) unchanged, S0 test still identical.
- `KeyedHippocampus`: competition is the parent's `_compete`. `S_all` (and `S_mid`, `S<l>`) no longer stores a key:
  the similarity is read from the episode's stored fp16 delta at recall, one layer at a time (exactly the centred
  cosine of the concatenated per-layer-normalised key; test). `familiarity=` / `margin=` set a per-instance bar;
  the class defaults are untouched. Pooled activity is kept only for the learned key.
- `DEV_FACTS36` + `DEV_NEVER_TOLD` (`facts_bank.py`): 36 new teacher-written facts (same format, own confusables,
  12 never-told questions), no shared answer word or cue text with the 48-fact test bank (asserted). Used only to fit.
- `brain/experiments/selection_realpath.py`: everything through `KeyedHippocampus.scores/recall/reinstate`.

## Pre-registration (written before the run; instrument fixed)

Setup. Models 0.1B and 0.4B (`/home/dant123/models/rwkv7/`), CPU. Seeds 42/43/44/100/101/102. TEST: the 48-fact bank,
split per seed as the 11-03 probe (`Random(seed)` shuffle; 24 stored, 12 untold); cues: same-words (orig), light
(para_a), heavy (para_b) rewording of each stored fact (72 told cues); untold cues = the 12 untold facts' questions +
the 12 NEVER_TOLD questions. DEV: the same on DEV_FACTS36 (24 stored, 12 untold) + DEV_NEVER_TOLD. Every episode is
fresh at each cue (age 0, strength 1, use boost undone after each cue).

Fit (DEV only, pooled over the six seeds, per key and model): grid FAMILIARITY -0.10..0.80 step 0.005 x MARGIN
{0, .005, .01, .02, .03, .05, .08, .12}; maximise J = P(fire and the right episode | told cue, all three cue types) -
P(fire | untold cue); ties to the larger bar, then larger margin. The fitted values are then frozen for TEST.

Arms on TEST, each through `recall()`: `forced` (bar and margin off: top-1), `default` (0.12 / 0.02), `refit`
(dev-fit), for S0 and S_all; plus `oracle` (right episode reinstated) and `none` (no memory). Recall = reinstated
episode (or none if recall did not fire), greedy 16-token answer, hit if the answer word is in it.

Criteria (each must hold at BOTH sizes):
- P1 selection: forced S_all > S0 on selection for para_a and for para_b, each on >= 5/6 seeds.
- P2 recall: refit S_all > refit S0 on para_b recall on >= 5/6 seeds, and the mean gap >= 0.10.
- P3 told-vs-untold under refit: J_test(S_all refit) > J_test(S0 refit) on >= 5/6 seeds. Reported, not a criterion:
  untold fire rate under refit; above 0.25 means the familiarity bar does not control false recall for that key.
- P4 replication: forced S_all para_b selection mean within 0.10 of the Gram probe (0.69 at 0.1B, 0.76 at 0.4B);
  otherwise the 11-03 numbers do not stand and this writeup replaces them.
- P5 memory: S_all bytes per episode == its fp16 delta bytes (before: measured from the 11-03 code).
- P6 consistency: every recall() decision equals the bar/margin rule applied to scores() (asserted in-run; a failure
  aborts the run).
Default recommendation rule: S_all is recommended as the Hippocampus default only if P1-P3 hold at both sizes, and then
only after a live-brain test (below); a lead is not a GO.

## Results

Files: `2026-11-03-meaning-selection-realpath/realpath_{0.1b,0.4b}_6seeds.json` (per seed, per arm, plus every
score row and answer), `memory.json`. Table: `python -m brain.experiments.selection_realpath_summary <json>`.
P6 held: no assertion fired in either run (every `recall()` decision matched the rule on `scores()`, and pass-1 scores
reproduced in pass 2). Refactor check: old vs new `Hippocampus` identical on 156/156 cues
(`hippocampus_refactor_check.py`, 0.1B, 24 stored).

Dev fits (DEV bank only, six seeds pooled): S0 0.1B bar 0.385 / margin 0.02, 0.4B 0.425 / 0.01; **S_all 0.185 / 0.05 at
both sizes**. Dev J: S_all 0.46 / 0.44, S0 0.38 / 0.10.

Mean over 6 seeds (min-max). sel = recall fired AND on the right episode; rec = answer word in the reply; untold fire =
recall fired on an untold cue (24 per seed); J_test = mean sel (3 cue types) - untold fire.

| size | arm | sel orig | sel para_a | sel para_b | rec para_a | rec para_b | untold fire | J_test |
|---|---|---|---|---|---|---|---|---|
| 0.1B | S0 forced | 0.92 | 0.57 | 0.33 (0.25-0.38) | 0.51 | 0.22 | 1.00 | -0.39 |
| 0.1B | S_all forced | 0.99 | 0.79 | 0.69 (0.58-0.83) | 0.70 | 0.46 (0.38-0.62) | 1.00 | -0.18 |
| 0.1B | S0 default bar | 0.88 | 0.55 | 0.27 | 0.50 | 0.18 | 0.72 | -0.16 |
| 0.1B | S_all default bar | 0.94 | 0.70 | 0.44 | 0.61 | 0.28 | 0.56 | 0.13 |
| 0.1B | S0 refit | 0.65 | 0.28 | 0.07 | 0.26 | 0.03 (0.00-0.04) | 0.16 | 0.17 (0.10-0.22) |
| 0.1B | S_all refit | 0.86 | 0.47 | 0.23 | 0.42 | 0.14 (0.08-0.25) | 0.22 (0.12-0.29) | 0.30 (0.19-0.43) |
| 0.1B | oracle / none | | | | 0.88 / 0 | 0.72 / 0 | | |
| 0.4B | S0 forced | 0.45 | 0.10 | 0.10 (0.04-0.17) | 0.10 | 0.05 | 1.00 | -0.78 |
| 0.4B | S_all forced | 0.99 | 0.82 | 0.76 (0.67-0.88) | 0.70 | 0.66 (0.54-0.79) | 1.00 | -0.14 |
| 0.4B | S0 default bar | 0.39 | 0.08 | 0.08 | 0.08 | 0.03 | 0.76 | -0.58 |
| 0.4B | S_all default bar | 0.96 | 0.62 | 0.49 | 0.50 | 0.38 | 0.53 | 0.15 |
| 0.4B | S0 refit | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.09 | -0.05 (-0.10-0.01) |
| 0.4B | S_all refit | 0.76 | 0.40 | 0.19 | 0.28 | 0.12 (0.08-0.25) | 0.15 (0.12-0.21) | 0.30 (0.25-0.33) |
| 0.4B | oracle / none | | | | 0.88 / 0 | 0.88 / 0 | | |

Told-vs-untold AUROC of the top weighted score (test): S0 0.62 / 0.54, S_all 0.65 / 0.62 (0.1B / 0.4B).

Memory per episode (`memory.json`, MB): 0.1B default Hippocampus 1.38 (fp16 delta 1.18 + fp32 S0 key 0.20); 11-03
S_all **3.58** (delta + 2.36 fp32 all-layer key + 0.04 pooled activity) -> now **1.18** (the delta only). 0.4B: default
3.41; 11-03 S_all **9.54** -> now **3.15**. S_all now costs less than the default (no separate key at all).

Criteria (per-seed wins from the summary):
- P1 PASS both sizes: forced selection S_all > S0 on para_a and para_b on 6/6 seeds (0.1B mean +0.22 / +0.36;
  0.4B +0.72 / +0.67).
- P2 PASS both sizes: refit para_b recall S_all > S0 on 6/6 seeds, mean gap +0.11 (0.1B), +0.13 (0.4B).
- P3 PASS both sizes: J_test refit S_all > S0 on 6/6 seeds (mean +0.12, +0.34). Untold fire under refit 0.22 (0.1B)
  and 0.15 (0.4B): below the 0.25 flag, but 3-5x the dev estimate (0.08 / 0.03): the dev fit is optimistic.
- P4 PASS: forced S_all para_b selection 0.694 (0.1B) and 0.764 (0.4B) reproduce the Gram probe (0.69 / 0.76): the
  11-03 S0/S_all numbers stand for forced selection; the probe lacked only the bar.
- P5 PASS: S_all bytes per episode == fp16 delta bytes (test + memory.json).
- P6 PASS (above).

## What it shows

- Through the real hippocampus, S_all is better than the default S0 key in every arm, at both sizes, on 6/6 seeds,
  and now stores less per episode than the default. At 0.4B the default key is broken (forced reworded selection 0.10)
  and S_all restores it (0.76-0.82).
- The bar matters as much as the key: S_all under S0's default bar (0.12 / 0.02) fires on 53-56% of untold questions.
  S_all needs its own bar; the dev fit gives 0.185 / 0.05 at both sizes (the same value at 0.1B and 0.4B).
- The familiarity bar is a poor gate for rewordings: at the refit bar, heavy-rewording recall falls from 0.46 / 0.66
  (forced) to 0.14 / 0.12, while untold fire stays 0.15-0.22. A reworded told question looks about as unfamiliar as an
  untold one on this score (AUROC 0.62-0.65). The key fixes WHICH memory; it does not fix WHETHER to recall.

## What it does not show

- Not run in the live `Brain` (no conversation context in the cue, no sleep, no weekly transcript). 24 stored
  one-sentence tellings; teacher-written rewordings; no 1.5B.
- "Recall" = selection x the cortex's answer rate given the right episode (oracle 0.72-0.88); it adds the reinstate
  and answer step, not an independent capability. The substring scorer cannot be fooled by echo here (no question
  contains its answer word), but it does not check perspective.
- The bar was fit on 36 dev facts written by the same teacher as the test facts (same style): dev->test gap shown
  above (untold fire 3-5x higher on test).

## Recommendation

S_all should become the hippocampus key, with its own bar (FAMILIARITY 0.185, MARGIN 0.05), not S0's; Hippocampus
defaults are NOT changed here (a lead, not a GO). Before flipping: the live-brain test. Six seeds x two arms
(`Brain` with default Hippocampus vs `KeyedHippocampus(key="S_all", familiarity=0.185, margin=0.05)`), same sleep
setting in both arms: tell 24 facts across a conversation, restart, ask same-words / reworded / never-told questions
inside the conversation; score correct-perspective answers (not substring, see the sleep-in-brain skeptic), false
recall on never-told questions, and a hippocampus-off lesion that changes the reply. Pair it with the false-recall
branch's question-familiarity gate: the key picks the memory, a separate familiarity signal must decide whether to
recall at all, since this bar cannot do both for reworded questions.
