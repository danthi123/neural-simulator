# Voice split read (2026-11-26): UNDEFINED by the registered rule. Sleep supplies the told fact; whether it also adds the user's voice is not shown, and the effect reverses across items

Prereg `brain/research/2026-11-26-voice-split-prereg.md` (Amendment R1, rule fixed before data). Harness db40ce062.
Data in `2026-11-26-voice-split/`: `vs_s*.json`, `vs_s*.log`, `vs_done_s*.txt`, `verdict.json`, ledgers in `lives/`.
Offline on K-P's KR night snapshots, CPU fp32, pool40/41/42 = s319/s320/s321 (node-mxl1462drl / node-dl4g243 /
node-dl5d243, the cascade screen's assignment), rc 0 on all three (10:00-11:00 UTC). DEV, 3 seen seeds, $0.

## Validity
- **Reproduction.** On every seed, the intact-B and hippocampus-off-B replies match the cascade screen's B rows on
  560/560 rows, and `brain_kw` is the same. No seed is void.
- **Engagement, recorded in the rows.**
  - Fetch lesion: hippocampus-off and both-off fetch 0 and recall 0 on every row. Intact and consolidated-off fetch
    150/145/132 and recall 217/215/213 (s319/s320/s321).
  - ctx: `told_first` is set on 336/336 told rows and on no row of any other arm. ctx knows 174/336, against both-off B
    knows 0/336.
  - The raw-synapse lesion (consolidated-off, both-off) has no direct trace. It shows only through behaviour:
    consolidated-off knows 54 against hippocampus-off 196.
- **Abstention.** "I don't know about that." appears 0 times in every arm and seed.

## Rule (told rows; totals as uv|knows)
| arm / cond | s319 | s320 | s321 | pooled P(uv\|knows) |
|---|---|---|---|---|
| hippocampus-off B | 12/71 | 19/59 | 15/66 | 0.235 |
| both-off ctx | 5/62 | 6/51 | 12/61 | 0.132 |
| consolidated-off B | 8/15 | 6/20 | 5/19 | 0.352 |
| intact B | 13/75 | 17/66 | 14/64 | 0.215 |
| both-off B | 0/0 | 0/0 | 0/0 | (knows 0) |

- **Gate.** 3 live seeds; hippocampus-off B and ctx knows are at least 51 on every seed. The UNDEFINED gate does not
  fire.
- **D (hippocampus-off B minus ctx)** = +0.1025, item-clustered 95% CI [-0.0126, 0.2019] (72 cells, 2000 draws, seed
  20261126). Both skeptics recomputed it independently and got the same result to bootstrap rounding.
- **SLEEP-LEARNED fails.** hippocampus-off > ctx on 3/3 seeds, but the lower bound is -0.013, not above 0.
- **NOT-SLEEP fails.** The upper bound is 0.20, outside ±0.10.
- **FETCH-PRIMED fails.** consolidated-off minus hippocampus-off is +0.117, CI [-0.039, 0.379]; the lower bound is
  below 0.
- **Verdict: UNDEFINED** (the final "anything else" branch). By the prereg, no lever follows from it. It is not
  rescued.

## Report only (not registered; these steer the next method, they do not decide anything)
- **The pooled D mixes a sign reversal across items.** Three items hold most of the knows rows. Pooled over seeds,
  P(uv|knows) for hippocampus-off vs ctx:
  - pixel (cat's name): 19/39 vs 3/30.
  - marguerite (sister's name): 13/42 vs 1/27.
  - volvo (car): 2/34 vs 6/29, reversed.

  Leaving one of them out moves the CI either way: without volvo it is [+0.036, 0.244], without pixel
  [-0.072, 0.144]. The wide CI is heterogeneity across items, not noise around one effect.
- **The ctx control is confounded with reply form.** With the telling in the preceding turn, ctx often answers in
  the second person ("Your dog is called Bramble", "You drive a red Volvo") or tersely ("Marguerite."). Its low
  P(uv|knows) on the name items reflects that, not only un-slept weights. D's sign is therefore not evidence about
  sleep.
- **On the name items the raw cortex already uses the voice.** both-off B (raw synapses, no fetch, knows 0) answers
  "What is my cat called?" with "My cat is called Fluffy / Mittens / Mr. Whiskers" and "What is my sister's name?" with
  "My sister's name is Sarah / Emily".
  - On pixel and marguerite told rows, both-off B is in the user's voice on 36/84 rows, against hippocampus-off on
    32/84.
  - Over all told rows it is 62/336 (18.5%) against 68/336 (20.2%); hippocampus-off minus both-off is +0.018, CI
    [-0.038, 0.078].
  - On volvo the raw cortex never uses it (0/42).
  - So, for name questions, the "My X is called ..." frame is the pretrained cortex's answer to "What is my X
    called?". Sleep supplies the content (knows 196 vs 0), not demonstrably the frame. This is two items on 3 seeds,
    not a general statement.
- **Instrument caveat.** `own2` counts question echoes as user voice ("What does my nephew do for work?"). That
  happens only in ctx (16 of its 51 told-row own2 hits; 0 in every other arm). It does not touch uv|knows or D (no
  echo has knows = 1). It does bias ctx's P(uv) over all told rows: hippocampus-off minus ctx there is +0.051 as
  reported and about +0.098 echo-free.
- **Fetch.** consolidated-off and intact have 20 right-episode fetches per seed on told rows. consolidated-off knows
  only 15-20 per seed, so its high P(uv|knows) (0.35) rests on few rows.

## What it means for the voice defect
- The registered question (is the user's voice in told answers sleep-learned?) is UNDEFINED, and the control chosen
  for it (fact in context) changes the reply form, so a sharper version of this contrast is not the next step.
- The source-hold DRAFT (`2026-11-24-self-other-tag-prereg-DRAFT.md`) stays BLOCKED. Its premise, that sleep binds
  the item without its source, is not shown. On the name items the voice is already present in the raw cortex.
- The next voice method needs a new before-build. It should work on the cortex's question-answer frame ("What is my
  X" answered as "My X ..."), which exists before any sleep, rather than on a night method. It should be judged by
  fresh-state P(uv|knows) and checked per item.

## Skeptics (two sonnet lenses, fresh context, REFUTER-LENS block)
- **Numbers and rule:** "PARTLY-REFUTED. The numbers and the UNDEFINED reading hold. The interpretive sentence about
  the user-voice frame does not hold as worded." The CI, the branches and the gate were applied by hand. It found
  the leave-one-item-out fragility and the volvo reversal.
- **Construction and instrument:** "PARTLY-REFUTED. The rule outcome UNDEFINED is correct. The interpretive gloss
  attached to it is not supported as worded." It found the ctx second-person confound, the echo bias in ctx P(uv), and
  that the raw-cortex voice is limited to name items.
- **All wording fixes are applied above:** "not shown" in place of "not sleep-learned", the item scope, the 36/84
  concentration, the echo caveat, and the hedged "sleep supplies the content".
- **Lenses:**
  - (a) Engagement is recorded for the fetch lesion and ctx; the raw-synapse lesion is behavioural only.
  - (b) No default-on candidate; 0 abstentions.
  - (c) `tools/where_ran.py` says "unknown" because it does not read the top-level `host` key. The JSONs carry the
    three pool nodes, and the 560/560 reproduction shows the inputs matched.
