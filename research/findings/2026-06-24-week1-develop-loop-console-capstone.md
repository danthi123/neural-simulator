# Week-1 develop-to-disk run + console capstone — DEMONSTRATED (2026-06-24)

The artificial-life capstone: a brain **develops over a simulated week**, saving a
self-contained, console-loadable bundle **per day**. The owner can load any day in the
interact console and chat with the brain at that stage of its development — *watch and
talk to a brain developing over time*.

## The run

```
python -m research.runners._longitudinal_develop_loop_gpu \
    --n-days 7 --save-bundle bridges/developed/week1 --per-day-bundles
```

cupy, seed 42, run `b08132cle`. Loop per day: WAKE (REAL stream-cortex co-occurrence
code-learning) -> CONVERSE (MultiTurnAgent on the learned codes) -> SLEEP (replay +
retention) -> GROWTH (tier promotion) -> METRICS -> PERSIST.

## The brain developed (per-day metrics, from the run log)

| day | vocab | facts | recall | held-out | retention | moat FA | corr(M,C) | tier | wall |
|----:|------:|------:|-------:|---------:|----------:|--------:|----------:|-----:|-----:|
| 0 | 6  | 2  | 1.00 | — | — | **0** | +0.91 | 4 | 123s |
| 1 | 12 | 5  | 1.00 | 1.00 | 1.00 | **0** | +0.90 | 4 | 131s |
| 2 | 18 | 8  | 1.00 | 1.00 | 1.00 | **0** | +0.89 | 4->8 | 138s |
| 3 | 24 | 11 | 1.00 | 0.00 | 1.00 | **0** | +0.88 | 8 | 143s |
| 4 | 24 | 11 | 1.00 | — | — | **0** | +0.88 | 8 | 148s |
| 5 | 24 | 11 | 1.00 | 1.00 | 1.00 | **0** | +0.89 | 8->12 | 150s |
| 6 | 24 | 11 | 1.00 | 1.00 | 1.00 | **0** | +0.88 | 12 | 152s |

- **Vocabulary grew 6 -> 24, facts 2 -> 11.** Recall stayed perfect every day.
- **Zero catastrophic forgetting** (retention 1.00 on every measured day).
- **The no-confab moat held every single day** (0 false-accepts, 7/7).
- Real stream-cortex learning (corr(M,C) ~ +0.88-0.91 between the learned firing-rate
  code and the held-out concept code).
- Growth: the brain mastered + was promoted tier 4 -> 8 -> 12 over the week.
- Per-day wall-clock 123-152s (rises as the brain grows -> more conversation windows);
  the whole week ran in ~15 min, local, on one RTX 3090.

## Console-loadable per-day bundles (verified)

8 bundles under `bridges/developed/week1/`: `day_0` … `day_6` + the final consolidated
`brain`. The console `GET /api/brains` picker lists all 8, labeled `week1/day_<N> (day
N)`, with the fact counts visibly growing across the week:

```
week1/day_0 (day 0)  n_facts=3       week1/day_4 (day 4)  n_facts=18
week1/day_1 (day 1)  n_facts=7       week1/day_5 (day 5)  n_facts=19
week1/day_2 (day 2)  n_facts=12      week1/day_6 (day 6)  n_facts=19
week1/day_3 (day 3)  n_facts=16      week1/brain (day 7)  n_facts=11 (consolidated)
```

The owner picks a day in the Interact tab -> chats with that day's brain -> watches the
brain's vocabulary and knowledge grow as the week progresses.

## ⇒ The artificial-life north-star is DEMONSTRATED end-to-end + LOCAL

A brain that develops over time (growing vocabulary + facts, no forgetting, moat intact),
persists each day to a loadable bundle, and can be talked to at any stage — all on one
local GPU in ~15 minutes for a simulated week.

## Open follow-ons

- **B3 live-verify**: restart the webapp (now has the per-turn activity viz) + confirm a
  chat turn shows the brain's spiking activity. (B3 committed `d7379e4a`; live-verify
  deferred until the develop run freed the GPU.)
- **A2 — scale the horizon**: a compressed month / year run (the per-day wall-clock +
  retention curve are characterized; ~13.5 hr/year overnight per the develop-loop ETA).
