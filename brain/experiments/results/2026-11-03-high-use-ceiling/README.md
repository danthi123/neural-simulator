# High-use facts, part 2: in-context ceiling, reachable-only rescore, use counted inside the hippocampus (2026-11-03)

Follows `../2026-11-03-high-use-facts/README.md` (NO-GO at 0.1B, "next step 1"). Branch `research/high-use-ceiling-1103`.

## PRE-REGISTRATION (written and committed BEFORE the six-seed run; instrument fixed)
- Instrument: `brain/experiments/use_strength_probe.py` as committed with this section. Scorer = answer key as a lowercase
  substring of the greedy hippocampus-off answer (the same scorer as `sleep_probe.ask`; the questions do not contain the key,
  so first-person echo of the user's line cannot score). No change after the run; if one is needed, everything is rerun.
- Reachable fact (0.1B) = its ORIGINAL question is answered correctly in context (tell it once after the birth context, no sleep,
  greedy). Facts = the first 12 reachable in pool order (`ceiling_probe.POOL` = the 12 old facts then 24 new distinctive ones).
- Design: 12 reachable facts, 4 told k=1, 4 told k=3, 4 told k=10 times THROUGH `UseHippocampus.store` (re-telling merges into
  one episode and adds strength 1.0; nothing is hand-set; the probe asserts strengths equal k). k per fact = seeded permutation.
  One sleep (240 steps, lr 5e-4, kl 1.0), hippocampus off. Arms: no_sleep, sleep_use (replay by learned strength), sleep_equal
  (control: same episodes and steps, all strengths 1). Seeds 42 43 44 100 101 102, 0.1B.
- **GO**: sleep_use recalls 4/4 of the k=10 facts on 6/6 seeds. (The facts are reachable by construction, so 4/4 is the bar.)
- Also reported, not part of GO: use vs equal contrast on k=10 (paired by seed), k=1 and k=3 recall, paraphrase recall (read only
  against facts whose reworded question is reachable in context), false recall, drift. Anything short of the GO bar is a lead or a
  negative, whatever the contrast shows.

## Verdict: NO-GO (pre-registered bar missed). A weak lead for "use helps k=10", not established.
k=10 reachable facts reach 4/4 on 1 of 6 seeds (s43). The bar was 6/6.

## (1) Per-fact in-context ceiling (`ceiling_0.1b.json`, `ceiling_0.4b.json`; greedy, deterministic, so no per-seed variation)
Tell one fact after the birth context, no sleep, no hippocampus, ask original and reworded question. Pool = 12 old facts + 24 new.
| model | old 12: original | old 12: reworded | all 36: original | all 36: reworded | both |
|---|---|---|---|---|---|
| 0.1B | 6 (lyon marguerite biscuit cello peanut oslo) | 4 (pixel lyon biscuit cello) | 20 | 11 | 8 |
| 0.4B | 9 (misses teal lyon volvo) | 9 | 20 | 18 | 14 |
The 0.4B old-12 ceiling here is 9/12, not the 11/12 in the scale-ladder note: different scorer/protocol (greedy, 16 tokens, substring);
not reconciled. Teal and volvo are unreachable at both sizes. The 0.1B ceiling is a property of single facts: the set that is held
differs from the set that is paraphrase-held (reworded questions are a separate, smaller ceiling).
Note: "spider" scores reachable by substring on a hedging answer; kept because the rule was fixed in advance.

## (2) Old six-seed results re-scored on reachable facts only (`rescore_old_reachable.txt`)
Old probe = hand-set strength k. Hits / reachable facts in each k group, pooled over six seeds (only 6 of the 12 facts are reachable).
| arm | k=1 | k=3 | k=10 |
|---|---|---|---|
| sleep_use | 3/11 | 4/11 | 7/14 |
| sleep_equal | 4/11 | 2/11 | 5/14 |
Even reachable k=10 facts are 50% recalled (use) vs 36% (equal); k=10 was 100% on 2 of 6 seeds (s42, s100, each with only 2 reachable
k=10 facts). So the old NO-GO is not an artifact of unreachable facts, but the old test had too few reachable facts per cell to say more.

## (3) Use counted inside the hippocampus, six seeds, 0.1B, 12 reachable facts (`s*.json`, `aggregate.txt`)
Facts used (first 12 reachable): lyon marguerite biscuit cello peanut oslo zephyr gustav alvarez stamp maple iceland.
The probe asserted one episode per fact with strength exactly k (re-telling merged, nothing hand-set).
Recall of the 4 facts per group:
| seed | use k1/k3/k10 | equal k1/k3/k10 |
|---|---|---|
| 42 | 0/1/1 | 3/0/1 |
| 43 | 2/2/4 | 2/1/2 |
| 44 | 2/3/2 | 1/1/0 |
| 100 | 2/0/2 | 2/1/2 |
| 101 | 0/1/2 | 3/2/1 |
| 102 | 2/2/3 | 1/2/2 |
| mean | 1.33 / 1.50 / 2.33 | 2.00 / 1.17 / 1.33 |
No-sleep recall is 0 on every group. Totals /12: use 5.17, equal 4.50. Paraphrase /4 per group (mean): use 0.50/0.83/1.17,
equal 1.50/1.17/1.17. False recall /4: use 3.0, equal 3.3 (any sleep gives false recall again). Drift KL 0.026 vs 0.029.
- k=10, use minus equal, seeds 42 43 44 100 101 102: 0, +2, +2, 0, +1, +1 (4 up, 2 ties, none down; one-sided sign test 1/16, n=6 so
  a lead only). k=10 4/4: use 1/6 seeds, equal 0/6.
- Use trades away the low-use facts: k=1 recall 1.33 vs 2.00 (use is worse on 2 of 6 seeds: s42, s101; better on 2: s44, s102), so total recall gain is small (+0.67/12).
- Paraphrase recall does not rise with use.

## What it shows
- Use counted inside the brain does what it should mechanically (strength = number of tellings, replay follows), and k=10 gains
  about one fact of four over equal replay, never losing to it. That is the expected sign, but small.
- It does not make a fact that the cortex can hold in context into a reliably consolidated one: with ~43 replays, 5 of 6 seeds
  still miss 1-3 of 4 reachable k=10 facts. The limit is not exposure to the fact. Likely candidates (untested here): 240 steps,
  rank-8 adapters shared by 12 facts plus KL guard, and cross-fact interference (false recall ~3/4 shows leaking names).
- "Reachable" in context is necessary but not sufficient (and the in-context check is with a fact told once, in a clean context).

## What it does not show
0.4B, the hippocampus-on path, spaced re-tellings, successful-retrieval boosts (inherited USE_BOOST, unit-tested only for re-telling),
any other step count. Strength is linear in tellings; a saturating or floored rule was not tried. Not a claim about "high-use facts never wrong".

## Next
Sweep the capacity/steps that bound consolidation on reachable facts alone (steps 480, rank 16, kl 1 vs 3) with the use arm; add a
replay floor for k=1 so use does not starve rarely told facts; fix false recall first (familiarity gate, `research/false-recall-1103`).
