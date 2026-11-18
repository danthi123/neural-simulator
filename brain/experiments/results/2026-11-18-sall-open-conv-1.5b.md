# S_all memory key on open conversation (1.5B), seeds 1126-1131: S1 FAILS; cause undetermined

Prereg: `brain/research/2026-11-18-sall-open-conv-1.5b.md`. Board #282. Raw: `results/2026-11-18-sall-open-conv-1.5b/`
(transcripts, rows, logs, `scores.txt`). Pool CPU fp32; both arms of a seed on the same node (rows `host`:
1126/1127 node-mxl1462drl, 1128/1129 node-dl4g243, 1130/1131 node-dl5d243). Arm d = S0 (default), arm k =
`--key S_all --key-bar size` (header "Brain defaults + key S_all bar size"; fire counts differ, so the switch engaged).

**Intact arm, day 2, summed over six seeds (from rows.json):**

| | d (S0) | k (S_all) | bar |
|---|---|---|---|
| S1 reworded right memory fetched | 3/18 | 3/18 | k >= d+4: **FAIL** |
| reworded: fired / right among fired | 15 / 3 | 6 / 3 | |
| direct: fired / right | 17 / 15 | 13 / 12 | |
| recall_fired (direct+reworded, both days) | 38 | 24 | low-fire branch: k < d-4 |
| S2 direct + reworded replies ok | 18 (11+7) | 21 (11+10) | k >= d-1: holds |
| S2 abstentions on memory questions | 6 (all seed 1130) | 0 | holds |
| S3 false + invented on never-told | 18 | 20 | k <= d+2: holds (at the edge) |
| S3 needless recall / intrusions | 17 / 4 | 4 / 2 | holds |

S4 (blind reader) was not run: no decision branch depends on it once S1 fails.

**Verdict.** S1 fails. S_all after a night does not fetch the right telling for reworded questions any more often
than S0 does. It fires less, at higher precision (3/6 vs 3/15).

**Skeptic (sonnet, refuter-lens block): PARTLY REFUTED.**
- The numbers hold.
- The prereg's low-fire branch ("the unslept bar is the cause") is a hypothesis, not a finding.
  - The rows record no S_all scores at the fire decision.
  - So "bar too high" and "no meaning selection after sleep" fit the data equally.
- Lesion read-out: reworded ok with the hippocampus off is 10, equal to k's 10 against d's 7.
  - k's reply gain is the removal of S0's wrong fetches, not better selection.
  - Wrong fetches by S0 cost reworded replies: the slept cortex alone does better than S0's recall.
- Lenses:
  - (a) Engagement is indirect only: different fire counts, no per-turn trace.
  - (b) Abstentions are k 0 vs d 6, so no answer-rate harm; no flip is proposed.
  - (c) `where_ran.py`: UNKNOWN for these files; the hosts above come from the rows and transcript headers.

**Next (follows the prereg's branch, with the skeptic's caveat).** Stage 1 of the slept-bar refit
(`research/2026-11-15-sall-slept-bar-1.5b.md`), running on the pool, seeds 42/43/44. It records the slept S_all score
of every episode for each paraphrase cue. That decides between the two readings:
- right episode ranked first on paraphrases → the bar is the cause; refit and rerun this test on fresh seeds;
- right episode not first → S_all does not select by meaning after a night at 1.5B → NO-GO on the method; next is a
  self-key on the slept cortex.
