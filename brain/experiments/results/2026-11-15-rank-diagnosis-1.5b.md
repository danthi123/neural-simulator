# Result: changing the adapter rank at lr 5e-4 does not lower false recall after sleep: rank 2 no lower, rank 32 loses the told facts (MIXED; R0 holds, R1-R3 fail)

Prereg: `brain/research/2026-11-15-rank-diagnosis-1.5b.md` (R0-R3, committed before any run).
Run: AWS batch 20261115-1318 (g6.xlarge L4, instance i-020093192a5d5cbcb), 18 jobs, all rc 0 (`run.log`). The JSONs
record no host (`tools/where_ran.py`: "30 file(s): unknown=30"); the AWS L4 provenance is the batch log and instance id only, and the rank-8 reference ran in another batch (20261114-2031). Reference: banked
rank-8 `nmoff_s*` / `nmon_s*` (`results/2026-11-14-near-miss-guard-1.5b/`). Every file re-scored
(`brain.experiments.rescore`) before reading; scorer `_scratch/score_rank.py` output in `table.txt`.

## Arm s (six-seed means, of 12)

| arm | rank | adapter norm (range) | told | slept | new | para | false | self | invented | told abstain |
|---|---|---|---|---|---|---|---|---|---|---|
| nmoff (ref) | 8 | 52-89 | 7.17 | 4.67 | 2.50 | 4.67 | **6.83** | 3.00 | 5.00 | 0.00 |
| nmon (ref, guard) | 8 | 47-102 | 6.00 | 3.50 | 2.50 | 3.17 | 3.83 | 3.00 | 6.67 | 1.17 |
| r2off | 2 | 34-41 | 6.83 | 4.50 | 2.33 | 4.17 | **7.83** | 3.00 | 4.17 | 0.00 |
| r32off | 32 | 90-400 | **2.00** | 1.67 | 0.33 | 1.33 | 5.67 | 3.00 | 6.33 | 0.00 |
| r32on (guard) | 32 | 104-342 | 3.67 | 3.00 | 0.67 | 2.67 | 5.50 | 2.00 | 6.17 | 0.67 |

False per seed (42/43/44/100/101/102): nmoff 11/4/10/5/7/4; r2off 11/7/10/2/12/5; r32off 8/0/12/3/7/4;
r32on 12/0/9/3/9/0.

## Criteria

- **R0 holds.** Every run records its rank (2 or 32); the adapter norms differ from rank 8 on every seed.
- **R1 fails (capacity does not set the frame).** Rank 2 false 7.83 vs 6.83 (+1.0), lower on 1/6 seeds. Told
  6.83 and para 4.17 are within the bars, so the narrow night learns the told facts about as well and still answers
  never-told questions with them.
- **R2 fails.** At rank 32 the guard lowers false only to 5.50 vs 5.67 (bar 0.6x); told 3.67 vs 2.00.
- **R3 fails, narrowly.** |r2off - nmoff| false = 1.00, at the < 1 bar; |r32off - nmoff| = 1.17.
- **Verdict: MIXED.** Per the prereg no next method is chosen from this decision rule.

## What it says

- **Rank 2 did not lower false recall.** A quarter of the adapter learns the told facts about as well (6.83 vs 7.17)
  and gives as many or more told facts to never-told questions (7.83 vs 6.83, lower on 1/6 seeds; per-seed swings are
  large, s101 12 vs 7, s100 2 vs 5). At unscaled lr the rank-2 step is also smaller (norms 34-41 vs 52-89), so this
  is a null at this lr, not a refutation of capacity: what it shows is that a narrower night at matched told does not
  cure the frame.
- **Rank 32 is confounded, not a capacity test.** The adapter has no alpha/rank scaling (`brain/plastic_cortex.py`:
  A random, B zero, no 1/r factor), so at the same lr 5e-4 a rank-32 night takes a larger step. Its norms run 90-400
  against 52-89 at rank 8 and told falls to 2.00 (no loss trace was recorded). R2 and the rank-32 half of R3 read that larger step,
  not only a wider adapter. "False" counts never-told questions answered with a told fact, so it can exceed told
  (5.67 vs 2.00): the facts leak into never-told answers even where the told questions fail. A matched test would scale lr (or the output) by 8/32; it is not worth running,
  given rank 2's answer.
- False recall stays at 5.5-7.8/12 at every rank tested: rank is not the lever at lr 5e-4 without scaling.
- The guard raises abstentions on told questions at both ranks (0.00 to 1.17 at rank 8, 0.00 to 0.67 at rank 32).
- Skeptic (sonnet, fresh context, lenses a-c): NARROWED; numbers, criteria and reference config (17 fields, zero
  differences) confirmed; the wording above is its narrowing.

## Next

Not chosen by this result's decision rule (MIXED). Chosen from the record instead: rank 2 gives no sign that the frame is
capacity, and the external research at this wall (Gekhman et al. 2024; Kang et al. 2025) predicts a content cause at
any rank: fine-tuning on new facts teaches the model to answer unfamiliar questions with the trained answers. The
10th method is the one the prereg named for a content verdict: the hippocampus-as-teacher night (the episodes in
context as the teacher of the never-told answers), with a no-training gate first. Pre-register before building.
