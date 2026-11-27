# Source-tag screen read (11-27): NO-ENGAGEMENT by the registered rule

Prereg: `brain/research/2026-11-26-source-tag-prereg-DRAFT.md` (R1 rule, R2 onset widening). Harness:
`brain/experiments/voice_tag.py`, commit 0359fa89. Three KR lives (s319/s320/s321) re-slept night 14 from the n13 store, on
pool40 (host node-mxl1462drl, CPU fp32; recorded in each JSON and `st_done_full_*.txt`). s320 was killed twice by the
20 GB memcap (rc 137, about 80 s into the first sleep); the third run at 22 GB (`MEMCAP_GB`, infra only) finished rc 0.

## Registered verdict: NO-ENGAGEMENT (`verdict.json`)
- Not UNDEFINED: reproduction rel_diff 0.25 / 0.26 / 0.07 (cosine 0.968 / 0.966 / 0.998), all under 0.5; A0 knows 46.
- Engagement: TAG mean onset KL 0.0023 (per seed 0.0016 / 0.0030 / 0.0024) < max(0.01, 2 x SHAM 0.0013 = 0.0027).
- The GO clauses are not scored once the engagement gate fails.

## Report only (not scored)
- Onset (log P(You/Your) - log P(My), hippocampus off), pooled over 67 told questions:
  TAG - A0 +0.42 [0.21, 0.62]; LESION - A0 +0.44 [0.23, 0.64]; TAG - SHAM +0.34 [0.09, 0.61];
  A0 - A0snap (re-sleep and host noise) -0.17 [-0.32, -0.01].
- Per seed the shift is not stable: TAG - A0 is +0.30 (s319), +1.21 (s320), -0.31 (s321). s320 carries the pooled number.
- LESION ~ TAG on every seed (diffs -0.01 / -0.01 / -0.04). What the data supports: the shift is carried by the adapters
  learned in the tagged sleep, not by the tag at test time. Whether the learned tag or just the sleep re-draw caused it is
  not separated (re-sleep noise is about 40% of the shift).
- P(user voice | knows), per seed, A0 vs TAG: 0.16 vs 0.00, 0.33 vs 0.08, 0.50 vs 0.77. Direction reverses on s321.
- Knows / refusals, hippocampus off: A0 46/6, TAG 44/7, LESION 47/5, SHAM 32/14. The fixed random tag harms answers.

## Blind hand-read (sonnet, 40 rows per arm, same questions, arms hidden until labelled)
User-voice replies 9/40 in every arm (A0, TAG, LESION, SHAM). Addresses-user 22 / 19 / 20 / 15. Refusals 4 / 3 / 2 / 7.
Degenerate text concentrated in s320 ("Dr Dr. Dr. Dr.", "laps to the floor"). User-voice rows are mostly role flips of
correct content ("My cat is Pixel"). The detector's per-row `ok` disagreed with the hand label on 31/160 rows (lenient
hand reading of vague answers).

## Skeptic (fresh sonnet, refuter-lens block): PARTLY-REFUTED
Upheld: the registered label and its arithmetic; LESION ~ TAG. Refuted wording, now fixed here: "not caused by the tag"
and "learned during a tagged sleep" overreach (see the report-only bullets); "does not engage" overstates a null, since
the tag norm was 0.14-0.26 over 2048 dims and engagement was read at one token. (a) engagement: tag trained and applied
(nonzero KL vs detached), no masked-token count recorded; (b) answer rate: not a default-on flip, TAG knows 44 vs 46;
(c) where-ran: `tools/where_ran.py` reports unknown (known gap, HANDOFF item 5); the JSONs record pool40 CPU, and the
cuda-to-CPU re-sleep is the A0 - A0snap noise above.

## Verdict on the method
A small learned residual vector on user-turn tokens, trained only on user-turn replay texts, did not engage. Expected
from the prereg review: every training text is a user turn, so the tag carries no contrast to learn from. This is a
verdict on this tag form, not on the brain's ability to track who is speaking (PREREQUISITES: Self and identity).
