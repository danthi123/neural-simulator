# Pre-registration: speaker contrast, follow-up: larger mu, judged on 36 fresh facts (1.5B oracle)

Written 2026-11-14 before the run. Follows `results/2026-11-14-speaker-contrast-1.5b.md` (NOT A LEAD by one held-out
fact: mu 1 dev 71 vs tell 58, echoes 13 -> 5, test told 8 vs 7, para 9 vs 7). Dev right rose with mu (58 / 65 / 71 at
0 / 0.5 / 1), and 12 held-out facts cannot resolve a few facts' difference. This is a new test with a larger,
untouched held-out bank, not a re-scoring: the 12 FACTS are excluded from it.

**Method.** `episode_format_probe.py`, formats tell and contrast_mu for mu 0.5 / 1 / 1.5 / 2 / 3, gains 0.25 / 0.5 /
0.75 / 1. Dev = DEV_FACTS36 (rows for tell, mu 0.5, mu 1 copied from the previous run; mu 1.5 / 2 / 3 run now).
Fresh = FACTS48 minus the 12 FACTS: 36 facts never used in this arc, 108 cues, every cell run (only the dev pick is
judged). Pool40 dev 1.5+2, fresh 2; pool41 dev 3, fresh tell+0.5; pool42 fresh 1+1.5+3. CPU fp32, deterministic.
Table `brain/experiments/speaker_contrast_table.py` (tested).

**Criteria.** Pick = the contrast (mu, gain) cell with the most dev right (ties -> smaller mu, then smaller gain);
tell = its best dev gain (0.5).
- S0 instrument: both cells present on fresh; tell right > 0. Else VOID.
- S1 fresh right: pick >= tell + 8 (of 108).
- S2 fresh echoes: pick <= half of tell.
- S3 no new failure mode: fresh generic replies pick <= tell + 5 (large mu could push into talk about the word).
LEAD -> the contrast in Brain.reinstate behind a switch (default OFF), live six-seed unslept test (pool).
