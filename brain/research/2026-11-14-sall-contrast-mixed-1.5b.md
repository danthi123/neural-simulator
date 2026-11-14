# Pre-registration: S_all + speaker contrast on a mixed day with sleep (1.5B, six seeds, AWS)

Written 2026-11-14 before the run. Record: unslept day, S_all + contrast GO (`results/2026-11-14-sall-contrast-1.5b.md`:
told 5.83 -> 6.83, para 2.17 -> 3.50, false 0.83 -> 1.17, self equal; skeptics in that file). Not yet tested: sleep (a
night replays the stored text; the reinstated change on slept facts is now the contrasted one), the day-2 facts told
after a night, and a GPU (fp16) cortex.

**Method.** `mixed_day_probe.py --arms c,s,r --self-q` as part B of `2026-11-14-key-default-1.5b.md` (1.5B, `--steps
480 --lr 5e-4 --kl-weight 3`, slept half counterbalanced, seeds 42/43/44/100/101/102), twice per seed in ONE AWS batch:
tellmix_s* (no contrast) and contmix_s* (`--speaker-contrast 3` for both telling days). Same instance type as part B
(g6.xlarge L4; the earlier part B run is a reproduction check, reported). Table
`brain/experiments/mixed_contrast_table.py` (tested). Host decision: none.

**Criteria (arm r = S_all + size bar, gain 0.5; six-seed means).** X0: contrast recorded on every contmix run and none
on tellmix; r's recalled episode the same on >= 95% of paired rows (else VOID). X1 told (12): contrast - tell >= 1,
>= on 5/6 seeds. X2 slept (6): contrast >= tell - 0.5. X3 false (12): contrast <= tell + 0.5. X4 self (3): contrast >=
tell - 0.5. GO = X0-X4 -> land-result, then flip the 1.5B defaults to S_all + size bar + speaker contrast 3 (one
package), shown in a 1.5B weekly transcript.

**Expected.** New facts gain most (unslept-like); slept facts may move either way (the contrast acts on a slept cortex).
