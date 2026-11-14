# Storing the lived exchange at 1.5B: NOT A LEAD (E1 +7 < 10). Echoes halve, but on dev they become non-answers

Pre-registration `brain/research/2026-11-14-lived-episode-1.5b.md` (E0-E3, commit 37da3d61, before the run). Data
`results/2026-11-14-lived-episode-1.5b/` (`dev_tell.json` pool40, `dev_lived.json` pool41, `test_both.json` pool42;
`table.txt`). Oracle routing, unslept 1.5B, CPU fp32, deterministic.

| format | dev right (108) at 0.25 / 0.5 / 0.75 / 1 | dev echoes at G* | test told / para at G* |
|---|---|---|---|
| tell (now) | 26 / **58** / 52 / 35 | 13 | 7 / 7 |
| lived (telling + own reply) | 31 / 64 / **65** / 48 | 6 | 10 / 7 |

G* tell 0.5, lived 0.75. E0 holds (tell reproduces the 11-13 refit exactly), **E1 fails** (+7 < 10), E2 holds (+3),
E3 holds (13 -> 6). **VERDICT: NOT A LEAD.** Fact bootstrap of the dev difference: +7, 95% CI -4..+18, 10 facts up and
6 down. Not established.

What it shows: with the brain's own reply stored in the episode, echoes nearly vanish at every gain (at 1: 42 -> 9),
and the best gain moves up a little. On dev the removed echoes mostly became non-answers (none 33 -> 31-40), not
answers; on the 12 held-out facts more of them became answers (told 7 -> 10 at G*, 11 at 0.5). The stored reply is
whatever the brain said ("Nugget sounds like a cute name for your hamster!"), so it carries the second-person frame
only sometimes.

No change to the brain. Verdict on this storage format at this margin, not on perspective. Next methods to rank:
the lived format in the live six-seed test as a new pre-registration (the live sampled replies echoed more than the
oracle, so the gain there may be larger), or perspective at recall time (reinstating the episode with the speaker
role bound, so "User: my X" is read as the listener's X).
