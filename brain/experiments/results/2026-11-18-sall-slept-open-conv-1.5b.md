# S_all + slept bar on open conversation (1.5B, seeds 1132-1137): selection MET (15/18 vs 4/18), replies flat, S4 fails, no flip

Prereg: `brain/research/2026-11-18-sall-open-conv-1.5b.md`, Amendment 1. Board #282, milestone 11-30 "recall from
reworded questions (meaning-based selection)".
- Raw: `results/2026-11-18-sall-slept-open-conv-1.5b/` (transcripts, rows, `scores.txt`, S4 key and reads).
- Pool CPU fp32, both arms of a seed on one node (rows `host`).
- Arm d = S0 (default); arm s = `--key S_all --key-bar slept` (0.165 / 0.02, fit on the DEV bank,
  `results/2026-11-18-sall-slept-fit-1.5b.md`).

**Intact arm, day 2, six seeds summed (rows.json):**

| | d | s | bar |
|---|---|---|---|
| S1 reworded right memory fetched | 4/18 | **15/18** | s >= d+4, s >= d on >= 5/6: **PASS** (6/6) |
| direct right memory fetched | 10/18 | 15/18 | |
| S2 direct + reworded replies ok | 24 (12+12) | 27 (13+14) | s >= d-1: holds |
| S2 abstentions | 0 | 0 | holds |
| S3 false + invented (never-told) | 25 | 25 | holds |
| S3 needless recall / intrusions | 14 / 2 | 7 / 0 | holds |
| hippocampus-off replies ok (lesion) | 29 | 27 | reported |
| S4 blind reader, s >= d | | 3/6 seeds | needs >= 5/6: **FAIL** |

S4 per seed (clean read, letters unblinded from `s4_key.json`), d vs s:

| seed | d | s |
|---|---|---|
| 1132 | 5 | 4 |
| 1133 | 4 | 4 |
| 1134 | 3 | 3 |
| 1135 | 5.5 | 5 |
| 1136 | 3 | 4 |
| 1137 | 3 | 1.5 |

- The readers judged 1133 and 1134 day 2 identical in both arms.
- On 1137, s collapses into "You're back!".
- A first S4 pass is void: a "needless-recall" marker leaked into the blinded files, more often in d. It gave the same
  3/6.

**Decision (prereg branch "S1 holds, S2-S3 hold, S4 fails"):**
- Selection is met as a read-out, and nothing flips.
- The milestone item "recall from reworded questions (meaning-based selection)" is met at the selection level on these
  six seeds. The registered-seed gate is running (Amendment 2): pool, seeds 42/43/44/100/101/102.

**What it is vs what it is not (3 sonnet skeptics, refuter lenses, all PARTLY REFUTED):**
- It is selection. With the slept bar, S_all fetches the right day-1 telling for reworded questions 15/18 against S0's
  4/18. S0's wrong fetches on told questions fall from 15 to 0, and needless recall halves.
- It is not a reply gain.
  - Replies move +3/36. The slept cortex already answers these questions with the hippocampus off: s intact 27 equals
    s off 27.
  - Under S0 the intact arm is worse than its own lesion (24 vs 29), so S0's wrong fetches cost answers. The new key
    removes that harm.
  - The right memory was fetched but the reply was still wrong on 6 turns (s) vs 3 (d).
  - Invented answers to never-told questions sit at 18/18 in both arms. This is the 1.5B ceiling defect, not touched.
- Day 1 differs between arms because the key acts on day-1 fetches, so the slept cortices differ. On seed 1136 the s
  arm's hippocampus-off cortex degenerates ("I'm Sam, nice to meet you"), the known seed-dependent collapse.
- Lenses:
  - (a) Engagement: the s header reads "key S_all bar slept"; fire and selection differ by arm.
  - (b) Answer rate: no abstentions in either arm; a "don't know" regex finds d 1/96, s 0/96. `tools/answer_rate.py`
    does not read these files.
  - (c) Where-ran: `where_ran.py` reports UNKNOWN; the hosts come from rows `host` (pool nodes).

**Next.**
- The gate on the registered seeds (selection read-out), then the reply-level problem, which is not selection:
  - the slept 1.5B brain makes up an answer for every never-told question;
  - it collapses into a repeated line on some seeds.
- Those are the K1/K2 defects of `results/2026-11-18-open-conv-defect-ranking.md`, KILLed at 1.5B for the in-context
  methods.
