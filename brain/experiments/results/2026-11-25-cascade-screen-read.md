# Cascade screen read: decaying own-write (`Brain(own_decay)`) KILLED at both beta (K2). Fading the brain's own replies makes it stop speaking, and most flip nights are not a cascade

Prereg `brain/research/2026-11-25-cascade-screen-prereg.md` with its review amendment. Build 268ea9041. Data are in
`2026-11-25-cascade-screen/`:
- `casc_s*.json`;
- K5 inputs `og_E_decay.json` and `og_B_decay.json`;
- `k5.json` and `table.json`.

Offline replay on the KR night snapshots: pool40/41/42, CPU fp32, 3 seen seeds. DEV only, never a claim.

## Run record
- **The stopped first start.** The first start (18:47 UTC 11-25, `casc_done_s*.txt`) was stopped by me after 24 min (rc 143) after I saw D25
  replies going empty. Stopping it was a mistake: that was the registered method behaving as registered. The run was
  restarted unchanged at 19:11 from the same commit, and the partial outputs are kept on the nodes as
  `casc_s*.aborted1.*`. The read uses only the restarted run, and nothing in the method or the bars changed between the
  two starts.
- **Where it ran.** `casc_done_s*.txt` records node, commit and rc 0 for s319 on pool40, s320 on pool41 and s321 on
  pool42. The JSONs carry `host`.
- **K5.**
  - E ran on pool40 (`homes_d_s1121/hippocampus_off`, `echo_onset.json` index 4, hippocampus and night off, 6 samples).
  - B ran on pool41 (`home_m1121`, `ocm_s1121.rows.json` index 0, 4 samples).
  - Both used script seed 1121, force 2, `--gains d0.5 d0.25`.

## Bars (`cascade_screen table`, pooled over 3 seeds and both lesion arms)
| beta | core | K5 | kills | flip arm-nights | V_A | V_A2 | V_D | B diagnosis (CASCADE / WEIGHTS / BETWEEN) |
|---|---|---|---|---|---|---|---|---|
| 0.5 (D50) | KILLED | PASS | K2 on s319, s320, s321 | 19 | 111 | 83 | 30 | 2 / 7 / 10 |
| 0.25 (D25) | KILLED | PASS | K2 on s319, s320, s321 | 19 | 111 | 83 | 6 | 2 / 7 / 10 |

**Told right on the intact arm** over valid nights (s320 night 9 intact is void: A's fetch did not reproduce the
banked fetch), per seed s319 / s320 / s321:

| | told right |
|---|---|
| A | 78 / 46 / 46 |
| A2 | 83 / 50 / 46 |
| B | 61 / 49 / 50 |
| D50 | 34 / 17 / 15 |
| D25 | 4 / 2 / 1 |

**K5 echo per sample:** E echoes 5/14 under d0.5 and 2/14 under d0.25 (bar 10); B echoes 0/14. The pass is hollow: the
free replies go empty after turns 4-7, so neither echoes nor question loops can register. The six E samples are
identical, so they count as one sample.

## What happened (the whole battery read for 4 nights in every condition, 321 n2 / n3 / n7 / n8; a sonnet reader)
- **The cut is empty replies, not right ones.**
  - Empty told replies over 726 told rows per condition (718 without the void night: 362 / 522): A 0, A2 0, B 0,
    D50 367, D25 528.
  - Replies match A on rows 0-8 (story, greeting, "My name is Nova."). From about the first told row (row 9-10) the reply
    is the empty string, and it stays empty to the end of the battery. D25 goes empty at row 9-10 on 83 of 84 arm-nights.
  - So K1's V cut (111 to 30 / 6) is the brain not answering. With no state from its own earlier turns it emits the
    turn end at once.
- **Fading the reply track removes the turn format.** As the arm S read and the prereg risk named, the state needs its own
  replies to keep the "User / Assistant" turn-taking. Fading them at 0.5 per turn is already enough to lose it.
- **Most flip nights are not a pure cascade (by the registered B bands).**
  - Of 19 flip arm-nights, only 2 are CASCADE (B, each question from a fresh state, has V <= 1). 7 are WEIGHTS (B
    V >= 4) and 10 are BETWEEN.
  - Pooled V in B (165) exceeds A's (141) across the arms. A fresh-state brain gives as many user-voice and refusal
    replies as the running conversation. This is the clean, unselected comparison.
  - Caveats: B is one sample per night; A vs A2 differ (111 vs 83), so flip nights picked on A regress (B 58 on them).
    BETWEEN (B 2-3) is a partial cascade, not "no cascade".
  - Example of a cascade night, 321 n3 intact: A says "My cat is called Pixel." at row 9 and keeps the "My ..." form; B says
    "Your cat is called Pixel.".
  - Example of a weights night, 321 n8 hippocampus-off: B alone gives "My cat is called Pixel.", "My favourite cheese is
    manchego.". A adds a loop on top ("I'm Dan, your Dan." for 12 rows). A2 does not loop but has the same V (6).
  - This narrows the 11-24 flip-night diagnosis ("mostly an in-battery cascade", on two M1 homes). On the K-P rerun
    homes the user-voice form is mostly in the weights.
- **Sampling noise at the flip point.** A vs A2 V is 111 vs 83. 321 n8 hippocampus-off loops under A and not under A2.

## Instrument notes (from the reader; not acted on here)
- The told-ok scorer credits garbled or wrong-pronoun replies: "Of course, Dan, you Dan?" and "Her name is Pixel." (the
  cat) are ok=True. An empty reply is ok=False, so D's ok here measures how often the brain answers at all.

## Skeptic (sonnet, fresh context, REFUTER-LENS block)
"The KILLED decision stands." Every number recomputes (`cascade_screen table` reproduces `table.json`). The
wording fixes above came from it: the void night, the hollow K5, the cascade caveats, the overclaimed next lever and
the start time. Lenses: (a) the decay engaged, shown indirectly by D's empties, V and ok against A, and A matches the
banked rows; (b) D sharply lowers the answer rate, and it stays default OFF; (c) where_ran reports unknown, but the
JSONs' `host` fields give node-mxl1462drl / node-dl4g243 / node-dl5d243 (pool40/41/42), and the inputs are node-held.

## Decision
**KILLED, as registered**:
- `core` KILLED by K2 on every seed at both beta;
- `k5` PASS;
- `Brain(own_decay)` stays default OFF.

The verdict is on this method (fading the brain's own replies in the conversation state), not on the capability. On
these homes most of the user-voice form survives a fresh state: B >= 4 on 7 of 19 flip nights and 2-3 on 10, and the
unselected B V is 165 against A's 141. That points the next method at what the weights hold, meaning how sleep
consolidates the told items and their voice. This is a pointer, not a tested result: no weights-side lever ran here.
