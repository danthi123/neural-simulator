# Routing screen 3 read. Part X: keeping permanent episodes out of the S_all recall competition is KILLED, because the freed rows fetch nothing. Part J: the J-space (Jacobian-lens) key is KILLED, because it ranks far below the plain hidden state

Prereg `brain/research/2026-11-26-routing-screen3-prereg.md` (two-lens review amendment before data). Build
42fff1b23. Data are in `2026-11-26-routing-screen3/`: `x_s*.json`, `x_verdict.json`, `x_done.txt`, `run_x.sh`.
Offline on arm S's own night snapshots, pool42 (node-dl5d243), CPU fp32, at about 5 min per seed. DEV, 3 seen seeds,
fetch level only.

## Part X

### Validity
- **Reproduction.** K reproduced the banked S fetch on every row of every night: 0 void nights, 0 dropped rows on all
  3 seeds. Its counts equal the live ones:
  - right fetches: 18, the same quantity as the live intact-frozen right fetches (18).
  - fetches of any permanent episode: 65 = 24 / 20 / 21. The live count of 65 counted only the self-name episode,
    so the numbers match but the definitions differ slightly.
- **Engagement.** KX fetched a permanent episode on 0 rows. There are 130 self rows (told rows where K's weighted
  winner is permanent), above the 10 required.
- **First launch.** The first launch, at 00:34 UTC, failed at once (rc 1, no output), because it loaded the default
  0.1B cortex without `--model 1.5b`. It was rerun with the flag, and that rerun is the data
  (`x_done.failed0.txt` on the node).

### Bars (KX vs K, told rows)
| | right | wrong | wrong non-perm | no fetch | fetched permanent | fires (fire rows) |
|---|---|---|---|---|---|---|
| K s319 / s320 / s321 | 6 / 7 / 5 | 31 / 23 / 24 | 7 / 3 / 3 | 43 / 40 / 44 | 24 / 20 / 21 | 31 / 23 / 27 |
| KX s319 / s320 / s321 | 7 / 9 / 8 | 13 / 6 / 3 | 13 / 6 / 3 | 60 / 55 / 62 | 0 | 18 / 3 / 3 |

**Self rows (130).** On these rows KX goes to right on 6, to wrong on 8, and to no fetch on 116.

**Kills:**
- X1: right 24 < 28 (K 18).
- X3: wrong non-permanent rises by 6 on s319 (the maximum is 5).
- X5: self rows going right are 6/130, below 1/3.
- X7: told no-fetch rises by 17 / 15 / 18 (the maximum is 5).

X2 holds: right rises on 3 of 3 seeds. X4 holds: fires fall. X6 holds: self rows going wrong are 8/130.

**Decision: KILLED.** No live SX life runs.

### What it says
- **With the self episode removed, the right fetch mostly does not appear.** Of the 130 self rows, 14 change to an
  episode (6 right, 8 wrong) and 116 go to no fetch. The self episode was capturing those rows, but on most of them
  nothing else would have fired either.
- **Why nothing fires is not measured here.** The harness does not log the right episode's weight, and the slept bar
  and the margin rule are not separated. That the told episodes' weights are too low (screen 2: median 0.36 on the KR
  stores) is an inference, not an observation from this run.
- **Report only.**
  - Fires on fire rows (open, greet, never-told, untold near-duplicate) fall from 81 to 24 pooled.
  - Excluding the permanent episodes removes most of the S key's false fires.
  - Identity rows lose the self episode by construction. This is not counted here.

## Skeptic, part X (sonnet, fresh context, REFUTER-LENS block)
"KILLED is CONFIRMED." Every number matches the rescore. Three wording fixes were applied above: the 65's
definition, "never what blocked" and "too low weight". Lenses: (a) the switch engaged (KX fetched a permanent
episode 0 times against K's 65); (b) offline only, no flip, though told no-fetch rose by 50 pooled; (c) pool42 per
`x_done.txt`, with the inputs held on the node.

## Part J

### Run and validity
- **J matrices.** AWS g5 batch 20261125-1937 (A10G, fp32, TF32 off, batched VJP), 24 held-out prompts of 57-62
  tokens, about 40 min. The forward check error is 0.0, and J_23 = I exactly (identity error 0.0). The meta is in
  `J.meta.json` (sha256 69e6e0bf...). The instance is terminated.
- **Smoke run first.** pool41 s320 with identity J. Jv equalled H and Jt equalled L on 70/70 rows, and S0 reproduced
  the ceiling.
- **Ranks.** pool40/41/42, one seed each, CPU fp32 (`j_done_s*.txt`: node-mxl1462drl / node-dl4g243 / node-dl5d243,
  commit 8d313e9ee), about 12 min per seed.
- **Validity.** 0 void nights here: S0 reproduced the ceiling's rank on all 217 rows (pooled S0 r1 90 = the ceiling's).
  s321 night 5 is void in the ceiling itself and contributes no rows. The result JSONs carry no `host`; it is in
  `j_done_s*.txt`.

### Bars (right@1, 217 told rows; non-H rows 89)
| key | layer 12 | layer 18 | non-H 12 / 18 |
|---|---|---|---|
| H (plain hidden state) | 121 | 134 | 62 / 62 |
| L (logit lens) | 27 | 19 | 24 / 14 |
| **Jt (J-lens readout)** | **47** | **33** | 23 / 12 |
| **Jv (J-mapped dense)** | **35** | **30** | 17 / 18 |
| Rv (random, J's spectrum; 42 / 43 / 44) | 79 / 64 / 75 | 132 / 114 / 126 | |

Reference keys:

| key | right@1 (217 rows) | non-H rows |
|---|---|---|
| S_all_live | 168 | 89 |
| S_all_rk | 184 | 88 |
| S12 | 160 | |
| S18 | 143 | |
| S0 | 90 | |

- **J1 fails for all four J keys.** The control (max of H, L and Rv) is 121 at layer 12 and 134 at layer 18, and every
  J key is 30-47.
- **J1n, J2 and J2n fail too.**
- **No H or L key clears J2.**
- **Decision: KILLED.** This J-lens method, on this cortex, at this scale.

### What it says (the cues' top J-lens tokens read for 2 nights, s319 n10 and others)
- **The J-lens does surface the concept behind a cue on this recurrent cortex.** Layer 12 Jt top tokens:
  - "What is my cat called?": "names", "nickname", "cats"
  - "What car do I drive?": "Chevrolet"
  - "What do I water every evening?": "plants"
  - "What is my sister's name?": "sisters"
- **But that concept mass is small against tokens shared by every cue.** " Хронологи", '?"', '?:' and
  broken-byte tokens lead most lists. So a top-25 key mostly matches what every text shares; centring removes the
  mean but not the noise.
- **The Jacobian map makes the dense key worse, not better.** Jv is 30-35 against H's 121-134. A random map with J's
  spectrum loses less (64-132), so J's specific directions hurt here.
- **Self capture.** The J keys put a permanent episode on top less often than L does (Jv 13 / 5 rows vs L 178 / 123).
  It does not help, because the right episode is not ranked either.

### Skeptics, part J (two sonnet lenses, fresh context, REFUTER-LENS block)
- **Numbers.** "KILLED stands." The table recomputes identically, every bar was applied by hand, and two wordings were
  fixed (above).
- **Construction (verify a negative).** "Artifact unlikely":
  - J's orientation is consistent (`h @ J.T` = J h, and J is not symmetric, so a transpose would matter).
  - Scale is harmless for the cosines.
  - Rv keeps J's spectrum and is applied the same way to store and cue. Rv18 is about H18 (132 vs 134), so it is J's
    own directions, not the spectrum, that cost Jv18 its rank.
  - The adapter-off J against adapter-on residuals is a real mismatch, but H, L and Rv on the same residuals rank
    fine.
- **Caveat.** J is a mean over 24 held-out prompts of 64 tokens and is badly conditioned at layer 12 (condition number
  about 1e5). A better-estimated J could differ. The verdict is on this method at this scale.

### Observations outside the bars (not pre-registered candidates; a pointer only)
- **S_all_rk = 184.** S_all re-keyed by reading every stored episode's text through tonight's cortex scores 184,
  against 168 for the stored-key S_all_live. On non-H rows it is 88 vs 89. The stored keys of older episodes may lag
  the slept cortex. The ceiling's S0drift found no such effect for S0. Still, this is rank only, and part X and screen
  2 show the gate, not the rank, is where told fetches are lost.
- **Excluding permanent episodes lifts H12 from 121 to 180.** This suggests the plain hidden-state key is heavily
  captured by the self episodes.

