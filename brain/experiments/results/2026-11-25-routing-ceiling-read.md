# Routing ceiling read: the all-layer key ranks the right episode first on 168 of 217 told rows (S0: 90), and on 89 of 89 non-repeated rows (S0: 56); DG and the registered cue offset dead

Prereg: `brain/research/2026-11-25-routing-ceiling-prereg.md` (kill rules and review fixes before data; harness commit
573bffc48). Data: `2026-11-25-routing-ceiling/` (per-seed rows, `table.json`, node logs, ship/run scripts).
A MEASUREMENT of rank order on the K-P slept store, not a method and not a fetch or reply result. All three seeds were
seen for design. Rank is the raw similarity order only: no strength weighting, no bar, no margin.

## Validity
- Population: the 217 non-void told rows of the 11-25 routing screen (s319 80, s320 70, s321 67; s321 night 5 void
  there). Every row scorable (the right episode is in every store).
- **The rows are not independent.** They are 68 (seed, item) clusters. Four H items (cat Pixel, work admin, sister
  Marguerite, car Volvo) are asked every night: 128 of 217 rows, 98 of the 136 A-wrong rows. The non-H rows (89) are
  reported separately below and carry the robust part of the result.
- Where ran: stored episodes, keys.pt and trunk deltas (fp16) come from the AWS g5 fp16 KR lives; cue reads ran on the
  pool nodes, CPU fp32 (log paths /home/node/brain-lane; `ship.sh` maps pool40/41/42 to s319/320/321).
  `tools/where_ran.py` reports host UNKNOWN for every artifact (no host field). K-sane below is a CPU-vs-CPU check.
- K-sane holds: S0's top-1 equals the screen replay's raw winner on every row of every night (0 insane nights), and S0
  right@1 = 90 = the raw winner's right count.
- Each night has 3 episodes with a plain routing change (`route_delta`, the self episodes); S_l / S_all read it, as the
  live key does (those are self episodes, competitors of the told tellings; S0 has no route_delta control row).
  3-5 question cues per night form the S0q mean (3 on the late nights).

## Pooled (217 told rows; A-wrong = the 136 rows where arm A fetched a wrong episode)

| key | right@1 | right@<=3 | right@<=5 | median rank | A-wrong right@<=3 | A-wrong rank > 5 | verdict (prereg 5) |
|---|---|---|---|---|---|---|---|
| S0 (default)  | 90  | 147 | 165 | 2  | 79/136  | 41 | baseline |
| S0q (cue offset) | 49 | 64 | 80 | 11 | 43/136 | 85 | dead (-41) |
| S6            | 141 | 181 | 185 | 1  | 108/136 | 28 | alive (3/3 seeds) |
| S12           | 160 | 204 | 213 | 1  | 127/136 | 2  | alive (3/3) |
| S18           | 143 | 198 | 205 | 1  | 128/136 | 5  | alive (3/3) |
| S23 (last)    | 21  | 50  | 76  | 9  | 36/136  | 86 | dead |
| S_all         | 162 | 192 | 205 | 1  | 116/136 | 10 | alive (3/3) |
| S_all_live    | 168 | 201 | 212 | 1  | 122/136 | 3  | alive (3/3) |
| S0drift       | 90  | 147 | 165 | 2  | 79/136  | 41 | vacuous (see below) |

Per seed right@1 (of 80 / 70 / 67): S0 37 / 25 / 28; S12 56 / 57 / 47; S_all 65 / 50 / 47; S_all_live 67 / 51 / 50.

**Non-H rows only** (89; per seed 36 / 26 / 27), right@1: S0 56 (26 / 13 / 17), S6 76, S12 68 (23 / 25 / 20), S18 68,
S_all 84 (34 / 26 / 24), S_all_live **89 (36 / 26 / 27)**. On these rows the prereg's deep-key rule (10% of the seed's
rows, 2 of 3 seeds) holds for S6, S_all and S_all_live; S12 on 2 of 3, S18 on 1 of 3.

**The four H items** (32 rows each, right@1 S0 / S12 / S18 / S_all_live): Pixel 0 / 31 / 19 / 28; Marguerite
2 / 30 / 10 / 18; Volvo 32 / 31 / 25 / 32; admin 0 / 0 / 21 / 1. "admin" is the item no all-layer key ranks first.

**DG(S0): dead.** No (E, f) cell gains at least 5 rows on all six map seeds with one sign; gains swing from -45 to +29
across map seeds (e.g. 4096 / 0.02: +5, -33, -30, -45, -30, +2). The hub rule is moot.
**Separation is not dead on this store**: under S12 / S_all_live the right episode is within rank 5 on all but 2-3 of
the 136 A-wrong rows.

## What it says
- **On the same question cues, an all-layer key ranks the right episode first far more often than the layer-0 key.**
  S_all_live: 168 vs 90 of 217, and 89 vs 56 of the 89 non-repeated rows (better than S0 on every seed). Single middle
  layers help on some items and not others (S12 fixes Pixel and Marguerite but not admin; S18 the reverse); the last
  layer is worse than S0. Whether cue form also limits routing is untested here (the one cue transform, below, was
  mis-specified).
- **Hub captures:** of A's 62 wrong rows on one of the night's three strongest episodes, S_all_live ranks the right
  episode first on 50 (S0: 16). 44 of the 62 are rows of the repeated H items, and all 12 misses are "admin".
- **A's competition loses a lot of S0's raw order.** Of the 90 rows S0 ranks right first, A fetches the right episode on
  24, a wrong one on 45 (strength weighting) and nothing on 21 (bar or margin); A is right on 20 rows S0 does not rank
  first. Net 90 -> 44. That is a separate effect from the key gain (+78 for S_all_live raw), and what the weighted, gated
  competition does to a deeper key is not measured here.
- **S0q as registered is dead (49 vs 90), and that says little about cue offsets.** The harness computed
  `centred_cos_one(K0, q0 - qmean)`, which also subtracts the store mean from the already-offset cue (the prereg text
  reads the same way). S0q's top-1 collapses onto a few episodes (median margin 0.116 vs 0.004). An offset with
  `q0 - qmean` as the centred cue was not run.
- **DG(S0) is dead by the prereg rule:** gains are sign-unstable across the six map seeds (pooled right@1 49 to 104
  across cells and map seeds). The cause (order preservation, as predicted) is not tested by this run.
- **Drift control is vacuous for S0, by construction.** Re-keying every episode through the night's cortex moves S0
  margins by at most 0.0006 and changes no rank or top-1: the layer-0 matrix memory is computed before any channel-mix,
  where the sleep adapters sit (`brain/plastic_cortex.py` header). The deeper keys compare a storage-night delta with a
  slept-cortex cue; their drift is UNMEASURED (the drift row is no control for them). Indirect support only: the deeper
  keys win most on the oldest episodes (right episode in the first quarter of the store: S0 42/144, S12 107/144,
  S_all_live 95/144; instrument refuter).

## What it does not say (caveats)
- Rank only. A live fetch is strength-weighted, gated by a bar and a margin. S_all was given its own bar before
  (11-03 realpath: S_all bar 0.185 / margin 0.05 vs S0 0.12 / 0.02), so any deeper-key arm needs its own bar, set
  without the KR told rows.
- On the slept gate, S_all moved reworded right-memory fetches 1/18 -> 14/18, replies only +4/36 with the lesion arm
  as good, and the intrusion bar failed (`results/2026-11-18-sall-slept-gate-1.5b.md`). At gain 0.5 the live default
  stayed S0 by 0.17 of a fact (`results/2026-11-14-key-default-1.5b.md`). A fetch gain is not a reply gain; the next
  test needs a reply-level bar and a lesion that changes the reply.
- Seen seeds (s319-321 shaped the screen design); 3 seeds, not 6. No generalisation claim.

## Refuters (sonnet x3, 11-25: recount, instrument, interpretation): all SURVIVES_WITH_CAVEATS, applied
Every pooled and per-seed number recomputed and confirmed. Fixed: the H-item clustering (non-H rows now reported);
the 90 -> 44 decomposition ("weighting costs as much as the key" withdrawn); S0q's double centring disclosed and its
"removes information" reading withdrawn; DG "as predicted" narrowed to the kill rule; drift reported as margins and as
unmeasured for the deep keys; the bar-scale figures replaced by a traceable record; the slept-gate citation corrected;
the 7-minute timing dropped (not recorded). Lenses: (a) every key engaged on all 217 rows, route_delta on 3 per night,
S0drift engaged with nil effect; (b) no default flip, no replies, answer-rate not applicable; (c) where_ran UNKNOWN,
split stated above.

## Next method (named, not built)
- **Routing screen 2 on the same KR snapshots, offline:** arm A vs an S_all_live arm (the robust key here) through A's
  own competition (strength-weighted, the existing margin), with the arm's bar set by a
  pre-registered rule that never reads a KR told row (candidate: the bar at which the arm fires on the night's fire
  rows no more often than A does), and K1-K4 as in the 11-24 screen plus a no-fetch count and non-H rows reported. Then,
  only if it passes, a live DEV K-P life with that key and a reply-level bar.
