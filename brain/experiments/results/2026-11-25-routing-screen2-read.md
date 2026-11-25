# Routing screen 2 read: the S_all key (slept bar) through the live competition is KILLED (K1, K2, K5); the right episode is ranked first but its weighted score does not clear the gate

Prereg: `brain/research/2026-11-25-routing-screen2-prereg.md` (bars and review fixes before data; harness commit
773718f39). Data: `2026-11-25-routing-screen2/` (registered run `s2_s*.json`, `verdict.json`; diagnostic rerun
`s2diag_s*.json`, commit f669ca403, which only adds per-row gate fields; `ship.sh` / `run_node.sh` there are the
diagnostic versions). Fetch level only, offline, seen seeds, 3 seeds.

## Validity
- Same population and void rule as the 11-24 screen: one void night (s321 n5, the same identity row); A reproduces the
  banked fetch on every row of the other nights. Status READ.
- Arm K was built by the brain's own code (`Brain._make_hippocampus("S_all")`, `key_bar="slept"`: bar 0.165, margin
  0.02, statement bar 0.45), FREEZE on; no episode changed (asserted per night).
- Where ran: `tools/where_ran.py` reports host UNKNOWN for every JSON; the `*_done_*.txt` files record the node per
  seed (s319 node-mxl1462drl, s320 node-dl4g243, s321 node-dl5d243; pool40 / 41 / 42 per `ship.sh`), CPU fp32, about
  4-5 min per seed. Inputs (snapshots, trunks, life JSONs) are on the pool nodes only.
- The diagnostic rerun reproduces every registered decision (A idx, K idx and K_sim_raw_max equal on 660 of 660 rows),
  and recomputing fire = (score_best >= bar and margin_w >= 0.02) from its fields reproduces K idx on every row.
- Void-night sensitivity: counting s321 night 5 gives s321 A 14, K 12 (still a fall of 2) and K1 50 vs need 56.

## Bars (pooled over s319 / s320 / s321)

| arm | right (A 44) | need | wrong | told no-fetch | false fires | K3 right->wrong | status |
|---|---|---|---|---|---|---|---|
| A (K-P as is) | 44 (16 / 15 / 13) | - | 136 | 37 (14 / 8 / 15) | 239 | - | - |
| K (S_all, slept bar) | 49 (22 / 16 / 11) | 55 | 40 | 128 (40 / 41 / 47) | 67 | 0 / 0 / 1 | **KILLED** |

Kills: K1 (49 < 55); K2 (rises on 2/3, falls by 2 on s321); K5 (no-fetch excess +26 / +33 / +32, max 5). K3 and K4 hold.
Non-H rows: right A 23, K 25 (of 89). The four H items (32 rows each, A right / K right): Pixel 0 / 4, admin 0 / 0,
Marguerite 21 / 20, Volvo 0 / 0. Identity rows fire A 40, K 51.

## Where the decision stops (a description, not an ablation)
On the 168 told rows where K's raw S_all similarity puts the right episode first (the ceiling's 168, reproduced):

| what happens to the raw-first right episode | rows |
|---|---|
| fired, right | 35 |
| same winner after weighting, but weighted score below the bar | 49 |
| weighting puts another episode first, which then falls below the bar | 37 |
| weighting puts another episode first, which fires (a wrong fetch) | 35 |
| weighting puts another episode first, blocked by margin | 8 |
| same winner, blocked by margin | 4 |

- Of the 133 raw-first rows K did not fetch right, 121 stop at or after weighting (below the weighted bar, or a rival
  wins on weighted score) and 12 at the margin only. The buckets are sequential: 43 of the 133 have weighted margin
  below 0.02 in all. Raw similarity below the bar itself: 6 rows. Whether raw ranking would have fetched them is not
  tested here, and R-raw with S_all was KILLED on 11-20 (`results/2026-11-20-rank-raw-screen-1.5b.md`).
- The right episode's weight `tanh(effective_strength)` has median 0.36 on the 168 (127 below 0.55), against 0.85 on the
  35 that fired right. By group: Volvo 0.25 (32/32 raw-first, K right 0), rotating 0.35 (64/64 raw-first, K right 8),
  near-duplicate 0.81 (25/25, K right 17). What made the weight low (age decay, no use boosts, stronger rivals boosted
  by S0's fetches) is not logged: no hub strength, uses or last_used per episode is in these files.
- K's other 14 right fetches come from rows where weighting put the right episode first although raw did not.

## What it says
- **On this S0-shaped store the all-layer key ranks the right episode first (168 / 217), but the weighted gate at the
  slept bar converts 49.** The raw bar passes nearly everything, so the weight carries both familiarity and the
  rivalry; this screen cannot separate accessibility from gating.
- **The slept bar was fit on young episodes** (the 2-day slept DEV bank, `results/2026-11-18-sall-slept-fit-1.5b.md`);
  these are 5-15 nights old. Bar-versus-age miscalibration is a candidate as much as the store's fetch history.
- **Records agree with the kill:** the 11-18 slept gate (live, 2 days old) shows fresh episodes are reachable at this
  bar; the 11-20 R-raw screen's weighted S_all arm matched A on right fetches (8 vs 8).
- **Pre-registered caveat 2 stays a hypothesis:** the store's strengths were shaped by S0's fetches; a live S_all life
  would boost what S_all fetches, but could equally let told episodes decay where it does not fetch (K no-fetch 128).
- Fetch-level trade: wrong fetches 136 -> 40, fires on fire rows 239 -> 67, told rows with no fetch 37 -> 128. Whether
  that is abstention in the reply is not measured.

## Caveats
- No banked tripwire for K (CPU fp32 cue vs fp16 stored deltas; `K_sim_raw_max` logged). Borderline rows unvalidated.
- The diagnostic fields were added after the registered read; they explain the kill and change no bar.
- Seen seeds, 3 seeds. Fetch level only: no reply was produced.

## Refuters (sonnet x3, 11-25: recount, instrument, interpretation): all SURVIVES_WITH_CAVEATS, applied
Every number recomputed and confirmed; KILLED on K1/K2/K5 as registered. Fixed: commit hash; the margin count (43 rows
involve the margin); "weighting blocks" restated as where the decision stops, not a cause; the S0-shaped-store story
marked a hypothesis (hub strengths not logged); the young-bar candidate and the 11-18 / 11-20 records added; "abstention"
withdrawn at fetch level. Lenses: (a) K engaged on every row; (b) no flip, no replies, not applicable; (c) where_ran
UNKNOWN, hosts from the done files.

## Next method (named, not built)
- **What blocks the right episode depends on the life's own history and on a bar fit on young episodes**, so an
  offline replay on an S0 life cannot settle it. The test is a live DEV K-P life with `key="S_all", key_bar="slept"` (an existing switch,
  nothing fitted) on fresh-for-this-question seeds, with its own prereg: told-fact reply recall, answer rate and false
  recall as bars, the hippocampus lesion as the reply-level control, and per-night strength of the right vs the hub
  episodes reported (strength, uses, last_used logged per night). Primary bar at reply level on nights >= 5, paired
  with lifeKP on the same seeds. That is a new test justified by this diagnosis, not a relaxed bar for this screen.
- Not proposed: changing the decay or the weighting form (strength is the declared accessibility mechanism; a refit
  would need its own record and prereg).
