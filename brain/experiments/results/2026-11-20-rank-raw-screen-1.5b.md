# R-raw screen at 1.5B (night-7 homes s310-s314): KILLED on K1 and K2

Prereg: `brain/research/2026-11-20-hub-rank-raw-prereg.md` (rules fixed before the run, no bar refit).
Runner: `brain/experiments/life_screen.py rank`, job `brain/experiments/jobs/2026-11-20-pool-rank-screen.sh`.
Hosts: pool40 (s310), pool41 (s311, s314), pool42 (s312, s313). Each arm is a frozen copy of the banked trunk
home (FREEZE_USE), the same 20-turn frozen battery per home (6 H, 2 rotating, 2 near-duplicate, 3 never-told, 7 open/greet). Data: `2026-11-20-rank-raw-screen-1.5b/screen_rank_s31?.{json,md}`.

Arms: A = S0 key, strength-weighted (as banked); D = S0, raw-ranked; B = S_all key, weighted, slept bar;
R = S_all, raw-ranked, slept bar.

## Pooled (s310-s314)

| arm | fires on open/greet/never | right-episode fetches | wrong-episode fetches | H right (of 30) | open intrusions |
|---|---|---|---|---|---|
| A | 32 | 8 | 23 | 12 | 13 |
| D | 41 | 14 | 20 | 11 | 13 |
| B | 12 | 8 | 9 | 10 | 13 |
| R | 11 | 6 | 3 | 10 | 13 |

Top-1 share of fetched episodes on the homes where A > 0.40:

| home | A | D | B | R (n fetched) |
|---|---|---|---|---|
| s311 | 0.714 (Bramble) | 0.143 | 0.5 | 0.40 (5) |
| s312 | 0.538 (Rashid) | 0.222 | 0.75 | 0.75 (4) |
| s313 | 0.588 | 0.125 | 0.429 | 0.25 (4) |
| s314 | 0.5 | 0.125 | 0.714 | 0.50 (4) |

Engagement: weighted-only winners are 0 in every D and R arm (A: 10/11/7/15/7), so the switch engaged on every home.

## Kill rules

- K1 selection: R right = 6, needs >= max(2*8, 8+4) = 16 and >= B (8). **FAIL.**
- K2 hub: R top-1 is 0.75 on s312 and 0.50 on s314 (> 0.40). **FAIL.** Caveat: R fetches only 4-5 times per home,
  so these shares are 3/4 and 2/4; the prereg set no minimum n and the rule is applied as written.
- K3 open feed: R 11 <= 0.5*32 = 16. Pass.
- K4 replies: H 10 vs 12 (-2, within 3); intrusions equal. Pass.

**R-raw is KILLED** (K1, K2) under the prereg as written. Verdict on the method (raw S_all ranking at the slept
bar, as one package), not on hub-free selection.

Skeptic caveats (3 sonnet refuters, fresh context; verdicts SURVIVES, PARTLY_REFUTED, PARTLY_REFUTED):
- K1 was close to unreachable under that bar: R fired only 20 times over 100 probes, fewer than the 16 right
  fetches K1 asks for plus misses. S_all raw sims top out at 0.28-0.36 against 0.55 for S0, so told-once facts
  rarely clear sim*tanh(strength) >= 0.165. The kill is mostly a verdict on the S_all-at-slept-bar combination,
  low-power on raw ranking's selection itself. The data cannot separate key from bar (no S_all-at-old-bar arm,
  no bar sweep), so "bar drift" is a hypothesis, not a measurement.
- K2 rests on 4 fetches per home (3/4 on s312, a generic "new route to work" episode; 2/4 on s314, the permanent
  identity episode). Applied as written, weak as hub evidence. K1 is the robust kill.
- Host: `tools/where_ran.py` reports host UNKNOWN for all screen files (the screen JSON records no host field);
  the pool attribution comes from the job script. The screen reads only the banked homes, so validity is not
  host-dependent.
- Answer rate: 0 abstentions in all 400 rows, every arm.

## What the arms say (attribution, reported not ruled)

- **S_all at the slept bar mostly stops fetching** (fires 32 -> 12, fetches 70 -> 31 pooled). It cuts wrong
  fetches (23 -> 9 -> 3) but never raises right ones (8 -> 8 -> 6). Key and bar changed together, so which one
  causes this is not separable from these arms.
- **Raw ranking alone (D) dissolves the hub**: top-1 <= 0.222 on all four hub homes, and right-episode fetches
  rise 8 -> 14 (s311: 0 -> 3, H 2 -> 5). It fails K1 by 2 (14 < 16) and K3 (41 fires > 16), so the prereg's
  "D alone passes K1-K3" clause does not trigger. D keeps firing on open turns.
- **Pooled, replies move little with the fetch**: H is 10-12 in every arm and open intrusions are 13 in every arm.
  Per home the swings are real and cancel (A to D: s311 H 2 -> 5, s312 H 4 -> 2), and reply text is the same in
  A and D on only 8-18 of 20 turns per home. Open-turn replies are the same in all four arms on s310 (6/6) but
  only 3-4/6 on s311-s314.
  Expected-fail 3 held: at night 7 the slept cortex answers, the fetch rarely changes the reply. The same shows in
  the dev lives: hippocampus-off H equals intact H on most nights (`2026-11-19-life-screens-1.5b.md`).

## Next

Hub lever count is now 2 (M-use killed, R-raw killed). Per before-build section 7, research before a third lever.
Priority call: since the fetch rarely moves the reply at night 7, the reply-level defect is the slept cortex
(adapter growth, first-person speech), which M-scale treats; the hub is second. Candidate third levers to research,
not build: D (raw ranking on S0) plus a fire gate for open turns; the decay-clock rule (no last_used reset on
strength-decided fetches).
