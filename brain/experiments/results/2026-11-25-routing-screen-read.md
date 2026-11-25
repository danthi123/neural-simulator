# Routing screen read (D+G, SC) on the K-P rerun night snapshots: both KILLED

Prereg: `brain/research/2026-11-24-routing-screen-prereg.md` (landed 6701dda0b, bars fixed before the KR data).
Data: `2026-11-25-routing-screen/` (replay outputs, validity files, verdict.json, node logs, ship/run scripts).
Fetch level only. No reply claim. Not held-out (all three seeds were seen for design).

## Validity (section 4 V)
- KR lives (AWS g5 fp16; s319 at commit 2e7a4f6a6, s320/s321 at 4864e3274, which only adds the screen files) equal
  lifeKP on every probe row: `mismatches: 0` per seed (`valid_s*.json`), over 60 probes and 1162 rows per seed (counted
  from the life JSONs; the validity files record only the mismatch count).
- Replay: pool40/41/42, CPU fp32, one seed per node, under `tools/memcap.sh 11`, about 2 min per seed.
- Night voids: one, s321 night 5. The mismatched row is an identity row ("What's your name?", read only, in no bar):
  banked fetched nothing, A fetched idx 0 (sim 0.3163 vs banked 0.3161). The whole night is dropped as section 2 says,
  although its told and fire rows were unaffected. On every other night A reproduces the banked fetch idx and recalled
  text on every row (640 rows). No seed void. Status READ.
- 5 of 660 rows have a raw top-1 that differs from the banked fp16 read: exact near-duplicate ties, fire rows only (every
  told row's raw winner matches). They can move D/D+G fire counts, which no bar reads for D+G, not K1-K3. A's reproduction
  does not validate D, D+G or SC decisions on other near-threshold rows.
- Denominators therefore differ from the prereg table on s321 only: told 67 (not 73), fire rows 113 (not 125), A right
  13 (not 14), A fires 85 (not 94). K1's need becomes max(ceil(1.25 x 44), 54) = 55.
- Tripwire, max |sim_raw - banked| per seed: s319 0.0004, s320 0.0051, s321 0.067. The 0.067 is one s321 night-15 told row
  ("What is my favourite cheese?", 0.168 vs banked 0.234). A and D+G fetch nothing there, as banked, but SC fetches idx 89
  on this deviant read, which is unvalidated. The SC kill does not hinge on it. The 0.0051 (s320) and 0.067 deviations are
  not fp16/fp32 drift-sized; they are reported, not explained.

## Bars (pooled over s319/s320/s321)

| arm | right (A 44) | need | K1 | K2 rises on | K3 new wrong on A-right (max 1/seed) | K4 fire excess (max 1/seed) | status |
|---|---|---|---|---|---|---|---|
| D+G | 54 | 55 | fail | 1/3 | 1 / 5 / 4: fail | vacuous | KILLED |
| SC  | 46 | 55 | fail | 1/3 | 0 / 0 / 0 | -19 / +7 / -7: fail | KILLED |

**D+G is killed by K3** (right -> wrong on 1 / 5 / 4 rows, max 1 per seed), on any reading. Its K1 (54 vs 55) and K2
(1/3) fails exist only because s321 night 5 is void. Sensitivity, not a bar: with that night counted, pooled right is A
45, D+G 57 (need 57, K1 passes), and K2 passes (2/3 seeds). SC is killed by K1, K2 and K4 on either reading. Bars are not
relaxed.

## Per seed (told rows; fire rows)

| seed | arm | told n | right | wrong | no fetch | A-right to no fetch | A-right to wrong (K3) | gained on A-wrong | fires / fire n | identity fires |
|---|---|---|---|---|---|---|---|---|---|---|
| 319 | A | 80 | 16 | 50 | 14 | - | - | - | 80/118 | 21 |
| 319 | D | 80 | 26 | 8 | 46 | 1 | 0 | 11 | 64/118 | 14 |
| 319 | DG | 80 | 27 | 26 | 27 | 0 | 1 | 12 | 62/118 | 13 |
| 319 | SC | 80 | 16 | 47 | 17 | 0 | 0 | 0 | 61/118 | 16 |
| 320 | A  | 70 | 15 | 47 | 8  | - | - | 74/128 | 10 |
| 320 | D  | 70 | 14 | 17 | 39 | 12 | 11 | 53/128 | 10 |
| 320 | DG | 70 | 15 | 21 | 34 | 5  | 10 | 45/128 | 4 |
| 320 | SC | 70 | 15 | 44 | 11 | 2  | 2  | 81/128 | 11 |
| 321 | A  | 67 | 13 | 39 | 15 | - | - | 85/113 | 9 |
| 321 | D  | 67 | 19 | 9  | 39 | 10 | 16 | 42/113 | 10 |
| 321 | DG | 67 | 12 | 23 | 32 | 5  | 8  | 54/113 | 4 |
| 321 | SC | 67 | 15 | 31 | 21 | 1  | 3  | 78/113 | 16 |

Pooled: D+G loses 20 of A's 44 right fetches (10 to no fetch, 10 to a wrong episode) and gains 30 of A's wrong rows.
D loses 23, all to no fetch.

## What it says
- **SC (strength cap at 1.0) barely moves routing.** Right 44 -> 46, wrong 136 -> 122. Capping the hub's weight at a
  told-once strength is not enough. Not the method.
- **D+G turns wrong fetches into no-fetches more than into right fetches.** Of A's 136 wrong fetches it makes 46 no-fetch
  and 30 right; 60 stay wrong. It loses 20 of A's 44 right fetches (10 to no fetch, 10 to a wrong episode): the loss the
  prereg predicted (weighted-only winners).
- **Arm D (reference, unbarred) is reported, not claimed.** Raw ranking with its own existing switch: right 59, wrong 34,
  false fires 159 vs A's 239, told no-fetch 124 vs A's 37. Read against these bars it clears K1 to K4, but only because
  K3 ignores rows lost to no fetch (23 of A's 44 right rows). That is a post-hoc reading of an unbarred arm on seen seeds,
  so it is no result. D was KILLED before on the unslept R-raw screen (false fires rose 32 -> 41); here false fires
  fell. The difference is not explained.
  - D's cost: told rows go unanswered from memory (37 -> 124 no-fetch). Whether that helps or hurts the reply is unknown.
- **What stands.** D+G and SC both failed their bars. The ranking changes here trade confabulation (wrong episode) for
  abstention (no fetch). Whether raw similarity's top-1 is right is open: where the recorded texts allow scoring it is
  right on 69 told rows and wrong on 77, with 77 unscored (interpretation refuter). The near-duplicate-separation
  prerequisite stays "missing" in PREREQUISITES.md; this screen neither confirms nor removes it.

## Next method (named, not built)
- A pattern-separated key (dentate-gyrus-like sparse expansion of the cue/episode key) is the prerequisite this screen
  cannot supply. That needs its own record check and prereg. Record check (11-25) points at an existing old-track asset
  to read first, not rebuild: `research/findings/2026-08-19-replay-pattern-separation-DG-separator-keeps-similar-memories-discriminable-6seed-GO.md`
  (sparse-expansive DG recoding, 6-seed GO on a LIF substrate with two Jaccard-0.60 memories; separation halves false
  recall, a reliability cliff below that). Its scale (two memories, LIF) differs from a 98-episode RWKV-key store.
- If D is to be tested, it needs its own prereg on fresh seeds, with a reply-level bar (answer rate and told recall),
  not this screen's numbers.

## Refuters (sonnet x3, 11-25: recount, instrument, interpretation): all SURVIVES_WITH_CAVEATS, applied
Both KILLED verdicts confirmed by independent recount from the replay JSONs. Fixed: the mislabelled lost column (D+G lost
20 of A's 44 right, not 10); D+G's K1/K2 fails depend on the void night, K3 kills it on any reading; "raw similarity is
mostly wrong" was unsupported and is withdrawn; the void row is an identity row; the 0.0051/0.067 deviations and the 5
raw top-1 ties are disclosed. Lenses: (a) every arm engaged (decisions differ from A on recorded rows); (b) no default
flip, no PASS, answer-rate not run; D's abstention cost (told no-fetch 37 -> 124) is the thing a future D test must bar;
(c) `tools/where_ran.py` reports host UNKNOWN for all artifacts (replays ran on pool40/41/42 CPU per the logs' paths and
`ship.sh`; KR lives on AWS g5, host field ip-172-31-*). Validity does not depend on host: the replay needs only the life
JSON and the shipped snapshots.
