# READ 3: age-ladder DEV row check, fresh TEST split v2 (life lifeAL_s322)

Source: _scratch/stage2/read/lifeAL_s322.json (+ .md). device cuda, host ip-172-31-32-13 (AWS g5), split dev, seed 322, 15 days,
model rwkv7-g1l-1.5b-20261031-ctx34304, flags: age ladder on. Design: brain/research/2026-11-22-fresh-test-split-design.md (section 6:
"DEV life is a row and timing check only"; no F1, no threshold change). Light read only: no model loaded, no life run. Tests run
(CPU): test_life_age_ladder, test_life_bank_test2, test_life_split = 61 passed, 2 skipped. Code is in HEAD (commit 31de906ca, ancestor).

## Verdict
Row check PASSES on every condition checkable on DEV. Two design conditions are structurally not checkable on DEV (controls, never
rows: the DEV bank has no matched-night pool and no NEVER_LADDER); the design says these are tested host-side on TEST-sized pools
(covered by the passing unit tests, not by this life). Split v2 MAY be hash-frozen on this evidence. The P0-v2/FM-v2 "UNDEFINED"
on DEV is expected, not a fault, and per design triggers nothing (no F1, no rule change).

## Conditions
| condition (design) | expected | read from life | ok |
|---|---|---|---|
| ladder items | 12 | 12 (d01 d02 f01 f02 f03 f06 f07 f08 f09 f14 f17 f18) | yes |
| every item told once | 1 telling | all 12 have 1 told day (days 2-6: d 3,5; f 2,2,3,3,4,4,5,5,6,6) | yes |
| items with an age >= 10 row | >= 8 (n_items_age10p) | 12 | yes |
| planned vs landed hippocampus-off rows | 58 (4+4 day-6 items, 5 x 10) | 58 planned, 58 landed, 0 missing, 0 extra, 0 duplicate, ages match plan on all 58 | yes |
| ages per item | >= 3 | 4 (day-6 items) or 5 | yes |
| bins (hipp-off rows) | 12 / 12 / 12 / 22 | age 1: 12; 2-4: 12; 5-9: 12; >=10: 22 | yes |
| arm nights | hipp-off 2-15; intact, both-off 5/10/15 | exactly that (14 + 3 + 3 probes) | yes |
| intact / both-off rows | 19 each (design ~21 "due") | 19 and 19 (>=10 bin 12 rows each, 12 items) | yes |
| ladder rows kept out of p["rows"] | none | 0 ladder-kind rows in p["rows"]; 0 non-ladder kinds in age_ladder | yes |
| no double asking | ladder items excluded from battery | plan exclude = the 12 items (and 0 controls) | yes (see caveat) |
| controls (matched-night age 1) | 0 on DEV (18-item F pool ends day 10) | 0 | yes (as designed; not checkable here) |
| never-told ladder items | none on DEV bank (no NEVER_LADDER) | never A/B empty; slot_counts name 6, thing 6; no never rows | n/a on DEV |
| P0-v2 readable | rows parse | life_bars ran: ladder instrument used, bins filled, P0-v2 row emitted | yes |
| FM-v2 readable | rows parse | emitted, UNDEFINED (0 never rows) | n/a on DEV |

## Recall by age bin (strict mark_item, raw ok/n)
| arm | age 1 | 2-4 | 5-9 | >=10 |
|---|---|---|---|---|
| hippocampus-off | 6/12 = 0.500 | 6/12 = 0.500 | 2/12 = 0.167 | 5/22 = 0.227 |
| intact (nights 5/10/15) | 2/3 | 0/2 | 1/2 | 4/12 = 0.333 |
| both-off (null) | 0/3 | 0/2 | 0/2 | 0/12 = 0.000 |
Item-weighted hipp-off: age 1 0.500 (12 items), 2-4 0.500, 5-9 0.167, >=10 0.208 (12 items). life_bars P0-v2 (instrument age-ladder):
drop 0.292 (>= 0.15), log-odds 3.489 (n 37) at age 1 vs 1.972 (n 46) at age >=10 (lower, as required), floor_margin None (no never rows),
matched-night drop None (no controls), seeds defined 0/1, pooled perm_stat 0.292, p 0.071. Not a verdict (DEV, row check only).
Forgetting is not total here (age >= 10 recall 0.21, both-off 0), so the Stage-1 failure mode (recall 0.0 at age >= 10) did not recur on DEV.

## Engagement
Both-off arm 0/19 on ladder rows vs hippocampus-off 19/58 (corrected from 13/58, refuter recount: 6+6+2+5) and intact 7/19: the ladder rows measure told content, not chance. Rows are
fresh single-turn conversations; example row: f17 "What colour is my front door?" age 1 night 2.

## Caveats
- One DEV seed, one life; all rates have tiny n (per-item means are 0, 0.5 or 1). Do not read the P0-v2 numbers as a result.
- No controls and no never rows exist in this life, so the control count (8), floor (24 questions, 60 rows), slot quotas and FM-v2 are
  verified only by unit tests (passed), not by a live life. First live read of those is the first TEST v2 life.
- Double-asking exclusion is read from the plan's exclude list; whether the battery's rotating/D draws actually skipped them was not
  re-derived from rows (the rotating-row kinds are present: 96 rotating, 106 neardup). Covered by tests.
- Probe timing: probe seconds include the whole probe (battery plus ladder): hipp-off mean 26.8 s (max 41.3) over 21 probes, intact 36.7 s
  (15), both-off 28.7 s (5); night seconds 618-685. Ladder-only cost not separable from the JSON; no timing overrun visible against the
  design's 13 min per seed.

## Freeze hash coverage (do NOT write yet; per design sections 4 and 6)
Hash files, each `sha256sum` of the single source file, matching how life.py and life_ladder.py check them:
1. brain/eval/life_bank_test2.sha256 -> brain/eval/life_bank_test2.py (checked by life.py TEST2_HASH_FILE; missing file = refuses to run)
2. brain/eval/life_cues_test2.sha256 -> brain/eval/life_cues_test2.py (checked by life_ladder.CUES_TEST2_HASH_FILE; fail closed)
Freeze commit (design: "hashes, code, this note, one commit") also carries the P0-v2 code and the design note
brain/research/2026-11-22-fresh-test-split-design.md. Do NOT hash or edit life_bank_test.py, life_cues.py (v1, registered sha256 2055dc92...)
or life_cues_split.py. Pending before the actual freeze: any edit to the two v2 files after hashing invalidates it; the design also says
the freeze slips if this life is not back by 11-25 12:00 (it is back, file dated 11-23).

## Refuter corrections (2 refuters, both NOT refuted on the verdict)
- Ladder-only probe time IS in the JSON (age_ladder_seconds): hippocampus-off ~2.4 s per probe (14 probes, max 6.4), intact ~4.3 s, both-off
  ~3.6 s, ~57 s total, far under the design budget. Any caveat above saying it cannot be separated is wrong.
- Log-odds n (37 / 46) are span rows, not ladder rows (life_bars n_lo_age1 / n_lo_age10p), as the design specifies.
- Double-asking exclusion re-derived from rows: 0 battery rows carry any of the 12 ladder items (96 rotating + 106 neardup rows present).
- Intact use_restored True on 17/19 ladder rows. Host ip-172-31-32-13 / cuda from the JSON fields (tools/where_ran.py: unknown); no code sha recorded.
