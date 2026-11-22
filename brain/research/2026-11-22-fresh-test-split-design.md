# Capability: Lifelong learning
# Record: brain/experiments/results/2026-11-21-stage1-registered-test-1.5b.md
# Expected-fail: P0-v2 NOT VALID if forgetting is total (age >= 10 recall 0.0 on 4 seeds in Stage 1); priced in section 6
# TEST split v2 and the P0 instrument for the Stage-2 registered tests (2026-11-22, revised after review)

Design rules: `2026-11-19-multiweek-life-design.md` §3 (splits), §5 (P0), §6 (battery), §9 (rotation: "Any method change
after a TEST read writes a fresh TEST split; the old split is never re-tuned on"). Record check: `before_you_build.sh`
stops at "STALE CORPUS" (published main not in local main; not recorded), so the record was read by hand.
**Writer isolation.** Sections 1-3 quote v1 TEST outcomes and keys. The blind bank writer gets section 4 only.

## 1. Record brief
- Stage 1 (TEST v1, six seeds) read P0 **NOT VALID, 0/6**. The verdict stands. Never-told false rate 0.40-0.76 per
  seed (45 rows: 3 questions x 15 nights), defined as "the reply names any told key". 30 of 158 hippocampus-off false
  rows are stock self-names; the rest quote other told keys. LOW recall 0.289 at age 1 (n 90 pooled), 0.0 at age >= 10
  on s44/s100/s101/s102; the age >= 10 bin holds 20 rows pooled (2-4 per seed).
- Why so few old rows: an age >= 10 row needs an item told once on days 2-6 and probed on nights 11-15. The items exist
  (about 16 per seed); the battery asks only 2 rotating items a night and the cue ladder puts each item in one cell.
- Timings (six Stage-1 JSONs, g5 A10G): night 649-662 s; about 2.1-2.5 s a battery turn with span reads; one life 4.1-4.2 h.

## 2. Old-age power: the age-ladder sub-battery
Options rejected: more tellings on days 2-5 (changes the lived script, store and replay shares, so every bar moves);
more rotating questions (random, no age guarantee, and battery rows feed collapse, D1, D2 and the reader).
**Chosen: a fixed age ladder, own rows** (module `brain/eval/life_age_ladder.py`, flag `--age-ladder`, default OFF).
- Ladder items: class F or D, told exactly once on days 2-6, not in `life_ladder.live_asked_late` (the existing
  definition) and not a same-day question. Fixed script: F days 2-6 (10) + D days 3, 5 (2) = **12 per seed** (checked
  with `build_life`, structure only, on v1 TEST seeds and DEV 310-312/319-321). E (same-day question) and A (re-told day
  11) are excluded.
- Probe nights for told day t: t, t+3, t+6, t+9, 15. Ages (nights slept, AGE_OFFSET 1): 1, 4, 7, 10, 16-t. Day-6 items
  get 4 ages. Per seed: age 1 12 rows, 2-4 12, 5-9 12, >= 10 **22 rows on 12 items** (day-2..5 items 2, day-6 items 1).
- Matched-night age-1 controls: F/D items told once on days 11-15, not the same-day question, probed the night of
  telling (age 1, nights 11-15). 8 per seed on TEST-sized pools (days 11, 11, 12, 13, 13, 14, 15, 15). DEV has 0
  (its 18-item F pool ends on day 10), so controls are tested host-side on TEST-sized pools only.
- Clipping for `days < 15`: probe nights t+3k kept only if <= days, last probe min(15, days); a 3-day smoke has
  ladder items of days 2-3 at ages 1-2 only. The ">= 8 items at age >= 10" check runs on 15-day plans (host-only).
- Where: inside the hippocampus-off probe copy, after the battery; also intact (for FM-v2) and both-off (null) on full
  nights 5, 10, 15. Each turn starts from fresh state (`b.state = base`); questions are not stored (no recall with the
  hippocampus off; intact copies are discarded). Rows go to `p["age_ladder"]`, never `p["rows"]`, so collapse, D1, D2,
  summaries and reader units are unchanged.
- Question: the item's `direct` text; scorer `mark_item` (strict `correct_answer`).
- No double asking (test2 under the flag only): ladder and control items are excluded from the battery's rotating
  draw and from the D near-duplicate pool (`random.Random(seed)` sample over `LB.D` minus ladder D items). Checked on
  DEV: the current draws hit ladder items (rotating 5-10 per life; the D pool overlapped `d01`/`d01`/`d06` on 319/310/311).
  The pooled "v1 observation set" (battery + cue-C0 + ladder) is deduplicated by (item, night), keeping the ladder row.

## 3. Never-told reference and false memory
**Two separate quantities, registered here before any v2 run.**
(a) **Chance-recall floor (P0-v2 floor).** This is a chance-recall model (how often the strict scorer would credit a
same-slot told key to a question with no answer), NOT a false-memory audit.
- `NEVER_LADDER` questions carry a slot: thing, name or place (job dropped: v2 ladder items have no job slot, enforced
  by the section-4 slot rule; v1 s101 had 2 job items, so the rule is now a test).
- Lure set for a question = keys of items told by that morning with that slot, minus STOCK = {H `dan`, H `nova`} (the
  stock self-names; their leak is tracked by the stock-leak read-out, not here). Other H keys count.
- `slot_false = any(correct_answer(reply, k, q, told=telling))` over the lure set, `telling` = the ledger telling (H:
  its SESSION_1 line), so first-person echoes, generic lines, disclaimers and wrong-subject replies do not score.
- Rule for the stated cases: "Your landlord is called <same-slot told name>." scores; "I'm Nova" and "Dan is a name" do
  not (STOCK); "My sister is called <her name>" to a landlord question does not (wrong subject / echo).
- Set: **24 distinct questions per seed** from a 48-tuple pool (thing 22, name 14, place 12), slot quotas
  proportional to the seed's 12 ladder items (largest remainder, at least 4 per slot present), drawn with
  `random.Random(f"ladder-never:{seed}")`, disjoint from the battery and interview never pools. Split in two halves of
  12 asked on alternate nights (half A even nights, half B odd), so nights 11-15 give **60 rows on 24 questions**.
- Floor = mean over questions of the per-question rate on nights 11-15 (each question 2-3 rows). Effective n 24.
(b) **FM-v2, a separate false-memory bar.** Any-told-key rate on the same never rows: the reply names any told key, as
a word, excluding keys in the question and STOCK. Computed per question then averaged, nights 11-15, on hippocampus-off
(every ladder night), intact and both-off (nights 5, 10, 15; night 15 is the bar). Registered bar: on >= 5/6 seeds,
intact rate <= both-off rate + 0.10 AND hippocampus-off rate <= both-off rate + 0.10 (excess over the untrained cortex
is false memory). It is reported as its own verdict and never gates savings. The v1-style rate (any key, STOCK
included) is computed from the same rows for comparability with Stage 1.

**P0-v2 (the §5 thresholds, unchanged), per seed, VALID on >= 5/6:**
- drop: item-weighted recall(age 1) - recall(age >= 10) >= 0.15. Each item's mean over its rows first, then the mean
  over items. Age-1 pool = 12 ladder items (nights 2-6) + 8 matched-night controls (nights 11-15) = 20 items.
- log-odds lower at age >= 10 than age 1 (existing span rows, unchanged);
- floor: item-weighted recall(age >= 10) - floor >= 0.10.
- Pre-stated caveats: per-seed recall near 0.3 on 20 items has SD about 0.10 against a 0.15 drop, so the 5/6 count is
  low power; a pooled item-level permutation test (items nested in seeds) is reported beside it. The matched-night drop
  (controls vs age >= 10, both nights 11-15) is reported; if it is below 0.15 while the pooled drop passes, the report
  labels the drop night-confounded (no verdict change). Power and exponential fits run on ladder age bins.
- Offset-0 sensitivity: not computed on ladder data (ladder ages are fixed at AGE_OFFSET 1; at offset 0 the >= 10 bin
  drops day-6 items). Reported "n/a (ladder)".
- If forgetting is total (rL about 0), the floor clause fails and P0-v2 is NOT VALID: a measured result, not a fault.

**Host-side wording check** (test, before the freeze), for every `NEVER` and `NEVER_LADDER` question: no key or gist
key of any v2, v1 or DEV item in it (`strict_scorer._base` folding); its subject noun is not the subject of any v2 item;
at most 1 content token (`life_cues.content`) shared with any v2 text, no shared bigram; ends with "?"; slot is one of
the three and at least 3 v2 F/D/AB items carry it.

## 4. Content rules for `brain/eval/life_bank_test2.py` (the only section the writer sees)
- Same `Item` dataclass and layout as the v1 TEST bank. Counts: H 6, AB 48, E 20, F 60, D 18, NEVER 12, OPEN 12,
  SMALL_TALK 30, GREETINGS 8, INCIDENTS 8 + 8 with X/Y/new texts, INCIDENT_DAYS unchanged, `ALL`, `split_stats`.
- H is identical to v1 H (day 1 is weekly SESSION_1, the sanity anchor). All other content is new.
- `NEVER` keeps its tuple shape `(id, text, lure_keys)`; `NEVER_SLOT` {id: slot}; `NEVER_LADDER` 48 tuples
  `(id, text, slot)` (thing 22, name 14, place 12); `NEVER_TOPICS` for the interview. E items carry
  `extra["detail_slot"]`, `extra["gist_slot"]`.
- Slots: F thing 38, name 11, place 11 (no job); D all name.
- Keys are distinctive words. Key check (test): `K(B) = {_base(k) for x in B.ALL.values() for k in x.keys + x.gist_keys}
  | {_base(k) for n in B.NEVER for k in n[2]}`; `K(T2) & (K(DEV) | K(T1)) == {_base(k) for h in T2.H for k in h.keys}`;
  no non-H v2 key occurs as a word in DEV or v1 text, and the reverse; no 5-gram shared with DEV or v1; no 4-gram with
  `life_heldout.py`.
- Writer: a separate sonnet agent that has never seen brain output and does not open v1 TEST content. It gets the
  dataclass, the counts, this section and the check messages (which name only the clashing word). C2/C3 cue texts for
  v2 E and F come from the same writer into `brain/eval/life_cues_test2.py`.
- Frozen: `life_bank_test2.sha256`, `life_cues_test2.sha256`, committed with the P0-v2 code before the first v2 run.
  `--split test2`; v1 (`--split test`) stays runnable and hash-guarded.

## 5. Code changes (every new switch default OFF; flag-off byte-identical)
| File | Change |
|---|---|
| `life.py` `use_split` | accept "test2": import `life_bank_test2`, check `TEST2_HASH_FILE`; dev/test unchanged |
| `life.py` `main` | `--split` choices dev/test/test2; `--age-ladder` |
| `life.py` `battery` | under `--age-ladder` only: exclude ladder + control items from rotating and D near-dup draws |
| `life.py` `probe`, `run` | call the age ladder (hippocampus-off nights 2-15; intact and both-off nights 5/10/15) into `p["age_ladder"]`; plan in `res["age_ladder"]` |
| new `life_age_ladder.py` | `age_ladder_plan` (items, controls, nights, ages, clipping, never halves; named apart from `life_ladder.ladder_plan`), `run_age_ladder`, `slot_false`, `any_key_false` |
| `life_cues.py` | `CUES["test2"]` from `life_cues_test2`; `overlaps` default bank by split (dev, test, test2) |
| `life_ladder.py` `load_cues` | hash-check `split in ("test", "test2")` against the split's own sha256 file |
| `life_interview.py` | `getattr(bank, "NEVER_TOPICS", SP.NEVER_TOPICS)`; v1 spec and hash unchanged |
| `life_bars.py` | if `res.get("age_ladder")`: `p0_seed`/`p0` use ladder rows (item-weighted, controls, floor (a)); `age_bins_pooled` and `_fit` from ladder bins; offset-0 "n/a (ladder)"; dedup v1 observation set; new `fm_v2` bar; else old path byte for byte |
| `life_table.py` | ladder recall, slot-floor, FM-v2 and v1-style rate columns when present |

Tests: `test_life_bank_test2.py` (v1 validator cloned; key, n-gram, wording, slot counts, H == v1 H);
`test_life_split.py` (edited test2 bank or cues refused; `overlaps("test2")` reads the v2 bank; dev/test unchanged);
`test_life_age_ladder.py` (15-day plans on test2 and v1 TEST seeds: 12 items, >= 8 with an age >= 10 row, every item
>= 3 ages, 8 controls, none in `live_asked_late`; no ladder/control item in rotating or D-pool draws under the flag;
clipping at days 3 and 7; ladder rows never in `p["rows"]`; trunk byte-identical after probes; `slot_false` positive and
negative controls from section 3; flag-off life JSON/md byte-identical to HEAD for dev and test, 0.1B 3 days);
`test_life_bars.py` (no `age_ladder` -> byte-identical P0; synthetic ladder JSON -> hand-computed P0-v2 and FM-v2;
`life_bars.py` on the six v1 JSONs reproduces the committed `bars.md`).

## 6. Decision rules, budget, timebox
- **Probe budget per seed:** ladder 58 + controls 8 + never 12 x 14 = 168 (hippocampus-off) + intact and both-off on
  nights 5/10/15 (per arm over the 3 nights: about 21 due ladder/control rows + 36 never, so about 114) = **about 350
  turns, about 13 min** at 2.2 s, about 5% of a 4.1 h life, about $0.22 per seed.
- **Timebox:** build 11-23 (code, tests; bank and cues by the writer in parallel). DEV 1.5B life (new dev seed, AWS g5,
  about 4.2 h, about $4) launched by 11-24 09:00. Freeze 11-25 (hashes, code, this note, one commit). If the DEV life
  is not back by 11-25 12:00, the freeze slips one day; the 12-04 launch date is unchanged.
- **DEV life is a row and timing check only.** It confirms rows land, counts match the plan and timings fit. A P0-v2
  failure on it does not trigger F1 (§9.2) and changes no threshold or rule; F1 needs its own record check on DEV seeds.
- If any registered seed's plan gives fewer than 8 old items, the planner rule is fixed on DEV and in tests, never by
  choosing seeds or editing items after the freeze.
- **Slip rule:** past 11-24 evening drop the intact/both-off ladder (FM-v2 then reported on hippocampus-off vs the
  battery's both-off never rows, labelled weaker); past 11-26 ask never questions on nights 11-15 only.
- **What P0-v2 NOT VALID means (pre-stated).** Expected if forgetting stays total. Then savings is UNDEFINED for B,
  K-P and D2-PR alike, and the 18 lives (B + K-P + D2-PR, six seeds each, §13 command + `--split test2 --age-ladder`,
  about 77 g5 hours under $50/day) are read on the bars that do not need P0: language guard (K-P's target), D2 excess
  intrusion and stock-reply leak (D2-PR's target), collapse, cued recall, FM-v2, self-report and the whole-transcript
  read. If P0-v2 fails on the floor with rL about 0, F1 (which adds forgetting) is not the fix; if it fails on the
  drop, F1 is the candidate, decided after the read.
- **Go/no-go for the 18-life launch:** go only if K-P or D2-PR has passed its own DEV prereg gate; P0-v2 is a rider and
  never a reason to launch or to hold. If neither method passes, run B on v2 only (6 lives) or nothing.
- After v2 is read, any method change writes TEST v3 (§9). v2 is never re-tuned on.

**Not changed.** No old split, default, other bar, threshold, arm or life-script turn. **Tuning disclosure:** the
ladder cadence, the nights 11-15 floor window, the slot-matched floor, the quotas and the choice to change the P0
instrument were chosen after seeing Stage-1 outcomes (recall 0 at age >= 10, the false-row decomposition, specific
false keys). That is allowed under §9 (fresh split, v1 verdict untouched), but it is tuning on TEST v1 and is stated.

## Review (11-22, nine defects)
1. Slot floor hides cross-slot false memory; H stock names inflate it. **Fixed:** FM-v2 separate bar with threshold
   (3b), STOCK excluded from the floor, positive/negative `slot_false` controls and stated rule, floor named a
   chance-recall model.
2. Floor pseudo-replicated (6 questions). **Fixed:** 24 questions, per-question rates, effective n 24. Job questions
   dropped; the "ladder has no job slot" premise was false on v1 s101, so the v2 slot rule now enforces it.
3. Drop clause power, night confound, unequal weights. **Fixed:** 8 matched-night controls (not about 10: 8 on
   TEST-sized pools, 0 on DEV), item-weighted means, permutation and matched-night drop pre-stated.
4. "Nothing tuned on TEST v1" overstated. **Fixed:** tuning disclosure; writer gets section 4 only; DEV outcome rule
   pre-stated (no F1 trigger).
5. Battery re-asks ladder items. **Fixed:** `live_asked_late` reused; ladder/control items excluded from rotating and
   D-pool draws under the flag; dedup of the pooled set; tests on TEST-sized pools.
6. Missing P0 pieces and code sites. **Fixed:** bins, fits, offset-0 n/a, `life_cues.py` `CUES`/`overlaps`, test2
   cue hash, argparse choices; the new planner is renamed to avoid `life_ladder.ladder_plan`.
7. Probe cost. **Fixed:** re-budgeted to about 350 turns, about 13 min; DEV-life slip rule added.
8. Expected-fail not priced. **Fixed:** NOT VALID meaning, bars still read, go/no-go rule.
9. Minor. **Fixed:** clipping for `days < 15`, 15-day host-only check; floor population (60 rows on 24 questions,
   nights 11-15) and the v1-style rate from the same rows stated.
