# AMENDMENT V1 (revision 2) to the recall-voice prereg section 5 stop rule: the stored form may go next

Status: 11-27, revised after the skeptic REJECTED revision 1. Amends
`2026-11-23-recall-voice-dev-prereg.md` section 5 and HANDOFF L18. **Not in force until a fresh skeptic APPROVES this
revision; only then is the pointer line added to the recall-voice prereg.** Nothing is registered or run by this file.

## 1. Decision
**Yes, amend, on one ground only.** The stop rule says no further perspective lever "before a ROUTING method", read
here as "before a routing method has been tried". Three routing methods are KILLED (S_all, skip-permanent, J-space
key) and the arc was stopped by its timebox on 11-26. That is the whole case.

The store-what-was-understood method may therefore be registered as perspective lever 6, the last at 1.5B (section 5).

**Routing: neither closed nor resumed.** Fetch-side routing work is neither closed nor resumed by this amendment. The
memory ceiling is UNDEFINED; reachable strength stays held. It only permits side S before a NEW routing method
(own prereg).

## 2. What this amendment does not claim
- **No one-cause claim.** Reinstating ANY raw user line raises user voice (SHAM +15, ORACLE +10, A +8), so the
  revision-1 shared-cause claim is dropped.
- **Nothing about the memory ceiling.** Its verdict is UNDEFINED; fetch-side work is neither closed nor resumed.
- **No side R.** Reinstatement is dropped; `memory_ceiling.py` is not touched.
- The self/other tag is still MISSING (`brain/PREREQUISITES.md`): a fail is an expected failure of this form.

## 3. The store filter is the blind hand-read
Revision 1's frame-blind filter (36 of 54 pass) stored user-voice and hedged replies. It is replaced by the blind
hand-read of all 54 answered exchanges, labelled before the key was opened
(`brain/experiments/results/2026-11-27-voice-stage0/handread.md`, `handread_labels.json`; single reader).

| seed | answered | LISTENER-CORRECT | NEUTRAL-CORRECT | user voice | hedged | wrong fact | wrong frame | other |
|---|---|---|---|---|---|---|---|---|
| 319 | 18 | 6 | 6 | 2 | 3 | 1 | 0 | 0 |
| 320 | 18 | 1 | 4 | 2 | 8 | 0 | 0 | 3 |
| 321 | 18 | 4 | 5 | 0 | 6 | 0 | 1 | 2 |

Rule:
- **Primary filter:** only exchanges judged LISTENER-CORRECT enter AE.
- **Second arm:** LISTENER-CORRECT plus NEUTRAL-CORRECT.
- All other labels are excluded from both.
- **Floor:** 8 exchanges per seed under the primary filter, else that seed is UNDEFINED and is not run. A seed that
  is UNDEFINED under the primary filter is not run on the second arm either.

**Read today:** the primary filter gives 6 / 1 / 4, so all three seeds are UNDEFINED-by-floor (the second filter,
12 / 5 / 9, would fail seed 320). Stage 1 cannot run on the days 1-14 store. Borderline calls could move a count by 1
or 2.


## 4. Stage-1 screen, side S only
KR seeds 319/320/321, hippocampus off, the 67 told questions. The exchange is the lived question plus the brain's own
reply. Measures and bars: design section 5.

| arm | night replay adds |
|---|---|
| A0 | nothing (re-sleep noise floor) |
| AE-L (primary) | LISTENER-CORRECT exchanges (the You/Your replies), reply-masked |
| AE-LN (second arm) | LISTENER-CORRECT plus NEUTRAL-CORRECT, reply-masked |
| AE-LN-M | a fixed draw from the AE-LN set, equal in count to AE-L |
| AE-LOO (lesion) | AE-L without exchanges about pixel or marguerite |
| SHAM | equal count and length of lived small-talk exchanges, same masking |
| NOCON | the AE-L replies as bare text, no question, no labels, whole-text CE |
| ALL | every answered exchange (control only; never carried to Stage 2) |

ENGAGED, UNDERPOWERED-LEAD and KILLED are the design's, read on AE-L.

**Own replies or own correct-frame replies.** AE-L against AE-LN-M at matched count: onset above, CI lower bound > 0,
gives "own correct-frame replies"; otherwise, with AE-LN engaged against A0, SHAM and NOCON, "own replies" with no
frame claim. AE-LN against AE-L is reported only.

## 5. Stop line (binding; copy verbatim into the Stage-1 prereg)
> **No perspective lever 7 at 1.5B.** This is perspective lever 6 and the last. Whatever Stage 1 returns that was
> actually run (ENGAGED, UNDERPOWERED-LEAD or KILLED), no further perspective lever is designed, registered or run at
> 1.5B.
> Perspective reopens only by a new written amendment with a fresh skeptic, after the self/other tag prerequisite is
> met by a new method class or at a different scale.

**UNDEFINED-by-floor does not count.** An outcome that never ran because the per-seed floor was not met
(UNDEFINED-by-floor) does not consume perspective lever 6 and does not trigger the no-lever-7 line. The stop line
binds only after a Stage 1 that ran.

## 6. Sequencing
1. Fresh skeptic APPROVES this revision; only then the pointer line in the recall-voice prereg and HANDOFF L18
   reworded to match section 1.
2. Code, default OFF, flag-off byte-identical: design section 3 items only.
3. Stage-1 prereg registered with the section 3 filter, section 4 arms and section 5 stop line; run per seed only
   where the floor is met (today: no seed).
4. Held: reachable strength; fetch-side routing work; any new routing method (own prereg).

Compute: 8 sleeps x 3 seeds.

## 7. Data for Stage 1
The days 1-14 KR store is below the floor (LISTENER-CORRECT 6 / 1 / 4 against 8 per seed), so Stage 1 waits for a
larger store of answered exchanges.

Candidate source: the running Stage-2 test2 lives (18 lives x 15 days, K-P / B / PR arms), usable ONLY after the test2
read is complete. No test2 replies are read before then, so that read stays blind. After it, a fresh blind
hand-read is done under the same labels (LISTENER-CORRECT, NEUTRAL-CORRECT, user voice, hedged, wrong fact, wrong
frame, other), and the same floor applies: 8 LISTENER-CORRECT per seed, else that
seed is UNDEFINED-by-floor.
