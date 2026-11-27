# AMENDMENT V1 (revision 2) to the recall-voice prereg section 5 stop rule: the stored form may go next

Status: written 11-27, revised the same day after the skeptic REJECTED revision 1. Amends
`2026-11-23-recall-voice-dev-prereg.md` section 5 and HANDOFF L18. **Not in force until a pointer line is added to the
recall-voice prereg.** Nothing is registered or run by this file.

## 1. Decision
**Yes, amend, on one ground only.** The stop rule says no further perspective lever "before a ROUTING method". It is
read here as "before a routing method has been tried". Routing was tried: three routing methods are KILLED (S_all,
skip-permanent, J-space key) and the arc was stopped by its timebox on 11-26. That is the whole case.

The store-what-was-understood method may therefore be registered as perspective lever 6, the last at 1.5B (section 5).

## 2. What this amendment does not claim
- **No one-cause claim.** Revision 1 said routing and voice share one cause (the stored raw user line). The record
  does not support that: reinstating ANY raw user line raises user voice (SHAM +15, ORACLE +10, A +8), so the rise is
  not tied to the right episode. The claim is dropped.
- **Nothing about the memory ceiling.** Its verdict is UNDEFINED; fetch-side work is not closed.
- **No side R.** The reinstatement side is dropped entirely. `memory_ceiling.py` is not touched.
- The self/other tag is still MISSING in `brain/PREREQUISITES.md`: a fail is an expected failure of this form.

## 3. The store filter is the blind hand-read
Revision 1's frame-blind "reply names the fact" filter (36 of 54 pass) stored user-voice and hedged replies. It is
replaced by the blind hand-read of all 54 answered exchanges, labelled before the key was opened
(`brain/experiments/results/2026-11-27-voice-stage0/handread.md`, `handread_labels.json`; single reader).

| seed | answered | LISTENER-CORRECT | NEUTRAL-CORRECT | user voice | hedged | wrong fact | wrong frame | other |
|---|---|---|---|---|---|---|---|---|
| 319 | 18 | 6 | 6 | 2 | 3 | 1 | 0 | 0 |
| 320 | 18 | 1 | 4 | 2 | 8 | 0 | 0 | 3 |
| 321 | 18 | 4 | 5 | 0 | 6 | 0 | 1 | 2 |

Rule:
- **Primary filter:** only exchanges judged LISTENER-CORRECT enter AE.
- **Second arm:** LISTENER-CORRECT plus NEUTRAL-CORRECT.
- User-voice, hedged, wrong-fact, wrong-frame and other replies are excluded from both.
- **Floor:** 8 exchanges per seed under the primary filter, else that seed is UNDEFINED and is not run. A seed that
  is UNDEFINED under the primary filter is not run on the second arm either.

**Read against this rule today:** the primary filter gives 6 / 1 / 4. All three seeds are UNDEFINED. The second
filter gives 12 / 5 / 9 and would fail seed 320. So the Stage-1 screen cannot run on the days 1-14 store. Three
borderline calls could move a count by 1 or 2.

The floor is not lowered and the filter is not loosened. The screen runs only on a later store holding 8
LISTENER-CORRECT exchanges on a seed, counted by a new blind hand-read.

## 4. Stage-1 screen, side S only
KR seeds 319/320/321, hippocampus off, the 67 told questions. The exchange is the lived question plus the brain's own
reply; the host writes no text. Measures and bars as design section 5.

| arm | night replay adds |
|---|---|
| A0 | nothing (re-sleep noise floor) |
| AE-L (primary) | LISTENER-CORRECT exchanges (the You/Your replies), reply-masked |
| AE-LN (second arm) | LISTENER-CORRECT plus NEUTRAL-CORRECT, reply-masked |
| AE-LN-M | a fixed draw from the AE-LN set, equal in count to AE-L |
| AE-LOO (lesion) | AE-L without any exchange about pixel or marguerite |
| SHAM | equal count and length of lived small-talk exchanges, same masking |
| NOCON | the AE-L replies as bare text, no question, no labels, whole-text CE |
| ALL | every answered exchange (control only; never carried to Stage 2) |

ENGAGED, UNDERPOWERED-LEAD and KILLED are the design's, read on AE-L.

**Own replies or own correct-frame replies.** AE-L against AE-LN-M separates the two at matched count:
- AE-L onset above AE-LN-M, CI lower bound > 0: the wording is "own correct-frame replies".
- Otherwise, with AE-LN also engaged against A0, SHAM and NOCON: the wording is "own replies"; no frame claim.
- AE-LN against AE-L is reported only (counts differ).

## 5. Stop line (binding; copy verbatim into the Stage-1 prereg)
> **No perspective lever 7 at 1.5B.** This is perspective lever 6 and the last. Whatever Stage 1 returns (ENGAGED,
> UNDERPOWERED-LEAD, KILLED or UNDEFINED), no further perspective lever is designed, registered or run at 1.5B.
> Perspective reopens only by a new written amendment with a fresh skeptic, after the self/other tag prerequisite is
> met by a new method class or at a different scale.

## 6. Sequencing
1. Pointer line in the recall-voice prereg; HANDOFF L18 reworded to match section 1.
2. Code, default OFF, flag-off byte-identical: the design's section 3 items only.
3. Stage-1 prereg registered with the section 3 filter, section 4 arms and section 5 stop line; run per seed only
   where the floor is met (today: no seed).
4. Held as before: reachable strength; any new routing method (own prereg).

Compute if it runs: 8 sleeps x 3 seeds.
