# AMENDMENT V1 to the recall-voice prereg section 5 stop rule: the stored form goes next

Status: written 11-27. Amends `2026-11-23-recall-voice-dev-prereg.md` section 5 and HANDOFF L18. **Not in force until a
fresh skeptic has read it and a pointer line is added to the prereg.** Nothing is registered or run by this file.

## 1. Decision
**Yes, amend.** The store-what-was-understood method may be registered before any further routing method. It is
perspective lever 6 and the last one: see section 4.

## 2. Why, from the record
The stop rule rested on one premise: wrong-episode fetch was the cost (28 of K-P's 41 fetch-hurt rows), so a routing
method would pay and a perspective lever would not. Three results remove that premise.

1. **Routing was tried.** Three routing methods are KILLED (S_all, skip-permanent, J-space key) and the arc was stopped
   by its timebox on 11-26. The rule's "before a routing method" has been met in letter.
2. **A perfect routing method is bounded.** The memory ceiling forced the right episode on all 217 rows. Told ok rose
   +17, but only +4 outside the four repeated items (bar 6, fail), and user voice rose +10 (fail). No routing method
   can beat its own oracle. Verdict UNDEFINED, so fetch-side work is **not closed** and this amendment does not close it.
3. **The oracle's cost is the stored line.** Blind hand-read: all 7 of ORACLE's user-voice answers copy the reinstated
   telling verbatim ("User: My cat is called Pixel." -> "My cat is called Pixel."). The rise comes with any
   reinstatement (A +8, SHAM +15). So a better fetch delivers more of the user's own sentence. Routing and voice share
   one cause: what is stored is the raw user line.

The voice direction ceiling closed the residual-tag class, so the remaining untried class is the stored data itself.
Stage 0 shows the material exists: 36 of 54 answered told questions name the fact, floor met on all three seeds
(16 / 9 / 11).

Against, stated plainly:
- UNDEFINED is not NO-HEADROOM. The design's own branch asked for exactly this amendment in that case; it is not a
  relaxed bar on the ceiling.
- 7 of the 36 replies start My/I. Storing them stores the defect. The filter stays frame-blind (the voice scorer never
  picks training text); the ALL arm and the hand-read measure the damage.
- The echo is a hippocampus-on effect; the design reads hippocampus-off. Section 3 adds the arm that tests the echo
  directly.
- Self/other tag is still MISSING in `brain/PREREQUISITES.md`. A fail remains an expected failure of this form at 1.5B.

## 3. Amended sequencing
1. Fresh skeptic on this amendment; pointer line in the prereg; HANDOFF L18 reworded.
2. Blind hand-read of the 36 exchanges (`own2`). Floor stays 8 confirmed per seed, else UNDEFINED and not run.
3. Code, default OFF, flag-off byte-identical: the design's section 3 items, plus forcing an answered-exchange
   episode in `memory_ceiling.py` by a new switch (the design said that file is untouched; this amends it).
4. Register **one** Stage-1 prereg with two sides (below). Then run.
5. Held as before: reachable strength (needs a clean HEADROOM); any new routing method (own prereg, after this read).

## 4. Stage-1 screen, updated
KR seeds 319/320/321, night-13 store. The answered exchange is the lived question plus the brain's own reply; the host
writes no text.

**Side S, sleep (design section 5, unchanged):** A0, AE, AE-LOO, SHAM, NOCON, ALL. Hippocampus off, 67 told questions.
ENGAGED-S, UNDERPOWERED-LEAD and KILLED exactly as the design states.

**Side R, reinstatement (new; A0 weights, no sleep, forward passes only, forced fetch as in the ceiling harness):**

| arm | forced into the reinstatement |
|---|---|
| R-OFF | nothing |
| R-TELL | the right telling (the ceiling's ORACLE; must reproduce the echo) |
| R-EX | the right answered exchange |
| R-EX-SHAM | an answered exchange about another item, matched length |
| R-EX-LOO (lesion) | R-EX with the reply span blanked, question kept |

Rows: told questions whose item has a confirmed exchange. Rows where the probe question equals the stored question
are reported apart and carry no claim (they would only repeat the old answer). Fewer than 20 remaining rows pooled, or
R-TELL user voice not above R-OFF: side R UNDEFINED.

**ENGAGED-R, all of** (item-clustered bootstrap, bars checked against counted maxima first):
1. user voice given knows: R-EX below R-TELL, paired CI upper bound < 0, same direction on 2/3 seeds;
2. R-EX user voice at most R-OFF + 2 rows (no reinstatement-presence rise);
3. told ok: R-EX at least R-TELL - 2 rows, and R-EX above R-EX-SHAM with CI lower bound > 0;
4. R-EX-LOO loses clause 3 (the reply span carries it);
5. blind 40-row hand-read confirms the listener frame and counts verbatim copies of the stored reply.

**KILLED-R:** R-EX user voice not below R-TELL, or told ok more than 5 rows under R-TELL.

**What each outcome allows:**
- S and R engaged: Stage 2 stores answered exchanges in the hippocampus and replays them reply-masked at night.
- R only: Stage 2 stores them in the hippocampus, with night replay off for those episodes.
- S only: the design's Stage 2 (sleep-only list).
- Neither: perspective closes at 1.5B. No lever 7 until the self/other tag prerequisite is met by a new method class.

Named risk for Stage 2, not tested by forced fetch: a telling and its exchange are near-duplicates, and near-duplicate
separation is MISSING. Side R is a ceiling, as the oracle was.

Odds: R engaged about 35%; S as in the design (15%, lead another 20%); DEV life pass about 15%.
Compute: side S 18 sleeps; side R five forward passes over about 200 rows on pool CPU, under $5.
