# Fable design review of the DG port (11-27)

**Verdict: DO-NOT-RUN the DG screen as built. Run a reply-level oracle ceiling instead (a measurement, not a 6th routing lever).**

## 1. Is DG the right mechanism? No.

- **It acts on the wrong stage.** Screen 2's diagnostic (`brain/experiments/results/2026-11-25-routing-screen2-read.md`) puts the loss after ranking. Of the 133 raw-first rows not fetched right, 121 stop at or after strength weighting, and only 6 have raw similarity below the bar. The right episode's `tanh(effective_strength)` has median 0.36, against 0.85 on rows that fired right. `dg_separation.py` says in its own header that strength and hubs are "outside this module".
- **The biology does not match the stored items.** DG separation orthogonalises similar inputs at write time, plastically (the 08-19 GO). Here Volvo and the rotating items are already ranked first on 32/32 and 64/64 rows and still fetch 0 and 8. The deficit is the banked "strength signal the told episodes can reach", not overlap.
- **A read-only random expansion is a score reshaper, not separation.** k-WTA overlap is close to a monotone convex function of the dense cosine, which is why DG(S0) died on rank. Any gain would be a sharpening exponent, and the screen has no control for that: DGlesion and K-refit are both linear cosine.
- **The record already declines it.** The routing arc was closed on 11-26 with "No 6th lever", and prereg §8 itself says the real candidates are write-side or strength.

## 2. Does the screen change what we do next? No.

- **KILL (the builder's own prior, D9)** changes nothing; §8 is already written.
- **PICK** earns a live life at fetch level, where the record is negative:
  - The hippocampus hurts on K-P: told 123/223 with it, 138 without.
  - S_all on the slept gate moved fetches 1/18 to 14/18 and replies +4/36, with the lesion as good.
  - On 11-13, always-reinstate scored 7.67 against 9.33 for hippocampus-off.
- **The one unknown arm is K-refit**, and it needs no DG and probably no new compute. `2026-11-25-routing-screen2/s2diag_s*.json` holds per-row weighted score and margin; I did not confirm the fire rows carry the same fields.
- **The bar rule is weak.** It fits to A's 239 false fires, so it buys right fetches with A's worst defect.
- The seven code-review fixes are correct but repair an experiment that should not run.

## 3. Best alternative: oracle reply ceiling on the KR snapshots

No routing lever has ever been tested at reply level on the slept K-P brain with the right episode forced; oracle runs exist only on unslept cortex (11-13, 11-14). The question is whether a right fetch improves the reply at all. That decides whether any fetch-side work (DG, write-side separation, strength, source tag) is worth building.

**Stage 0 (about an hour, no new compute).** Rescore the s2diag files with the §5 bar rule for K. Report right, wrong, no-fetch and fires at the A-matched bar and at half A's fires. Report only.

**Stage 1 (extend `routing_replay --reinstate`, no new mechanism).**
- **Population:** s319/s320/s321, nights 5-15, the 217 told rows plus fire rows, same void rules.
- **Arms**, all through the brain's own reinstatement path at shipped gain, with no host text:

| arm | what |
|---|---|
| OFF | hippocampus lesion |
| A | banked fetch (reproduction check against banked replies) |
| ORACLE | right episode forced |
| SHAM | a forced other told episode, matched on strength tercile and age; the no-contrast control |
| ORACLE-rekey | right episode forced, delta re-read through tonight's cortex (the 184/217 pointer; separates drift from content) |
| K-refit | Stage 0 decision reinstated |

- **Measures:** strict told scorer; user-voice rate with the existing voice-split scorer; blind hand-read of 40 rows per arm; never-told fire rows for OFF, A and K-refit. Report pooled, per seed, the 89 non-H rows, and the 68 (seed, item) clusters, because 128 rows come from four items.
- **Pass rule, fixed before data:**
  - **HEADROOM** needs all of: ORACLE − OFF ≥ +15 of 217; ≥ +6 on the 89 non-H rows; positive on 3/3 seeds; cluster sign test with wins > losses; ORACLE − SHAM ≥ +15; user-voice not above OFF by more than 5.
  - **NO-HEADROOM** if ORACLE − OFF < +8 or ORACLE − SHAM < +8.
  - Anything between is UNDEFINED, with no rescue.
- **Compute:** about 1,100 replies on pool40/41/42 CPU fp32, one seed per node, under `tools/memcap.sh`. Per-reply time is unmeasured: time night 5 first, and move to the 3090 in fp16 if the projection exceeds 20 hours.
- **Operational fixes carried over from the code review:** per-night atomic files with skip and merge; a workstation guard; INCONCLUSIVE if any control arm is missing; verdict tests on synthetic records.

**What each outcome does:**
- **NO-HEADROOM:** close all fetch-side work, including the DG and source-monitor ports. Bank that the slept cortex already carries what reinstatement adds. The lever is what sleep stores and replays (both speakers' turns), after test2.
- **HEADROOM:** build the mechanism the diagnosis names, not DG. That is reachable strength: nightly replay strengthens the replayed episode's same trace, with homeostatic scaling of strengths to a set-point (`BIO/repetition-retrieval-strengthen-same-trace.md`, `BIO/homeostatic-scaling-relative-strength.md`). It is trained only by the brain's own replay schedule.
  - Arms: replay-strengthen plus scaling; scaling only; shuffled-strength sham; fetch-boost lesion.
  - Bars: right fetches and oracle-validated replies; false fires at most half of A's.
  - Scope: own prereg, six seeds live.
- **If a separator is still wanted after HEADROOM:** write-side only (orthogonalise a new telling against its nearest stored neighbours at `hippocampus.store`), scored on the 25 near-duplicate rows. Never a random expansion again.

## 4. Summary

Keep `key='dg'` default OFF, add no SCAFFOLDS row, and leave the prereg DRAFT marked "not run: wrong stage (strength, not rank or shape); outcome non-decisional". This is a verdict on the fixed read-only DG, not on separation.

Files read:
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/research/2026-11-27-dg-separation-port-prereg-DRAFT.md`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/dg_separation.py`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/experiments/dg_screen.py`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/experiments/routing_replay.py`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/experiments/results/2026-11-25-routing-ceiling-read.md`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/experiments/results/2026-11-25-routing-screen2-read.md`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/experiments/results/2026-11-26-routing-screen3-read.md`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/experiments/results/2026-11-20-rank-raw-screen-1.5b.md`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/experiments/results/2026-11-13-confidence-gate-1.5b.md`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/PREREQUISITES.md`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/HANDOFF.md`
