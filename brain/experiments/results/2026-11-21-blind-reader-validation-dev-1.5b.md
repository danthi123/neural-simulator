# Design §7 blind reader validated on DEV (s316, s317, s318): canaries 6/6, decoys flagged, pairs consistent 9/9 after a position-bias fix; readers prefer the null brain to the intact brain in week 3 on 3/3 DEV seeds

DEV only (seeds 316 and 317 from the 15-night defaults run checks, 318 from the all-flags rehearsal). It validates the
instrument; it makes no registered claim. Code: `brain/eval/life_read.py`, spec `brain/eval/life_read_spec.py`
(sha256 6cf3690a68e1...), `--lived-context all`. Readers: 2 independent sonnet readers, blind (units T0..T28 shuffled,
sealed key). Material: `_scratch/stage1/readdev2/` (score in `score.md`; the first read is in `readdev/`).

## Validity (§7 bars)

| check | reader 1 | reader 2 |
|---|---|---|
| canary recall strict / loose (bar >= 5/6) | 6/6 / 6/6 | 6/6 / 6/6 |
| decoys (refuse, generic, keyword stuffer) flagged and at or below both-off | 3/3 | 3/3 |
| pairwise consistent across both A/B orders | 9/9 | (pairwise kappa 0.67) |
| duplicate spread (MDE, defects per unit) | 11 | 12 |
| answer-verdict controls (subject / echo / refuse -> no; answer -> yes) | pass | pass |

Agreement between the readers:
- Spearman of per-unit defect counts: 0.67 on totals, 0.92 on battery turns.
- Per type: from 0.39 (wrong) to 0.87 (experience claim).
- Kappa on coherence 0.85, on answer verdicts 0.94.
- Per-type kappa is weak for off-topic (-0.04) and repetition (0.0), so those two types are report-only.

## Instrument fixes made during validation (DEV, before any TEST read)

1. **Pairwise position bias.** In the first read, both repeated pairs (the same comparison shown again with A and B
   swapped) named different winners for both readers. Every pair is now judged in both orders, and a winner counts
   only when the two orders agree; otherwise the pair is a tie.
2. **Reader agreement.** Spearman of per-unit counts was added; the "any defect" kappa is uninformative because every
   unit has some defect.
3. **Fixes from the review before the first read** (`life_read.py` docstring):
   - each arm's battery now follows the trunk's lived days, so a unit's length no longer reveals its arm;
   - duplicates are never placed next to their copy;
   - canary facts come from inside the unit, and canaries no longer stand out by their format;
   - canary matching is strict on the exact planted turn;
   - defects are compared per turn across weeks.

## What the DEV read shows (report-only; DEV, n = 3)

- With pairs consistent in both orders, both readers prefer the both-off null brain to the intact brain in week 3,
  on 3/3 seeds. The §7 bar "intact preferred on >= 4/6" would FAIL on these DEV lives.
- The intact brain's week-3 defects:
  - experience claims go from 0 in week 1 to 12 (reader 1);
  - invented facts from 16 to 27;
  - stuck tails such as "That's a lovely place. I work as a baker." on about 17 consecutive lived replies (s316).
- RIGHT (strict scorer AND the reader's verdict):
  - intact 24/57, hippocampus-off 21/57, both-off 0/57;
  - the verdict never vetoes a strict pass (veto 0), so on these lives it adds little beyond the strict scorer.
- This is the cortex-side drift that Stage-2 D2-PR targets (`brain/research/2026-11-21-replay-priority-prereg.md`).
