# membrane-census pool RESULT `[census-pool]` (2026-10-26)

## What ran
- 60 dev census shards: seeds 1-12 x 5 groups (ask/conv/emo/infer/inner), `intact_a`, at main
  `79c6f5128` (provisioned pool40/pool41/pool42/wspool); dispatched 19:10-21:10, **60/60 exits
  rc=0**; server sync lagged node completion (last s12 files ~23:2x).
- Landed in four data commits (`72e24d3f5` s1-s4, `f051b505e` s4-s8, `608fac7c4` s8-s12,
  `5594f6090` s12 final) plus 12 per-seed aggregates (this cycle), each provenance-stamped by the
  runner.

## Per-seed aggregate (runner `--aggregate`; all clean: zero FLAGGED lines)
| seed | organs | out-of-range membranes | bridge instances |
|---|---|---|---|
| 1 | 32 | 13 | 189 |
| 2 | 32 | 13 | 189 |
| 3 | 32 | 12 | 189 |
| 4 | 32 | 12 | 189 |
| 5 | 32 | 12 | 189 |
| 6 | 32 | 12 | 189 |
| 7 | 32 | 13 | 189 |
| 8 | 32 | 13 | 189 |
| 9 | 32 | 12 | 189 |
| 10 | 32 | 12 | 189 |
| 11 | 32 | 13 | 189 |
| 12 | 32 | 12 | 189 |

## Reading (descriptive; no bar registered for this diagnostic)
- The non-physical-membrane defect **replicates on every dev seed**: 12-13 of 32 organs carry
  out-of-range membranes, 189 bridge instances each, with no aggregate flags — not a dev-7
  curiosity.
- Count note: `2026-10-26-membrane-census-dev7.md` reports **12 production** organ labels on dev 7
  while this census counts **13** labels on s7; the definition gap is flagged for the substrate
  packet, not resolved here.
- Reply impact is NOT measured (round2-1026(3)): substrate packet next; no organ tuning until it
  lands.

## Housekeeping
- Three stale `.part.json` pairs (s2 ask, s10 emo, s12 conv) remain untracked; the final files are
  complete and committed.
