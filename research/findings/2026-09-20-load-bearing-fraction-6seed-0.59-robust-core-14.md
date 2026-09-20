---
type: finding
status: live
lane: load-bearing
date: 2026-09-20
---

# Load-bearing fraction — 6-SEED: 0.59 ± 0.02 <!--derived--> robust core of 14 (+2 borderline) (2026-09-20)

The single-seed baseline (16/26 = 0.615 <!--derived-->, seed 42, `2026-09-19-load-bearing-fraction-baseline-16of26-first-reading.md`)
now has its mandated 6-seed error bar (42/43/44/100/101/102). It **refines the headline down and honestly**: the
single seed was the high end, not the center.

## Result

**mean = 0.5897, std = 0.0181** (n=6), range 15/26–16/26. Every seed deterministic (0 dirty null-controls).

| seed | fraction | | seed | fraction |
|---|---|---|---|---|
| 42  | 15/26 | | 100 | 16/26 |
| 43  | 15/26 | | 101 | 16/26 |
| 44  | 15/26 | | 102 | 15/26 |

- **14 faculties load-bearing in ALL 6 seeds** (the robust core): comprehension-monitor,
  comprehension-learned-animacy-cue, comprehension-learned-verb-selects, affect-drives-response,
  da-mode-drives-response, confidence-forthcomingness, pragmatic-implicature, metacog-monitor, worldmodel-forward,
  curiosity-followup, reconsolidation, gnw-multistep-deliberation, self-initiated-utterance,
  vision-identity-spiking-hmax.
- **2 seed-dependent / borderline**: `affect-marker-spiking-wta` and `source-provenance-honesty` — load-bearing in
  some seeds, not others. These two are what move the fraction between 15 and 16.

## Honest reading

- The robust #1-metric estimate is **~0.59 <!--derived--> (14 solid + 2 borderline)**, not a crisp 0.615 <!--derived-->. The
  0.615 <!--derived--> single-seed reading was a favorable point, not wrong but not central.
- **Local seed-42 read 16/26; AWS seed-42 read 15/26** — same seed, different value. This is not a seed-threading
  bug (all runs deterministic, seed verified to reach the substrate below); it is the two BORDERLINE faculties
  flipping under local-vs-AWS numerics (BLAS/numpy build differences). It is itself evidence those two are
  marginal, not robustly load-bearing.
- Functional read-out only (a lesion changes the produced decision on the probe); asserts nothing phenomenal.

## Method / provenance

- Ran on an AWS r7i.4xlarge CPU box (numpy, no cupy), 6 seeds in parallel, BLAS capped to 2 threads/proc (a first
  attempt oversubscribed 6×full-core → load ~96; capping fixed it). Full brain (corpus present → XEDGE intact,
  matching the local baseline — a first run silently degraded without the corpus and was discarded).
- Seed control: seed 42 reproducible + seed 43 substantively different substrate (da_level, eff_threshold,
  mean_magnitude all differ) — the seed genuinely reaches the substrate, not a seed-trap.
- Produced via branch `research/seed-threading-lbf` @ `5b718e73` (BRAIN_CHAT_SEED threaded end-to-end;
  byte-identical at the default seed 42). **NOT yet merged to main** — pending a default-path no-regression confirm
  before the production `server.py` changes land.
- Aggregate (mean/std/per-seed/robust/borderline): research/findings/raw/_load_bearing/load_bearing_6seed_aggregate.json
- Per-seed reports: research/findings/raw/_load_bearing/load_bearing_s42.json (+ s43, s44, s100, s101, s102).
- Raw run log: research/findings/raw/_load_bearing/_6seed_aggregate.txt

## Next

1. Update the ledger `headline.load_bearing_integrated` with the 6-seed error bar + the 2 borderline faculties.
2. No-regression confirm of the seed-threading default path, then merge the branch.
3. The 2 borderline faculties (affect-marker, source-provenance) are the natural next hardening targets alongside
   the hollow set.
