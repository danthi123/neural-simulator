# Direction 4 PRODUCTION decisive multi-seed — Adversarial Reviewer Verdict (2026-05-26)

Fresh adversarial reviewer (no shared session history) reproducing the
DIRECTION_4_PASS verdict from raw data and exercising the 9 pre-registered
scrutiny items per
`docs/plans/2026-05-26-direction-4-production-adversarial-reviewer-prompt.md`.

The reviewer is ESPECIALLY alert because 3 of 4 recent NEGATIVES in this
arc turned out to be bugs (D5 sparse-pattern uniformity, D4 same-pattern
analog, etc.). A "clean PASS" after a bugfix is exactly the situation
where a different bug could produce artifactually-clean results, so all
9 items are checked with independent recomputation.

---

## Per-item adjudication

### Item 1 — Bug fix correctness (activity distinctness) — PASS

- `_DIRECTION_4_BRIDGE_LABEL_SEED_OFFSETS` map has **5 entries spaced
  100k apart**: A_nouns=0, B_verbs=100000, C_adj=200000, D_spatial=300000,
  E_functional=400000. Confirmed by direct import.
- Activity-distinctness check across ALL 10 cross-bridge pairs at seed 42:
  cosines range **0.021 to 0.052** — well below 0.99 BLOCK threshold,
  also well below 0.50 (no cross-bridge collision).
- Same A_nouns-vs-others check repeated across seeds 42, 43, 44:
  - seed 42: range 0.025 to 0.052
  - seed 43: range 0.037 to 0.072
  - seed 44: range 0.026 to 0.031
- Bug-fix diff is well-documented in the file header, explicitly cites
  the D5 c4e18f2 analog, and uses fixed integer offsets (deterministic +
  reproducible). No hash/RNG drift.
- Activity vectors are biologically plausible (sparse ~4-5% non-zero,
  per-neuron spike count max 9-12, mean ~0.1) — NOT the byte-identical
  pattern that defined the D5/D4 bug.

### Item 2 — Multi-seed reproducibility at production scale — PASS

- `direction_4_5bridge_production_bugfix.json` shows **15 / 15 training
  cells** completed (5 bridges × 3 seeds), all status="OK", zero failures.
  Each bridge has n_pool_union=3200 (full V14 scale). Per-cell wall ranges
  18.65-19.53 min. Total training wall = **285.47 min (4.76 hr)**, matches
  the reported ~5 hr.
- `direction_4_cross_bridge_production_bugfix.json` `per_seed` contains
  **3 entries** for seeds [42, 43, 44] — each with `per_load` keyed by
  {"2","3","5"}, each cell with `n_trials=200`. No missing cells.
- V=80, n_bridges=5, decoder_order_bearing=parallel_population_matching_batched,
  decoder_order_invariant=marginal_sum_phase_similarity_batched,
  mean_centring=per_bridge_local.

### Item 3 — Smell-test recomputation — PASS

Independent multi-seed-mean recomputation from per-seed values:
- L=2: OB=1.000000 (per-seed [1.0, 1.0, 1.0]); OI=1.000000 (per-seed [1.0, 1.0, 1.0])
- L=3: OB=1.000000 (per-seed [1.0, 1.0, 1.0]); OI=1.000000 (per-seed [1.0, 1.0, 1.0])
- L=5: OB=1.000000 (per-seed [1.0, 1.0, 1.0]); OI=0.976667 (per-seed [0.965, 0.99, 0.975])

All match JSON `aggregate` field byte-exactly (within 0.000001 tolerance).

### Item 4 — OB PASS at every cell — PASS

Multi-seed OB at L=2/3/5 = 1.000/1.000/1.000. Each ≥ 0.80, each = 1.000.
Smoke OB was also perfect (1.000); production matches identically. No
capacity-edge degradation at L=5 cross-bridge composition.

### Item 5 — OI characterization — PASS (key differentiator confirmed)

- L=2 multi-seed OI = 1.000 (clears 0.80; smoke also 1.000)
- L=3 multi-seed OI = 1.000 (clears 0.80; smoke also 1.000)
- L=5 multi-seed OI = **0.977** (clears 0.80 by +0.177 margin; smoke 0.983)
- Per-seed L=5 OI = [0.965, 0.990, 0.975] — tight clustering (std ~0.013),
  no seed-pathological dropout.

The L=5 OI of 0.977 is the strongest cross-bridge order-invariant result
in the project to date.

### Item 6 — Comparison to D5 hybrid + pillar n=95 — PASS

- D5 hybrid production (`direction_5_cross_bridge_production_bugfix.json`)
  L=5 OI = **0.790**.
- Pillar n=95 (cross_bridge_mode_unification_probe via OPTION 4) L=5 OI
  was also boundary at 0.790.
- D4 production L=5 OI = **0.977** — **dramatic outperformance**:
  +0.187 margin (23.7% absolute improvement) over both D5 hybrid AND
  pillar n=95.
- L=2 / L=3 also at 1.000 (vs D5 hybrid 1.000 / 0.998) — equivalent at
  the easier loads; the differentiator is at L=5 where D5 hits the
  marginal-sum top-K ceiling.

Biology insight (confirmed): the v14/v16 dedicated-pool architecture
produces CLEANER cross-bridge phase patterns at high load than D5's
shared sparse pool. The hybrid was a workaround for the bug, not a
necessary architectural feature.

### Item 7 — Parallel-matching primitive byte-unchanged — PASS

- `git log --oneline -- research/findings/raw/cross_bridge_mode_unification_probe.py`
  shows only 2 commits (cd30fc6 = pillar n=95 ship, 3e73ce3 = launch).
  No modifications since cd30fc6.
- `git diff cd30fc6..HEAD -- research/findings/raw/cross_bridge_mode_unification_probe.py`:
  **EMPTY DIFF** (no output). Primitive is byte-identical.
- D4 cross-bridge probe last modified at d162dc3 (Direction 4 Tasks 4-5,
  CPU code) and not touched since. The bugfix touched only
  `direction_4_bridge_builder.py`.
- `git diff HEAD~5 HEAD -- sim/ research/runners/`: **EMPTY** — no
  protected/frozen project modules modified.

### Item 8 — Builder fix non-default-breaking — PASS

- `_DIRECTION_4_BRIDGE_LABEL_SEED_OFFSETS["A_nouns"] = 0` preserves prior
  single-bridge behavior (a builder invocation with the canonical
  `label="A_nouns"` and `seed=42` yields `bridge_seed=42`, exactly the
  pre-bugfix value).
- Defensive fallback (`abs(hash(label)) % 900000 + 100000`) only triggers
  for unknown labels, never for the 5 canonical labels.
- `_build_bridge_core` signature unchanged; existing call sites unaffected.

### Item 9 — Score-tuning / threshold-tampering check — PASS

- `_DIRECTION_4_OB_MIN = 0.80` (frozen design value)
- `_DIRECTION_4_OI_MIN = 0.80` (frozen design value)
- `_DIRECTION_4_LOADS = (2, 3, 5)` (pre-registered ladder)
- `_DIRECTION_4_MIN_SEEDS = 3` (frozen)
- JSON metadata confirms: seeds=[42, 43, 44], bar_ob=0.8, bar_oi=0.8,
  loads=[2, 3, 5]. No post-hoc adjustment.
- Smoke → production consistency check: smoke L=5 OI = 0.983, production
  L=5 OI = 0.977 (delta 0.006 within reasonable seed-sampling variance).
  No "smoke peeking + threshold lowering" pattern visible.

### Bonus adversarial check — frozen verdict module on actual JSON

```
compute_verdict(per_seed_for_verdict) → "DIRECTION_4_PASS"
```

Verdict module returns PASS using the frozen 0.80 thresholds against
the production JSON, with proper key reconstruction from `"L=2"/"L=3"/"L=5"`
format. Independent confirmation of the runner-reported verdict.

---

## Final verdict

**CLEAR — all 9 items PASS.**

Direction 4 production decisive multi-seed result is **DIRECTION_4_PASS**:

- L=2: OB 1.000 / OI 1.000
- L=3: OB 1.000 / OI 1.000
- L=5: OB 1.000 / OI 0.977

The cross-bridge composition extends to 5 × 16 = 80 biology-faithful
concepts on bio_brain_regions substrates (v14/v16 dedicated-pool recipe),
with order-invariant accuracy at L=5 = 0.977 (well above 0.80 bar) and
dramatic outperformance of both D5 hybrid (0.790) and pillar n=95
(0.790) at the same cell.

The bug fix is correct (5 distinct bridge seeds via 100k offset map),
preserves single-bridge default, and the cleanliness of the result is
NOT a different-bug artifact (activity vectors are biologically
plausible, cross-bridge cosines are 0.02-0.05 range, smoke↔production
agree to within 0.006).

## Pillar n=108 candidacy

**VALIDATED quality (PASS, not BOUNDARY).** L=5 OI = 0.977 is dramatically
above the 0.80 bar (+0.177 margin) and significantly above both D5
hybrid and pillar n=95 (+0.187 over each). This is the strongest
cross-bridge composition result in the project to date.

Pillar n=108 framing: Direction 4 dedicated-pool bio_brain_regions
cross-bridge composition (5 bridges × V=16 = 80 concepts).
OB perfect every cell + OI 1.000/1.000/0.977 at L={2,3,5}.
Hybrid architecture's shared sparse pool was a workaround for the
cross-bridge uniformity bug, NOT a necessary architectural component;
pure dedicated-pool wins decisively at the L=5 capacity edge.

## Concerns noted (non-BLOCK)

None material. All 9 items clear; reviewer-triggered defensive checks
(full pairwise cosines, activity statistics, smoke↔production
consistency, verdict-module re-run on JSON) all consistent with a
genuine PASS rather than a different-bug artifact.
