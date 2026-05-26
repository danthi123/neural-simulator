# Direction 5 PRODUCTION decisive multi-seed — Adversarial Reviewer Verdict (2026-05-26)

Fresh adversarial reviewer (no shared session history) reproducing the
DIRECTION_5_PARTIAL verdict from raw data and exercising the 9 pre-registered
scrutiny items per
`docs/plans/2026-05-25-direction-5-production-adversarial-reviewer-prompt.md`.

---

## Per-item adjudication

### Item 1 — Bug fix correctness — PASS

- `_BRIDGE_LABEL_SEED_OFFSETS` map has **5 entries spaced 100k apart**:
  A_nouns=0, B_verbs=100000, C_adj=200000, D_spatial=300000, E_functional=400000.
- Independent regeneration of `generate_sparse_patterns(n_concepts=16,
  n_pool=2000, pattern_size=100, seed=42+offset)` yields **5 distinct**
  `pattern_0` sets across the 5 bridges (sorted-first-5 indices all
  different).
- Pairwise pattern_0 overlap matrix: max 10.0%, mean ~5.7%. Well below the
  80% BLOCK threshold.

### Item 2 — Multi-seed reproducibility at production scale — PASS

- `direction_5_5bridge_production_bugfix.json` shows **15 / 15 training
  cells** completed (5 bridges × 3 seeds), all status="OK", zero failures.
  V_total=80, n_bridges=5, seeds=[42,43,44].
- `direction_5_cross_bridge_production_bugfix.json` `per_seed` contains
  **3 entries** for seeds [42, 43, 44] — each with `per_load` keyed by
  {"2","3","5"}, each cell with `n_trials=200`. No missing cells.

### Item 3 — Smell-test recomputation — PASS

Independent multi-seed-mean recomputation from per-seed values:
- L=2: OB=1.0000, OI=1.0000
- L=3: OB=1.0000, OI=0.9983 (=mean of [1.0, 1.0, 0.995])
- L=5: OB=1.0000, OI=0.7900 (=mean of [0.685, 0.795, 0.890])

All match JSON `aggregate` field **byte-exactly** (>0.001 tolerance).

### Item 4 — OB PASS at every cell — PASS

Multi-seed OB at L=2/3/5 = 1.000/1.000/1.000. Each ≥ 0.80, each ≥ 0.95.
Smoke OB was also perfect (1.000); production matches identically. No
capacity-edge degradation.

### Item 5 — OI characterization — PASS

- L=2 multi-seed OI = 1.000 (clears 0.80; smoke was 1.000 — matches)
- L=3 multi-seed OI = 0.998 (clears 0.80; smoke was 0.840/0.972 — production
  IMPROVES on smoke due to higher M_OBS=16 vs 8)
- L=5 multi-seed OI = 0.790 BOUNDARY (just below 0.80; smoke was 0.195
  raw / 0.463 topK)
- Per-seed L=5 OI = [0.685, 0.795, 0.890] — monotonically improving across
  seeds; mean exactly at the n=95 boundary value.

The L=5 OI boundary is the EXPECTED pattern per pillar n=95.

### Item 6 — Comparison to pillar n=95 — PASS

- Pillar n=95 reference (`2026-05-24-cross-bridge-OI-load-ceiling-map...md`):
  L=5 OI = 0.770 (global_mean) / 0.752 (per_bridge_mean) at V=160 on the
  extended map; original n=95 probe (LOADS=[2,3,5]) reported L=5 OI = 0.790
  (global) / 0.785 (per_bridge_mean).
- D5 production cross-bridge OI L=5 = **0.790** — exact match with the
  n=95 LOADS=[2,3,5] global-mean value.
- Qualitative pattern matches: OB perfect every load + OI L=5 boundary
  cell, with OI L=2/L=3 above bar.

### Item 7 — Parallel-matching primitive byte-unchanged — PASS

- `git diff cd30fc6..HEAD -- research/findings/raw/cross_bridge_mode_unification_probe.py`:
  **EMPTY DIFF** (no output). The pillar n=95 primitive is byte-identical.
- D5 probe imports `batched_phase_similarity` and
  `verify_batched_equivalent_to_scalar` at line 115-116 of
  `direction_5_cross_bridge_probe.py` (reuse-by-import; not
  re-implementation).
- Additionally verified: `concept_pool_sparse_distributed.py`,
  `text_minimal_isolation.py`, `sim/bridge.py` all show byte-empty diffs
  since pillar n=95 commit cd30fc6.

### Item 8 — Top-K decoder fix is genuine ENHANCEMENT, not artifact — PASS with note

- `_consolidate_topK_binary` in `direction_5_cross_bridge_probe_topK.py`
  correctly selects top-K=100 indices via `np.argpartition` and binarizes
  (sets exactly K entries to 1.0; all others 0.0). Verified line-by-line.
- OB is perfect (1.000) in BOTH raw and topK probes at all loads — topK
  does NOT degrade OB. **First BLOCK criterion not triggered.**
- Smoke comparison (probes on same SMOKE cache): topK ratios are
  L=3: 0.972/0.840 = **1.16x**; L=5: 0.463/0.195 = **2.37x**. Both ratios
  in [1.0, 5.0] range. **Second BLOCK criterion (ratio > 5x) not triggered.**
- **Note (non-BLOCK)**: the topK probe hardcodes `tag = "smoke"` at line
  393 of `direction_5_cross_bridge_probe_topK.py`. The
  `direction_5_cross_bridge_topK_production_bugfix.json` filename is
  misleading — the file's `tag` field reads "smoke" and the log shows
  M_OBS=8 (smoke setting, not production M_OBS=16). The numerical results
  are byte-identical to the smoke-tagged file. This is a labeling
  inconsistency, NOT a tampering issue: the topK probe was DESIGNED as a
  one-time diagnostic on the smoke cache; the CANONICAL verdict
  (DIRECTION_5_PARTIAL) comes from the RAW probe on the FULL/production
  cache (M_OBS=16, tag=full), which I independently re-derived (Items 3
  + 4). Recommend renaming the topK production file to
  `_topK_smoke_rerun.json` for clarity in future runs.

### Item 9 — Score-tuning / threshold-tampering check — PASS

- `bar_ob` = 0.8, `bar_oi` = 0.8 in result JSON (frozen at design value).
- `min_seeds` = 3 in result JSON.
- `seeds` = [42, 43, 44] (canonical reproducibility set; no cherry-picking).
- Search across `direction_5_*.py` runners and probes: no `_attempt_1`,
  `_softer_verdict`, `np.clip`, or `fallback` patterns found.
- Search across the 3 production JSONs: no `_attempt_1`, `_softer_verdict`,
  `_softer`, or `_attempt` keys.
- `direction_5_verdict.py` thresholds match exactly (0.80 / 0.80 /
  (2, 3, 5) / 3). Module modified only once (initial scaffolding commit
  7ff60a7); no post-result tampering.
- Frozen verdict module run against JSON data returns
  **DIRECTION_5_PARTIAL** (matches runner-reported verdict exactly).

---

## Summary of per-item adjudication

| Item | Result |
|---|---|
| 1. Bug fix correctness | PASS |
| 2. Multi-seed reproducibility | PASS |
| 3. Smell-test recomputation | PASS |
| 4. OB PASS at every cell | PASS |
| 5. OI characterisation | PASS |
| 6. Comparison to pillar n=95 | PASS |
| 7. Parallel-matching primitive byte-unchanged | PASS |
| 8. TopK decoder fix is genuine (with labeling note) | PASS |
| 9. Score-tuning / threshold-tampering check | PASS |

**Total: 9 / 9 PASS.**

---

## Verdict: CLEAR

The D5 bugfix PARTIAL (BOUNDARY) result is GENUINE: bio_brain_regions
HYBRID architecture supports cross-bridge composition at OB perfect +
OI passing L=2 / L=3 multi-seed, with the same FHRR capacity-envelope
L=5 OI boundary as pillar n=95 G.20 sparse cross-bridge. **Pillar n=106
BOUNDARY candidate APPROVED for promotion as a BOUNDARY-class pillar**
(5 of 6 cells PASS at 0.80 bar; L=5 OI characterised at the capacity
edge — multi-seed mean 0.790 exactly mirroring n=95's 0.790).

---

## Pillar n=106 BOUNDARY framing

```
Pillar n=106 (BOUNDARY): Direction 5 hybrid sparse-distributed
bio_brain_regions cross-bridge composition (5 bridges × V=16 = 80
concepts) — OB PERFECT every load, OI PASSES L=2/L=3, OI L=5 BOUNDARY
(capacity edge mirroring pillar n=95 G.20 sparse cross-bridge OI L=5
= 0.790; D5 production = 0.790 exact match). Unifies pillar n=98 /
n=105 (dedicated biology-faithful pools) with pillar n=95 (sparse
K-of-N cross-bridge composition). First architecture supporting BOTH
biology-faithful dedicated pools AND sparse-distributed cross-bridge
composition on a single substrate.
```

---

## Reviewer-flagged secondary issue (non-blocking)

The topK probe's hardcoded `tag = "smoke"` causes the
`direction_5_cross_bridge_topK_production_bugfix.json` file to read the
SMOKE cache. The file's data is byte-identical to the smoke-tagged file
and the verdict module returns DIRECTION_5_VOID_MALFORMED (due to the
runner passing per_seed in a non-standard shape; the underlying topK
accuracies, however, would compute to DIRECTION_5_PARTIAL via
independent recomputation). This labeling inconsistency does NOT affect
the PARTIAL verdict: the canonical RAW probe on M_OBS=16 production
data is what establishes the verdict. Recommend a follow-up commit to:

1. Make the topK probe accept a `--tag` argument (or default to "full"
   when production caches exist).
2. Use distinct output filenames per cache tag.
3. Fix the topK runner's per_seed shape to satisfy the frozen verdict
   module's `L=...` key contract.

None of these is a precondition for the n=106 BOUNDARY promotion since
the RAW probe is the canonical source of truth.

---

Reviewer: fresh adversarial pass with no shared session history.
Discipline: only standard library + numpy + json + git for verification;
no project module modified.
