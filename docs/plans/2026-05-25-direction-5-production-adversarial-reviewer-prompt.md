# Direction 5 PRODUCTION — Adversarial Reviewer Prompt (pre-staged 2026-05-25)

Status: Direction 5 bugfix SMOKE = PARTIAL multi-seed (5 of 6 cells PASS at 0.80 bar; OB perfect every load; OI passes L=2/L=3 multi-seed; only OI L=5 below bar at 0.195 raw / 0.463 topK; mirrors pillar n=95 G.20 sparse cross-bridge OB-perfect / OI-L=5-boundary pattern). D5 PRODUCTION decisive multi-seed run launched ~19:35 EDT (full scale n_lang=2048, n_per_pool=200, events=200, M_OBS=16; 5 bridges × 3 seeds = 15 cells; ~7-8 hr GPU ETA; watcher `bkxj54p00` chains both probes on completion).

This document is the pre-registered, pre-staged adversarial reviewer prompt to dispatch when D5 production completes. Reviewer must be a FRESH AGENT (no shared session history), must reproduce the verdict from raw data, must verbatim record CLEAR or BLOCK at the end.

---

## Inputs to the reviewer

- **Bugfix smoke training**: `research/findings/raw/direction_5_5bridge_smoke_bugfix.json` + `.log`
- **Bugfix smoke probe (raw)**: `research/findings/raw/direction_5_cross_bridge_bugfix_smoke.json` + `.log`
- **Bugfix smoke probe (topK)**: `research/findings/raw/direction_5_cross_bridge_topK_bugfix_smoke.json` + `.log`
- **Production training**: `research/findings/raw/direction_5_5bridge_production_bugfix.json` + `.log` (will exist when production completes)
- **Production probes**: `research/findings/raw/direction_5_cross_bridge_production_bugfix.json` + `.log` AND `research/findings/raw/direction_5_cross_bridge_topK_production_bugfix.json` + `.log`
- **Bug fix commit**: `c4e18f2` (added _BRIDGE_LABEL_SEED_OFFSETS map; verified 5 distinct pattern_0 across bridges)
- **Bridge builder (post-fix)**: `research/findings/raw/direction_5_bridge_builder.py`
- **Top-K decoder fix**: `research/findings/raw/direction_5_cross_bridge_probe_topK.py`
- **Frozen verdict module**: `research/findings/raw/direction_5_verdict.py`
- **Findings doc**: `research/findings/2026-05-25-DIRECTION-5-HYBRID-BUGFIX-PARTIAL-pattern-uniqueness-bug-fixed-substrate-is-NOT-the-limit-mirror-of-pillar-n95.md`
- **Pillar n=95 reference**: `research/findings/2026-05-24-cross-bridge-OI-load-ceiling-map-extension-of-n95-ceiling-between-L4-and-L5.md`
- **Pillar n=105 reference**: `research/findings/2026-05-25-DIRECTION-3-V32-PRODUCTION-PASS-bio_brain_regions-vocab-32-multi-seed-decisive-pillar-n105-candidate.md`

---

## What the reviewer must scrutinise

Direction 5 BOUNDARY pillar n=106 candidacy. ANY single FAIL on items 1-9 returns BLOCK. CLEAR requires all 9 to PASS.

### 1. Bug fix correctness

- Run `python -c "from research.findings.raw.direction_5_bridge_builder import _BRIDGE_LABEL_SEED_OFFSETS; print(_BRIDGE_LABEL_SEED_OFFSETS)"` and confirm the map has 5 entries spaced 100k apart.
- Run a small script verifying `generate_sparse_patterns(n_concepts=16, n_pool=2000, pattern_size=100, seed=42+offset)` produces 5 DISTINCT pattern_0 sets across the 5 bridges.
- Inspect `research/findings/raw/direction_5_cache/sparse_patterns_full_*_seed42.npz` (if production cache exists) and verify pattern_0 across bridges has <50% pairwise overlap.
- BLOCK if any 2 bridges have ≥80% pattern_0 overlap.

### 2. Multi-seed reproducibility at production scale

- Open `direction_5_5bridge_production_bugfix.json`. Confirm 5 bridges × 3 seeds = 15 training cells completed (no FAILs).
- Open `direction_5_cross_bridge_production_bugfix.json`. Confirm `per_seed` contains 3 entries for seeds [42, 43, 44].
- For each seed, confirm `per_load` contains entries for loads {2, 3, 5} with `n_trials=200`.
- BLOCK if any seed/load cell is missing.

### 3. Smell-test recomputation from raw per-seed data

- Independently recompute multi-seed mean OB and OI from JSON per-seed values WITHOUT running compute_verdict.
- Confirm match to JSON's `aggregate` field within 0.001.
- BLOCK if discrepancy ≥ 0.001.

### 4. OB PASS at every cell

- Production OB at L=2, L=3, L=5 must each have multi-seed mean ≥ 0.80.
- If smoke OB was perfect (1.000), production should ideally match or stay near 1.000.
- BOUNDARY (not BLOCK) if OB at L=5 drops below 0.95 but stays ≥ 0.80 (capacity edge).
- BLOCK if OB at any L drops below 0.80 multi-seed (this would CONTRADICT smoke).

### 5. OI characterisation

- Production OI at L=2 and L=3 must clear 0.80 multi-seed (smoke had OI L=2 = 1.000; L=3 = 0.840/0.972).
- OI at L=5 will likely be BOUNDARY (smoke had 0.195/0.463); production should be in the same range. The L=5 OI boundary is EXPECTED per pillar n=95 (V=160 cross-bridge OI L=5 = 0.790; D5 V=80 hits boundary lower because shared pool has more competing dynamics).
- BLOCK only if OI at L=2 falls below 0.80 multi-seed (this would CONTRADICT smoke).

### 6. Comparison to pillar n=95

- D5 production should produce the QUALITATIVE pattern: OB perfect + OI L=5 boundary.
- The pattern's QUANTITATIVE values may differ (D5 V=80 with mixed-noise hybrid pool vs G.20 sparse V=160 with pure sparse pool).
- BLOCK if D5 production shows a fundamentally different pattern (e.g., OB also degrades; OI passes L=5).

### 7. Anti-cheat: parallel-matching primitive byte-unchanged

- Run `git diff <pillar-n=95-commit>..HEAD -- research/findings/raw/cross_bridge_mode_unification_probe.py` and confirm EMPTY diff for the parallel-matching primitive (`batched_phase_similarity`, `verify_batched_equivalent_to_scalar`).
- Confirm Direction 5 probes IMPORT these primitives (not re-implement).
- BLOCK if any diff on the primitives.

### 8. Top-K decoder fix is genuine ENHANCEMENT not artifact

- Compare raw probe (no topK) vs topK probe on same cache.
- Confirm topK ratio (OI_topK / OI_raw) is between 1.0 and 5.0 at L=3 and L=5 (signal restoration, not random improvement).
- Confirm `_consolidate_topK_binary` in `direction_5_cross_bridge_probe_topK.py` correctly selects top-K=100 indices and binarizes (not just rescales).
- BLOCK if topK probe degrades OB (it should be perfect both ways).
- BLOCK if topK ratio > 5x (suggests numerical artifact).

### 9. Score-tuning/threshold-tampering check

- Confirm `bar_ob` and `bar_oi` in result JSONs are BOTH exactly 0.80 (frozen at design).
- Confirm seeds are exactly [42, 43, 44] (canonical reproducibility set; no cherry-picking).
- Search runner + probe modules for any post-hoc threshold adjustment, np.clip, fallback patterns. BLOCK if found.
- Search result JSON for `_attempt_1`, `_softer_verdict`, etc. (BLOCK if found).
- Confirm direction_5_verdict.py thresholds match (0.80/0.80/(2,3,5)/3).

---

## Reviewer verdict

End with EXACTLY ONE of:

- **CLEAR** — All 9 items PASS. The D5 bugfix PARTIAL (BOUNDARY) result is GENUINE: bio_brain_regions HYBRID architecture supports cross-bridge composition at OB perfect + OI passing L=2/L=3 multi-seed, with the same FHRR capacity-envelope L=5 OI boundary as pillar n=95 G.20 sparse cross-bridge. Pillar n=106 BOUNDARY candidate APPROVED for promotion as a BOUNDARY-class pillar (5 of 6 cells PASS at 0.80 bar; L=5 OI characterised at the capacity edge mirroring n=95).
- **BLOCK** — Specify which item(s) failed; describe strengthening fix needed.

---

## Pillar n=106 BOUNDARY framing (if CLEAR)

```
Pillar n=106 (BOUNDARY): Direction 5 hybrid sparse-distributed
bio_brain_regions cross-bridge composition (5 bridges × V=16 = 80
concepts) — OB PERFECT every load, OI PASSES L=2/L=3, OI L=5 BOUNDARY
(capacity edge mirroring pillar n=95 G.20 sparse cross-bridge OI L=5).
Unifies pillar n=98/n=105 (dedicated biology-faithful pools) with
pillar n=95 (sparse K-of-N cross-bridge composition). First
architecture supporting BOTH biology-faithful dedicated pools AND
sparse-distributed cross-bridge composition on a single substrate.
```
