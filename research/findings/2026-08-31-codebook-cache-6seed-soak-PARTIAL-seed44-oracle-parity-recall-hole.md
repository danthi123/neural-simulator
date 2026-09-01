---
status: qualified
lane: gap#66
type: finding
date: 2026-08-31
---

# 6-seed codebook-cache production-load soak — PARTIAL (technical bar met by all 6; oracle-parity gate fails on seed 44)

**STATUS: UNDEFINED (NOT GO)** — `go=False`, `n_seeds_ok=5`, `n_seeds_fail=1`, `elapsed_s=8385.23s`.

## Result table (all 6 seeds, cupy, codebook-cache ON, bundle wikidata_100k: 78,857 facts, D=128, vocab 23,914, 395 shards)

| seed | verdict | recall_rate | moat_confab | lat_median | lat_p95 | oracle_mismatches |
|------|---------|-------------|-------------|------------|---------|-------------------|
| 42   | GO      | 0.9933      | 0           | 841.3     | 901.4   | 0                 |
| 43   | GO      | 1.0         | 0           | 842.2     | ~901    | 0                 |
| 44   | UNDEF   | 1.0         | 0           | 830.0     | 895.3   | **2**             |
| 100  | GO      | 1.0         | 0           | 829.2     | ~890    | 0                 |
| 101  | GO      | 1.0         | 0           | 827.6     | ~890    | 0                 |
| 102  | GO      | 1.0         | 0           | 828.1     | ~891    | 0                 |

## Verdict logic

- The declared technical bar (median < 1000ms, recall >= 0.99, moat = 0) is **met by all 6 seeds** (s42 recall 0.9933 >= 0.99; all others 1.0; all lat_med 827–842ms; all moat 0).
- However, the **Verdict gate is AUTHORITATIVE** and includes `RECALL parity: 0 mismatches vs the tractable flat oracle (held agent set)`. Seed 44 fails this gate with **2 mismatches**, making its status UNDEFINED and the overall soak NOT GO.
- All other seeds (42, 43, 100, 101, 102) have 0 oracle mismatches.

## Seed 44 oracle-parity mismatch (the residual)

MISMATCHED CUE: `berkeley_county_virginia` + `located_in_the_administrative_territoria` →
- **live tiered store: None (miss)**
- **flat oracle: 'culture_of_west_virginia'**
- **ground-truth (direct from bundle facts.json): 'culture_of_west_virginia'**

The tiered store MISSES a fact that exists in the bundle. Both the flat oracle and the ground-truth dict (computed directly from the same `facts.json` the tiered store loaded from) return `'culture_of_west_virginia'`, so the **tiered store is the defect** (not the oracle).

### Why the scale_recall (oracle-free) shows 1.0 for s44

The scale_recall battery uses a **different, randomly-sampled 150-probe set** than the oracle-parity battery (the oracle agents' complete fact sets). The s44 scale_recall sampling happened not to include this specific cue, so its recall_rate reads 1.0. The oracle-parity battery (a much larger held set) DID catch the miss. This means **the tiered store's recall hole is real but under-sampled by the 150-probe scale battery** — the gate (0 mismatches on the larger held set) is the more faithful detector.

## Interpretation (NO-DEFER)

This is a **METHOD-level residual** (a recall hole in the tiered store's routing or query path for certain agent/action pairs), not a CAPABILITY wall. The technical bars (latency, scale recall, moat) are all met; the residual is a **correctness parity** issue that the authoritative gate correctly flags. The codebook-cache lever itself (the latency wall) is closed; the tiered store's fact-routing has a small miss rate that must be investigated.

## Next concrete step (NO-DEFER — investigate the recall hole)

1. Locate the `berkeley_county_virginia located_in_the_administrative_territoria → culture_of_west_virginia` fact in the bundle's `facts.json` and confirm its shard placement.
2. Check whether the tiered store's **agent-hash routing** places this fact in the correct shard, or whether the **query path** fails to reach it.
3. Determine if this is a **routing miss** (agent mapped to wrong shard) or a **query-ordering/first-match discrepancy** (the tiered and flat stores have different first-match semantics for the same fact set).
4. Fix the tiered store's routing/query to eliminate the hole, then re-run the 6-seed soak (all 6 must pass the oracle-parity gate to close the wall).

**Artifact:** `research/findings/raw/_knowledge_scale_100k_cacheon_6seed.json` (per-seed: `s42.json`, `s43.json`, `s44.json`, `s100.json`, `s101.json`, `s102.json`).