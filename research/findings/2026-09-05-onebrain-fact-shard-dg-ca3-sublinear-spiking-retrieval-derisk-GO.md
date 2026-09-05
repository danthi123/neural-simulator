---
type: finding
status: go
claim_check: measured-result
date: 2026-09-05
mechanism: dg-ca3-sparse-index-over-fact-blocks (sublinear spiking retrieval for the onebrain composer)
lane: knowledge-integration
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_fact_shard/derisk_404_6seed.json
  - research/runners/_onebrain_fact_shard_derisk.py
  - research/runners/one_brain_composer.py
  - research/biology/dg-ca3-sparse-index.md
  - research/findings/2026-09-05-slotbinder-L3-wirein-derisk-NOGO-perstep-cost-dominates-latency.md
  - research/findings/2026-08-27-knowledge-100k-sublinear-sharded-retrieval-verified-no-flip.md
  - research/findings/2026-08-28-dg-shard-escalation-root-cause-noise-calibration-mismatch-two-levers-refuted.md
---

# The onebrain composer's O(k_max) recall wall falls to a DG-CA3 sparse index over the FACT BLOCKS: sublinear spiking retrieval, 6-seed parity + moat clean, ~402x faster at 404 co-resident facts (de-risk; additive default-off, NOT a production flip)

**Board/lane: rank-1 composer-latency residual.** Rank-1 established the spiking `OneBrainComposer` is CORRECT
(6/6, recall==rf, moat clean) and FITS memory (563 MiB @ 404 facts) -- its ONLY residual is per-query LATENCY:
recall is an **O(k_max) LINEAR SCAN** over the co-resident fact blocks (~114 s/recall @ 404 facts), which is why
`k_max` is pinned at 32. This de-risks rank-1's own named next mechanism -- a **sharded / sublinear SPIKING
retrieval** (DG->CA3 sparse index) -- on the fact-block axis. It is the lever that could retire the host FHRR
composer, the single biggest host shortcut on the conversational spine.

## TL;DR

A per-role **DG-CA3 sparse index over the fact blocks** routes a query cue to a small SHARD of candidate blocks;
only the shard is decoded on the spiking substrate. At the real 404-fact scale, 6 seeds (42/43/44/100/101/102):
**GO, 6/6.** Correctness parity vs the full O(k_max) scan is **6/6 (540/540 checks: 30 queries x 3 kinds x 6
seeds)**, the no-confab moat is **6/6 (0/96 new confabulation)**, the shard is a mean **1.17 blocks (max 4) vs 404**
decoded per recall (sublinear), and wall-clock drops from a **149 s (median) full O(k_max) scan to 0.37 s (median)
sharded** -- a **~402x** speedup, landing inside FHRR's ~0.9 s interactive band. The scramble control collapses
recall to 0/30 every seed (**100% of recall attributable to content routing**, `tools.lab.attributable_to`), and a
real public-API anchor (`comp.query_patient`/`query_agent`/`ask_yes_no`, seed 42) equals the full reference 3/3.
This is a **de-risk (additive, default-off)**, NOT a production flip -- the wire-in to `OneBrainComposer` is the
named next rung.

## Why the existing `enable_sparse_index` did NOT already close this

`OneBrainComposer.enable_sparse_index` (the DG index, `research/biology/dg-ca3-sparse-index.md`) shards the
**VOCABULARY axis (V)**: it routes each role's recovered phasor to a small shard of the V-wide concept CODEBOOK so
the per-block CLEANUP is O(shard_V) not O(V). But `_read_blocks_indexed` STILL loops `for i in
range(len(self.kb))` -- it decodes EVERY fact block. So the **FACT-COUNT axis (k_max)** stayed a full linear scan.
This de-risk attacks that missing axis. The two indices are orthogonal and compose (fact-count shard x vocab
shard).

## Why this is distinct from `ShardedPhasorStore` (the tiered LTM, already sublinear-at-scale)

`ShardedPhasorStore` IS sublinear at 100k (2026-08-27 finding), but by a **HOST-HASH agent-router** -- a python
dict keyed on the cued agent -> one of ~395 shards. That is (i) a DECLARED host shortcut (`scaffold_retired`=0),
(ii) NOT on the spiking one-brain composer, and (iii) its reverse lookups (`query_agent`) fan out to ALL shards
(that finding's residual #2). This de-risk is the **brain-based** version ON the spiking `OneBrainComposer`: a
DG-CA3 sparse index (pattern separation + conjunctive routing, the `DGSparseIndex` class the composer already
imports), with a PER-ROLE inverted index so **reverse lookups shard too** (`query_agent`, cue = action+patient --
measured 30/30 parity every seed).

## Why it works where the `RFPhasorComposer` DG port FAILED (99.5% escalation, 2026-08-28)

That port routed on the **NOISY RECOVERED phasor** read off the substrate (sigma=1.27 rad -> misroute 98.3% of
queries). This de-risk routes on the **CLEAN cue-word concept code**: the caller ASSERTS `agent="dog"`, so the key
is `comp.comp.concepts["dog"]` (sigma=0, the exact stored code). The block that stored `agent="dog"` was indexed
under that SAME code -> deterministic, by-construction hit. **No recovery noise -> no misroute.** This is the
structural reason the fact-block index is a fundamentally easier (and correct-by-construction) problem than the
recovered-phasor vocabulary shard on real codes -- and it is why parity is 6/6 exact, not "0.98 with escalation".

## Mechanism (per-role DG-CA3 inverted index; reuse-by-import of `DGSparseIndex`)

* **Index build (encoding-time role knowledge -- legitimate, the fact's roles are known when it is stored):** for
  each MAIN role r in (agent, action, patient), a `DGSparseIndex` over the (K, D) matrix whose row i is the concept
  code of block i's filler in role r. Bucket-member id = the block index. DG expansion + hard per-band k-WTA + CA3
  conjunction routing (`m ~ K^(1/g)` -> O(1) bucket occupancy) -- the SAME class + math the vocabulary index uses.
  Build cost 0.01 s @ 404 facts (amortized, once per store mutation).
* **Query:** route each asserted cue role's CLEAN code to its DG shard of candidate blocks; **INTERSECT** the
  per-role shards (conjunctive cue). The intersection is what tightens the shard to ~1 even with reused fillers
  (agent recurs ~3x, action ~10x, but their intersection is the unique (agent,action) block), and it is also what
  gives the moat for free: an out-of-store combination of valid words intersects to the EMPTY set -> abstain. The
  shard is a **SUPERSET of the true matches BY CONSTRUCTION** (block i with the cued filler in role r routes to the
  SAME bucket its filler was stored in). Extra collisions are harmless -- decoded and rejected.
* **Decode:** only the shard blocks, via the composer's EXISTING spiking `_read_block` (reconstruct + unbind +
  cleanup on FIRING NEURONS -- the CA3 pattern-completion, restricted to the routed ensemble), first-match in
  ascending block order (== the full scan's first-match). The answer role is read OFF THE SPIKING DECODE, never
  off `kb`.

## The second lever: `no_batched_region` (right-size the bridge) -- built here, additive default-off

The as-is composer sizes a `k_max*(n_roles*D + cb)` **batched region** for `_read_all_blocks` (the batched O(k_max)
scan). That region is **dead weight for a per-block sharded read** yet dominates `n_total` (measured 660,856
neurons @ k_max=420/V=308), inflating the per-step resonate cost of EVERY per-block read AND every store O(n_total)
-- on CPU it makes the as-is 404-fact composer intractable to even build+store (a k_max=420 as-is bridge was killed
after ~8 CPU-min on store alone). The new `no_batched_region` param drops it, shrinking the bridge **11.6x** (to
56,896 neurons: the store region + one per-block op region). Per-block reads are **byte-identical** (exact-compared,
all blocks: the `[q_base:c_base+cb]` region is unchanged); the default (flag off) layout is byte-identical to before
(same `n_total` arithmetic, verified). This is both the de-risk enabler and a core wire-in lever: a fact-shard
composer never batches, so it should never build the batched region.

## Results (404 facts, D=128, g=2/G=4/c=8, `_onebrain_fact_shard_derisk.py`, SIM_BACKEND=numpy)

| seed | store (s) | full O(k_max) scan (s) | s/block | shard mean/max | sharded recall (s) | speedup | parity P/A/YN | full recall vs truth | moat new-confab | scramble recovered |
|---|---|---|---|---|---|---|---|---|---|---|
| 42  | 120 | 471.7 | 1.17 | 1.19/4 | 3.26 | 144x | 30/30 30/30 30/30 | 404/404 | 0/16 | 0/30 |
| 43  |  97 | 154.5 | 0.38 | 1.03/2 | 0.38 | 409x | 30/30 30/30 30/30 | 404/404 | 0/16 | 0/30 |
| 44  | 111 | 153.5 | 0.38 | 1.18/2 | 0.42 | 362x | 30/30 30/30 30/30 | 404/404 | 0/16 | 0/30 |
| 100 |  98 | 144.5 | 0.36 | 1.21/4 | 0.35 | 407x | 30/30 30/30 30/30 | 404/404 | 0/16 | 0/30 |
| 101 |  97 | 144.3 | 0.36 | 1.24/3 | 0.36 | 401x | 30/30 30/30 30/30 | 404/404 | 0/16 | 0/30 |
| 102 |  99 | 143.1 | 0.35 | 1.18/3 | 0.35 | 403x | 30/30 30/30 30/30 | 404/404 | 0/16 | 0/30 |

<!--derived: every cell is read directly from research/findings/raw/_onebrain_fact_shard/derisk_404_6seed.json
(per_seed[*]); no cell is computed from another cell. speedup = full_perblock_scan_seconds / shard patient-latency
median, as the runner records it.-->

<!--derived-->
<!-- Aggregate numbers below are rounded from the aggregate block of derisk_404_6seed.json (block scope opened by
the bare <!--derived--> marker above, per tools/claim_check.py: e.g. shard_patient_latency_median 0.36879 -> 0.369 s,
speedup_median 401.97 -> 402x, full_perblock_scan_seconds_median 148.978 -> 148.98 s, bridge_shrink 11.615 -> 11.6). -->
**Aggregate (`derisk_404_6seed.json`):** shard mean **1.17** (max **4**) vs **404** blocks; full O(k_max) scan
median **148.98 s**; sharded recall median **0.369 s**; speedup median **402x**; bridge shrink **11.6x**; parity /
moat / full-recall / anchor-where-run / scramble / sublinear ALL 6/6. **anchor (seed 42): 3/3** -- the real
public-API `comp.query_patient`/`query_agent`/`ask_yes_no` equal the host-matched full reference, proving the
reference IS the real full path.

**Seed 42 is a machine-LOAD outlier, not a mechanism difference:** it ran while system load was 8-37 (concurrent
jobs), giving 1.17 s/block vs 0.35-0.38 s/block on the low-load seeds -- SAME shard size (1.19). Its speedup is
still 144x, and blocks-decoded (the load-independent core result) is identical. At low load the sharded recall is
~0.37 s, inside FHRR's ~0.9 s band.

## Anti-cheats (adversarial, wired into the runner + a documented verify-go pass)

* **(a) content-addressable:** the routing key is the CUE WORD's concept code, never the answer id; the answer is
  read off the spiking `_read_block` decode; `kb` is touched ONLY at index-build (encoding).
* **(b) parity vs the full scan is a hard gate** -- a fast-but-wrong index is a NO-GO. Anchored: the real
  public-API calls (seed 42) == the host-matched full reference 3/3, AND the full path's own recall == ground
  truth 404/404 every seed. Chain closed: sharded == reference == real-API == ground truth.
* **(c) scramble control:** permuting the query band-winner tuple collapses recall to 0/30 every seed;
  `attributable_to(real, scrambled)` = 100% -- the content routing is load-bearing, not luck.
* **verify-go (documented, 5 refutation angles, all SURVIVE):** (1) parity-is-circular -- refuted by the anchor +
  recall-vs-truth chain; (2) shard-~1-is-trivial -- refuted by scramble collapse + routing-on-cue-not-answer; (3)
  moat-untested -- 2/3 of moat probes are valid-word UNSTORED COMBINATIONS (exercise the intersection), 0/96
  confab; (4) unfair-baseline -- the 149 s scan is a genuine O(k_max) recall (matches the ~114 s rank-1 cited), and
  blocks-decoded (1.17 vs 404) is bridge-independent; (5) seed-42-outlier -- a load artifact (same shard size),
  ratio still 144x.

## Brain-based (honest)

The in-shard reconstruct/unbind/cleanup IS the composer's on-substrate op (unchanged, over fewer blocks) -- the CA3
completion. The DG sparse PROJECTION is the SAME declared host-rate stand-in the vocabulary index already uses
(`research/biology/dg-ca3-sparse-index.md`: fixed random sparse granule bands + hard k-WTA; named spiking burn-down
= the granule-cell WTA in `_riii_ca3_completion_specificity_derisk.py` / `cortex_dg_ca3_cleanup_probe.py` /
`_gap5_emergent_dg_selection_derisk.py`). NO `sim/` edit. The answer always comes from the spiking decode.

## Honest residuals / scope (NO-DEFER -- named next rungs, not walls)

1. **This is a DE-RISK, not wired.** It is GO at runner level; it is NOT reachable from `/api/brain-chat`, NOT
   on-by-default, NOT scaffold-retired (per `docs/TERMS.md`). Next rung: wire `enable_fact_shard` +
   `no_batched_region` into `OneBrainComposer` as an additive default-off fast path (the `enable_sparse_index`
   pattern -- route the cue-known query methods `query_patient`/`query_agent`/`ask_yes_no`/`_seq_block` through the
   shard), then a no-regression verify (byte-identical off, moat unchanged), then the owner's default-on review.
2. **The DG projection is a host-rate stand-in** (declared). Its spiking burn-down (granule-cell WTA) is named
   above and unchanged by this de-risk.
3. **Synthetic FHRR codes, not a grounded bundle.** The routing is on CLEAN codes, so the real-vs-synthetic noise
   calibration that killed the RFPhasorComposer port DOES NOT apply here (deterministic routing on clean cues, not
   recovered noisy phasors). A real-bundle (day_33 404-fact) verify is a named follow-on, not a risk to the
   mechanism.
4. **Latency measured on numpy CPU.** A GPU re-verify would refine the absolute numbers (store + full scan drop
   further); it would not reverse the sublinear verdict (blocks-decoded is backend-independent).
5. **First-match semantics on non-unique cues.** The production store is unique-(agent, action); the sharded path
   returns the same ascending first-match as the full scan even under duplicates (shard is a superset, iterated
   ascending) -- parity holds by construction, and was measured 6/6 with reused agents/actions/patients.

## Sources

Code: `research/runners/_onebrain_fact_shard_derisk.py` (new; `FactShardIndex` reuse-imports `DGSparseIndex`, and
the scramble control calls `tools.lab.attributable_to`), `research/runners/one_brain_composer.py` (additive
`no_batched_region`). Biology: `research/biology/dg-ca3-sparse-index.md` (Kandel: DG pattern separation + CA3
completion). Prior: `2026-09-05-slotbinder-L3-wirein-derisk-NOGO...` (Path B, latency NO-GO),
`2026-08-27-knowledge-100k-sublinear...` (host-hash sharding), `2026-08-28-dg-shard-escalation-root-cause...` (the
recovered-phasor misroute this avoids).
