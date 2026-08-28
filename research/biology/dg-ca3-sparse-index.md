---
type: biology
id: dg-ca3-sparse-index
mechanism: Dentate-gyrus sparse pattern separation (expansion + k-WTA) routes a cue to a small CA3 ensemble that pattern-completes -- content-addressable retrieval is O(shard), not a linear scan of every stored memory
status: established
last_verified: 2026-08-26
current_finding: research/findings/raw/four_day/_sparse_indexed_retrieval_6seed.json
current_status: "A DG-like sparse index (random sparse granule bands + per-band k-WTA -> CA3 conjunctive band-tuple bucket keys) routes a cue to a SMALL candidate shard; the existing matched-filter cleanup runs only within the shard. De-risk 6-seed GO at V in {10k,50k,200k}: top-1 parity vs the full linear cleanup 0.988-0.997, shard ~constant (~20 rows) as V grows (rows-speedup to 9840x, wall to ~396x at 200k), scramble->chance, out-of-store cues abstain under both with 0 new confab. WIRED into research/runners/one_brain_composer.py as an ADDITIVE DEFAULT-OFF fast path (enable_sparse_index / env BRAIN_SPARSE_INDEX_RETRIEVAL); no-regression verify GO (composer answers + moat BYTE-IDENTICAL OFF vs ON, all seeds). DG projection = host-rate stand-in; spiking-granule-WTA burn-down named below."
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "pattern separation results from the divergence"
    note: "DG expansion IS pattern separation: '...pattern separation results from the divergence / of entorhinal inputs onto a larger number of granule / cells in the dentate gyrus.' -- the mechanism the index reuses (project the cue's feature vector through more, sparsely-sampling granule bands to decorrelate)."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "of entorhinal inputs onto a larger number of granule"
    note: "the divergence/expansion target -- a LARGER number of granule cells than the afferent EC input dimension (the index's L bands x m granule cells >> the 2D cue feature dim)."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "in different environments differ more extensively in"
    note: "decorrelation is measured: 'Neural activity patterns recorded / in different environments differ more extensively in / the dentate gyrus and CA3 than they do one synapse / upstream in the entorhinal cortex' -- the pattern-separated DG code is what makes distinct cues route to distinct buckets (the scramble anti-cheat verifies this is load-bearing)."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "the recurrent excitatory connections of CA3"
    note: "CA3 pattern completion via recurrence: Marr proposed 'the recurrent excitatory connections of CA3 / pyramidal cells' store a memory as a cell assembly; 'the / reactivation of a subset of this / stored cell assembly would be sufficient to activate / the entire original neural ensemble.' -- the in-shard matched-filter cleanup IS this completion, restricted to the routed ensemble."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "referred to as pattern completion."
    note: "the named result: 'This restoration is / referred to as pattern completion.' -- a few cues retrieve the whole memory, over the routed subset, not a global scan."
implemented_by:
  - research/runners/_sparse_indexed_retrieval_derisk.py
  - research/runners/one_brain_composer.py
  - research/runners/rf_phasor_composer.py
  - research/runners/sharded_phasor_store.py
findings:
  - research/findings/raw/four_day/_sparse_indexed_retrieval_6seed.json
  - research/findings/raw/dg_shard_escalation/diag_seed42_real500k.json
---

# DG sparse pattern separation routes a cue to a small CA3 ensemble; retrieval is O(shard), not O(V)

**The claim the code must respect.** The hippocampus does not linearly scan memory. The dentate gyrus performs
sparse **pattern separation** -- in Kandel's words, "pattern separation results from the divergence of entorhinal
inputs onto a larger number of granule cells in the dentate gyrus," and the resulting activity patterns "differ
more extensively in the dentate gyrus and CA3 than they do one synapse upstream in the entorhinal cortex." CA3
then performs **pattern completion**: Marr proposed that "the recurrent excitatory connections of CA3 pyramidal
cells" store a memory as a cell assembly, so that "reactivation of a subset of this stored cell assembly would be
sufficient to activate the entire original neural ensemble" -- "referred to as pattern completion." A cue
content-addressably routes to a small ensemble and completes there.

**How the index realizes it.** DG expansion + sparsification: the cue's feature vector `x = [cos(phi), sin(phi)]`
is projected through `L` fixed random **sparse** granule bands (each granule cell samples only `c` input dims --
DG afferents are sparse), then a hard **per-band k-WTA (k=1)** yields the sparsest pattern-separated code (one
active granule of `m` per band). CA3 conjunction routing: the `L` bands are partitioned into `G` groups of `g`;
each group's bucket key is the `g`-tuple of its band winners (a memory is the CO-activation of a specific granule
ensemble). A fact drops its id into its `G` group-buckets at store time; a query's candidate **shard** is the
union of its `G` group-buckets. With `m ~ V^(1/g)` the bucket occupancy is O(1), so the shard stays ~constant as
`V` grows and the matched-filter cleanup runs over the shard only -> **sublinear retrieval** (the standard
banded-LSH sublinearity, re-derived as the DG->CA3 sparse-conjunctive code).

## What is established, and where the shortcut still stands

**Established (6 seeds, `_sparse_indexed_retrieval_derisk.py`, GO):** at `V` in {10k, 50k, 200k} synthetic FHRR
concepts at production `D`, the DG-indexed retrieval returns the same top-1 as the full linear cleanup (parity
0.988-0.997), the shard stays ~constant (~20 rows; rows-speedup to 9840x, wall to ~396x at 200k), permuting the
band-winner tuple collapses accuracy to ~chance (the routing is load-bearing, not luck), and a genuinely
out-of-store cue abstains under BOTH paths with **0 new confabulation** (the shard is a SUBSET of the codebook, so
its peak score <= the full peak; if the full scan abstains, the shard abstains). Wired into the production
`OneBrainComposer` as an additive default-off fast path (`enable_sparse_index` / env
`BRAIN_SPARSE_INDEX_RETRIEVAL`).

**2026-08-28 port to `RFPhasorComposer`/`ShardedPhasorStore` (board #66 knowledge-scale vocab-latency wall).** The
2026-08-28 vocab-latency-wall finding showed the tiered LTM's shard composer (`RFPhasorComposer`, the engine
behind `ShardedPhasorStore`) pays the SAME O(V) cleanup cost the DG index already fixes for `OneBrainComposer` --
every routed shard shares ONE global codebook, so a query's cleanup step still scans the FULL vocabulary
regardless of the shard's own ~200-fact size (1.37s@24k words -> 20.7s@347k -> 33.8s@581k). The identical
mechanism (same `DGSparseIndex` class, reused by import, not reimplemented) was ported into `_cleanup`/
`_cleanup_all` behind `enable_sparse_index` / env `BRAIN_SHARD_SPARSE_INDEX` (a DISTINCT env var from
OneBrainComposer's, since the two composers' defaults are reviewed independently). One new consideration this
port introduces: `ShardedPhasorStore(share_codebook=True)` already grafts ONE shared `{word: phases}` codebook
across all S shards (avoiding an S-fold memory blow-up, `sharded_phasor_store.py`'s own documented rationale) --
the DG index is grafted the SAME way (`_dg_index_source`), so the index is built ONCE for the whole store, not
once per shard. Measured effect (real Wikidata bundle, 347,695-word vocab, paired same-session): correctness
exact (parity + moat unchanged), peak RSS LOWER with the accelerator on, warm-query latency reduced ~25% (median
33.21s -> 24.92s, excluding the one-time index-build query) -- a real but MODEST reduction, not a close to
sub-second; the residual is plausibly frequent DG-shard escalation on real (non-synthetic) entity codes and/or
session memory pressure, named as the next lever, not resolved. See
`research/findings/2026-08-28-shard-composer-dg-sparse-index-port-modest-latency-reduction.md`.

**2026-08-28 escalation-rate measurement + two REFUTED levers (board #66/#192).** The residual above was
measured directly: `research/runners/_dg_shard_escalation_diagnostic.py` calibrates the REAL RF-resonate
recovery noise through the production `store()`/`_unbind_phases()` path (not assumed) and instruments
`_dg_shard_select` on the real bundle. Result (6-seed, real `wikidata_500k_fast`): **99.50% escalation**, and
the root cause is a NOISE-CALIBRATION mismatch, NOT real-vs-synthetic code geometry -- a matched-scale synthetic
sweep escalates at an almost identical 99.4%, refuting that hypothesis directly. The de-risk above validated GO
at `sigma=0.30` rad; production's real recovery noise measures **sigma=1.27 rad (~4.1x larger)**, at which the
true stored code is a member of its own DG-routed shard only 1.7% of queries (a genuine MISROUTE, not an
under-confident match). Two levers were tested and REFUTED: lowering `conf_floor` (crashes parity to 1.4%-39%
at any floor that meaningfully cuts escalation) and doubling the multi-probe group count `G` (true-in-shard
hit rate rises only 1.1%->3.4%; closing the gap would need G in the thousands, already hitting the ~4GB RSS
budget at a mere 2x). No code shipped; the already-merged accelerator's correctness is unaffected (parity 1.0
whenever it does decide). Named next levers: decouple codebook caching from the DG index (plausibly explains
the port's own ~25% win independent of shard routing); decouple the granule width `m` from `V^(1/g)`-driven
occupancy; raise D toward the de-risk's own validated D=256 operating point (production runs D=128). See
`research/findings/2026-08-28-dg-shard-escalation-root-cause-noise-calibration-mismatch-two-levers-refuted.md`.

**Declared shortcut, and the named burn-down.** The in-shard matched-filter cleanup IS the composer's existing
on-substrate op (the complex-synapse cleanup matvec + WTA select), just over fewer rows. The **DG sparse
projection here is a RATE/host shortcut** -- a fixed random sparse projection + hard argmax-WTA, computed on the
host. Its biological burn-down is the SPIKING DG granule-cell layer already validated in the trisynaptic-loop
probes: `research/runners/_riii_ca3_completion_specificity_derisk.py` (CA3 partial-cue completion specificity),
`research/runners/cortex_dg_ca3_cleanup_probe.py`, and `research/runners/_gap5_emergent_dg_selection_derisk.py`
(emergent DG k-WTA selection). The burn-down replaces the host argmax-WTA with the spiking granule competition
(the same NEF-WTA the cleanup Stage-2 already uses); the routing is content-addressable and sparse exactly as the
granule layer is.

## What this entry cannot catch

No `constraints_config`. The two properties that matter are inequalities and call-graph shapes, not scalar
equalities, so `biology_check --config` (an equality matcher) cannot pin them without firing on a legitimate
re-tuning (the de-risk runs at D=256, the composer at D=128 -- both valid): (1) the FHRR separation the
shard-subset abstain proof needs -- a stored match scores ~D while a random code scores ~sqrt(D/2), so D must be
LARGE ENOUGH (>=128) that a near-ceiling shard hit is provably the global max; and (2) the property that matters
most -- that the routing key is computed from the CUE VECTOR via the DG sparse projection and NEVER from the
ground-truth answer id (content-addressable, anti-cheat a). Both live as RUNNER anti-cheats, not config gates:
parity vs the full scan is a hard gate (a fast-but-wrong index is a NO-GO), the scramble control must collapse to
~chance, and the wiring's no-regression verify (`_wire_sparse_index_verify.py`) proves byte-identical composer
answers + moat with the flag OFF vs ON.
