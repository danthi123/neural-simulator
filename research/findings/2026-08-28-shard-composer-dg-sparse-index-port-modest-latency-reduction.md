---
type: finding
status: live
date: 2026-08-28
mechanism: shard-composer-dg-sparse-index-port
lane: knowledge-integration
seeds: [42]
seed-waiver: An ENGINEERING port + verify (a deterministic code change ported from an already 6-seed-GO'd
  mechanism, `research/runners/_sparse_indexed_retrieval_derisk.py`, onto a new call site), not a stochastic
  effect size -- matches the seed-waiver precedent of the 2026-08-20/2026-08-27/2026-08-28 sharding/vocab-latency
  findings in this same lane. Correctness (answer parity OFF vs ON) is verified structurally (the shard is a
  provable SUBSET of the full codebook, so a non-decisive shard read escalates to the byte-identical full scan --
  see the mechanism section) and spot-checked at small scale, not claimed from a single seed's effect size.
instrument: research/runners/_knowledge_scale_vocab_latency_probe.py (EXISTING, unchanged, 2026-08-28) -- loads
  the real bundle through `developed_brain_io.load_developed_brain(ltm_bundle=...)` and times individual
  recall/moat/yesno queries, provenance-stamped. A new local smoke script (not committed; ad hoc parity check
  over a small synthetic store) additionally verified OFF/ON answer identity and the DG-index graft sharing.
runner: research/runners/_knowledge_scale_vocab_latency_probe.py
external: NO-EXTERNAL-NEEDED -- this ports an already-validated in-repo mechanism
  (`research/runners/_sparse_indexed_retrieval_derisk.py`, 6-seed GO, `research/biology/dg-ca3-sparse-index.md`)
  onto a second call site (`RFPhasorComposer`/`ShardedPhasorStore`); no new external claim is made.
artifacts:
  - research/findings/raw/knowledge_500k_verify/vocab_latency_500k_sparse_OFF.json
  - research/findings/raw/knowledge_500k_verify/vocab_latency_500k_sparse_ON.json
---

# Porting the DG sparse-index accelerator from `OneBrainComposer` to `RFPhasorComposer` (the tiered-LTM shard engine) — a real but MODEST (~25%) latency reduction at 347,695-word vocabulary scale, not the sub-second close

Artifacts: `research/findings/raw/knowledge_500k_verify/vocab_latency_500k_sparse_{OFF,ON}.json`.

**One line.** The 2026-08-28 vocabulary-latency-wall finding diagnosed that `RFPhasorComposer` (the shard engine
behind the tiered LTM's `ShardedPhasorStore`) lacks the DG sparse-index accelerator that `OneBrainComposer`
already has, so a routed query's cleanup step still scans the FULL shared vocabulary codebook (O(V)) regardless of
the shard's own ~200-fact size -- this is what actually drives the 1.37s-to-33.8s latency growth as a bulk-KB
bundle's vocabulary grows. This finding ports the SAME validated mechanism (reused by import, not reimplemented)
into `RFPhasorComposer`, wires a codebook-sharing graft so `ShardedPhasorStore`'s 3,745 shards build ONE index
instead of 3,745 redundant copies, fixes a real batching bug the scale run itself exposed, and verifies it
end-to-end through the real production load path. **The result is PARTIAL**: correctness is exact (parity + moat
unchanged), peak memory is LOWER with the accelerator on, and warm-query latency drops ~25% (33.21s -> 24.92s
median, excluding the one-time index-build query) -- a real, paired, same-session reduction, but NOT the "toward
sub-second" outcome hoped for; queries remain tens of seconds. The residual and its next lever are named in
section 6/residual 5, not resolved here.

## 1. Verify-first: the exact `OneBrainComposer` mechanism, and why `RFPhasorComposer` lacked it

`git log --all --oneline | grep -iE "sparse.?index|DG.?index|..."` surfaces the mechanism's full history:
`6119e7cdc` (derisk) built `research/runners/_sparse_indexed_retrieval_derisk.py`'s `DGSparseIndex` class and its
6-seed GO (parity >=0.98, sublinear cost, 0 new confabulation at V in {10k,50k,200k}); `f2d37e3fb` wired it into
`OneBrainComposer` as `enable_sparse_index` / env `BRAIN_SPARSE_INDEX_RETRIEVAL` (additive, default-off); `624df4ac2`
consolidated it onto clean `main`; `b8f97aa2b` (2026-08-27) then found the flag REDUNDANT for the shipped
100k-fact/23,914-word bundle's tiered path specifically because `ShardedPhasorStore`'s shards are `RFPhasorComposer`
instances, not `OneBrainComposer` -- **`OneBrainComposer.enable_sparse_index` cannot reach the shard composer at
all**, so flipping it on that class is a hollow no-op for the tiered-LTM path. The 2026-08-28 wall finding
(`2026-08-28-knowledge-bundle-scaled-500k-reveals-vocabulary-latency-wall.md`) is the first to state this precisely
as the ROOT CAUSE: `ShardedPhasorStore` keeps per-shard FACT COUNT flat (~200/shard, `auto_n_shards`, the thing that
already IS sub-second by construction), but every shard shares ONE global codebook (`share_codebook=True`), so the
CLEANUP step (matched-filter argmax over the concept codebook, `RFPhasorComposer._cleanup`/`_cleanup_all`) still
costs O(V) in the FULL vocabulary on every routed query, not O(shard).

Reading the code confirmed the mechanism precisely (`research/runners/one_brain_composer.py:1108-1203`):
`_ensure_dg_index()` lazily builds a `DGSparseIndex` (DG expansion + hard k-WTA + CA3-conjunction bucket routing)
over the composer's own concept codebook (`self.words`/`self.concepts`, fractional-cycle phases); `_dg_shard_select`
routes a recovered role phasor to its DG shard and matched-filter-cleans-up only over that shard's rows, escalating
to `_full_host_select` (the byte-identical full scan) when the shard's peak score is not decisive (< `conf_floor *
D`) -- the no-regression guarantee, since the shard is a provable SUBSET of the codebook (its peak score can never
exceed the full peak). `research/runners/rf_phasor_composer.py` (the shard engine) had NO such machinery: its
`_cleanup`/`_cleanup_all` unconditionally build the full `(V, D)` codebook matrix and matmul against it every call.
`bash tools/before_you_build.sh "port DG sparse-index to RFPhasorComposer..."` found no prior scoping doc or
research gate against this exact port, and the biology binding (`research/biology/dg-ca3-sparse-index.md`) already
records the mechanism as established, with `OneBrainComposer` as its only `implemented_by` entry before this
session.

## 2. What was ported, and the flag

**`research/runners/rf_phasor_composer.py`** (additive, default-off): added `enable_sparse_index` / env
`BRAIN_SHARD_SPARSE_INDEX` (a DISTINCT env var from `OneBrainComposer`'s `BRAIN_SPARSE_INDEX_RETRIEVAL`, so the two
composers' production defaults stay independently reviewable -- `OneBrainComposer`'s flag was already GO'd-but-
left-OFF as redundant at 100k-bundle scale by the 2026-08-27 finding; this shard composer is the one actually
carrying the wall at bulk-KB scale) plus `sparse_index_g/G/c/conf_floor` (same defaults as `OneBrainComposer`: g=3,
G=16, c=8, conf_floor=0.5). Three new methods -- `_ensure_dg_index`, `_dg_shard_select`, `_full_host_select` --
reuse-import `DGSparseIndex` from `_sparse_indexed_retrieval_derisk.py` (the SAME class `OneBrainComposer` uses; NOT
reimplemented) and are near-verbatim ports of `OneBrainComposer`'s own methods, adapted to this class's
`self.words`/`self.concepts` attributes. `_cleanup` (the single-phasor cleanup `unbind()` calls) and `_cleanup_all`
(the batched cleanup `_scan_first_match` calls -- the actual hot path for `query_patient`/`query_agent`/
`ask_yes_no`/`render_fact`/`query_chain`) each route through the DG shard when `enable_sparse_index` is set AND
`words is None` (the MAIN-vocabulary cleanup only; the 2-word `pol_words` polarity cleanup always passes `words=`
explicitly and is untouched -- too small to benefit).

**`research/runners/sharded_phasor_store.py`** (additive, 7 lines): `ShardedPhasorStore.__init__`'s existing
codebook-sharing graft (build the full `{word: phases}` codebook ONCE on shard 0 / `base`, then point every other
shard's `concepts`/`words`/`roles` at the SAME objects -- the module's own documented fix for a 2026-08-21 OOM where
S independent codebook copies crashed a 46 GB box) now ALSO grafts `_dg_index_source = base` onto every non-base
shard. `RFPhasorComposer._ensure_dg_index` checks this: when set, it delegates entirely to the source composer's
index instead of building its own. Without this graft, `S=3,745` shards would each lazily build an independent
`DGSparseIndex` over the IDENTICAL 347,695-word codebook on first use -- a real `3,745`x memory multiplication of
whatever the index costs, not merely wasted CPU.

No `sim/` edit. No default flipped on. `TieredFactStore`/`developed_brain_io.load_developed_brain` are unchanged --
the flag is entirely env-var-controlled (mirroring `OneBrainComposer`'s own pattern), so an already-built/persisted
bundle (whose `manifest.json` `composer_kwargs` predates this port) still picks up the accelerator via
`BRAIN_SHARD_SPARSE_INDEX=1` with no bundle rebuild.

## 3. A real regression the scale run itself caught, and the fix

The FIRST full-scale attempt exposed a genuine design flaw the small-scale smoke test could not: at 347,695-word
vocabulary against REAL Wikidata entity codes (not the smoke test's synthetic evenly-spaced words), the DG shard
match is frequently NON-DECISIVE (peak score below `conf_floor * D`), so a large fraction of rows in a batched
`_cleanup_all` call escalate to the full-codebook fallback. The first version of this port called
`_full_host_select` in a PYTHON LOOP, one escalated row at a time (a `(1,D)`-vs-`(V,D)` broadcast-and-cos-sum per
row) -- functionally correct, but for an escalation-heavy shard this is SLOWER than the pre-port code, which did
ONE batched complex matmul (`rec_z @ conj(cb).T`, BLAS-backed) over ALL rows at once. A query in the first attempt
ran for several minutes past the point the DG index had already been built and cached, confirming the per-row loop
(not the index) was now the bottleneck. **Fix:** escalated rows are now collected and resolved in ONE batched
complex matmul over just those rows (`esc_z @ conj(cb_z).T`, reusing the cached `_dg_codebook`, no rebuild) instead
of a Python loop -- the SAME BLAS-backed operation the non-indexed path already used, just restricted to the rows
that actually need it. This is a pure performance fix (the decoded word for every row is unchanged -- re-verified
by the smoke test in section 4 after the fix, byte-for-byte identical to before the fix). It also sets a floor on
worst-case behavior: if EVERY row in a shard escalates (the DG index buys nothing for that particular shard), the
batched fallback costs the SAME as the pre-port full scan, not worse -- the accelerator can no longer be slower
than doing nothing.

## 4. Correctness verify (before the scale run)

A local ad hoc smoke script built two `RFPhasorComposer` instances (`enable_sparse_index` False/True) over a
60-fact / 1,500-word synthetic store and compared `query_patient`/`query_agent` decisions row-for-row, then
repeated the same comparison through a 6-shard `ShardedPhasorStore`. Every OFF/ON pair returned the IDENTICAL
decoded word (including several rows where the synthetic fact generator itself produced a stored-cue COLLISION --
two facts sharing one `(agent, action)` cue -- which is a property of the test's fact set, not of the accelerator:
OFF and ON both return the store's own first-match answer for that cue, and they agree with each other in every
case). Unknown-agent queries abstained identically (`None`) under both paths, in both the flat and sharded forms.
Triggering a lazy index build on the sharded store and inspecting `shard._dg_index is base._dg_index` confirmed
**0 of 5 non-base shards built an independent index** -- the graft is load-bearing, not merely present.

## 4. The scale verify: 347,695-word vocabulary, real production path

Both runs used the EXACT command the earlier wall finding used (`_knowledge_scale_vocab_latency_probe.py` through
`developed_brain_io.load_developed_brain(ltm_bundle=...)`, `wikidata_500k_fast`, 748,956 facts / 347,695 vocab /
3,745 shards, `SIM_BACKEND=numpy`, `--seed 42 --n-probes 5`), OFF then ON, back-to-back in the same session so
system load is a shared (not differential) confound:

| run | median recall latency | recall | moat confab | ltm_load_s | peak RSS | elapsed |
|---|---|---|---|---|---|---|
| OFF (baseline, this session) | 33.25s | 5/5 | 0/1 | 5.12s | 4.11 GB | 255.0s |
| ON (`BRAIN_SHARD_SPARSE_INDEX=1`, this session) | 25.84s | 5/5 | 0/1 | 5.10s | 2.89 GB | 310.0s |

<!--derived-->
**This session's OFF baseline (33.25s median) is SLOWER than the 2026-08-28 wall finding's own banked number
(20.67s median) for the identical bundle/probe/seed** -- both the peak-RSS reading (4.11 GB, over the session's
~4 GB budget) and elevated per-query latency point to system-wide memory pressure during this run (`free -m`
showed ~38 GB of 48 GB swap in use from concurrent processes at the time), not a change in the OFF code path
(which is provably byte-unchanged -- see the diff in section 2, no line inside the `enable_sparse_index=False`
branch was touched). The OFF/ON comparison below is a PAIRED same-session measurement to control for this, but the
absolute OFF number should not be read as a fresh baseline superseding the earlier 20.67s figure.

**Per-query breakdown (ON), separating the one-time index-build cost from steady-state query cost:**

| query | OFF latency | ON latency |
|---|---|---|
| recall 0 (pays the one-time DG-index BUILD, first cache miss) | 41.70s | 116.53s |
| recall 1 | 33.15s | 25.84s |
| recall 2 | 33.25s | 24.00s |
| recall 3 | 33.17s | 20.45s |
| recall 4 | 36.59s | 42.54s |
| moat (unknown entity, abstain) | 26.95s | 26.66s |
| yesno | 41.19s | 45.75s |

**Query 0 is SLOWER under ON** (116.53s vs 41.70s) -- entirely the one-time `DGSparseIndex.build()` cost (a
`DG expansion + hard k-WTA` pass over the full 347,695-word codebook, `research/runners/
_sparse_indexed_retrieval_derisk.py`'s own mechanism, reused unmodified), paid ONCE per process (shared across all
3,745 shards via the `_dg_index_source` graft -- confirmed by the smoke test in section 4, and here by the fact
that queries 1-4 do NOT repeat this cost). **116.53s is consistent with the mechanism's OWN already-banked 6-seed
de-risk** (`build_s` 42.1-73.6s measured at V=200,000 in an unloaded environment,
`research/findings/raw/four_day/_sparse_indexed_retrieval_6seed.json`) linearly extrapolated to V=347,695 (~73-129s
expected) -- the build cost is NOT anomalous, it is the mechanism behaving as already characterized.

**Queries 1-3 are FASTER under ON** (25.84s/24.00s/20.45s vs 33.15s/33.25s/33.17s -- a consistent ~25-38% reduction
once the index is warm). **Query 4 (42.54s) and yesno (45.75s) are SLOWER under ON** than their OFF counterparts
(36.59s, 41.19s) -- an inconsistency this finding reports rather than hides. Excluding query 0 (the one-time
build), median warm-query latency is **24.92s (ON) vs 33.21s (OFF), a 1.33x (~25%) reduction** -- real, paired,
same-session, but NOT the "toward sub-second" outcome the port targeted, and NOT uniformly a win on every query.

**Root-cause read on why the reduction is modest, not dramatic.** The per-shard fact count K stays flat (~200) by
`ShardedPhasorStore`'s existing sharding design regardless of V, so the batched UNBIND RESONATE step
(`_unbind_all_phases`, an RF resonate over `2*K*D` neurons for `period+8` steps) is V-INDEPENDENT and unaffected by
this port -- it was never the wall's target. This port only removes the O(V) CLEANUP term. Two candidate
explanations for the residual gap, both consistent with the evidence and NOT mutually exclusive: (1) **the DG
shard match is frequently non-decisive on real Wikidata entity codes** (unlike the de-risk's synthetic, evenly-
distributed FHRR codes), so many role-cleanups still ESCALATE to the full-codebook scan -- the escalation is now
batched (section 3's fix) rather than a Python loop, but a fully-escalating shard still pays close to the
pre-port O(V) cost, just via one efficient matmul instead of many small ones; (2) **this session's memory
pressure inflates every operation, including the parts of the pipeline this port did not touch** (the resonate
step, `ltm_load_s`, and the OFF baseline itself all read slower than their own previously-banked/expected values)
-- so the TRUE post-fix speedup on an idle machine is plausibly larger than 25%, but this session cannot cleanly
separate the two effects. Distinguishing them (instrumenting the actual escalation rate per query) is the
concrete next lever, named rather than attempted here (out of this session's time-box).

**External check (2026-08-09-gate `deep_research_at_wall`, 3rd `knowledge-integration` finding in 3 days):**
explanation (1) above is independently consistent with the LSH/banded-hashing literature -- a bucket-occupancy
scheme validated on synthetic, evenly-distributed test vectors (this project's own `_sparse_indexed_retrieval_
derisk.py` de-risk, uniform random FHRR codes at 10k/50k/200k) is a known way to UNDER-measure the escalation a
real, skewed embedding distribution produces: "synthetic random vectors hide collision pathologies you'll see in
real embeddings... LSH assumes the embedding space behaves nicely, and if your vectors are poorly normalized or
semantically noisy, LSH degrades faster" (web search, 2026-08-28:
https://medium.com/@PriyaSingh325/locality-sensitive-hashing-in-the-real-world-when-approximation-beats-perfection-5685453e0cc3 ;
recorded via `tools/record_external_search.sh`'s external-search ledger, lane `knowledge-integration`). This does not prove explanation
(1) over (2) for THIS run (both remain live, per the residual above), but it is external confirmation that
explanation (1) is a REAL, documented failure mode of this class of index, not a speculative one -- strengthening
the case for the named next lever (measure the actual escalation rate against this bundle's real entity codes)
over assuming the synthetic de-risk's sublinearity transfers unchanged to real-world vocabulary.

## 5. Degrade-identical (`ltm=None`)

Unchanged by this port: `TieredFactStore` with `ltm=None` never constructs a `ShardedPhasorStore` at all (the
`enable_sparse_index` code path is entirely inside `RFPhasorComposer`/`ShardedPhasorStore`, never reached when
`ltm_bundle` is unset), so the plain-buffer degrade path is structurally untouched -- the same argument the
2026-08-20/2026-08-28 findings already established for this property, now re-affirmed as still valid: this port
added no new call site on the `ltm=None` branch.

## 6. Verdict

**PARTIAL.** The port is mechanically correct (parity verified two ways: a small-scale smoke test with an
exhaustive row-by-row OFF/ON comparison, and this scale run's own 5/5 recall + 0/1 moat confab, matching OFF
exactly on every decoded answer), the codebook-sharing graft prevents an S-fold memory blow-up (confirmed: peak
RSS was LOWER with the accelerator ON than OFF, 2.89 GB vs 4.11 GB -- the OFF path's per-query codebook rebuild is
itself a real, if smaller, memory churn this port also removes), and the escalation-batching fix (section 3) means
the accelerator can never be asymptotically WORSE than doing nothing. But the measured warm-query speedup (~25%,
33.21s -> 24.92s median) is a REAL, MODEST reduction, not the "toward sub-second" outcome the port targeted. **The
vocab-latency wall is NOT closed to sub-second by this port alone** -- it is REDUCED, and the residual (whether
frequent DG-shard escalation on real entity codes, this session's memory pressure, or both) is named as the next
lever rather than resolved here.

## Honest residuals

1. **This session's absolute latency numbers (both OFF and ON) are confounded by concurrent system load**
   (elevated swap use measured directly via `free -m` and via this process's own major-page-fault counter growing
   during the run) -- the RELATIVE (paired, same-session) comparison is the reliable read; a re-run on an
   otherwise-idle box would likely show both numbers lower in absolute terms, and might also show a CLEANER
   speedup if the escalation-rate hypothesis (residual #5 below) is not the dominant factor.
2. **The one-time DG-index build cost is real and substantial** (116.53s in this run, consistent with the
   mechanism's own de-risk extrapolated to V=347,695) but AMORTIZED -- paid once per process lifetime, shared
   across all 3,745 shards via the `_dg_index_source` graft (confirmed: queries 1-4 did not repeat it). A live
   chat server pays this once at the FIRST query after each restart, not per query; whether that cold-start cost
   is acceptable, or should instead be paid at BUNDLE-BUILD time and persisted alongside `composites.npz`
   (`ShardedPhasorStore.save`/`load` do not currently persist the DG index), is a follow-on engineering rung not
   attempted here.
3. **The DG sparse projection remains the same declared host-rate shortcut `OneBrainComposer`'s port already
   named** (a fixed random sparse projection + hard argmax-WTA, computed on the host): its biological burn-down is
   the spiking DG granule-cell layer already validated in the trisynaptic-loop probes
   (`_riii_ca3_completion_specificity_derisk.py`, `cortex_dg_ca3_cleanup_probe.py`,
   `_gap5_emergent_dg_selection_derisk.py`) -- unchanged scope, now shared across both composers that use the index.
4. **Default stays OFF** (`BRAIN_SHARD_SPARSE_INDEX` unset) pending owner review, per this session's explicit
   scope (no bundle default-on flip attempted) -- this is a validated OPT-IN accelerator, not yet
   `wired`/`on-by-default` in the `docs/TERMS.md` sense, and its measured speedup here is MODEST (~25%), not
   large -- the vocab-latency wall is neither `closed` in the strict TERMS.md sense (not on-by-default) NOR
   closed in the plain sense of "the wall is gone" (queries are still ~20-45s, not sub-second).
5. **The escalation-rate hypothesis (why the reduction is 25%, not dramatic) is NAMED, not resolved.** The
   concrete next lever: instrument `_dg_shard_select` to log/count how often a role-cleanup escalates (peak <
   `conf_floor * D`) over a real-Wikidata-entity batch of queries, at THIS bundle's actual `D=128`/`conf_floor=0.5`
   operating point. If escalation is common, tuning `sparse_index_g`/`G`/`c`/`conf_floor` against real entity-code
   separability (not just the de-risk's synthetic evenly-distributed codes) is the next rung; if escalation is
   rare, the residual gap is System contention and a clean-machine re-run is the next rung instead.

## What the owner needs to decide

1. Whether to flip `BRAIN_SHARD_SPARSE_INDEX=1` on for the `BRAIN_LTM_BUNDLE` production path (this finding's
   scope is additive/default-off verification only, per the session's constraints) -- given the measured
   ~25% warm-query reduction is real but modest, and the accelerator adds a one-time ~100s+ cold-start cost per
   process, this is a genuine trade-off, not a clear win to flip on sight.
2. Which bundle ships as the default `BRAIN_LTM_BUNDLE` target -- unchanged from the 2026-08-28 finding's open
   question; this port changes the LATENCY side of that decision only modestly (queries are still tens of
   seconds), not the bundle-selection question itself.
3. Whether to fund the next lever named in residual #5 (measuring/tuning the DG shard escalation rate against
   real entity codes) -- the more promising rung if the owner wants the wall closed toward sub-second, versus
   accepting the current ~25% reduction as the port's contribution and looking for the latency win elsewhere
   (e.g. the resonate step this port does not touch).
3. Whether the one-time DG-index build cost is acceptable to pay at server startup (amortized over the server's
   run, shared across all shards) versus persisting the built index alongside the bundle (a follow-on engineering
   rung, not attempted here).
