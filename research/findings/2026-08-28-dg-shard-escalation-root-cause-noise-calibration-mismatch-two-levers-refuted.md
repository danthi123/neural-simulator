---
type: finding
status: live
date: 2026-08-28
mechanism: dg-shard-escalation-diagnostic
lane: knowledge-integration
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_dg_shard_escalation_diagnostic.py (NEW, this session) -- instruments the PRODUCTION
  `RFPhasorComposer._dg_shard_select` directly (imported/called, not reimplemented) against a real bundle's own
  codebook plus a resonate-recovery noise level CALIBRATED from the production `store()`/`_unbind_phases()`
  path (not assumed). Cross-checked against a literal one-line reproduction of the same scoring formula for a
  bare synthetic FHRR codebook (documented in the module docstring as the one place this script duplicates
  rather than calls production code, since a synthetic codebook has no composer instance to call a method on).
runner: research/runners/_dg_shard_escalation_diagnostic.py
external: Lv, Josephson, Wang, Charikar & Li 2007, VLDB "Intelligent Probing for Locality Sensitive Hashing:
  Multi-Probe LSH and Beyond" (http://vldb.org/pvldb/vol10/p2021-lv.pdf) -- the canonical multi-probe-LSH
  technique the DGSparseIndex `G` parameter already implements; recorded via `tools/record_external_search.sh`
  into the external-search ledger, lane `knowledge-integration`.
artifacts:
  - research/findings/raw/dg_shard_escalation/diag_seed42_real500k.json
  - research/findings/raw/dg_shard_escalation/diag_seed43_real500k.json
  - research/findings/raw/dg_shard_escalation/diag_seed44_real500k.json
  - research/findings/raw/dg_shard_escalation/diag_seed100_real500k.json
  - research/findings/raw/dg_shard_escalation/diag_seed101_real500k.json
  - research/findings/raw/dg_shard_escalation/diag_seed102_real500k.json
  - research/findings/raw/dg_shard_escalation/floor_sweep_misroute_seed42_real500k.json
  - research/findings/raw/dg_shard_escalation/lever_more_probes_G32_seed42_real500k.json
---

# The DG-shard escalation wall is a NOISE-CALIBRATION mismatch (the de-risk's own test noise was ~4x too small), not real-vs-synthetic code geometry — 99.5% escalation (6-seed), two levers tested, both REFUTED as production fixes

Artifacts: `research/findings/raw/dg_shard_escalation/diag_seed{42,43,44,100,101,102}_real500k.json`,
`research/findings/raw/dg_shard_escalation/floor_sweep_misroute_seed42_real500k.json`,
`research/findings/raw/dg_shard_escalation/lever_more_probes_G32_seed42_real500k.json`.

**One line.** On the real `wikidata_500k_fast` bundle (347,695-word vocab, D=128, production defaults), the
DG-shard cleanup escalates to the full-codebook scan on **99.50% of cleanups (6-seed mean, 99.37%-99.70%, n=3000
queries/seed)** -- not because the shard's confidence threshold is miscalibrated (the escalate-vs-decide
`conf_floor` gate), and not because real Wikidata entity codes have different geometry than synthetic ones (a
matched-scale synthetic sweep escalates at an almost identical 99.4%), but because the ORIGINAL de-risk's own
assumed query noise (sigma=0.30 radians) is **~4.1x smaller** than what this composer's actual RF-resonate
recovery produces in production (measured sigma=1.27 radians, calibrated through the real `store()`/
`_unbind_phases()` path, consistent across three cue roles). At that real noise level the DG hash's discrete
per-band winner-take-all is far more brittle than the smooth linear cleanup it replaces: **the true stored code
is a member of its own DG-routed candidate shard only 1.1%-2.0% of queries (6-seed mean 1.7%)** -- a genuine
MISROUTE, not an under-confident-but-correct match. Two levers were tested against this (lowering `conf_floor`;
doubling the multi-probe group count `G`); both are REFUTED as production fixes, with the reasons quantified
below. Per this session's own scope, the measured diagnosis + a named next lever is the deliverable.

## 1. Verify-first

Read `research/findings/2026-08-28-shard-composer-dg-sparse-index-port-modest-latency-reduction.md` (the port
that ADDED the DG-indexed fast path to `RFPhasorComposer`/`ShardedPhasorStore`, measuring a real but modest ~25%
warm-query latency reduction and naming, without measuring, "the DG shard match is frequently non-decisive on
real Wikidata entity codes" as one of two live candidate explanations). Read `research/runners/
_sparse_indexed_retrieval_derisk.py` (the original DG-index mechanism: `DGSparseIndex`, `gen_fhrr_phases`,
sigma=0.30 rad as the de-risk's own default query noise, D=256 as "the production console op-point" --
**already stale**: the shard composer that was later ported to runs at D=128, per `wikidata_500k_fast/
manifest.json`). Read `research/runners/rf_phasor_composer.py`'s `_ensure_dg_index`/`_dg_shard_select`/
`_full_host_select`/`_cleanup_all` (lines 700-846) and `research/runners/sharded_phasor_store.py`'s
`_dg_index_source` graft (lines 89-126). Read the biology binding `research/biology/dg-ca3-sparse-index.md`.
`bash tools/before_you_build.sh "sparse-index shard escalation real entity codes latency"` found no prior
scoping doc against this exact defect (only the port finding's own residual #5, which named the lever --
"instrument `_dg_shard_select` to log/count how often a role-cleanup escalates" -- but did not attempt it).

## 2. Method (efficient -- does not resonate/store the full 748,956-fact bundle)

The naive approach (run the full conversational pipeline hundreds of times) is intractable: at the measured
~25-45s/query, hundreds of real end-to-end queries would take hours. Instead `research/runners/
_dg_shard_escalation_diagnostic.py` measures the DG routing DIRECTLY, three steps:

1. **Calibrate the REAL noise** (production path, small and bounded). Build ONE `RFPhasorComposer` over the
   REAL, FULL bundle codebook (same seed as the bundle manifest -> byte-identical codes to production,
   `sharded_phasor_store.py`'s own documented "the codebook regenerates byte-identically from seed+vocab"
   property). Store 300 REAL facts sampled from the bundle's `facts.json` through the ACTUAL production
   `RFPhasorComposer.store()` (a real RF resonate bind of agent+action+patient[+polarity], with the genuine
   intra-fact crosstalk multi-role superposition introduces). For each, `_unbind_phases` the cue role and
   compare the RECOVERED phase to the TRUE stored concept phase -- this calibrates the REAL resonate-recovery
   noise sigma, measured rather than assumed.
2. **Escalation sweep** (REAL codes). Sample 3000 REAL words from the bundle vocab, add the calibrated noise,
   call the PRODUCTION `RFPhasorComposer._dg_shard_select` directly (imported, not reimplemented, with
   `conf_floor` temporarily forced to -999 so it always returns the shard's raw local top-1 -- letting many
   candidate floors be swept from ONE cached table with no rebuild, no re-routing). Also record whether the
   TRUE word's own index is a MEMBER of the DG-routed candidate set (distinguishes MISROUTE from a merely
   under-confident-but-present match).
3. **Matched-scale synthetic comparison.** The identical sweep over a synthetic uniform-random FHRR codebook at
   the IDENTICAL V=347,695/D=128 (the original de-risk's own `gen_fhrr_phases`/`DGSparseIndex`, reused by
   import), same calibrated sigma -- isolates whether escalation is about real-entity-code GEOMETRY or the
   D=128/V=347,695/noise OPERATING POINT itself.

Memory: builds ONE (V,D) codebook (~340MB at V=347,695/D=128) + ONE `DGSparseIndex` bucket set at a time (real
and synthetic run sequentially with an explicit `del`+`gc.collect()` between them). Peak RSS across the 6 baseline
seed runs: 2.82-3.18GB, within the ~4GB session budget (the `more_probes` lever run at G=32 reached 4.06GB --
see section 5).

## 3. THE CORE NUMBER: 99.50% escalation, 6-seed, real production bundle

`escalation_frac`/`mean shard size`/`decisive parity` below are read verbatim from each seed's `real` block in
`diag_seed<N>_real500k.json`. `calibrated sigma (rad)` is `sigma_fractional_cycle_calibrated * 2*pi` (the JSON
stores the fractional-cycle value only). `true_in_shard` (the misroute check) is transcribed from this session's
own run console output (the `_log` line each run printed) -- it was NOT persisted into the JSON artifact (an
instrumentation gap section 10's residual 5 flags), so it is reproducible by re-running
`_dg_shard_escalation_diagnostic.py` at the same seed but is not independently checkable from the committed
artifact alone. The 6-seed mean/std row is computed across the six per-seed files. The whole table is therefore
DERIVED (unit conversion, transcription, and cross-file aggregation), not a verbatim single-artifact quote:

<!--derived-->

| seed | escalation_frac | true_in_shard (misroute check) | calibrated sigma (rad) | mean shard size | decisive parity |
|---|---|---|---|---|---|
| 42 | 99.60% (2988/3000) | 1.1% (34/3000) | 1.2743 | 24.5 | 1.0 (12/12) |
| 43 | 99.40% (2982/3000) | 1.9% (57/3000) | 1.2669 | 24.7 | 1.0 (18/18) |
| 44 | 99.53% (2986/3000) | 1.7% (52/3000) | 1.2696 | 24.4 | 1.0 (14/14) |
| 100 | 99.70% (2991/3000) | 1.8% (55/3000) | 1.2660 | 24.3 | 1.0 (9/9) |
| 101 | 99.40% (2982/3000) | 2.0% (61/3000) | 1.2807 | 24.2 | 1.0 (18/18) |
| 102 | 99.37% (2981/3000) | 1.8% (55/3000) | 1.2606 | 24.6 | 1.0 (19/19) |
| **mean** | **99.50%** (std 0.12%) | **1.74%** | 1.2697 | 24.5 | 1.0 |

Every escalation is `low_peak` (shard non-empty, peak below `conf_floor*D`); `escalated_empty_shard=0` at every
seed -- the DG hash always returns SOME candidates, they are usually just not the right ones. The FEW decisive
cases (9-19 per seed at default `conf_floor=0.5`) are ALWAYS correct (`decisive_top1_self_agree=1.0`, all 6
seeds), and a full-codebook self-consistency floor check (12-19 samples/seed) also reads 1.0000 -- the EXISTING,
already-shipped (default-OFF) mechanism remains CORRECT as characterized in the port finding; this session finds
no new correctness bug, only that it rarely fires.

## 4. WHY: a noise-calibration mismatch, confirmed NOT real-vs-synthetic geometry

The port finding's residual #5 credited the escalation to "the DG shard match is frequently non-decisive on
real Wikidata entity codes... unlike the de-risk's synthetic, evenly-distributed FHRR codes" and cited external
LSH literature on real-embedding skew as consistent with that. This session's matched-scale comparison refutes
the geometry-specific version of that hypothesis directly: at the IDENTICAL V=347,695/D=128 operating point and
the SAME calibrated noise, a SYNTHETIC uniform-random codebook (the de-risk's own `gen_fhrr_phases`,
statistically IDENTICAL in construction to how `RFPhasorComposer.__init__` builds `self.concepts` -- both are
i.i.d. `rng.uniform` draws per word) escalates at **99.4%** (2981/3000, seed 42) -- essentially the SAME as the
real bundle's 99.6%. Real and synthetic codes are not geometrically distinguishable here because they are built
by the identical random construction; the wall is an OPERATING-POINT property (D, V, and noise), not something
peculiar to real entity embeddings.

The actual driver: the original de-risk (`_sparse_indexed_retrieval_derisk.py`) validated its 6-seed GO at
`sigma=0.30` radians of query phase noise -- but this session's calibration, run through the ACTUAL production
`RFPhasorComposer.store()`/`_unbind_phases()` RF-resonate path (not assumed, not a synthetic jitter model) on
300 real bundle facts, measures the TRUE recovery noise at **sigma=1.27 radians (6-seed mean), ~4.1x larger**.
Cross-validated across three different cue roles on a smaller bundle (`wikidata_core_15k_smoke`, V=206) to rule
out a single-role artifact: agent=0.197, action=0.208, patient=0.203 fractional-cycle (~1.24-1.31 rad) --
consistent regardless of which role is unbound, as expected since the noise source (RF resonate settling +
intra-fact multi-role crosstalk from `_encode`'s bundle-of-binds) does not depend on total vocabulary size.

A direct sensitivity check (bare `DGSparseIndex`, V=206, D=128, G=16, n=500) makes the mechanism unambiguous:
escalation is **0%** at the de-risk's own sigma=0.30 rad, and **85.6%** at the calibrated sigma=1.27 rad --
the SAME index, SAME codebook, SAME V/D, only the noise magnitude differs. The de-risk's GO was real and
correctly measured; it was measured against a noise level roughly 4x smaller than production's own RF dynamics
actually produce.

**Escalation is a MISROUTE, not a threshold-calibration issue.** At production defaults (G=16), the true
stored code's own index is a member of its DG-routed candidate shard in only 1.7% of queries (6-seed mean) --
the rest can NEVER be answered correctly by that shard at ANY `conf_floor`, because the correct candidate simply
is not present in the returned set. This is why the two levers below behave the way they do.

## 5. LEVER 1 -- `lower_conf_floor`: REFUTED (unsafe)

A cheap post-hoc sweep of the SAME cached raw-scores table (no rebuild) at floors 0.15-0.50 (seed 42, real
bundle, `research/findings/raw/dg_shard_escalation/floor_sweep_misroute_seed42_real500k.json`). Values below are
read from the artifact's `floor_sweep` list, rounded to 1-4 significant figures for the table:

<!--derived-->

| floor | escalation_frac | n_decisive | parity (decisive top1 == true) |
|---|---|---|---|
| 0.15 | 19.5% | 2416 | **0.0141** |
| 0.20 | 77.7% | 669 | **0.0508** |
| 0.25 | 97.1% | 87 | **0.3908** |
| 0.30 | 98.8% | 37 | 0.9189 |
| 0.35 | 98.9% | 34 | 1.0 |
| 0.40 | 98.9% | 32 | 1.0 |
| 0.45 | 99.0% | 29 | 1.0 |
| 0.50 (default) | 99.6% | 12 | 1.0 |

The only floors that meaningfully cut escalation (<=0.25) crash parity to 1.4%-39% -- because as the floor
drops, the mechanism starts ACCEPTING misrouted shards' spurious local top-1 as if it were decisive (the shard's
own best-scoring candidate under noise is often simply the wrong word, and a low enough floor lets it through).
Floors that preserve exact parity (>=0.35) barely move escalation at all (99.6% -> 98.9%, i.e. essentially no
practical gain). There is no floor value that both meaningfully reduces escalation AND preserves the required
exact recall -- REFUTED as a production fix.

## 6. LEVER 2 -- `more_probes` (double `G`, the multi-probe OR-amplification knob): REFUTED (infeasible at scale)

Doubling `G` from 16 to 32 (`research/findings/raw/dg_shard_escalation/lever_more_probes_G32_seed42_real500k.json`,
seed 42, real bundle) is the standard multi-probe-LSH technique this project's own `G` parameter already
implements (Lv, Josephson, Wang, Charikar & Li 2007, VLDB "Intelligent Probing for Locality Sensitive Hashing:
Multi-Probe LSH and Beyond" -- probe multiple nearby buckets instead of building more hash tables). It DOES
help, directionally: true-in-shard hit rate rises **1.1% -> 3.4%** (a real ~3x improvement), and escalation at
default `conf_floor=0.5` drops **99.6% -> 98.9%**. But the gain is tiny in absolute terms and the per-probe hit
rate is far lower than typical multi-probe LSH deployments assume (the paper's probing-sequence construction
targets small, controlled hash-bit perturbations, not phase noise this large against an `m=71`-way per-band
winner-take-all). Modeling the G disjoint OR-groups as roughly independent draws (`self._groups` are disjoint
band partitions by construction), the per-group hit probability implied by the two measurements is ~0.07%-0.2%
-- reaching even a modest 90% true-in-shard hit rate at that rate would require **G on the order of several
thousand** (`L = g*G` bands, bucket-table memory ~ `V*G` entries). This is not just slow: **memory is already at
the session's ~4GB budget at G=32** (peak RSS 4.06GB, up from 2.82-3.18GB at G=16 for the baseline runs) --
pushing G an order of magnitude further would blow the budget long before it closed the gap. REFUTED as a
production fix at this scale.

## 7. A structural observation (code-reading, not re-tested this session): the port's own measured ~25% speedup is plausibly NOT from shard short-circuiting

Given escalation is ~99.5%, almost every `_cleanup_all` call still pays a near-full-V matmul either way -- so why
did the port finding measure a real, paired, same-session ~25% latency reduction at all? Reading the two code
paths side by side in `rf_phasor_composer.py` suggests a DIFFERENT, simpler explanation than shard
short-circuiting. The non-indexed `_cleanup_all` (around line 843-844) rebuilds the FULL `(V,D)` complex codebook
from a Python dict comprehension **on every call**: `cb = np.stack([np.exp(2j*np.pi*self.concepts[w]) for w in
words])` -- a V=347,695-iteration Python loop with a dict lookup per word, paid 2-3 times per query (once per
cue role in `_scan_first_match`). The indexed path's escalation branch (around line 834) instead reuses
`self._dg_codebook`, a raw-phase `(V,D)` numpy array built ONCE in `_ensure_dg_index` (`_ensure_dg_index`,
line ~722), so its `cb_z = np.exp(2j*np.pi*self._dg_codebook)` is one vectorized `np.exp` call over an
already-materialized array, not a per-call Python dict-comprehension rebuild. This is a genuine speed win that
is STRUCTURALLY INDEPENDENT of whether the DG index routes correctly at all -- consistent with a real ~25%
reduction persisting even though shard-level routing is escalating on ~99.5% of rows. This is NOT re-verified
by a new experiment this session (it would be a third lever, outside this session's cap); it is named here as
the concrete, cheap, and separately-testable next step in section 8.

## 8. Named next levers (not attempted -- per this session's 2-lever cap)

1. **Cache the codebook independently of the DG index.** Build `self._dg_codebook`-equivalent (a raw-phase
   `(V,D)` array, built once and invalidated only when `len(self.words)` changes) for the NON-indexed
   `_cleanup_all` path too, whether or not `enable_sparse_index` is set. If section 7's hypothesis is right,
   this alone recovers most or all of the previously-measured ~25% win, WITHOUT the DG index's misrouting risk,
   memory cost, or one-time ~100-150s build. Cheap, low-risk, directly testable.
2. **Decouple the DG granule width `m` from `V^(1/g)`-driven occupancy.** `m` is currently chosen ONLY to keep
   bucket occupancy O(1) as V grows (`m ~ V^(1/g)`, giving `m=71` at this bundle's V=347,695) -- not chosen for
   noise robustness. A hierarchical/coarse-to-fine index (route through a small, FIXED, noise-robust `m`
   regardless of V, controlling occupancy by adding routing STAGES instead of widening each band) would let
   noise robustness and vocabulary scale vary independently.
3. **Raise D.** The original de-risk's own validated operating point was D=256 ("the production console
   op-point" per its docstring); the shard composer this was ported to actually runs at D=128. A larger D gives
   the DG projection's per-band k-WTA more input dimensionality (`c` afferents sampled from `2D`) and the linear
   cleanup a wider noise margin; whether it meaningfully improves per-band winner stability under the measured
   1.27rad noise is untested this session.

## 9. Verdict

**NO-GO on both tested levers as production fixes; the diagnosis is the deliverable.** The wall is real,
precisely characterized (99.50% escalation, 6-seed, real production bundle), and its cause is now a measured
noise-calibration mismatch (production RF-resonate recovery noise ~4.1x larger than the de-risk's own test
assumption) driving a genuine DG-hash MISROUTE (true candidate present in its own shard only 1.7% of queries),
not a threshold-calibration issue and not real-vs-synthetic code geometry. Neither `lower_conf_floor` (unsafe:
parity crashes to 1.4%-39% at any floor that meaningfully cuts escalation) nor `more_probes` (infeasible: the
per-group hit rate is so low that closing the gap needs thousands of probe groups, already hitting the session's
RSS budget at a mere 2x increase) is a viable fix. No code change ships from this session -- the already-merged
`enable_sparse_index`/`BRAIN_SHARD_SPARSE_INDEX` accelerator remains exactly as it was (default-OFF, correctness
intact where it does fire), and this finding adds no risk to it. The wall stays OPEN; section 8 names three
concrete next levers, ordered by expected cost/risk, none of which this session's 2-lever cap permits testing.

## Honest residuals

1. **The 6-seed escalation number (99.50%) is tight (std 0.12%) and robust**, but each seed samples only 3000
   of 347,695 vocabulary words as query cues (uniform random, not frequency-weighted) -- a genuinely different
   sampling scheme (e.g. weighted toward words that actually appear as query cues in live chat) could shift the
   number somewhat, though the underlying MECHANISM (noise >> de-risk's assumption, `m=71`-way k-WTA too
   fine-grained to survive it) would not change.
2. **The noise calibration used 300 real facts and the `patient`/`agent`/`action` cue roles** (cross-validated
   across all three at the smoke scale, not re-cross-validated at full 500k scale for time -- the mechanism does
   not depend on V, so this is a reasonable but not exhaustively re-verified extrapolation).
3. **Section 7's "codebook caching explains the port's ~25% win" claim is a code-reading inference, not a
   re-run experiment this session** -- flagged explicitly as the first, cheapest item in section 8's next-lever
   list rather than asserted as established.
4. **The `more_probes` lever's G=32 build reached 4.06GB peak RSS**, at the edge of this session's ~4GB budget;
   G=64+ was not attempted (would very plausibly exceed it), so the "thousands of probes needed" extrapolation
   in section 6 is analytic (from the measured per-group hit-rate trend), not empirically confirmed at higher G.
5. **The `true_in_shard` misroute statistic (1.1%-2.0% per seed, section 3) is NOT persisted in the committed
   JSON artifacts** -- `_dg_shard_escalation_diagnostic.py` only logs it to stdout (this session's own run
   output, transcribed faithfully into section 3's table) and does not write it into the `real` dict the JSON
   dump saves. Re-running the script at the same seed reproduces it deterministically, but a reader auditing
   the committed artifact alone cannot verify this specific number without re-running. A follow-on to this
   script should add `true_in_shard_frac` to the saved JSON so the number is artifact-backed, not just logged.

## What the owner needs to decide

1. Whether to fund the cheapest named next lever (section 8.1, decouple codebook caching from the DG index) --
   plausibly recovers the ~25% win already banked WITHOUT the DG index's misrouting risk or one-time build cost,
   and is a small, low-risk change.
2. Whether the DG-shard accelerator (`BRAIN_SHARD_SPARSE_INDEX`) is worth keeping wired at all given it now
   escalates on ~99.5% of production-scale queries -- it remains CORRECT (parity 1.0 whenever it does decide)
   and harmless (default-OFF, no regression), but this session found no lever that makes it deliver its intended
   sublinear-retrieval benefit at this bundle's actual noise/scale operating point.
3. Whether raising D (section 8.3) or the hierarchical-index redesign (section 8.2) is worth the larger
   engineering investment those options would require, versus accepting the wall as characterized and
   redirecting effort toward a different real-time-recall lever entirely (e.g. the resonate/unbind step this
   port never touched, per the original port finding's own residual).
