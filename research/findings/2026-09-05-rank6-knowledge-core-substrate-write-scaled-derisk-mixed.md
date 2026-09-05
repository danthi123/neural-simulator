---
type: finding
status: mixed
date: 2026-09-05
mechanism: knowledge-core-substrate-write
lane: scaffold-retirement
board: scaffold_retirement_backlog#6
runner: research/runners/_rank6_knowledge_core_substrate_write_derisk.py
seeds: [42, 43, 44, 100, 101, 102]
seed-waiver: The 6-seed battery validates the STOCHASTIC claim below (recall+moat answer-parity between the
  numpy-kb and substrate-store paths depends on the seeded codebook, so it is run at all 6 mission seeds). The
  memory/time COST slope and the save/reload structural probe are deterministic resource/structural
  measurements (a peak-RSS reading; an exception or its absence) with no seed-dependent stochastic component to
  average over, so they are reported at seed 42 only, matching this project's own `tools.lab.project_cost`
  convention (one finished unit is ground truth for a resource projection; re-seeding a memory reading adds no
  information).
artifacts:
  - research/findings/raw/_rank6_knowledge_core_substrate_write_derisk/full_run.json
---

# RANK-6 knowledge-core WRITE de-risk: per-fact synaptic-substrate parity is GO (6/6) at real curated-fact scale; persistence-format + curation-selection are the actual residuals, and the literature already validates the architecture at larger scale than our target

## Verdict

**MIXED, by design (a characterization + a scoped de-risk, not a capability flip).** `scaffold_retirement_backlog.md`
rank-6 named the ~78k-fact knowledge core as "written by a closed-form host routine." Reading the substrate
first (per this project's own standing lesson) shows that claim is **half right**: the per-fact VSA bind is
already genuinely neural, and the actual host residual is narrower — data PERSISTENCE and CURATION SELECTION,
not the bind computation. The already-existing, already-validated synaptic-weight write
(`enable_substrate_store=True`) is confirmed **GO (6/6 seeds)** for answer+moat parity against the shipped
numpy-kb path, now at REAL curated-fact scale (not the small synthetic batteries the original 2026-06-05
Phase-2 de-risk used). Extending it from the small conversational buffer to the actual ~78,857-fact bulk core
surfaces concrete, previously-uncharacterized costs: a MEASURED marginal-memory slope that projects to **3.78
GB at full scale — genuinely affordable, not a blocker**; a query-time cost (loses the batched scan, falls back
to a per-fact loop) confirmed by code-reading but not separately benchmarked; and a real save/reload structural
gap (`TypeError: cannot pickle 'mappingproxy'`) that IS the concrete, previously-undocumented reason the LTM
class excludes this path today. None of these is a capability wall — the literature (Crawford-Gingerich-
Eliasmith 2016) already demonstrates the identical per-fact-population architecture holding MORE facts
(117,659) than our target, in spiking neurons, at a fraction of the neurons/fact our generic-bridge
implementation spends — so the residual is engineering efficiency (a lean per-fact population + a real
serializer), not architecture. Also confirmed: the "at minimum, honest" fallback the task named already has its
data dependency sitting unused in production. Default-off throughout; **the production store path is unchanged
by this file.**

## Premise check (read the substrate before theorizing)

1. **The ~78k-fact core is real and identified.**
   The box-local bundle directory `sim-data/knowledge_bundles/wikidata_100k` (a sibling of this repo, NOT
   tracked in git — the runner's own `--bundle` default) has a `curation_report.json` recording `n_facts:
   78857` (the actual number of qualifying (subject,relation) candidates the curator
   (`research/runners/_knowledge_core_curate.py`) found under `--top-entities 25000 --top-relations 60`, despite
   the directory's "100k" name — a follow-on scale bundle, not the currently-shipped default,
   `wikidata_core_15k`/15,000 facts, per the 2026-08-26 knowledge-core-ship finding). Both bundles are built by
   the identical mechanism this file probes.

2. **The per-fact BIND is already genuinely neural, not closed-form.** `curation_report.json`'s own `"fast":
   false`, `"build_seconds": 1913.5` for 78,857 facts (~24.3ms/fact) confirms `wikidata_100k` was built with the
   genuine resonate bind: `RFPhasorComposer.store()` → `_encode()` → `_bind()`/`_bundle()` → `_resonate()` steps a
   real `SimulationBridge` of RESONATE_AND_FIRE neurons per fact (`rf_phasor_composer.py:346-423`), NOT
   `tiered_fact_store.encode_fast()`'s closed-form `np.exp(2j·π·(...))` shortcut (that shortcut IS used,
   correctly out of `ship_ready` scope, for the separately-built 500k/1M `_fast` bundles — confirmed via their
   own `curate_report_*_fast.json`, `"fast": true`). The composer's spiking computation (bind/unbind/bundle/
   cleanup) was already established fully-on-substrate for the small conversational buffer by the 2026-07-20
   finding (`composer-factstore-host-persistence-is-the-VSA-idealization-scoping.md`); point 3 below is that
   finding's own named residual, now tested at real bulk-knowledge scale for the first time.

3. **The actual host residual is DATA PERSISTENCE, one layer down from the bind.** The composite `_encode()`
   produces is committed to `self.kb` — a bare host Python list holding a numpy array (the "numpy-kb fast
   path"). `RFPhasorComposer` already has an additive, default-off alternative for exactly this
   (`enable_substrate_store=True` → `_store_substrate`: the composite lives in a persistent `(1+D)`-neuron RF
   bridge's complex synaptic weights — "the Crawford-Eliasmith weight-store," Phase-2 GO at small N,
   `2026-06-05-phase2-substrate-store-derisk-GO.md`). But `sharded_phasor_store.py` — the class the LTM/
   knowledge-core actually uses — says outright, in its own `save()` docstring: *"Numpy fast path only
   (enable_substrate_store=False, the LTM default)."* The validated synaptic write is explicitly excluded from
   ever reaching the knowledge core. This file asks, empirically, what lifting that exclusion would cost.

4. **CURATION SELECTION is a third, separate, deeper residual — NOT attempted here.** Which ~78,857 of
   wikidata5m's 5M triples make the cut is a closed-form host frequency/degree ranking
   (`_knowledge_core_curate.curate()`), run once, outside any conversational/experiential context — no
   salience, curiosity, or reward signal the brain itself produces has any say in what gets encoded. Closing
   this needs an autonomous reading/attention loop over the corpus, a materially larger mechanism than a
   store-persistence swap. Named honestly as the deeper next rung, per the no-defer rule — not scoped away.

## Disambiguation from the existing `semantic-store-cortical-capacity` biology entry

`research/biology/semantic-store-cortical-capacity.md` already tracks a DIFFERENT, complementary axis of the
same store: bind-computation THROUGHPUT (closed-form-fast `encode_fast` vs the genuine per-op resonate,
established 6-seed GO on N=100k SYNTHETIC triples at D=512, "median faithful spiking ~35 f/s" for the genuine
resonate) and names its own next rung as "the faithful per-fact write cost is reported separately... GPU-batched
resonate." That is the BIND question. This finding is the PERSISTENCE question one layer downstream — given a
fact's composite already exists (by whichever bind path), where does it then live: a host list, or synaptic
weight — over REAL curated wikidata facts rather than synthetic triples. The two are independent and both
matter; neither substitutes for the other.

## (a) Parity: numpy-kb vs synaptic-substrate-store, REAL curated facts, 6/6 seeds — GO

240 real facts sampled directly from the bundle's `facts.json` (not synthetic) + 40 held-out moat probes
(facts whose agent never appears among the stored 240 — a genuine "never taught this" cue, not merely an
out-of-vocabulary word) were built into a `ShardedPhasorStore` twice per seed — once exactly as production does
(`enable_substrate_store=False`) and once with the candidate (`=True`) — at all 6 mission seeds:

| seed | stored recall (baseline) | stored answer-agree | moat abstain (baseline) | moat answer-agree |
|---|---|---|---|---|
| 42  | 240/240 | 240/240 | 40/40 | 40/40 |
| 43  | 240/240 | 240/240 | 40/40 | 40/40 |
| 44  | 240/240 | 240/240 | 40/40 | 40/40 |
| 100 | 240/240 | 240/240 | 40/40 | 40/40 |
| 101 | 240/240 | 240/240 | 40/40 | 40/40 |
| 102 | 240/240 | 240/240 | 40/40 | 40/40 |

Every one of the 6 mission seeds is clean: the baseline recalls every stored fact (no vacuous-abstain floor
issue), the candidate's answer matches the baseline on all 240 stored cues, and both abstain identically on all
40 held-out moat probes. Full per-seed build-time detail is in the cited artifact (`parity.per_seed`); note
those wall-clock numbers ranged 7.6s-33.5s for the IDENTICAL N=240 baseline build across seeds run minutes
apart on this box — see the contention note in (b) below before reading anything into build-time deltas here.

**Verdict: GO** (`decide(go=all_agree)` — the runner's own `tools.lab.Verdict`, not a metric lifted from a
different check). A `knob()` lever precondition confirms `enable_substrate_store` is not a no-op (a plain-array
handle under `=False` vs a non-array handle under `=True` for the identical stored fact); a `floor()`
precondition confirms the baseline itself genuinely recalls (>=90% of) the stored facts on every seed, so the
agreement above is not two stores trivially abstaining in lockstep. `curiosity/salience-driven curation` is
declared `disabled` in scope (point 4 above) — this verdict is about the WRITE mechanism, not the SELECTION.

## (b) Cost: measured, not estimated (the marginal memory/time the synaptic write actually costs)

A naive "build N facts in a fresh process, read peak RSS, divide by N" measurement is swamped by ~300MB of
FIXED process cost (numpy/`SimulationBridge` import + the codebook allocation, which scales with vocabulary,
not fact count) at any N cheap enough to run quickly — this file's own smoke test hit exactly that trap first
(N=5 → 310216 KB, N=20 → 310332 KB, projecting to an absurd ~1166 GB at 78,857 facts) and is kept in the runner
as a documented, earned lesson. The fix: pay the fixed cost once per subprocess, take a baseline peak-RSS
reading immediately after construction (before any fact is stored), then grow ONE store continuously to the
largest checkpoint, so every checkpoint's delta against that baseline is a clean marginal reading. `False` and
`True` still run in separate processes (a high-water mark never falls, so running both variants in one process
would let the first contaminate the second).

Measured at N=[500, 2,000, 8,000] real facts, seed 42 (raw checkpoints in the cited artifact's `cost` array):

**Memory — the reliable half of this measurement (RSS reflects genuine allocation, not CPU scheduling).**
- Substrate-store marginal slope: **50.26 KB/fact** (N=500→8,000; monotonic: +0.0MB→+88.8MB→+368.1MB above the
  post-construction baseline). Projected to the full 78,857-fact core: **3.78 GB** peak RSS above the fixed
  import+codebook floor.
- numpy-kb (current production path) marginal slope over the SAME N range: **0.0 KB/fact measured** — RSS sat
  flat at every checkpoint. This is NOT a real zero cost; it means the numpy path's per-fact footprint (a single
  `complex128[128]` array ≈ 2KB, i.e. ~25x smaller than the substrate path's measured slope) is below this
  instrument's resolution at N up to 8,000 — the allocator absorbed 8,000 x ~2KB ≈ 16MB of small array growth
  without requesting new pages the `ru_maxrss` high-water mark would register. A `substrate-store costs Nx the
  numpy-kb path` multiple is therefore NOT reported (dividing by an unresolved ~0 would manufacture a number);
  the honest comparison is "measured 50.26 KB/fact" vs "an unresolved but almost certainly much smaller number,
  bounded below by the ~2KB/fact raw array size."
- **3.78 GB for the full core is affordable** on this project's own consumer-hardware reference point (a single
  workstation, not an exotic multi-GPU box) — memory is NOT the blocker to shipping `enable_substrate_store` at
  bulk scale. The blockers are (c) below and query-time batching (next paragraph).

**Time — CONFOUNDED by concurrent system load on this box, reported for completeness, NOT verdict-bearing.**
Measured substrate slope 66.6 ms/fact and numpy-kb slope 121.1 ms/fact (projecting to 1.5h / ~2.65h respectively
at 78,857 facts) are internally inconsistent (the numpy-kb path measured SLOWER per fact than the substrate
path, which does strictly more work per fact — not physically sensible) and BOTH exceed the actual ground truth:
the bundle's own `curation_report.json` itself records **1913.5s for the identical 78,857-fact genuine-resonate
bind** (24.3ms/fact, no substrate persistence at all) from the ORIGINAL uncontended build. This run's own
N=240 parity-phase build times for the IDENTICAL baseline operation ranged **7.6s to 33.5s across seeds
executed minutes apart** (a >4x spread for identical work) — this box was carrying heavy concurrent load during
this run (340 active agent worktree directories observed). **Conclusion: the wall-clock time slope in this
measurement is contention-inflated 2.7x-5x and should not be read as a clean substrate-vs-numpy time
comparison; only the memory slope above is load-bearing here.** A clean re-measurement would use CPU time
(`resource.getrusage().ru_utime`) rather than wall-clock, or run on an idle box — named as a concrete, small
methodology fix for next time, not attempted in this pass.

**Query-time cost (named, not separately benchmarked here).** Code-reading confirms `RFPhasorComposer.
_can_batch_scan()` requires `not enable_substrate_store` — so `query_patient` under the candidate path falls
back from ONE batched resonate over the whole shard to a PER-FACT PYTHON LOOP (`_iter_facts` → `_retrieve_
substrate` → an actual `rf_kick`+`rf_resonate_steps` per candidate fact scanned), the real cost this parity
check exercised (every stored/moat query in (a) ran through this loop for the candidate arm) but did not time
against the baseline's batched scan. This is likely the more load-bearing of the two costs at real query
volume and is the natural next measurement, not attempted in this pass.

## (c) Structural probe: does `.save()`/`.load()` survive `enable_substrate_store=True`? — NO, and now precisely

`ShardedPhasorStore.save()` was written assuming `handle` is a numpy array (`comps.append(np.asarray(handle))`)
and this combination (bulk sharded store + substrate-store) had never been exercised end to end. Empirically:
`store.save(path)` under `enable_substrate_store=True` raises **`TypeError: cannot pickle 'mappingproxy'
object`** (inside `np.savez`'s object-array pickling path, triggered by trying to serialize a live
`SimulationBridge`/`CoreSimConfig` object graph rather than a plain array of phases) — a clean, loud failure AT
THE CALL, not silent corruption of the caller's control flow. But `save()` writes its three files in sequence
(`manifest.json`, `facts.json`, then `composites.npz` last) and the exception fires mid-write of the last file,
so it ALSO leaves a PARTIAL bundle on disk: `manifest.json` and `facts.json` complete and well-formed,
`composites.npz` a truncated/corrupt zip container (confirmed directly: `np.load(..., allow_pickle=True)` on
the leftover file raises `EOFError: Ran out of input`, a DIFFERENT and less legible error than the informative
`TypeError` the caller who actually checked would have seen). A caller that does not check for the raised
exception — or runs `save()` somewhere the exception is swallowed — is left with a bundle directory that LOOKS
complete (all three expected files present) but fails confusingly at LOAD time instead of informatively at
SAVE time. This sharpens why the exclusion exists and why it should stay excluded until fixed properly: the
class was never given a serializer for a bridge handle, and the write is not atomic across its three files.
Closing it needs either (i) a dedicated serializer that persists each bridge's `cp_rf_w_re`/`cp_rf_w_im` weight
arrays directly (bypassing the pickle of the whole bridge object) plus a write made atomic (temp-dir + rename,
or write `composites.npz` first so a failure never leaves a bundle that appears loadable), or (ii) the
2026-07-20 finding's own recommended deeper fix — a single PERSISTENT per-shard store-weight tensor disjoint
from the per-op `cp_rf_w_*`, which would also remove the per-fact bridge multiplication problem in (b) above.

## Literature check: is a spiking per-fact store even the right architecture at this scale?

`research/findings/2026-06-05-substrate-held-memory-literature-synthesis.md` (an existing deep-research pass
this arc's own workflow requires — re-read here rather than re-derived) already answers this: its recommended
mechanism for a bulk knowledge KB is **"one small dedicated population (~20 neurons) per fact, the fact's bound
vector in static weights... zero cross-fact crosstalk, capacity linear in #facts"** — citing
**Crawford, Gingerich & Eliasmith (2016, Cognitive Science)**, who held **117,659 role⊗filler bindings** at
D=512 in exactly this shape (spiking cleanup/associative-memory populations, ~2.5M neurons total, ≈21.2
neurons/fact) and recalled them by spiking unbind + cleanup. That is **MORE facts than this project's own
78,857-fact target**, already demonstrated in the field, in spiking neurons, with the SAME per-fact-population
architecture `enable_substrate_store`/`_store_substrate` already implements. So a per-fact synaptic store for
bulk facts is not an open architectural question — it has precedent at larger scale than ours.

What IS different: our `_store_substrate` spends `1+D=129` neurons per fact (a full generic `SimulationBridge` —
config object, heterogeneity setup, connectivity/position generation even at `connections_per_neuron=0`, RNG
state) to hold ONE bound vector, versus Crawford et al.'s ≈21.2 neurons/fact purpose-built NEF cleanup ensemble
— roughly 6x more neurons, and (per (b) above) a much heavier PER-FACT OBJECT, not merely more neurons. The
measured cost in (b) is therefore a property of THIS ENGINEERING REALIZATION of the architecture (a full
bridge object per fact) rather than evidence the architecture itself does not scale — the literature's own
number says it does, at larger N than we need. The next rung this ordering implies: a lean, purpose-built
per-fact cleanup population (or the shared per-shard weight-tensor design in (c)) sized like Crawford's, not a
full `SimulationBridge`, before concluding anything about feasibility at 78,857+ facts.

## (d) Provenance characterization: the "at minimum, honest" fallback's data dependency already exists, unused

`TieredFactStore.query_patient_source()` (shipped 2026-08-27 for `webapp/gnw_two_organ_bus.py`'s
`BRAIN_GNW_ORGANB_LTM_EXEMPT`) already returns which TIER answered a query: `"buffer"` (conversationally taught
this session) vs `"ltm"` (bulk-curated background knowledge). Confirmed directly: querying a fact just taught to
the buffer returns tier `"buffer"`; querying a bulk-curated LTM fact returns tier `"ltm"` — the distinction is
exact, for free, off state the store already tracks. This is currently UNCONSUMED by the shipped
provenance-honesty framing: `BrainConversationalAgent.known_fact_record` labels every recalled fact
`PROVENANCE_PERCEIVED` regardless of tier (board #129/#140's PERCEIVED/GENERATED axis discriminates
single-fact-recall vs multi-hop-inference — a different question from write-origin, and this file does NOT
extend that monitor's judged vocabulary; a 3-way discrimination would need its own accuracy validation exactly
like the existing 2-way axis earned one). The practical effect: closing the "honest, provenance-neutral" gap
the task named as an acceptable minimum is now a **wire a signal** problem (pass the tier `query_patient_source`
already computes through to a framing decision), not a **build a mechanism** problem — the data dependency is
already produced, in production, on every LTM-era recall.

## Honesty boundary / residual (no-defer)

Not a phenomenal claim; not a production flip. Four residuals, named plainly rather than parked:
1. **Persistence engineering** — a lean per-fact cleanup population (Crawford-sized, not a full
   `SimulationBridge`) + a real serializer for it. Scoped, buildable, not attempted here.
2. **Query-time cost** — `enable_substrate_store` loses the batched resonate scan (`_can_batch_scan()` requires
   it off), falling back to a per-fact loop; confirmed by code-reading, not separately timed against the
   baseline in this pass (the wall-clock confound in (b) makes this run's own timings unsuitable for that
   comparison). The natural next measurement.
3. **Curation selection** — the closed-form host frequency ranking that decides WHICH facts exist at all.
   Materially larger (an autonomous reading/attention mechanism); named, not scoped away.
4. **Provenance framing wire-in** — the tier signal exists; no self-report consumes it yet. The natural next
   de-risk, and now cheaper than it looked (a wire, not a monitor to validate from scratch).

This file changes no production default and edits no `sim/` code; `enable_substrate_store` is an EXISTING,
already-default-off constructor kwarg exercised only from this new standalone runner.

## Reproduce

```
SIM_BACKEND=numpy /home/dant123/Projects/sim/.venv/bin/python -m research.runners._rank6_knowledge_core_substrate_write_derisk \
    --seeds 42 43 44 100 101 102 --n-stored 240 --n-moat 40 --cost-points 500 2000 8000 \
    --out research/findings/raw/_rank6_knowledge_core_substrate_write_derisk/full_run.json
# --smoke --skip-cost : fast end-to-end sanity pass (10 facts, 2 seeds, no cost probe)
```
