---
type: finding
status: live
date: 2026-09-05
mechanism: knowledge-core-substrate-write
lane: scaffold-retirement
board: scaffold_retirement_backlog#6
runner: research/runners/_rank6_knowledge_core_substrate_write_derisk.py
seeds: [42, 43, 44]
seed-waiver: This finding's own claim is a DETERMINISTIC structural regression -- `save()` either raises or it
  does not, and a loaded store's recall answers either string-match the pre-save store's answers or they do
  not. There is no seed-dependent stochastic component to average over, matching this project's own
  established convention for exactly this class of claim (the cited rank-6 finding waives its OWN save/reload
  structural probe the identical way, reporting it "at seed 42 only... a resource/structural measurement...
  with no seed-dependent stochastic component"). Verified here at 3 seeds (42/43/44,
  `tests/test_sharded_phasor_store.py`, 6/6 green) plus the project's own runner's real-bundle smoke probe
  (seeds 42/43 against the actual 78,857-fact `wikidata_100k` bundle). The underlying 6-seed recall+moat
  PARITY claim this fix unblocks was already established 6/6 (42/43/44/100/101/102) by the cited rank-6
  finding and is UNCHANGED by this persistence-only fix -- this file does not re-run that claim.
artifacts:
  - research/findings/raw/_rank6_shardedphasorstore_pickle_fix/smoke_save_reload_probe.json
---

# `ShardedPhasorStore.save()`'s `TypeError: cannot pickle 'mappingproxy' object` is FIXED — the substrate-store write path now survives a save/load roundtrip byte-for-byte on recall

## Verdict

**GO.** The rank-6 finding (`research/findings/2026-09-05-rank6-knowledge-core-substrate-write-scaled-derisk-mixed.md`,
section (c)) named this exception as "the concrete, previously-undocumented reason the LTM class excludes
[the already-6-seed-GO'd `enable_substrate_store=True` write] path today." The bug is reproduced exactly as
described, root-caused to a specific class of object (`SimulationBridge`) carrying specific attributes
(`MappingProxyType` instances set directly on `sim/bridge.py` instances), fixed at the persistence layer with
no `sim/` edit, and PROVEN via a new regression test plus a real-scale run of the project's own existing
derisk runner against the actual 78,857-fact bundle. `enable_substrate_store` remains default-OFF; this file
closes only the save/load structural gap, not the production-flip decision (see Scope below).

## Reproduce (before the fix)

```
SIM_BACKEND=numpy /home/dant123/Projects/sim/.venv/bin/python -c "
from research.runners.sharded_phasor_store import ShardedPhasorStore
store = ShardedPhasorStore(n_shards=4, seed=42, D=128, enable_substrate_store=True)
store.store('dog', 'go', 'north')
store.save('/tmp/bundle')   # TypeError: cannot pickle 'mappingproxy' object
"
```
Confirmed with a minimal standalone script before touching any code (drift-#12 discipline — verified against
the actual `save()` source, not just the plan doc's description): the exception fires exactly as the rank-6
finding recorded, and `save()` leaves a PARTIAL bundle on disk (`manifest.json` + `facts.json` complete,
`composites.npz` truncated) — confirming that finding's own note that the write is not atomic across its
three files.

## Root cause

`ShardedPhasorStore.save()` (`research/runners/sharded_phasor_store.py`) iterates every fact's stored
`handle` and calls `comps.append(np.asarray(handle))`. Under the DEFAULT (`enable_substrate_store=False`)
path `handle` already IS the composite phase array, so this is a no-op cast. Under the CANDIDATE
(`enable_substrate_store=True`) path, `RFPhasorComposer._store_substrate()`
(`research/runners/rf_phasor_composer.py:1348-1361`) returns a live `SimulationBridge` object — the composite
lives in that bridge's `(1+D)`-neuron RF synaptic weights, not a numpy array (the Crawford-Eliasmith
weight-store).

`np.asarray()` on an arbitrary Python object does NOT raise — it silently wraps the object in a 0-d `dtype=
object` array. `np.stack()` then produces a 1-d object array of live bridge instances, and the failure
surfaces two calls later, when `np.savez()` tries to serialize that object array: numpy's `.npz` format
pickles object-dtype arrays (`numpy/lib/_format_impl.py: pickle.dump(array, fp, protocol=4, ...)`). Pickling a
`SimulationBridge` walks its full object graph and hits several INSTANCE attributes set to
`types.MappingProxyType(...)` directly on the bridge (`sim/bridge.py`, confirmed by direct grep):
`self.snr_packet_bindings` (lines 435, 1844, 2945, 4247, 12596), `self.snr_packet_kernel_parameters` (lines
676, 2948, 3091, 3315, 4248), `self.snr_packet_hh_phi` (lines 677, 2949, 3092, 3322, 4249) — an unrelated SNR
(source-normalized-rate) packet subsystem that every `SimulationBridge` initializes to an empty
`MappingProxyType({})` regardless of the RF-phasor use case. A `mappingproxy` has no `__reduce__`/pickle
support (by design — it is a deliberately read-only view), so `pickle.dump` raises exactly
`TypeError: cannot pickle 'mappingproxy' object`, matching the rank-6 finding's own text verbatim.

The deeper point, which shaped the fix below: serializing the WHOLE bridge object graph was never actually
the intent. The bridge is a vehicle — the only information a fact's substrate handle carries that matters to
`ShardedPhasorStore` is the D-dimensional composite phase vector, identical in kind to what the numpy-kb path
already stores directly. Everything else on the bridge (the SNR subsystem, connectivity/position arrays, RNG
state, config objects) is deterministic scaffolding that a fresh `_build_rf_bridge(1+D, seed)` call already
reconstructs byte-for-byte from the seed alone.

## Fix

`research/runners/sharded_phasor_store.py`, no `sim/` edit:

- **`save()`**: before handing anything to `np.asarray`, read the composite phase vector BACK OUT of a
  substrate handle via `sh._retrieve_substrate(handle)` — the identical call every substrate-store QUERY
  already makes (`_iter_facts`, `_find_cued_fact`, etc.). A numpy-kb handle is used as-is (unchanged
  behavior). `composites.npz` therefore only ever receives plain real-valued phase arrays — the SAME kind of
  array the numpy-kb path has always written, whether or not a substrate handle produced them.
- **`load()`**: the mirror image. After reading a fact's composite array back from the `.npz`, rebuild the
  substrate handle via `sh._store_substrate(comp)` — the identical call `RFPhasorComposer.store()` makes on
  first write — gated on `sh.enable_substrate_store`. That flag needed no new manifest field: it already comes
  from `manifest["composer_kwargs"]`, which `ShardedPhasorStore.__init__` already threads into every shard's
  `RFPhasorComposer(..., **composer_kwargs)` call, so a bundle saved with `enable_substrate_store=True`
  reloads shards that already report `enable_substrate_store=True` before `load()`'s per-fact loop even
  starts.

This is deliberately the MINIMAL fix at the point of failure, not a `SimulationBridge.__getstate__`/
`__setstate__` patch. Adding pickle support to `SimulationBridge` itself would (a) touch a class used
throughout `sim/` for many unrelated purposes, (b) risk masking OTHER unpicklable state on the same object
graph (thread-adjacent config, RNG generators) that this bug happened not to reach first, and (c) still
serialize ~129 neurons' worth of scaffolding per fact for no benefit — the actual content is the D-dim vector,
which is already exactly what gets serialized on the numpy-kb path. Confining the fix to
`sharded_phasor_store.py`'s own save/load boundary is both safer and matches this file's own established
persistence design (the module docstring already documents rebuilding shards from seed+vocab on load, `not`
literally freezing live state).

## Proof

**1. Regression test** (`tests/test_sharded_phasor_store.py`, new file, numpy backend only). Two
parametrized (`seed in [42, 43, 44]`) test functions build a 4-shard store over a small fixed fact set
(`[("dog","go","north"), ("cat","run","south"), ("river","look","apple")]`), save it, load it back, and assert
every read the conversational API exposes — `query_patient` (agent-cued recall), `query_agent` (the reverse
fan-out lookup), `render_fact`, `ask_yes_no` — plus the no-confab moat on a genuinely-unstored agent, returns
IDENTICALLY before vs. after the roundtrip:

  - `test_save_load_roundtrip_numpy_kb_baseline`: the pre-existing default path, confirming the fix changes
    NOTHING about the already-working case (3/3 seeds pass).
  - `test_save_load_roundtrip_substrate_store_survives_and_matches`: THE GATE. `enable_substrate_store=True`
    save/load, asserting (a) `save()` does not raise, (b) all three bundle files are present, (c) the
    reloaded shards still report `enable_substrate_store=True` (confirming `load()` genuinely rebuilt
    substrate handles rather than silently downgrading to plain arrays), and (d) recall/moat parity (3/3 seeds
    pass).

**Confirmed the test fails in the correct direction.** Before applying the fix (`git stash` on
`sharded_phasor_store.py` only), the substrate-store test failed 3/3 with the EXACT original exception:
```
FAILED tests/test_sharded_phasor_store.py::test_save_load_roundtrip_substrate_store_survives_and_matches[42]
FAILED tests/test_sharded_phasor_store.py::test_save_load_roundtrip_substrate_store_survives_and_matches[43]
FAILED tests/test_sharded_phasor_store.py::test_save_load_roundtrip_substrate_store_survives_and_matches[44]
TypeError: cannot pickle 'mappingproxy' object
```
while the numpy-kb baseline test still passed 3/3 (correctly scoped: the fix's job is to add substrate-store
support, not to touch the unaffected default path). With the fix restored: **6/6 pass.**

**2. No regression on the existing suite.** `tests/test_rf_phasor_composer.py` (48 tests, including the 3-seed
`test_rf_phasor_composer_substrate_store_parity` this fix's design mirrors) and `tests/test_tiered_fact_store.py`
(the LTM tier that actually USES `ShardedPhasorStore` in production): **57 passed, 4 skipped**, unaffected.

**3. Real-scale, end-to-end confirmation via the project's own existing derisk runner** (not just a hand-rolled
script). `research/runners/_rank6_knowledge_core_substrate_write_derisk.py --smoke --skip-cost` already
contains a structural probe (part (c) of the cited rank-6 finding) that builds a `ShardedPhasorStore` from
REAL facts sampled from the actual `wikidata_100k` bundle (78,857 curated facts) and round-trips it through
`save()`/`load()`. Run against this fix (`SIM_BACKEND=numpy`, no GPU):
```
=== (c) STRUCTURAL PROBE: does save()/load() survive enable_substrate_store=True? ===
  save() under enable_substrate_store=True: no exception raised
  load() succeeded; reloaded answers MATCH the pre-save answers
```
Artifact: `research/findings/raw/_rank6_shardedphasorstore_pickle_fix/smoke_save_reload_probe.json`
(`save_reload_probe: {"attempted": true, "save_ok": true, "save_error": null, "load_ok": true, "load_error":
null, "reloaded_answers_match": true}`; overall runner verdict `GO`). This is the SAME code path, SAME bundle,
and SAME probe the rank-6 finding used to first surface the bug — now green.

## Scope / honesty boundary

This fix closes ONLY the persistence-format gap the rank-6 finding named as the concrete blocker on
`ShardedPhasorStore`'s side. It does NOT flip `enable_substrate_store`'s default (still `False` everywhere in
production), and it does NOT touch the other two residuals that same finding named:
1. **Query-time cost** — `enable_substrate_store` still loses the batched resonate scan (`_can_batch_scan()`
   requires it off), falling back to a per-fact loop. Unaffected by this fix.
2. **Persistence engineering (footprint)** — each fact under the substrate store still costs a full
   `~129`-neuron `SimulationBridge` (the measured 50.26 KB/fact -> 3.78 GB projected at 78,857 facts) rather
   than a lean, Crawford-Eliasmith-sized (~21 neurons/fact) purpose-built population. This fix makes that
   footprint SURVIVE a save/load cycle; it does not shrink it.

**What this DOES unblock**: the rank-6 finding's own framing — "fixing this pickle bug is what unlocks scaling
the already-GO'd write path past its current default-off ceiling" — is now literally true at the mechanism
level: the write path can be saved and reloaded without data loss or crash, at the real curated-fact scale,
with a real regression test guarding it. Whether to actually FLIP the production default is a separate
decision gated on the two residuals above, not attempted here.
