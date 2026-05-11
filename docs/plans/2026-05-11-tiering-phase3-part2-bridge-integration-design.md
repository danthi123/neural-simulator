# Tiering Phase 3 part 2 — bridge integration design

**Date:** 2026-05-11 03:55 EDT
**Status:** DESIGN — companion to the foundational
`docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md`. Spells out
HOW to integrate TieredSynapseStore (shipped commit `33ca704`) into
SimulationBridge so the eviction policy actually fires per
simulation step.
**Prereqs:**
- `sim/synapse_storage.py` (✅ shipped 2026-05-11)
- `sim/backend.py` xp abstraction (✅ shipped 2026-05-11)
- NumPy backend chat_repl end-to-end (✅ shipped 2026-05-11)

---

## The integration gap

Today (post-Phase 3 part 1):

- `TieredSynapseStore` is standalone infrastructure with 25 tests.
- It tracks pathway activity, evicts dormant pathways to NVMe shards,
  pages them back in on access, and lineage-saves/loads consistently.
- **But the bridge doesn't use it.** The bridge has a single big CSR
  matrix `self.cp_connections` containing ALL synapses across ALL
  pathways. There's no per-pathway access pattern.

To make tiering actually save VRAM/RAM, the bridge needs:

1. **Per-pathway CSR matrices** (instead of one monolithic CSR)
2. **Per-pathway access** in the simulation step (compute synaptic
   conductance from EACH pathway separately)
3. **Per-pathway firing detection** (so we know which pathways "fired"
   this step → feed to store.step())
4. **Per-pathway weight updates** (STDP / Hebbian apply per pathway,
   not to the monolith)

This is a substantial refactor — the current `_run_one_simulation_step`
uses sparse matrix-vector multiply once on the monolithic CSR. We'd be
replacing it with N matvecs (one per pathway), each potentially loading
from SSD.

## Three integration strategies (ranked by risk)

### Strategy A — Full per-pathway refactor (highest payoff, highest risk)

**Scope:** ~3-4 weeks of focused work.

Replace `self.cp_connections` (monolithic) with `self.pathway_store: TieredSynapseStore`.
Each pathway in `RegionPathway` becomes a separate CSR in the store.

**Per-step changes:**
```python
# OLD (monolithic):
synaptic_current = self.cp_connections @ neuron_state

# NEW (per-pathway with paging):
synaptic_current = xp.zeros(n_neurons)
fired_pathways = set()
for pathway_name in self.pathway_store.pathway_names():
    csr = self.pathway_store.get_pathway(pathway_name)  # transparent page-in
    pathway_current = csr @ neuron_state[pre_indices_for_pathway]
    synaptic_current[post_indices_for_pathway] += pathway_current
    if pathway_current.sum() > FIRING_THRESHOLD:
        fired_pathways.add(pathway_name)
self.pathway_store.step(fired_pathways)
```

**Pros:**
- True memory tiering: dormant pathways live on SSD
- Activity-aware: hot pathways stay in VRAM/RAM
- Maximum VRAM savings (could cut active memory 5-10× at 64-word arch)
- Biologically grounded (per-pathway connectivity matches anatomy)

**Cons:**
- Massive refactor; touches all of `_run_one_simulation_step`
- Per-pathway matvec is slower than monolithic (more overhead per matvec)
- STDP / Hebbian / structural plasticity need per-pathway versions
- Checkpoint format needs to handle multi-file pathway shards
- Lineage save/load multi-file extension (per the tiering design doc)
- Risk of subtle correctness bugs that don't show up at toy scale

**Falsification:** if a 200-step smoke on the Tier 1 arch (n_lang=2048,
n_motor=500) is more than 5× slower than the monolithic implementation,
the per-pathway approach is too expensive.

### Strategy B — Shadow per-pathway tracking (low risk, lower payoff)

**Scope:** ~3-5 days of focused work.

Keep the monolithic `cp_connections`. ALSO maintain a per-pathway
`TieredSynapseStore` that mirrors the data. Use the store ONLY for:
- Tracking activity per pathway (cheap; activity update each step)
- Persisting per-pathway shards as a serialization format (alongside
  the .h5)
- Lineage save/load can write pathway shards next to the .h5

The monolithic CSR remains the inference path; the store is an
observational shadow. No actual paging happens yet — but the
infrastructure is in place for Strategy A later.

**Pros:**
- Low risk; preserves all current inference behavior
- Lineage gets per-pathway shards "for free" (extension of current format)
- Phase 4 activity tracker can be built independently
- Sets up future migration to Strategy A

**Cons:**
- No memory savings (monolithic CSR still in VRAM)
- Storage savings only at save-time (per-pathway shards instead of one big h5)
- Two sources of truth that must stay in sync

### Strategy C — Pathway-as-shard at save/load only (lowest risk, lowest payoff)

**Scope:** ~1-2 days of focused work.

`TieredSynapseStore` is used ONLY at lineage save/load boundaries.
The save dumps per-pathway shards; load reconstructs the monolithic
CSR from those shards. Activity tracking + eviction policy are NOT
used during simulation.

**Pros:**
- Zero changes to `_run_one_simulation_step`
- Trivial to implement (already most of the way there with
  `save_all_shards` and `load_shard_index`)
- Sets up future tiering work without committing

**Cons:**
- No active memory tiering at all
- No activity-driven optimization
- Phase 4 (auto-tiering) cannot build on this — needs Strategy B at least

## Recommended approach: incremental — C → B → A

Given the strategic uncertainty (Path 1/2/3 still open), the right
move is **incremental commitment**:

1. **Ship Strategy C first** (1-2 days):
   - Add `lineage.save_shards()` / `lineage.load_shards()` helpers
   - Use `TieredSynapseStore.save_all_shards` at lineage save
   - Use `load_shard_index` + page-in-all at lineage load
   - No simulation-time changes
   - Tests: round-trip preserves accuracy

2. **If usage justifies, ship Strategy B** (additional 3-5 days):
   - Add per-pathway activity tracking
   - Mirror to store but keep monolithic CSR for compute
   - Build Phase 4 (auto-tiering policy) on top
   - Test that activity tracking matches expected biology

3. **If memory becomes the bottleneck, ship Strategy A** (additional
   3-4 weeks):
   - Full per-pathway compute refactor
   - Validate at 64-word arch (current local ceiling)
   - Validate at 96-word+ (cloud-anchored, the prize this unlocks)

This incremental path means:
- Phase 3 part 2 is finishable in 1-2 days of focused work
- Each strategy is independently testable
- We don't commit to the high-risk refactor until evidence justifies it

## Decision criteria for going beyond Strategy C

- **Strategy B trigger:** "we want activity-aware optimization without
  full refactor" (e.g. Phase 4 auto-tiering is wanted)
- **Strategy A trigger:** "active memory is the bottleneck" (e.g. 96+
  word arch on local hardware, OR cloud-anchored 1B+ params)

If neither trigger fires, Strategy C is sufficient.

## File-level changes per strategy

### Strategy C (minimum viable)

| File | Change |
|------|--------|
| `sim/lineage.py` | New `save_shards()` + `load_shards()` helpers; multi-file save |
| `sim/bridge.py` | New `_extract_per_pathway_csrs()` helper that splits cp_connections |
| `tests/test_lineage.py` | Round-trip with shards |
| `docs/plans/2026-05-11-bridge-lineage-design.md` | Update to mention shard format |

### Strategy B (medium scope)

All of C, plus:

| File | Change |
|------|--------|
| `sim/bridge.py` | Add `self.synapse_store: TieredSynapseStore`; track activity per step |
| `sim/bridge.py::_run_one_simulation_step` | Per-step activity update |
| `tests/test_synapse_storage.py` | Bridge-integration smoke test |

### Strategy A (full refactor)

All of B, plus:

| File | Change |
|------|--------|
| `sim/bridge.py::_run_one_simulation_step` | Per-pathway matvec loop (replaces monolithic matmul) |
| `sim/bridge.py` | Per-pathway STDP, Hebbian, structural plasticity |
| `sim/bridge.py` | Per-pathway weight update + per-pathway eligibility |
| Performance benchmarks | Verify per-pathway ≤5× monolithic overhead |

## Risks specific to per-pathway compute

1. **Matvec overhead explosion.** Each pathway matvec has Python+kernel
   launch overhead. At 30 pathways × N matvecs per step, the overhead
   could dominate vs the monolithic single-matvec.

   *Mitigation:* benchmark early on real arch. If too slow, fall back to
   Strategy B and accept that active memory tiering requires bigger
   per-matvec work to amortize.

2. **STDP cross-pathway correctness.** STDP fires on (pre, post) edge
   pairs. If a pre-neuron is in pathway A and a post-neuron is in
   pathway B, the edge is only in B's CSR. Per-pathway STDP just
   needs to iterate pathways correctly — should be fine.

3. **Lineage save consistency.** During lineage save, we need a
   consistent snapshot. If pathways are evicted mid-save, the saved
   state may be inconsistent. *Mitigation:* `save_all_shards` already
   pages in dormant pathways first.

## Open questions

1. **Pathway naming convention** — do we use `RegionPathway.name` (as
   declared) or auto-generate `<from_region>_to_<to_region>`?
   Current Phase 1.4 BRANCH A arch has 24 pathways with explicit names;
   future arches may have more.

2. **Compaction trigger** — when pathways grow via structural plasticity,
   when do we re-compact CSR? Per-pathway has the advantage that compaction
   is local (one pathway at a time) rather than blocking on the
   monolithic.

3. **Cross-backend portability** — `TieredSynapseStore` is currently
   scipy.sparse-only. CuPy backend would need a parallel path
   (cupyx.scipy.sparse). For Strategy C this doesn't matter (save/load
   only). For Strategy A, a CuPy-side TieredSynapseStore is needed.

## Provenance + dependencies

- This doc: `docs/plans/2026-05-11-tiering-phase3-part2-bridge-integration-design.md`
- Foundational design: `docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md`
- Strategic context: `docs/plans/2026-05-11-strategic-reevaluation.md`
- Storage module: `sim/synapse_storage.py` (Phase 3 part 1, shipped 2026-05-11)
- Tests: `tests/test_synapse_storage.py` (25 tests)

**Next autonomous-arc-friendly unit:** Strategy C implementation
(1-2 days; CPU-only design + tests; demonstrates the multi-file
lineage save/load extension). Defer Strategy B+A until the Path 1/2/3
decision lands and we know whether active memory tiering is on the
critical path.
