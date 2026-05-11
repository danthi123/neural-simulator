# What's New — User Guide for the 2026-05-11 Autonomous Arc

**Date:** 2026-05-11 05:30 EDT
**Status:** USER GUIDE — every new user-facing capability shipped in
this autonomous arc, with copy-paste commands. ~60 commits, ~330 tests.
**Read this first when you come back.**

---

## Most useful single command to try

Open the dashboard (webapp on port 8765 should still be running from
the auto-restart earlier). The Lineages tab now has detail panels
that surface bindings, shards, and growth events. New endpoints:

- `GET /api/lineages` — list all lineages
- `GET /api/lineages/{name}` — full metadata + snapshots + growth events
- `GET /api/synapse-tiering/{name}` — per-pathway shard inventory
- `GET /api/bridge-memory/{name}` — memory bindings + consolidations

If no `main` lineage exists yet, the endpoints will 404 (expected).
Run any of the commands below to create one.

## NumPy backend (hardware independence)

The chat REPL now runs end-to-end on CPU. No NVIDIA / CUDA dependency.

```bash
# Force NumPy backend (Mac M-series, GPU-less Linux, CI)
SIM_BACKEND=numpy python -m research.runners.chat_repl --mode tier1 \
    --seed 42 --train-events 5 --scripted-words "north,east" --from-scratch
# → sim hears 'north' → activates motor_N (correct)
# → sim hears 'east'  → activates motor_E (correct)

# Or :speak (A→W generative)
SIM_BACKEND=numpy python -m research.runners.chat_repl --mode tier1 \
    --seed 42 --train-events 5 \
    --scripted-words ":speak N,:speak E" --from-scratch
# → "north" (top-1=0.15), "east" (top-1=0.27)

# Default backend (CuPy if available; falls back to NumPy automatically)
python -m research.runners.chat_repl --mode tier1
```

## Continuous-learning chat REPL (Bridge Lineage Manager)

The sim "lives" between sessions by default. First run creates the
'main' lineage; subsequent runs load it (skipping training).

```bash
# Default: continuous mode using lineage 'main'
python -m research.runners.chat_repl --mode synonym

# Science mode: from-scratch, no lineage interaction
python -m research.runners.chat_repl --mode synonym --from-scratch --seed 42

# Fork a branch for experiments
python -m research.runners.chat_repl --mode synonym --fork-lineage exp_v3

# Specify a different lineage
python -m research.runners.chat_repl --mode synonym --lineage user_alice
```

## Lineage management CLI

```bash
python -m research.runners.bridge_lineage list
python -m research.runners.bridge_lineage show main
python -m research.runners.bridge_lineage history main
python -m research.runners.bridge_lineage growth-log main --write
python -m research.runners.bridge_lineage list-shards main
python -m research.runners.bridge_lineage fork main experiment_v1
python -m research.runners.bridge_lineage rollback main --to <snapshot_id>
python -m research.runners.bridge_lineage prune main --keep-last 10
python -m research.runners.bridge_lineage diff main \
    --from <snapshot_id> --to current
```

## Auto-growth demo

The TierPromoter orchestrates tier promotions (4-word → 8-word →
12-word → ...) based on accuracy mastery. Strategy B demo uses
synthetic train/transfer:

```bash
python -m research.runners.auto_grow_chat \
    --initial-tier 4 --max-promotions 3 --lineage growth_demo

# Also available from chat_repl directly:
python -m research.runners.chat_repl --mode tier1 --from-scratch \
    --auto-grow --auto-grow-max-promotions 3
```

Output:
```
[AUTO-GROW] Starting at tier 4 (target: tier 16 in 3 promotions)
[EPOCH 7] tier=4 acc=0.920 (1/3 pass)
[EPOCH 9] tier=4 acc=0.930 (3/3 pass) -> PROMOTING 4 -> 8
...
```

## BridgeMemory (Path 3 LLM-callable memory)

The biology-grounded sim as a memory subsystem an LLM could call via
tool-use. Real ops shipped (Phase 3.1.6).

```bash
# End-to-end demo (NumPy or CuPy)
SIM_BACKEND=numpy python -m research.runners.bridge_memory_demo \
    --seed 42 --lineage memory_demo --out demo.json

# Programmatic usage
python -c "
from sim.bridge_memory import BridgeMemory
mem = BridgeMemory(lineage_name='alice', mode='synonym')
mem.store('alice', 'north', n_events=50)
print(mem.recall('alice'))
print(mem.speak('N'))
print(mem.stats())
"
```

API summary:
- `mem.store(key, value, n_events=50)` — bind via embodied-Hebbian
- `mem.recall(key, top_k=5)` — W→A inference; ranked motor pools
- `mem.speak(action, top_k=4)` — A→W generative; ranked words
- `mem.forget(key, decay_rate=0.5)` — weight decay (Phase 3.2 stub)
- `mem.consolidate(n_sleep_cycles=3)` — sleep replay (Phase 3.2 stub)
- `mem.stats()` — current state
- `mem.list_keys()` — known vocab

## Synapse tiering (SSD paging)

Activity-tracked per-pathway eviction policy. Opt-in.

```python
# In a CoreSimConfig:
cfg.enable_brain_region_framework = True   # required
cfg.enable_synapse_tiering = True
cfg.synapse_tiering_evict_idle_steps = 1000
cfg.synapse_tiering_ram_budget_bytes = 1_000_000_000  # optional

# Inspect at runtime
print(bridge.synapse_store.stats())
# {n_pathways: 24, n_in_memory: 18, n_on_disk: 6,
#  n_pageouts_lifetime: 8, n_pressure_evictions: 2, ...}

# Export per-pathway shards (works independently of runtime tiering)
from sim.lineage import BridgeLineage
lineage = BridgeLineage("main")
lineage.export_shards(bridge)
# Writes <lineage>/shards/<pathway_name>.npz files
```

## Webapp dashboard

The Lineages tab now lazy-loads ALL three subsystem subsections in
the detail panel (click a lineage):
- Growth events timeline
- Accuracy history
- History snapshots
- Synapse tiering shard inventory
- Bridge memory bindings

Endpoints live at:
- http://localhost:8765/api/lineages
- http://localhost:8765/api/lineages/{name}
- http://localhost:8765/api/synapse-tiering/{name}
- http://localhost:8765/api/bridge-memory/{name}

## Tests

All new tests are CPU-only:

```bash
# Lineage subsystem (96 tests)
python -m pytest tests/test_lineage.py tests/test_bridge_lineage_cli.py -v

# Backend abstraction (37 tests)
python -m pytest tests/test_backend.py -v

# Synapse storage / tiering (31 tests)
python -m pytest tests/test_synapse_storage.py -v

# Auto-growth (37 tests)
python -m pytest tests/test_auto_growth.py tests/test_auto_grow_chat.py -v

# Bridge memory (21 tests)
python -m pytest tests/test_bridge_memory.py -v

# NumPy backend integration (12 tests, slower)
python -m pytest tests/test_numpy_backend_integration.py -v

# Full subset
python -m pytest tests/test_backend.py tests/test_lineage.py \
    tests/test_bridge_lineage_cli.py tests/test_auto_growth.py \
    tests/test_auto_grow_chat.py tests/test_synapse_storage.py \
    tests/test_chat_repl.py tests/test_chat_demo_aggregate.py \
    tests/test_webapp_server.py tests/test_numpy_backend_integration.py \
    tests/test_bridge_memory.py
```

## Strategic Path 1/2/3 (open decision)

The strategic re-eval recommended 3 paths. Tonight's work is
foundational for all three. See:

- `docs/plans/2026-05-11-strategic-reevaluation.md` for the framing
- `docs/plans/2026-05-11-path3-bridge-memory-api-design.md` for Path 3
- `docs/plans/2026-05-11-phase-a2-chat-repl-auto-grow-design.md` for
  Path 1's auto-growth

**Path 3** (LLM + bio memory) is the lowest-risk + soonest-to-ship.
The BridgeMemory scaffold (commit `3ae13d8`) + demo runner
(commit `3788c08`) are the on-ramp. Phase 3.2 = pick a local LLM
(Phi-3-mini / Llama 3.2 1B / Qwen2.5) + wire to tool-use.

## What to read first

1. **This doc** (you are here) — quick action list
2. `research/findings/2026-05-11-autonomous-arc-comprehensive-summary.md`
   — full inventory + commit log
3. `docs/plans/2026-05-11-strategic-reevaluation.md`
   — open strategic question
4. Run `git log --oneline -60` to see all commits

## Webapp restart hint

If you've started a fresh session: the webapp on port 8765 from
earlier may already be running with all the latest endpoints. If not:

```bash
python -m uvicorn webapp.server:app --host 0.0.0.0 --port 8765 --reload
```

Then open http://localhost:8765 and click the Lineages tab.

## Acknowledgements

This guide captures what's in the codebase as of commit `3788c08`.
Everything is committed + pushed to origin (GitHub) + gitea
(self-hosted). All 330+ new tests pass. CuPy regression: zero.
NumPy backend regression: zero.

**Open the dashboard, run any of the commands above, browse the
findings docs.** Project is in a clean state.
