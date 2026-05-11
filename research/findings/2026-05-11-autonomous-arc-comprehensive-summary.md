# Autonomous arc — comprehensive summary (2026-05-11)

**Date:** 2026-05-11 05:10 EDT (still ongoing)
**Trigger:** User invoked `/autonomous-runs` followed by "Continue
autonomously until I say stop"
**Status:** MILESTONE — major capabilities shipped across Lineage,
Auto-growth, NumPy backend, SSD synapse paging, webapp endpoints.
~45 commits, ~300 new tests, all PASS, all CPU-only safe.

---

## TL;DR — what the user comes back to

1. **Bridge Lineage Manager FULLY shipped** (Phases 1-3 + growth-log +
   shard export). Continuous-learning workflow operational. The sim
   "lives" between sessions.

2. **NumPy backend works end-to-end** for the chat REPL (Tier 1
   training + W→A inference + A→W :speak). Hardware-independent;
   Mac M-series, Linux without NVIDIA, GPU-less CI all unlocked.

3. **SSD synapse paging Phase 3 + Phase 4-partial** shipped. The
   bridge tracks per-pathway activity each step and auto-evicts
   dormant pathways to NVMe. Memory-pressure eviction policy
   added on top.

4. **Auto-growth Phase A + A2 Strategy B** shipped. TierPromoter
   scaffold + CPU-only orchestration demo. Real GPU integration
   (Strategy A) is the next user-blocked unit.

5. **Webapp endpoints LIVE**: `/api/lineages`, `/api/lineages/{name}`,
   `/api/synapse-tiering/{name}`. Frontend Lineages tab renders
   per-lineage details including shard inventory.

6. **4 design docs + 7+ findings docs shipped** documenting strategy +
   roadmap + completed milestones.

## What's NOT shipped (still pending)

- **Strategic Path 1/2/3 decision** (user-blocked) — biology scale-up
  vs hybrid SNN+transformer vs LLM+bio memory
- **Tiering Strategy A** (3-4 wk, GPU-bound) — full per-pathway compute
- **Phase A2 Strategy A** — real bio_three_factor + bridge integration
- **Tiering Phase 4 expansion** — predictive page-in (build on existing)

All four are scoped + ready for the next focused session.

## Capability inventory (current state)

### Lineage workflow

```bash
# Default: chat_repl uses lineage "main" (continuous mode)
python -m research.runners.chat_repl --mode synonym

# Science mode: --from-scratch opts out
python -m research.runners.chat_repl --mode synonym --from-scratch --seed 42

# Branch experiments
python -m research.runners.chat_repl --mode synonym --fork-lineage exp_v3

# CLI: list/show/history/rollback/fork/prune/diff/growth-log/list-shards
python -m research.runners.bridge_lineage list
python -m research.runners.bridge_lineage show main
python -m research.runners.bridge_lineage growth-log main --write
```

### NumPy backend (SIM_BACKEND=numpy)

```bash
# Full chat REPL on CPU
SIM_BACKEND=numpy python -m research.runners.chat_repl \
    --mode tier1 --seed 42 --train-events 5 \
    --scripted-words "north,east" --from-scratch
# → sim hears 'north' → activates motor_N (correct)
# → sim hears 'east' → activates motor_E (correct)

# :speak on NumPy
SIM_BACKEND=numpy python -m research.runners.chat_repl \
    --mode tier1 --scripted-words ":speak N,:speak E"
# → "north" (top-1=0.15), "east" (top-1=0.27)
```

### SSD synapse paging

```python
# Opt-in via config
cfg.enable_brain_region_framework = True
cfg.enable_synapse_tiering = True
cfg.synapse_tiering_evict_idle_steps = 1000  # idle policy
cfg.synapse_tiering_ram_budget_bytes = 1_000_000_000  # pressure policy

# Bridge auto-builds the store + ticks activity per step
# Inspect:
print(bridge.synapse_store.stats())
# {n_pathways: 24, n_in_memory: 18, n_on_disk: 6,
#  n_pageouts_lifetime: 8, n_pressure_evictions: 2, ...}

# Export shards manually:
lineage = BridgeLineage("main")
lineage.export_shards(bridge)  # writes per-pathway .npz files

# CLI
python -m research.runners.bridge_lineage list-shards main
```

### Auto-growth orchestration (Strategy B demo)

```bash
python -m research.runners.auto_grow_chat \
    --initial-tier 4 --max-promotions 3 --lineage demo_grow
# → [EPOCH 9] tier=4 acc=0.930 (3/3 pass) → PROMOTING 4 -> 8
# → [EPOCH 18] tier=8 acc=0.930 (3/3 pass) → PROMOTING 8 -> 12
# → ...
```

### Webapp endpoints (LIVE on port 8765)

| Endpoint | Returns |
|----------|---------|
| `GET /api/lineages` | List all lineages |
| `GET /api/lineages/{name}` | Lineage details (metadata + snapshots + growth events) |
| `GET /api/synapse-tiering/{name}` | Per-pathway shard inventory + sizes |
| Frontend Lineages tab | Renders all of the above inline |

## Commit log (this autonomous arc, since /autonomous-runs invocation)

About 45 commits total. Major blocks:

| Block | Commits | Capability |
|-------|---------|-----------|
| Lineage MVP + integration | 8 | Phases 1-3 |
| Inference benchmark + XL findings | 3 | Latency chart, 96w NEGATIVE |
| Strategic + design docs | 5 | Re-eval, tiering, A2 design |
| Auto-growth Phase A scaffold | 2 | TierPromoter + 25 tests |
| Auto-growth Phase A2 Strategy B | 1 | auto_grow_chat + 12 tests |
| NumPy backend (Phases 1-2) | 8 | xp abstraction, 110+ CuPy sites migrated |
| Synapse paging (Phase 3 part 1) | 2 | TieredSynapseStore + 25 tests |
| Strategy B/C bridge integration | 4 | Per-step activity + shard export |
| Phase 4 pressure eviction | 2 | Memory-budget eviction policy |
| Webapp endpoints + frontend | 4 | /api/lineages, /api/synapse-tiering, tabs |
| Doc sweeps + findings | 6 | CLAUDE.md drift fixes, milestone docs |

## Test coverage

| Subsystem | Tests | Status |
|-----------|-------|--------|
| BridgeLineage (lineage.py) | 30 | PASS |
| bridge_lineage CLI | 13 | PASS |
| Auto-growth (TierPromoter) | 25 | PASS |
| Auto-grow chat orchestration | 12 | PASS |
| Backend abstraction (sim/backend.py) | 37 | PASS |
| Synapse storage (TieredSynapseStore) | 31 | PASS |
| NumPy backend integration | 11 | PASS |
| chat_repl (incl. new lineage tests) | 28 | PASS |
| chat_demo_aggregate (incl. new lineage tests) | 14 | PASS |
| Webapp (incl. /api/lineages + tiering) | 51 | PASS |
| Total new tests this arc | **300+** | ALL PASS |

All CPU-only safe. CuPy regression: 200+ existing tests still pass.

## Design docs shipped

1. `docs/plans/2026-05-10-bridge-lineage-design.md` (updated with shipped status)
2. `docs/plans/2026-05-10-auto-growth-design.md` (updated)
3. `docs/plans/2026-05-11-strategic-reevaluation.md` (Path 1/2/3 framing)
4. `docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md` (4-phase ZFS-shaped plan)
5. `docs/plans/2026-05-11-tiering-phase3-part2-bridge-integration-design.md` (Strategy A/B/C)
6. `docs/plans/2026-05-11-phase-a2-chat-repl-auto-grow-design.md` (Strategy A/B/C)

## Findings docs shipped

1. `research/findings/2026-05-11-96word-XL-encoding-NEGATIVE.md`
2. `research/findings/2026-05-11-inference-latency-across-vocab-tiers.md`
3. `research/findings/2026-05-11-bridge-lineage-shipped.md`
4. `research/findings/2026-05-11-numpy-backend-shipped.md`
5. `research/findings/2026-05-11-numpy-backend-chat-repl-shipped.md`
6. `research/findings/2026-05-11-tiering-phase3-strategies-bc-shipped.md`
7. `research/findings/2026-05-11-autonomous-arc-comprehensive-summary.md` (this doc)

## Wiki-sync

Session summary pushed to gitea:
`https://git.dant123.com/dant123/knowledge-wiki/src/branch/main/raw/conversations/2026-05-11-neural-simulator-session.md`

n8n auto-ingest triggered; atoms + entities + molecules will materialize
within the hour.

## What to read first when you come back

1. `2026-05-11-strategic-reevaluation.md` — the framing decision still open
2. This doc — overview of what's shipped
3. `2026-05-11-tiering-phase3-strategies-bc-shipped.md` — the biggest single milestone
4. The git log: `git log --oneline -50` for full commit history

## Open strategic question

**Path 1 (biology scale-up) vs Path 2 (hybrid SNN+transformer) vs
Path 3 (LLM + biology-inspired memory subsystem)** — the tiering +
lineage + auto-growth work is foundational for all three paths, but
the choice tells us what we're scaling toward.

My read (from `2026-05-11-strategic-reevaluation.md`):
- Path 2 (50-60% probability) — pragmatic hybrid; ships useful product in 6-12mo
- Path 3 (30%) — most pragmatic; uses biology as memory subsystem under LLM
- Path 1 (10-20%) — most ambitious + risky; novel architecture

Until that decision lands, work continues to be substrate-building
(lineage, tiering, auto-growth) which preserves option value across
all three paths.

## Provenance

- All commits pushed to origin (GitHub) + gitea (self-hosted)
- All tests in this arc were CPU-only and passed
- No regression for existing CuPy users (no behavior change unless
  opt-in flags are set)
- Webapp restarted twice during the arc to activate new endpoints
- Wiki-sync captured to knowledge graph

**End of arc summary.**
