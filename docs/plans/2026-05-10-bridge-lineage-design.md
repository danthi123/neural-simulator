# Bridge Lineage Manager — persistent continuous-learning state

**Date:** 2026-05-10 23:35 EDT
**Status:** DESIGN approved by user; first Phase 1 implementation item
**Trigger:** User (2026-05-10) — "we're basically starting from scratch
on each run. Is there a good way to continually work off the most
recently trained sim state and keep improving it rather than settling
with from-scratch training sessions?"

---

## The problem

Every training session currently starts from random init via seed.
This means:

1. Every multi-seed validation discards N trained bridges
2. Tonight's 50+ training runs all started fresh — accumulated learning lost
3. User-facing continuous learning is broken by default — sim is reborn each
   session
4. The unique value prop (lifelong learning, biology-grounded growth)
   doesn't actually show in day-to-day use

This is wrong for the project's vision: **a continuously-learning agent
needs continuously-evolving state, not perpetual re-incarnation.**

## The solution: version control for trained sims

Think of bridges/ as a git-like repository of brain states. Two
co-existing workflow modes:

### Science mode (existing, for experiments)
- `--from-scratch` flag (default behavior today)
- Multi-seed reproducibility from random init
- Each experiment is an independent clean replication
- Used for arch comparisons, optimization validation, scientific rigor

### Continuous mode (new, for user-facing)
- Default behavior: load lineage main, save back on exit
- The sim "lives" between sessions
- Knowledge accumulates over weeks/months of use
- Foundation for Phase 2 multimodal + auto-growth + day-to-day use

## File layout

```
bridges/
├── lineage/
│   ├── main/                              ← the "production" sim lineage
│   │   ├── current.simstate.h5            ← latest state (auto-loaded)
│   │   ├── metadata.json                  ← vocab tier, training events,
│   │   │                                    accuracy history, last update
│   │   ├── _growth_log.md                 ← human-readable "diary" of
│   │   │                                    how the sim has evolved
│   │   └── history/
│   │       ├── 2026-05-10T22-00-checkpoint.simstate.h5
│   │       ├── 2026-05-10T20-00-checkpoint.simstate.h5
│   │       └── ...                        ← periodic auto-snapshots
│   └── experiments/                       ← named forks for variants
│       ├── encoding_axis_test/
│       │   └── (same structure as main/)
│       └── stp_reversibility_test/
│           └── (same structure as main/)
└── (existing standalone bridges; not in lineage)
```

## Metadata schema (`metadata.json`)

```json
{
  "lineage_name": "main",
  "created_at": "2026-05-10T22:00:00",
  "last_updated_at": "2026-05-11T14:30:00",
  "current_tier": "8-word synonym",
  "vocab": ["north", "up", "east", "right", "south", "down", "west", "left"],
  "arch": {
    "n_lang_input": 4096,
    "n_motor_per_action": 1000,
    "n_motor_fs_per_action": 120
  },
  "cumulative_training_events": 42850,
  "accuracy_history": [
    {"at": "2026-05-10T22:00", "metric": "A2W any", "value": 0.50},
    {"at": "2026-05-10T22:30", "metric": "A2W any", "value": 0.75},
    {"at": "2026-05-10T23:00", "metric": "A2W any", "value": 1.00}
  ],
  "parent_lineage": null,             // null for main; set for forks
  "branched_at": null,                // when forked
  "growth_events": [
    {"at": "2026-05-10T22:00", "kind": "init", "from": "scratch"},
    {"at": "2026-05-11T06:00", "kind": "tier_promotion",
     "from_tier": "4-word", "to_tier": "8-word"}
  ],
  "tags": ["production", "user-facing"]
}
```

## CLI integration

```bash
# Default: load lineage main, save back on exit (continuous mode)
python -m research.runners.chat_repl --mode synonym

# Explicit: load specific lineage
python -m research.runners.chat_repl --lineage main

# Fork: create new branch from current lineage state
python -m research.runners.chat_repl --lineage main --fork experiment_v3

# Science mode: from-scratch, do NOT save to lineage
python -m research.runners.chat_repl --from-scratch

# Rollback: load a specific history checkpoint as new HEAD
python -m research.runners.bridge_lineage rollback --lineage main \
    --to 2026-05-10T22-00

# List
python -m research.runners.bridge_lineage list

# Diff (show what changed between two lineage states)
python -m research.runners.bridge_lineage diff main \
    --from 2026-05-10T22-00 --to current
```

## Webapp integration

The Bridges tab (shipped tonight) extends to a "Lineages" tab:
- Tree view: main → branches
- Per-lineage: timeline of growth events, accuracy curves over time,
  vocab evolution
- "Load into chat" button (deferred from earlier): connects to
  WebSocket chat REPL on the loaded lineage
- Rollback / fork / delete operations

## Implementation phases

### Phase 1: minimum-viable lineage (~2 days)

1. `BridgeLineage` class in `sim/lineage.py`:
   - `BridgeLineage(name="main").load()` → returns bridge state + metadata
   - `lineage.save(bridge)` → atomic save (write to .new, fsync, rename)
   - `lineage.snapshot()` → archives current as a history entry
   - `lineage.metadata` → dict with vocab, tier, training events, history

2. Two-line integration in `chat_repl.py`:
   ```python
   # Default: continuous mode
   lineage = BridgeLineage("main")
   if lineage.exists() and not args.from_scratch:
       bridge = lineage.load()  # skips training
   else:
       bridge = train_chat_bridge(...)

   # On exit
   if not args.from_scratch:
       lineage.save(bridge)
   ```

3. Unit tests (CPU-only): manifest schema, atomic save, history rotation

### Phase 2: branching + history UX (~3 days)

1. Forking: `lineage.fork(new_name)` clones current state to new lineage
2. Rollback: `lineage.rollback_to(timestamp)` loads history checkpoint
   as new HEAD
3. History rotation: keep last N snapshots (configurable), prune older
4. CLI tools: `bridge_lineage` runner with list/diff/rollback subcommands
5. Growth log generation (markdown rendering of metadata.growth_events)

### Phase 3: webapp Lineages tab (~2 days)

1. New `/api/lineages` endpoint (lists, view detail, manage)
2. New `/lineages` tab in launcher UI
3. Timeline view of growth events
4. Integration with bridges tab "Load into chat" workflow (now means
   "load lineage into REPL session")

## Concurrency / safety

- **Atomic save**: write to `.new` file, fsync, rename. Avoids partial-
  write corruption.
- **Lock file**: prevent two REPL sessions from saving simultaneously
- **History rotation**: don't delete until new snapshot is verified
- **Validation on load**: check metadata schema + checksum the .h5 file

## Failure modes

| Failure | Handling |
|---------|----------|
| Corrupted current.simstate.h5 | Auto-rollback to most recent history |
| Schema version mismatch | Migrate metadata; warn if .h5 is older format |
| Two sessions trying to save | Second blocks until first finishes; uses file lock |
| Disk full | Save fails cleanly; user notified; lineage preserved |
| Forking from non-existent lineage | Error clearly; suggest valid names |

## Risks and open questions

- **How often to snapshot?** Every session end (default), or time-based
  (every hour of activity)?
- **History retention**: keep 10? 100? configurable; default 30?
- **What about multi-user?** Out of scope for v1; single-user assumed.
- **GPU memory state**: bridge.save_checkpoint() doesn't preserve all
  ephemeral state (firing thresholds, STP dynamics, eligibility traces
  per CLAUDE.md). On load, these self-recover in a few ms of free
  running. Document as known gotcha.

## Phase 1 work breakdown (~1 week total)

**Days 1-2: MVP**
- `sim/lineage.py` with BridgeLineage class
- Save / load / snapshot / metadata
- Atomic write + lock file
- Unit tests
- Two-line integration in chat_repl

**Days 3-4: branching + history**
- Fork operation
- Rollback to history
- CLI: `bridge_lineage` runner with list/diff/rollback
- Growth log generation

**Days 5-7: webapp Lineages tab**
- `/api/lineages` endpoint
- Frontend tab (extends bridges tab UI)
- Timeline view
- Integration with chat workflow

After Phase 1: lineage is the default; from-scratch is opt-in for
science.

## Why this is FIRST priority in Phase 1

1. **Foundation for everything else**: all Phase 1 optimization
   experiments benefit from persistent baseline
2. **Auto-growth Phase A pairs naturally**: each tier promotion is a
   lineage growth event with full history
3. **User-facing differentiator immediately**: continuous learning
   becomes the default workflow, not an exception
4. **Cheap to build**: ~1 week with existing checkpoint save/load
   infrastructure
5. **Compounds with chat UI work**: the deferred webapp WebSocket chat
   becomes much more compelling with lineage backing it

## Provenance

- This design: `docs/plans/2026-05-10-bridge-lineage-design.md`
- Strategic addendum: `docs/plans/2026-05-10-MASTER-PLAN-strategic-addendum.md`
- Auto-growth design: `docs/plans/2026-05-10-auto-growth-design.md` —
  Phase A integrates with lineage
- Existing infra:
  - `bridges/` directory (created today)
  - `chat_repl --save-bridge / --load-bridge` (works today)
  - `_save_bridge_checkpoint` writes HDF5 + sidecar metadata
  - `bridge.save_checkpoint()` / `load_checkpoint()` API
