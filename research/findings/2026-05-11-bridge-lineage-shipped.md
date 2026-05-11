# Bridge Lineage Manager — Phases 1-3 shipped in one autonomous arc

**Date:** 2026-05-11 00:00 EDT
**Status:** Phases 1-3 of the lineage design (1-week scope) shipped in
~3 hours of autonomous execution. Phase 3 frontend Lineages tab is the
only piece deferred.
**Provenance:** Design doc at
[`docs/plans/2026-05-10-bridge-lineage-design.md`](../../docs/plans/2026-05-10-bridge-lineage-design.md);
user request: "we're basically starting from scratch on each run.
Is there a good way to continually work off the most recently trained
sim state and keep improving it?"

---

## What problem this fixes

Every training session before this change started from random init via
seed. Multi-seed validation discarded N trained bridges. Tonight's 50+
training runs all started fresh — accumulated learning lost. The
user-facing continuous-learning differentiator was broken by default.

After this change, the chat REPL "lives" between sessions. Knowledge
accumulates over weeks/months of use. Science mode is still available
opt-in via `--from-scratch`.

## What shipped

### Phase 1 (MVP) — commits `3030517`, `ee9040a`, `5f5b360`

- `sim/lineage.py` — `BridgeLineage` class + `LineageMetadata` dataclass.
  Atomic save (`.new` + `os.replace`), millisecond-precision history
  timestamps, schema-version field, growth-event + accuracy-history
  tracking.
- `research/runners/chat_repl.py` integration:
  - `--lineage NAME` (default `main`)
  - `--from-scratch` opt-out
  - `--fork-lineage NAME` to branch experiments
  - Mode/arch compatibility check on load (no shape-mismatch crashes)
- `chat_demo` / `chat_synonym_demo` / `chat_speak_synonym_demo`:
  opt-in `--lineage NAME` + `--save-to-lineage`. Defaults to fresh
  training (batch demos remain seed-deterministic).

### Phase 2 (branching + history UX) — commit `7b477fd`

- `research/runners/bridge_lineage.py` — CLI runner with subcommands:
  - `list` — All known lineages with tier/events/snapshot counts
  - `show NAME` — Detailed metadata for one lineage
  - `history NAME` — List of history snapshots
  - `rollback NAME --to SNAPSHOT_ID` — Restore a history snapshot
  - `fork PARENT CHILD` — Branch a new lineage
  - `prune NAME --keep-last N` — Trim old snapshots (default 30)
  - `diff NAME --from A --to B` — Compare two metadata states

- Bug fix during the arc: `sim/lineage.py` `_snapshot_current_to_history`
  was writing metadata files at `<snap_id>-checkpoint.simstate.metadata.json`
  but `rollback_to` and `prune_history` expected
  `<snap_id>-checkpoint.metadata.json`. Standardised on the cleaner
  naming. Caught by `test_diff_current_vs_history`.

### Phase 3 (webapp endpoints) — commit `7bb9bcf`

- `GET /api/lineages` — summary list (tier, vocab_size,
  cumulative_events, n_snapshots, parent, arch)
- `GET /api/lineages/{name}` — full metadata + snapshot inventory +
  growth events + accuracy history
- TestClient tests PASS; live webapp will activate at next restart
  (WatchFiles is unreliable on Windows per CLAUDE.md).

## Test coverage

| Suite | Tests | Status |
|-------|-------|--------|
| `tests/test_lineage.py` | 21 | All PASS |
| `tests/test_bridge_lineage_cli.py` | 13 | All PASS |
| `tests/test_chat_repl.py` | 28 (3 new) | All PASS |
| `tests/test_chat_demo_aggregate.py` | 14 (3 new) | All PASS |
| `tests/test_webapp_server.py` (lineage subset) | 2 new | PASS |

**Total: 78 tests across the lineage subsystem, all PASS, no GPU.**

## File layout (now operational)

```
bridges/lineage/
└── <lineage_name>/
    ├── current.simstate.h5         ← auto-loaded
    ├── metadata.json               ← tier, vocab, arch, events,
    │                                 accuracy_history, growth_events
    ├── _growth_log.md              ← (future) human-readable diary
    └── history/
        ├── 2026-05-11T00-00-12-456-checkpoint.simstate.h5
        ├── 2026-05-11T00-00-12-456-checkpoint.metadata.json
        └── ...                     ← last 30 by default
```

## Usage examples

```bash
# Default continuous mode (after this commit lands)
python -m research.runners.chat_repl --mode synonym
# → loads bridges/lineage/main/ if it exists, else trains
# → saves back to lineage on exit
# → snapshots previous state to history/

# Fork a branch for experimentation
python -m research.runners.chat_repl --mode synonym --fork-lineage experiment_v3

# Multi-seed science mode (no lineage interaction)
python -m research.runners.chat_repl --mode synonym --from-scratch --seed 42

# Inspect lineages
python -m research.runners.bridge_lineage list
python -m research.runners.bridge_lineage show main
python -m research.runners.bridge_lineage history main

# Roll back to a prior snapshot
python -m research.runners.bridge_lineage rollback main \
    --to 2026-05-11T00-00-12-456

# Webapp (after restart)
curl http://localhost:8765/api/lineages
curl http://localhost:8765/api/lineages/main
```

## Compatibility / gotchas

- Lineage stores mode + arch in metadata. Loading a `tier1` lineage
  with `--mode synonym` triggers a "fallback to training from
  scratch" warning — no crash.
- `save_checkpoint` doesn't preserve firing thresholds / STP / eligibility
  per CLAUDE.md. On load, dynamics self-recover in ~10ms of free running.
  This is fine for inference; documented as a known limitation.
- Single-user assumed; no file lock yet. Multi-user / concurrent-save
  protection is a Phase 4 item.

## Why this matters

1. **Continuous-learning differentiator unlocked.** The unique value
   prop (a sim that learns over weeks/months of use) is now the default
   workflow, not a manual `--save-bridge` / `--load-bridge` dance.
2. **Auto-growth Phase A pairs naturally.** Each tier promotion can be
   recorded as a `growth_event` on the lineage; rollback is free.
3. **Cheap to extend.** Phase 2 of the auto-growth design (tier
   promotion via checkpoint reload) sits on this foundation.
4. **Multi-seed science workflow preserved.** `--from-scratch` keeps
   reproducibility intact for experiments.

## What's deferred

- Webapp Lineages tab (frontend UI). Endpoints are wired; tab is the
  remaining ~1 day of work.
- File-lock for concurrent-save protection.
- Checksum validation on load.
- Auto-rollback on corrupt current state (manual rollback via CLI works).
- `_growth_log.md` markdown generator (metadata.growth_events is the
  source of truth; rendering is cosmetic).

## Provenance / next steps

- This findings doc:
  `research/findings/2026-05-11-bridge-lineage-shipped.md`
- Design:
  `docs/plans/2026-05-10-bridge-lineage-design.md`
- Master plan:
  `docs/plans/2026-05-10-MASTER-PLAN-strategic-addendum.md`
- Phase 1 optimization:
  `docs/plans/2026-05-10-phase1-local-optimization-design.md`
- Auto-growth (the next pillar):
  `docs/plans/2026-05-10-auto-growth-design.md`

Phase A of auto-growth (tier promotion via checkpoint reload) can now
move forward — lineage is the substrate it sits on.
