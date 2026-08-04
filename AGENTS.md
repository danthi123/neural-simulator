# Codex Project Entry Point

This is a long-running autonomous research project. Chat history is not the task ledger.

At the start of every session, after compaction, and before choosing new work:

1. Read `HANDOFF.md`, the current-state block in `GAP_CLOSURE_MISSION.md`, and
   `docs/AUTONOMOUS-EXECUTION.md`.
2. Run `python tools/autonomous_coordinator.py check` and
   `python tools/autonomous_coordinator.py next`.
3. Reconcile every warning before proceeding. A delegated lane in `ready`, `planned`, or
   `running` state must have a live assigned agent. Register dispatched agents immediately
   and finish them in the workboard when they return.
4. Keep independent, disjoint roadmap lanes moving concurrently across agents, local CPU,
   the RTX 3090, and the mini-PC pool. Scientific gates and GPU leases still control when a
   lane is eligible; idle hardware is not permission to invent or widen an experiment.
5. The controller owns decisive experiments and reads their receipts. Delegate bounded
   research, audits, tests, and disjoint implementation slices.
6. Continue through setup, implementation, verification, evidence, workboard update,
   commit, and verified pushes to both remotes. Do not end at a written next action that can
   be executed safely now.

Before a commit or handoff, run `bash tools/workflow_check.sh`. When a lane changes state,
priority, ownership, blocker, or next action, update `research/coordination/workboard.json`
through `tools/autonomous_coordinator.py` in the same cycle.
