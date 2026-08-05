# Autonomous Execution

This project uses a persistent workboard because chat history is not a reliable task manager. The current board is
[research/coordination/workboard.json](../research/coordination/workboard.json). It records the active lanes, their
priority, resource needs, file ownership, next action, blockers, recovery actions, and delegated agents.

The generated [research/coordination/HANDOFF.md](../research/coordination/HANDOFF.md) is a compact resume view. The
JSON workboard is authoritative; the generated Markdown is for quick reading.

## Resume A Session

From the repository root, run:

    python tools/autonomous_coordinator.py status
    python tools/autonomous_coordinator.py next
    python tools/autonomous_coordinator.py check

`tools/workflow_check.sh` runs the same coordinator check automatically before its legacy parallelism, research,
and pool checks. A commit or heartbeat that carries a ready lane without an assigned agent, or a stale running lane,
therefore fails loudly instead of relying on chat memory.

Read the top of HANDOFF.md, then the active mission header in GAP_CLOSURE_MISSION.md. The first concrete action
comes from the highest-priority unblocked lane. Do not replace this with a new list in chat.

## Lane Rules

Each lane must have:

- a status: ready, running, blocked, or completed;
- a priority and resource class;
- a disjoint write set when another worker may touch the repository;
- one concrete next action;
- a recovery action when blocked;
- a heartbeat while running.

An active delegated lane also records the agent ID that owns it. One active agent does not satisfy a different ready
lane; the coordinator checks the assignment per lane.

ready means work can start now. blocked means the blocker is recorded and the recovery action is the next
thing to check. A blocked GPU lane does not block CPU, research, documentation, or agent lanes.

## Parallel Work

Independent lanes are dispatched concurrently. The controller owns long experiments and multi-seed sweeps; an agent
may build a runner, audit sources, or prepare a bounded change, but it must not leave a detached sweep running and
must return its write set and exact command. Two workers may not edit the same file set.

Register delegated work immediately:

    python tools/autonomous_coordinator.py register-agent \
      --id <agent-id> --task "<bounded task>" --write-set tools/example.py tests/test_example.py

Record completion with the result and the next action:

    python tools/autonomous_coordinator.py finish-agent \
      --id <agent-id> --status completed --result "<what was verified>"

The status check warns when ready agent lanes exist with no active agent. That is an explicit dispatch reminder, not
a judgment that every lane must be filled regardless of conflicts.

## Resource Classes

- local_cpu: RAG refreshes, validation, aggregation, small screens, and documentation work.
- local_gpu: preregistered GPU experiments and confirmations, protected by the shared lease and queue readiness.
- local_cpu_plus_pool: independent CPU candidates and seeds, split across processes and mini-PC capacity.
- agent_cpu_network: source discovery, code audits, packet drafting, and bounded implementation in disjoint files.
- local_gpu_gated: the optional local model. It stays stopped while GPU experiments need the RTX 3090.

The coordinator reports live CPU load, GPU telemetry, queue depth, and matching processes. A heartbeat is evidence of
observation, not evidence that a scientific run succeeded. Scientific results still require their own receipts,
controls, lesions, held-out data, seeds, and gate verdicts.

Queue claims are not liveness. `research/queue/*.running` is a claim ledger used for recovery; it is not proof that a
process is still executing. The coordinator therefore reports claim counts separately from matching local research
processes and warns about stale or unclaimed GPU state. GPU runners must use the project interpreter
`/home/dant123/Projects/sim/.venv/bin/python` (or the equivalent project `.venv/bin/python` from the canonical
checkout), not whichever system `python` happens to be first on `PATH`; an import failure before the first step is an
environment failure, not a scientific negative.

An empty CPU queue is also not automatically a reason to invent a rerun. When all current CPU-compatible lanes are
banked and no new question is authorized, `.lane_waiver` may carry `scope=no-ready-work`. The workflow check verifies
that no CPU lane is ready, rejects preference-based wording, and expires the waiver after six hours. The next
research-packet or experiment-controller handoff must replace it with a concrete question before CPU work resumes.

## Heartbeats And Handoffs

Refresh a lane while it is active:

    python tools/autonomous_coordinator.py heartbeat --lane <lane-id>

Write a resumable snapshot before a session switch or after a meaningful transition:

    python tools/autonomous_coordinator.py snapshot
    python tools/autonomous_coordinator.py handoff

Generated status.json, heartbeats.jsonl, and HANDOFF.md stay local. The tracked workboard should be updated and
committed when a lane changes priority, status, ownership, blocker, or next action.

## Current RAG Boundary

LlamaIndex is the maintained primary retrieval path because the labeled evaluation remains hit@3=1.0 and
MRR=0.9231. SOMA is installed and automatically refreshed as a secondary local path. The current full SOMA rebuild
is fresh and loadable, but its present labeled result is lower (hit@3=0.5, MRR=0.4938), so it may organize or
triage material but does not replace the primary path or scientific source reading.

## Research And Experiment Boundaries

`tools/research_packet.py` describes an external search result with precise questions, prior-work matches,
sources, locators, structured values, and review state. `research_escalation.py handoff-packet` attaches that
record to the relevant wall without silently accepting it. Even an explicitly reviewed packet cannot resolve the
wall until its cited source has a durable, retrievable catalog/RAG intake record.

`tools/experiment_controller.py` plans and materializes an exact candidate-by-arm set without selecting seeds or
touching held-out data. `tools/experiment_executor.py` validates the already-expanded sealed jobs, creates durable
pending receipts, atomically claims work across local CPU, leased local GPU, and mini-PC lanes, and accepts success
only after checking the exact output and provenance. It does not choose candidates or interpret results.

`tools/experiment_observation.py` consumes those successful receipts under a separate preregistered contract. It
requires complete arm and seed coverage, rechecks artifact and provenance hashes, applies only explicitly mapped
scalar reductions, and emits verdict-free rows for the adaptive design. It cannot dispatch a successor or update
the design in place. That design-version-and-proposal transition remains explicit, so the generic engine is not yet
an unattended perpetual research loop.

Stage B has a specialized bounded execution path beyond the generic controller. It can generate the exact deterministic
512-point Sobol manifest, compile candidates with separate authority policies, write compact authenticated traces,
batch candidates on the GPU for engineering screening, persist campaign receipts, and strictly triage the five
resolved subgates. The fresh complete screen ran all 2,560 candidate-arm traces: 421 candidates failed, 91 were
inconclusive, and none passed. GPU results accelerate screening but are not scientific authority; an eligible
candidate would still require NumPy/CPU confirmation.

`tools/v14_stageB_campaign_supervisor.py` now provides one durable, fail-closed
campaign step. It authenticates the campaign and existing trace archives,
reconciles valid receipt additions, records digest-bound state, selects the
next declaration in the fixed NaP-first order, executes at most one GPU batch,
and runs strict triage only after all declarations are complete. `--status`
and `--dry-run` are read-only. Execution fails unless it is already running
inside the sanctioned project virtual environment with a usable CuPy device.

This is not a fully autonomous research loop. The supervisor is invoked once
per batch and has no persistent daemon, distributed lock, automatic CPU
confirmation, successor generation, or scientific decision policy. Until
those pieces are integrated, the controller owns confirmation and iteration,
and unspecified subgates remain unavailable.

## Failure Policy

An idle or stale lane is not silently accepted. Run check, dispatch a disjoint ready lane, or record a real blocker
and its recovery action. A positive experiment is not accepted because a process stayed alive; read its terminal
verdict and provenance. A completed task is not enough by itself: update the next action so the following session has
somewhere concrete to start.
