# The self-maintaining project OS — enforcement layer (2026-08-21)

**Why this doc exists.** Across a design conversation the owner named a coherent system, not a pile of patches. This
captures it durably so (a) the vision can't rot in chat (the exact failure it fixes) and (b) the agents that build it
work against ONE spec, not a memory of many messages. Owner directives, verbatim intent:
- Rapid, high-parallelism development — fan out to ALL independent ready work, not a few at a time; sequential is the
  exception that must be justified.
- Usage-conscious but NOT usage-hoarding — don't hold back to make genuine progress, but make the MOST progress per
  token by pushing work onto everything at our disposal (free machinery), reserving agent tokens for genuine judgment.
- A single pane of glass (Vikunja) — human-readable (Gantt, in the context of goals/walls/blockers) AND machine-usable
  (priority, dependencies, parallelizability, lane) — kept current mechanically, on time, low-effort; not when-asked.
- Everything must be ENFORCED (a check that blocks/fires), not remembered. Heartbeats exist but "never enough" because
  they inform rather than enforce, and can't self-check.
- Tools rot (experiment engine, the pool) and only get revived on manual prompting — readiness must be a checked
  property.

## The two root failure modes (both mechanical, both fixable)
1. **Compaction drops load-bearing info.**  2. **A full context dilutes attention** so in-context info doesn't drive
behaviour. Plus the operational failures they cause: sequential-not-parallel, work-not-surfaced, tools-rot,
over/under-token-use. Every one is "remembered, not enforced."

## Components (each a selftest-backed guard in `tools/gates/`-style registry; selftest MUST fail in its failing direction)

1. **Backlog generator** (`tools/backlog.py`) — the mechanical "many angles." One scanner fn per source, extensible:
   ledger de_risked&!on_by_default · roadmap walls-ledger open walls · FAILURE_LOG unclosed · findings' open
   residuals/next-levers · Vikunja open tasks+rungs · scaffold-burndowns · **tool-health (smoke results)**. Emits a
   ranked, de-duped list of INDEPENDENT ready items, each: {id, what, source+anchor, files/branch, verify-cmd,
   **dependencies**, **cheapest fitting lane**, rough dates}. ANTI-FABRICATION: every item traces to a real source
   line; empty is fine; never invent filler.

2. **Lane inventory + cheapest-lane router.** Free lanes (~0 tokens, fan out UNCONSTRAINED): mini-PC pool (36 cores),
   gpu_queue (3090, singleton-locked), headless runners, the experiment engine / `neural-simulator.py --auto-tune`,
   scheduled/cron tasks, cloud (AWS) when a run justifies it. Costly lane (agents/workflows — tokens count): genuine
   builds/wiring/design/adversarial-verify only, METERED against remaining usage. Each backlog item routes to the
   cheapest lane that can actually do it; agent only when it needs a mind.

3. **Fan-out ratchet** (blocking). Given the backlog + live capacity + remaining usage: auto-produce the launch list;
   treat a turn that leaves independent ready work undispatched with free capacity as a failure to justify. Fan-out is
   the default. **Continuous refill**: on each completion, immediately dispatch the next items to the freed lane
   (pipeline stays full — "a workflow per task, back-to-back"). Autonomy knob (config, owner-set default): fully
   autonomous auto-launch vs generate-and-confirm.

4. **Tool-health smoke tests.** Every free lane/tool has a cheap "runs against CURRENT state" check (experiment engine
   → tiny current-preset sweep; pool → trivial dispatch; gpu_queue → status+singleton; `--auto-tune` → `--quick`; pool
   checkout → git-current+import; cloud → reachability, no spend). Enforced periodic run (heartbeat every N cycles /
   scheduled, on free lanes) → `⛔ TOOL ROTTED: <name>` → auto-becomes a repair backlog item. Rot is caught by a check,
   not by the owner asking. Use itself exercises hot tools; smoke covers the idle ones (where rot hides).

5. **LIVE-STATE re-injection** — anti-compaction-loss + anti-dilution. A capped (~1-2 KB) file (frontier · last
   decision+why · ordered next-actions · live runs · hard constraints), mechanically re-injected by a hook at
   turn-start AND immediately post-compaction (re-read from the file, not the summary). Selftest: fires post-compaction;
   rejects an over-cap file (dilution guard). Updated on every landing via a PostToolUse nudge.

6. **Noise cut** — suppress the completed-task dump (biggest per-turn dilution); compress the heartbeat to a one-line
   delta carrying LIVE-STATE's next-action; keep agent transcripts / ledger dumps out of context (files/subagents).

7. **Vikunja = the single pane of glass.** Human: plain titles/leads + Gantt (start/target dates + **dependencies**) +
   area/ladder labels + priority. Machine: the same deps (→ parallelizability), lane, `(Ref:)` detail. Dependencies do
   double duty (timeline + independent-vs-blocked). ENFORCED sync: on every landing a gate updates the affected task's
   ladder/status/done + adds next-rungs (strengthen the existing freshness gate so a status-changing commit BLOCKS
   until the board reflects it); every heartbeat the generator reconciles drift. Backfill rough dates + convert the
   walls-ledger to first-class blocker tasks so the Gantt renders and blockers show in-context.

8. **gpu_queue singleton** (DONE, commit 21b83194 on research/gpu-queue-singleton — one shared queue/dpid/flock daemon,
   selftest passes + fails-correctly) + a **max-runtime guard** (a job over ~2x budget is killed + requeued-at-back so
   one slow job can't monopolize the single GPU).

## How they cohere (the loop)
Heartbeat cycle = **regenerate backlog (incl. tool-health) → reconcile the pane of glass → ratchet dispatches all
independent ready work to the cheapest fitting lane (free unconstrained, agent budgeted) → refill on completion →
re-inject LIVE-STATE → one-line delta.** Every guard selftest-backed + registry-discovered ("easy to update as needs
arise"). Result: work is surfaced, prioritized, parallelized, and cost-routed mechanically; the pane stays current for
both readers; tools stay ready; context survives compaction — none of it dependent on me remembering or the owner
asking.

## Build order
1. backlog generator (in flight) → 2. lane router + tool-health smoke → 3. fan-out ratchet (+ continuous refill) →
4. LIVE-STATE re-injection + noise-cut → 5. Vikunja deps/dates/walls + enforced sync → 6. adopt gpu_queue singleton +
max-runtime guard. Each lands as a tested guard; the heartbeat is rewired to the loop above once 1-3 exist.
