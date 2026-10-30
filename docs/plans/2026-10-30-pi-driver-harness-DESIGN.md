# Pi driver harness — DESIGN (draft for owner approval)

Date: 2026-10-30. Status: DRAFT, awaiting the owner's approval. Nothing here is built yet.

## Why

The owner asked on 10-30 for a driver that runs the project's compute smoothly:
- the workstation (W) GPU and the mini-PC pool;
- loading and unloading the local model on W;
- cloud models used only as needed.

It should make as much progress as possible at low cost, with high visibility for the owner and a way to steer without
constant upkeep. A custom harness is acceptable if it lowers overall maintenance. The owner approved the design on the
condition that the local models show real promise in taking over much of the existing Claude and cloud-model work.

### The gating evidence (10-29/30 hard-task trial)

Details: `sim-rev/llm-trial/`, memory `project_2026_10_29_local_model_strata_trial`.

- **Execution tasks: local Q2_0 = MiMo.** These are tasks with a checkable answer.
  - Q2_0 (125B MoE, Strata, ~100 tok/s) passed every one: H4 3/3 and H5 3/3. H5 is a real multi-day bug from our
    history, graded by a hidden test.
  - MiMo Pro also passed 6/6.
  - Times varied widely: 15 min to 3 h 53 min for the same task, and most of the long tail was the model running
    simulations.
- **Judgment tasks: no cheap model is close to the expert key.** H6 is an adversarial review of a real design packet with
  six known blocking defects, blind-graded.
  - Q2_0 scored 0 / 1.0 / 1.5 (mean 0.8) with 5 false alarms.
  - MiMo scored 2.5 / 1.0 / 1.0 (mean 1.5) with 0 false alarms.
  - No model found two of the six defects.
- **IQ3_XXS is dropped.** It runs about half Q2_0's speed with slow turns; one review timed out at 3 h.
- **Conclusion:** the local model can do most execution work (the cheap-cloud share and much of Claude's), for free.
  Judgment stays with Claude: packet approval, verify-go, rulings, merges. The harness is built around that split.

The 27B and Q2_0-medium-effort results are pending and may change the default local model, but not the design.

## Shape

```
              homelab-server (S) — always on
  ┌──────────────────────────────────────────────────────────┐
  │ pi-driver container (pinned pi 1.4.x, RPC mode)           │
  │   supervisor wrapper (ours, systemd)                      │
  │   extensions: router · compute · checkpoint · guard ·     │
  │               steer-inbox · status                        │
  │   workspace: the sim repo (same tree Hermes uses)         │
  └──────────────────────────────────────────────────────────┘
        │ LAN (when W is up)        │ internet
        ▼                           ▼
  W: Strata Q2_0 :8100        OpenRouter (MiMo, capped)
  W: gpu_queue / llm yield    `claude -p` (owner's plan) for checkpoints
  pool41/42 + wspool via the existing dispatcher (unchanged)
```

The driver lives on S, so W can unload its model, run a GPU job or reboot into Windows without killing the driver.

## Components

### 1. Router: which model does this step

Built as `registerVirtualModel`; routing is by step class, and the agent declares the class.

| Step class | Model | Fallback |
|---|---|---|
| execute (code, debug, run, harvest) | W Q2_0 (local, free) | MiMo if W is down and the step is urgent; otherwise wait for W |
| first-pass review / second opinion | MiMo | Q2_0 |
| checkpoint (packet approval, GO/flip verify-go, ruling, merge) | Claude via `claude -p` (the checkpoint tool) | queue it; never downgraded |

- **Health check:** W's endpoint is probed before each turn. A down W never stalls the loop silently: it switches or parks.
- **Spend caps:**
  - MiMo has a daily budget, enforced from OpenRouter key usage. The key also keeps its own weekly cap of $25.
  - Claude escalations have a daily count cap.
  - Hitting either cap parks the step and raises a NEEDS line. The cap is never bypassed.
- **Context:** `models.json` sets `contextWindow` 131072 for Q2_0, because Pi's default is 128000.

### 2. Compute: experiments get the whole machine

These are tools wrapping what already exists; nothing is re-implemented.

- **`run_pool(cmd, mem_gb)`:** appends to `pool.queue` through the existing pool-submit path. The dispatcher stays the
  only thing that launches pool jobs.
- **`run_gpu(cmd)`:** submits a GPU job through `gpu_queue.sh` on W. If the local model holds the GPU, the tool asks W to
  `llm yield`.
  - While the job runs, the driver switches to MiMo for urgent execution or simply waits.
  - When the job ends, `llm` reloads Q2_0 and the driver resumes on the same session. This is the owner's
    unload → run → reload → continue loop.
- **Waiting is not polling by the model.** The tool returns a job id. An extension watches job status
  (`/var/lib/host-status/sim-pool-jobs/exits.json`, the gpu_queue log) and injects a follow-up message on completion.
  The model sleeps; no tokens are spent.
- **Yield only for long GPU jobs (more than ~10 min).** Reloading Q2_0 costs ~1–3 min, so short CPU tests run inline.

### 3. Checkpoint: judgment goes to Claude, as a packet

- **Packet:** the agent writes `research/coordination/checkpoints/<id>.md` with the question, the evidence paths and the
  bar.
- **Call:** the tool runs `claude -p` in the repo with the right skill (verify-go, ruling or merge review), a turn cap and
  a timeout. Claude's verdict is written beside the packet.
- **On the plan, not API billing:** Claude Code is logged into the owner's plan inside the container, with no API key.
  The owner does that login; I cannot enter credentials.
- **Rules that do not move to the driver:** merges to main, default flips and GO claims need a Claude verdict file. The
  guard extension blocks the git command otherwise.

### 4. Guard: permissions

- **Container boundary:** only the workspace and the needed sockets are mounted. The secrets present are the scoped
  OpenRouter key and the Claude login. Pi has no built-in permission system and its docs recommend containers.
- **`tool_call` guard on bash as well as write/edit.** The example protected-paths extension misses bash.
  - It reuses `tools/agent_merge_policy.py` for reserved paths.
  - It blocks `--no-verify`, force-push, `git reset --hard` and direct pushes (push only through `tools/push_both.sh`).
- **The repo's own gates stay authoritative.** The pre-commit hooks run unchanged.

### 5. Supervisor, visibility and steering

The supervisor is a small systemd service wrapping `pi --mode rpc`. It:
- feeds the next task from the lane's board items and the runbook;
- restarts on crash with `--continue`, and repairs dangling tool calls after an abort (upstream issue #9986);
- detects a stall (no events for N minutes) and re-kicks it;
- writes `status.json`: current task, model, spend today, last event, open checkpoints.

Visibility:
- `status.json` and the transcript are published to S's host-status. Hermes's 08/16/22 report includes them.
- A minimal read-only status page goes behind Authentik.
- The existing attention watcher alerts on a dead or stalled driver.

Steering:
- **`pi-steer "text"`** writes to an inbox, which an extension injects as a `steer` message on the next turn.
- **Full takeover:** stop the service, attach interactively with `pi --continue` in tmux, then hand back.
- Pi has no multi-client attach, so a live shared terminal is out of scope.

## Failure handling (the "no constant upkeep" part)

| Fault | Handling |
|---|---|
| W down / gaming | router switches or parks; no alarm unless urgent work waits more than 6 h |
| Strata crash | `llm` restarts it; the router retries after the health check |
| pi crash / hang | supervisor restarts it with `--continue`; a stall triggers a re-kick |
| Context full | Pi compaction (reserve 16k, keep 20k recent), plus a task-decomposition rule: one board item per session |
| Spend cap hit | park the step and raise a NEEDS line; never bypass |
| Checkpoint refused | the verdict goes back to the driver as the next instruction |

## Pilot (one week, after approval)

- **Scope:** Pi drives ONE board lane on S; Hermes keeps every other lane plus homelab duty. The lane is chosen at the
  start: a bounded build item with tests, from the board's p4 list.
- **Measures:**
  - owner and Claude interventions per day;
  - defects Claude's checkpoints catch;
  - items landed;
  - spend (MiMo $, Claude escalations);
  - driver uptime and faults.
- **Compared against:** Hermes's lanes over the same week.
- **Exit:**
  - If interventions are under ~1/day and items land at Hermes's rate or better with no checkpoint-caught regressions
    reaching main, propose migrating more lanes.
  - Otherwise, bank what broke and decide.

## What this would retire (if the pilot succeeds and lanes migrate)

- The workstation-side local-model driver mode (`llm.sh claude`).
- Lane by lane, the Hermes `sim-driver` cron for the migrated lanes. Hermes keeps homelab and reports.

## Build plan and cost

1. **Week 0, about two sessions.**
   - Container and supervisor.
   - Six extensions.
   - Checkpoint tool.
   - Crash-resume test: kill mid-tool and confirm a clean resume, since #9986 is the open risk.
   - Spend-cap test.
2. **Owner steps.** Approve this design. Log Claude Code into the container. Confirm the MiMo daily budget (proposed $1/day).
3. Pin Pi 1.4.x. Upgrade deliberately, because releases are weekly and the API is still moving.

## Open risks

- Pi is one month past 1.0 (MIT, active). API churn is likely, so we pin the version.
- Resuming an interrupted turn cleanly is unverified upstream; the week-0 test settles it.
- 131k context against long tasks; mitigated by one item per session and compaction.
- Judgment quality is the binding constraint, not the harness. If checkpoints become the bottleneck, the Claude plan
  question (Max $100 vs $200) comes back with real numbers.
