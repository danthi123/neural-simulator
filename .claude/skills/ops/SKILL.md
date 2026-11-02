---
name: ops
description: Use when launching, watching or stopping compute (GPU queue, pool, AWS, long runs, subagents), when changing infra (tools/, hooks, gates, .claude/ config, homelab-server services), and when touching the frozen old track (sim/, research/runners, the Stage-2 retake). Resource caps, agent tiering, watchers and server conventions in one checklist.
---
# ops: compute, infra and the frozen old track

Each `(...)` names the incident behind a step. Detail: `.claude/skills/_reference/cost-routing.md`.

## Compute
- GPU: one brain-loading job at a time, only via `tools/gpu_queue.sh` (`pause --now` / `resume` for gaming; gaming
  often means W rebooted to Windows, so W, its watcher and lanes are offline).
- RAM: `tools/mem_ok.sh <gb>` then `tools/memcap.sh <gb> -- <cmd>`; never stack onto a live training
  (memory memcap_no_oom: 09-18 OOM killed the session).
- A fresh worktree has no untracked `data/`: check `data/corpus/` before launch, or the job degrades silently
  (FAILURE_LOG 2026-09-23, 2026-10-27, 2026-10-28: the same failure three times).
- Before queuing a sweep, print the expanded commands: a loop variable left out made jobs identical (FAILURE_LOG 2026-07-31).
- Real or decisive runs on GPU (CuPy); numpy only for smoke. A numpy-only check hid a crashing GPU production
  chat (FAILURE_LOG 2026-08-19; memory gpu_not_numpy).
- Sweeps go to `tools/sweep_pool.sh` or the controller's `--seeds`, not to agents (memory minimize_plan_usage 08-06).
- Stock lanes with `.venv/bin/python tools/lane_check.py`; never write a waiver to quiet a stall check
  (FAILURE_LOG 2026-08-01, false lane waiver).

## Watching
- Arm `.venv/bin/python tools/attention_watch.py --loop` in the background at session start and after compaction;
  re-arm after each event (FAILURE_LOG 2026-09-24, GPU idle 7 h overnight; 2026-10-28, ruled tasks stalled 2 days).
- No sleep/tail poller loops; completion notices arrive (memory no_stale_pollers). Before calling a run crashed or
  done, check the PID (`ps -p` / `kill -0`): an empty log can be buffered output.
- Kill by PID, never `pkill -f <pattern>` (matches your own shell and other users' jobs).

## Agents and cost
- Tier every agent: haiku = mechanical, sonnet = moderate, opus = hard judgment; never inherit Opus
  (FAILURE_LOG 2026-08-19: half the weekly limit in 1.5 days). `tools/cost_audit.py` checks workflows.
- At most ~5 agents at once, WIP commits, build agents in their own worktree and commit before verifying
  (memory budget_aware_agent_fanout, build_agents_commit_before_verify).
- Give a cleanup agent exact paths: a stray `rm -rf` argument deleted ~/.claude/projects (FAILURE_LOG 2026-09-24).
- Never send project data to providers that train on or store it; ZDR (memory no_training_on_our_data).

## AWS
- brain/ six-seed runs: `bash tools/aws_brain_batch.sh launch <jobs.txt>` (g5.xlarge, Deep Learning AMI, self-stopping:
  dead-man + stop when the jobs end, so it is safe with the workstation off), then `pull` next session (copies
  results, terminates). GPU runs use fp16 weights (CPU runs fp32): compare like with like. Keys stay on W; the
  server's rules forbid credentials in its agent-writable checkout (OWNERSHIP-MAP section 11).
- On the server, never run git or project tools as dant123 inside `~/Projects/neural-simulator` (agent-controlled
  hooks/config). Claude's changes land via a `claude/<topic>` branch on GitHub + `tools/land_claude_branch.sh`,
  which the owner runs (or approves) at a check-in.
- Bills while running: `bash tools/aws_gpu.sh stop` when done; cap about $50/day. Pull results BEFORE any stop:
  an idle-stop stranded flip-deciding seeds (FAILURE_LOG 2026-09-25). Stop invalid runs early; let valid failing
  runs finish (memory aws_runs_cut_invalid_finish_valid).

## Infra and the server
- homelab-server: off-the-shelf, deployed per vendor docs; read `~/apps` OWNERSHIP-MAP, the new-service template,
  backup.sh and fw.nft first. No sudo-capable unit may run scripts from an agent-writable workspace (FAILURE_LOG 2026-09-30).
- Never edit a running bash script in place: write a temp file and `mv` it over (FAILURE_LOG 2026-09-25,
  rewriting gpu_queue.sh killed the dispatcher).
- Tests must stub systemd/local-llm: a test run stopped the owner's live local model (FAILURE_LOG 2026-09-26).
- Bash tool is zsh: unquoted `$var` does not word-split; run loops as `bash -c '...'`.
- `/tmp` is RAM (wiped on reboot): scratch goes on disk under `.claude/worktrees/` or the scratchpad.
- A new gate only for costly or irreversible harm with a mechanical trigger and a test; otherwise a skill line.
- Hermes and Pi cannot push; they publish via sim-sync / sim-publish. Claude merges and runs `tools/push_both.sh`.

## Frozen old track
- Old-track runs: pass `CoreSimConfig(..., seed=42)`; `actual_seed_used` only reports a seed and seeds nothing.
- `sim/`, `research/runners/` and the old roadmap are FROZEN (RULING `pivot-1102`). Only exception: the Stage-2
  retake (Hermes finishing it, scored by 11-21). No new old-track arcs; reuse old mechanisms from the new core
  by reading them. Old procedures: `.claude/skills/_reference/neural-simulator.md` (read only the section you need).
