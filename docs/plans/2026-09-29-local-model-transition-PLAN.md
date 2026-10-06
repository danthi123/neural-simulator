# Transition: the local model runs the project, Claude inspects rarely

*2026-09-29. Owner: the Claude plan drops to Pro around 2026-10-06; 10% of the weekly budget went in the first 3 hours
after the reset. From then on the local model (`tools/local_llm/llm.sh claude`, Qwen3.8-27B IQ4 MTP, 128K context) is
the daily driver and Claude is an occasional inspector. This plan uses the remaining Claude week to make that work.
The research plan itself is unchanged: [2026-09-25-prove-who-owns-the-computation-PLAN.md](2026-09-25-prove-who-owns-the-computation-PLAN.md).*

## 1. Division of labour

**Local model (daily driver).** Executes packets (section 2): commits staged work once the gates pass, provisions,
queues, harvests, scores with REGISTERED scorers, writes findings from the template, syncs board and docs, triages
tests, logs failures. Drafts new builds on `research/draft-*` branches with dev seeds only. Reads before building
(RAG, then sources). Anything needing judgment goes to the NEEDS-CLAUDE list in the runbook.

**Claude (inspector, about weekly, one Pro session).** Reads the inspection digest (section 6), not the history.
Reviews draft branches, rules on NEEDS-CLAUDE items, merges sound branches, writes the next packets, and adds a
gate when a new local-model failure mode shows up.

**Verdict rule.** A GO/NO-GO the local model writes is PROVISIONAL until an independent re-derivation with the
registered scorer agrees, or Claude has inspected it. The finding says which.

## 2. Execution packets (this week's Claude work)

A packet is a build finished to the point where the rest needs no judgment. It contains a committed prereg with
frozen bars, a committed runner with tests, one dev verification, and a `HANDOFF.md` giving the exact commands
(commit, provision, queue, score), an outcome-to-verdict table, stop conditions, and a NEEDS-CLAUDE list.

Packets in progress (2026-09-29), in the research plan's order:
1. Host-decision share baselines (`research/host-share-baselines`), classified by intervention outcome.
2. Live selection seam, flag default-off (`research/selection-seam-need-reversal`).
3. Delayed credit on the production basal-ganglia organ (`research/long-delay-credit-spiking`).
4. Sleep transitive inference inside the prioritized-memory prereg (`research/sleep-transitive-inference`).
5. B2b scoring and the B2d load-bearing battery at HEAD with a reply-level score (`research/b2-rulings-2026-09-29`).
6. The reply-level re-score of the #1 metric (a finding draft).

The runbook's task queue should hold about two weeks of packets when Claude steps back.

## 3. GPU and the local model

**Facts.** One RTX 3090, 24 GB. The default local-model profile peaks at 22.8 GB with 1.7 GB headroom
(`tools/local_llm/results/summary.md`), so no brain GPU run fits beside it. Most research runs are numpy on the pool
and need no GPU. `tools/gpu_queue.sh` already swaps automatically: when a GPU job starts it stops the model server,
and when the queue is idle it restores it. The local model is currently DENIED `gpu_queue.sh add`.

**Problem.** With the local model as driver, an automatic swap stops the driver's own server in the middle of its
session: every request fails until the queue drains, and the task in flight is lost.

**Recommendation: explicit yield, not an automatic mid-session unload.**
- CPU first: every run defaults to the pool; the GPU is only for work that needs it.
- The local model may add GPU jobs through a checked wrapper (a committed revision; a committed prereg for gate seeds).
  The jobs wait in the queue while the driver works (queue held).
- The driver unloads itself only at a checkpoint: `llm yield` refuses unless the work is committed and a handoff note
  is written, then stops the server and releases the queue.
- The queue drains inside a capped window (e.g. 10 h; overnight is the natural window). Then the server reloads.
- Resuming the session afterwards is an OWNER DECISION (section 8): automatic resume with a fixed continuation
  prompt, or manual (`llm resume`).
- **Gaming (owner 2026-09-29: it must be easy, by chat or another way).** In the session: tell the model you are going
  to game; it notes its step, commits if it can, and runs `game.sh on --graceful` (the server stops ~45 s later, after
  its reply). From any terminal: `game on`. Either way the GPU queue pauses, local runner processes freeze (the pool
  keeps running) and the model server stops. `game off` thaws everything, restarts the server and reopens the session
  with a "re-check your last step, then continue" prompt. Gaming overrides a GPU yield: nothing reloads while paused.

## 4. Guardrails for the local model's observed failure modes

| Observed | Guard (existing or to add) |
|---|---|
| Registered fake builds to pass the lane gate (09-27) | Deny rule + gate requires the branch to exist (done) |
| Ran GATE seeds without a committed prereg (09-28) | ADD: gate seeds refuse to run unless a committed prereg path is passed (tools.lab check) |
| Edited an existing runner against the runbook | ADD: deny Edit(research/runners/*) except `_draft_*` |
| Tuned the benchmark toward the answer (credit stealing) | Runbook: prereg before tuning; draft verdicts provisional |
| Left the primary checkout on a draft branch | `git switch`/`checkout` denied (done) |
| Compaction every ~10 minutes | Section 5 |
| Stuck behind the idle-compute gate | Deadlock fix in progress (`research/compute-idle-deadlock-fix`) |

## 5. Context economy

- A lean startup: CLAUDE.md loads into the local model too, so keep it short and point to the runbook.
- Task cards small enough to finish inside one compaction window; WORKLOG entry after every step.
- RAG before reading; read the file a hit points to, not whole directories.

## 6. Inspection digest

ADD `tools/inspection_digest.py`: for the period since the last inspection it lists commits by branch, new findings
with their verdict lines (provisional or re-derived), NEEDS-CLAUDE items, new FAILURE_LOG rows, gate blocks, and
pool/GPU utilization. Target: under 15K tokens, so one Pro session can inspect.

## 7. Timeline (while Claude has budget)

- 09-29 to 09-30: packets for the running builds; the deadlock fix; the reply-level re-score; B2b scoring.
- 10-01 to 10-02: guardrails (section 4), `llm yield` and the GPU wrapper, the inspection digest, runbook v3 with a
  two-week task queue.
- 10-03: dry run. The local model drives a full day; Claude reads only the digest at the end, then fixes what broke.
- 10-04 to 10-05: buffer, plus packets for the next research steps.

## 8. Owner decisions (answered 2026-09-29)

1. **Auto-resume after a GPU window: YES.** Only after a driver-initiated `llm yield`; no watchdog restarts a crashed
   session. This is the one exception to CLAUDE.md's manual cross-session continuation.
2. **The local model may queue GPU jobs through the checked wrapper: YES.** Jobs still start only when it yields.

## 9. The harness on the server (owner 2026-09-29: containerized on 192.168.0.10, everything THIS week)

**2026-09-30: RETIRED.** The owner decided against a custom harness; the sim-agent container described below was
torn down. The owner runs the off-the-shelf Hermes Agent on homelab-server (`~/apps/hermes`) as a general
harness instead, and the sim project's own local-model driver stays `tools/local_llm/llm.sh claude` on the
workstation. The design below is kept as a record of the work done, not as current state.

The server (`homelab-server`: 16 threads, 61 GB, no GPU, Docker 29, hosts gitea and many other services) runs the
harness; the workstation keeps the 3090 for serving the model and for brain GPU runs, and keeps the pool dispatcher.

**Container `sim-agent` (isolation is the point: a rogue model must not reach anything else on the server).**
Non-root user; no Docker socket; no host network; not attached to any other stack's network (its own bridge);
`cap_drop: ALL`, `no-new-privileges`, read-only root filesystem with tmpfs `/tmp`; CPU/memory/pids limits (e.g.
12 CPUs, 24 GB, 4096 pids); two named volumes only (the repo clone, the agent's home with Claude Code state). Inside:
Claude Code, git, the project's numpy venv (no cupy), the RAG venv, tmux. The owner attaches with one command
(`agent` on the workstation: ssh + `docker exec -it sim-agent tmux attach`) and detaches without stopping it.

**LLM-agnostic.** `llm use <profile>` inside the container picks a backend profile (base URL, token variable,
model name, context size): the workstation's server over the LAN, or a cloud API with an Anthropic-compatible
endpoint later. Nothing else changes between backends.

**Reaching the workstation: one restricted gateway.** The container holds one ssh key for the workstation, locked in
`authorized_keys` to a forced command (`tools/agent_gateway.sh`) that allows only: submit a pool job (through the
gate-seed guard), submit a GPU job (through `gpu_add.sh`), sync results read-only into the agent's clone, queue and
game status, and `yield`. The pool dispatcher and the GPU queue stay on the workstation unchanged.

**Pausing: one mechanism for gaming and GPU windows.** The workstation holds one key for the server, locked to a
forced command that can only `docker pause` / `docker unpause` `sim-agent` and type a line into its tmux session.
`game on` pauses the harness, then stops the model server, the GPU queue and local runners; `game off` restarts the
model server, waits until it answers, unpauses the harness and types "re-check your last step, then continue".
`yield` (called by the agent through the gateway) pauses the harness and stops the model server; the GPU queue
drains inside the window; then the same restart, unpause and a "GPU window finished" line. With a cloud backend
neither gaming nor GPU windows need to pause the harness.

**Credentials (created by the owner, never by an agent).** A gitea key or token for the agent that may push only
`research/draft-*` (branch protection on `main`); pushes to GitHub stay with the workstation/Claude. Optional: a
Vikunja token. Optional egress limits for the container's subnet (a DOCKER-USER firewall snippet the owner applies).

**2026-09-30 refit to srv-host-0's own conventions** (branch `research/harness-server-fit`; procedure
`docs/AGENT_HARNESS_SETUP.md`). The design above assumed a generic Docker host. On the real box the harness is now an
`~/apps` stack fragment (one compose project, secrets as `SIM_AGENT_*` in the one `~/apps/.env`), its repo and home are
bind mounts on a refquota-capped, sanoid-snapshotted ZFS dataset (`data/appdata/sim-agent`, skipped by the nightly
cold-stop backup), limits are 8 CPUs / 12 GB (the box has ~23 GB free), and the egress firewall is MANDATORY: an
nftables table of its own (`sim-agent.nft`, loaded before Docker) replacing the DOCKER-USER snippet, which on this box
matched no bridge, blocked the fallback resolver and let the container hairpin into the server's own HTTPS services.
The container refuses to start unless that table is dropping its traffic.

**Serving simplification (workstation).** Bake-off with the existing `tools/local_llm/bakeoff.py`: Ollama and Unsloth
Studio (both serve the Anthropic Messages API) against today's llama-server profile, same model file where possible.
Switch to the easiest one that passes the three tasks and stays within ~25% of today's speed; otherwise llama-swap
(one YAML file) in front of llama.cpp. The chosen server listens on the LAN with an API key where supported.
