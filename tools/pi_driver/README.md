# Pi sim driver

The design is [`docs/plans/2026-10-30-pi-driver-harness-DESIGN.md`](../../docs/plans/2026-10-30-pi-driver-harness-DESIGN.md).

[Pi](https://github.com/earendil-works/pi) 1.4.0 (pinned) drives one board lane on the server. It works through three
pieces:
- **this extension**, which handles routing, compute, checkpoints and the guard;
- **a supervisor**, which feeds the lane, takes steering, restarts Pi and writes the status file;
- **the workstation's local model**, served by llama-swap.

Hermes keeps every other lane, plus its homelab duties.

## What does which job

| Job | How |
|---|---|
| Coding, debugging, running and harvesting experiments | Every request goes to W's `strata-q2_0` (Qwen3.8-Flash-Next Q2_0, ~100 tok/s, free) through llama-swap at `w_base_url`. When W is down (yielded to a GPU job, gaming, rebooting) the run **parks**, and the supervisor retries every `park_retry_min`. |
| Independent second opinion | The `second_opinion` tool. It reads the named files itself and sends their full text to MiMo in one call, metered against `mimo_daily_usd`. |
| Judgment: GO/PASS, default flips, prereg changes, method changes, merge reviews | The `checkpoint` tool. It writes a packet to `research/coordination/checkpoints/` and runs `claude -p` on the owner's plan; the verdict lands next to the packet. Capped at `claude_daily_calls` per day. |
| CPU jobs | `run_pool`, which calls `tools/pool_queue.sh add`. The dispatcher is unchanged. |
| GPU jobs | `run_gpu`, which calls `tools/gpu_remote.sh add` and optionally `yield`. The model unloads, the job runs, and llama-swap reloads the model on the next request. |
| Waiting | `wait_for` (file, gpu or shell) persists a waiter in `state_dir`. The model ends its turn and is messaged when the job finishes. Waiting costs no tokens. |
| Repo rules | `guard_policy.ts` checks bash as well as write and edit. It blocks: `--no-verify`, `git push` (the server publishes research/* branches itself), `reset --hard`, `clean`, touching main, host and process control, compute control outside the tools, deleting or overwriting results, editing gates, hooks, `CLAUDE.md` or this harness, and reading secrets. Tests: `node --test tools/pi_driver/extensions/sim-driver/guard_policy.test.ts`. |

A merge checkpoint yields a verdict on an exact sha. Landing on main stays with the existing flow, `land_claude_branch.sh`
or a Claude check-in, because the main checkout belongs to uid 99.

## Smoke test, 2026-10-31

This ran on the workstation against live Strata, with a throwaway workspace and on-disk state. Each item below passed:
- the lane item was fed;
- the guard refused `git push`;
- a `wait_for` woke the driver when its flag appeared;
- after a `kill -9` of Pi in the middle of a 150-second command, the supervisor restarted it with `--continue`, and the
  session re-ran the interrupted command and finished the item. Upstream issue #9986 (dangling tool calls) did not
  show.
- the killed Pi's orphaned tool command was reaped (the supervisor is a child subreaper);
- a steer sent just before a crash was re-delivered and obeyed;
- `second_opinion` cited real lines ($0.002);
- a Claude checkpoint returned `VERDICT: APPROVE` in 14 seconds.

Three defects the test caught and fixed:
- **Fabricated review.** A "review" routing mode let MiMo critique a file it had not read. The local model then repeated
  the fabrication. Fixed by the grounded `second_opinion` tool.
- **Lost steer.** A steer sent just before a crash was lost. Fixed by re-delivering unsettled steers on restart.
- **Endless re-feed.** The supervisor would re-feed the same lane item forever. Fixed: an item fed 3 times without its
  line changing is reported under `needs`.

## Deployment (server)

The deployer follows the host conventions (`~/apps` OWNERSHIP-MAP, its service template, backup and firewall). The
driver needs the following.

**Runtime**
- Node ≥ 22.19 (the docs use `node:24-bookworm-slim`).
- `npm install -g --ignore-scripts @earendil-works/pi-coding-agent@1.4.0`.
- Python 3 for `supervisor.py`.
- git, bash and ripgrep.
- The repo's Python venv, for tests: the worktree has none, so the lane prompt names the main checkout's.
- The Claude Code CLI (`claude`).

**Workspace (`workspace`)**
- A git worktree of the server repo on branch `research/pi-lane`, owned by the uid the driver runs as. That uid must be
  able to commit there. It gets no push credentials: the remotes' deploy keys stay with the `sim` user, and
  sim-publish pushes every research/* branch within 15 minutes. (Deployed 2026-10-31 as uid 99, compose stack
  ~/apps/pi-driver; OWNERSHIP-MAP section 11 "Pi lane"; results_root is the main checkout, which sim-sync fills.)
- `tools/pool_queue.sh add` writes the shared queue through the git common dir, so the worktree must share the main
  checkout's `.git`.
- The repo's pre-commit hooks must stay installed. They are the gates.

**Reads**
- `results_root` (`/home/sim/runtime`, where `sim-sync` lands pool and GPU results), read-only.

**Network**
- W's llama-swap at `192.168.0.68:8080`.
- `openrouter.ai`.
- Claude's API, which the CLI uses.
- The git remotes.
- SSH to the workstation, for `tools/gpu_remote.sh` (the same key Hermes uses for it).

**Secrets**
- The OpenRouter key, as the file `/run/secrets/openrouter.key` (`models.json` reads it with `!cat`). This account has
  zero data retention enforced, and `models.json` also sends `data_collection: deny`.
- The Claude Code login. **The owner does it**, interactively, once.

**State**
- `state_dir` (`/var/lib/pi-driver`): `config.json` (from `config.example.json`), `agent/models.json` (from
  `models.json`), sessions, waiters, the ledger, `inbox/`. Back it up.

**Status**
- `status_dir` (`/var/lib/host-status/pi-driver/status.json`): read by `tools/attention_watch.py` and by a read-only
  status page, if one is added behind Authentik.

**Service**
- One long-running process, `PI_DRIVER_CONFIG=<state_dir>/config.json PI_BIN=pi python3 tools/pi_driver/supervisor.py`,
  run from the repo (the supervisor sets Pi's agent dir to `<state_dir>/agent`).
- Restart it always. Stop with SIGTERM: Pi sessions resume with `--continue`.

**Workstation side (done)**
- `tools/local_llm` serves `strata-q2_0` through llama-swap: `llm on strata-q2_0`. It loads lazily on the first request
  after `mem_ok.sh 24`, and idle-unloads after 15 minutes.

## Operating it

- **Status:** `tools/pi_driver/pi-steer --status`, or the status page. Its fields:
  - `status`: working, waiting, parked, idle or stuck;
  - `needs`: what only the owner or Claude can unblock;
  - `agent.ledger`: today's MiMo dollars and Claude calls;
  - `agent.waiters`: open waits.
- **Steer:** `tools/pi_driver/pi-steer "message"`. While Pi is working the message is delivered as a steer; when it is
  idle, as a prompt.
- **Give it work:** add `- [ ] <item>` lines to `research/coordination/PI_LANE.md` on the lane branch. The driver marks
  each item `- [x]` (done, with branch, sha and verdict file) or `- [!]` (waiting on something).
- **Take over:** stop the service, then run `pi --continue` interactively in tmux, in the workspace, with the same
  `PI_DRIVER_CONFIG` and `PI_CODING_AGENT_DIR`. Restart the service when done. Pi has no multi-client attach.

## Pilot

Run it for one week on one lane: a bounded build item with tests, from the board's p4 list.

**Measure:**
- interventions per day;
- defects that checkpoints caught;
- items landed;
- MiMo dollars and Claude calls;
- restarts and parks.

Compare these against Hermes's lanes over the same week. Exit criteria are in the design.
