# OPENHANDS TAKEOVER — running this project on local Qwen when Claude usage is out

This is the owner's operating manual for **OpenHands** driving this repo on the local **Qwen3.8-27B**
brain, for when Claude usage is exhausted. It replaces **Hermes** (`tools/hermes_takeover.sh`,
`docs/HERMES_HANDOFF.md`) as the recommended local fallback — Hermes stays installed and working
until OpenHands is live-verified end-to-end (see "Status" below), so nothing is lost if this needs a
fallback of its own.

**Run ONE driver at a time: Claude, Hermes, or OpenHands.** Hand the current one back before starting
another.

## What "OpenHands" actually means here

The OpenHands *brand* today points most people at either a deprecated CLI or "Agent Canvas" (a
Node/npm multi-agent visual workspace) — neither is what runs here. This project uses the actively
maintained **`openhands-sdk`** Python library directly: one continuous, resumable conversation against
the local model, no Docker, no Node, no separate web server. There is **no browser UI** — you talk to
it from a terminal, and it keeps scrolling in one session like a chat log, which is the property the
owner asked for over Hermes' fresh-session-per-turn design. See
[`docs/2026-09-06-openhands-harness-prototype.md`](2026-09-06-openhands-harness-prototype.md) for the
full evaluation that led here.

## 0. One-time build (skip if `tools/openhands_proto/.venv` already exists)

The install lives in its own isolated Python 3.12 venv — never the repo's main `.venv` — so it is
fully reversible (`rm -rf tools/openhands_proto/.venv`) and never conflicts with anything else on the
box.

```bash
cd tools/openhands_proto
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -U openhands-sdk openhands-tools
```

Takes under a minute; ~485 MB on disk. `tools/openhands_takeover.sh on` checks for this and tells you
to run it if missing.

## 1. Start: hand the project to OpenHands

```bash
bash tools/openhands_takeover.sh on
```

**Run this from the canonical checkout (`/home/dant123/Projects/sim`), not a git worktree.** Every
piece here (`qwen_serve.sh`, `qwen_supervisor.sh`, the local GPU queue) is a *singleton* shared by the
whole box — a worktree has its own, separately-checked-out copy of `research/queue/*`, so running the
takeover from a worktree would make decisions off a stale/disconnected queue snapshot while the real
GPU and model are elsewhere. (This is a pre-existing limitation shared by `hermes_takeover.sh` too,
not something new here.)

This does three things: marks OpenHands the active driver, starts the shared VRAM supervisor (the same
one Hermes uses, watching a second sentinel now — see "How the GPU handoff works" below), and launches
`tools/openhands_proto/openhands_loop.py` in the background. Qwen then loads once the local GPU queue
is idle. **Run `on` when the GPU is actually free for the model** — it does not preempt a running
research job the way Hermes' takeover does.

## 2. Give it a task

Every invocation below talks to the **same persisted conversation** — there is nothing to "open", you
just send the next message:

```bash
tools/openhands_proto/.venv/bin/python tools/openhands_proto/run_turn.py \
    --prompt "Read GAP_CLOSURE_MISSION.md and fix the failing test in tests/test_foo.py"
```

The transcript prints to your terminal as it works (tool calls, file edits, shell output) — that is
the "scrollable session," and each `run_turn.py` call continues where the last one left off, even
across a GPU-offload cycle in between. With no `--prompt`, it sends a generic "continue the session"
instruction that re-reads `research/coordination/live_state.md` and `GAP_CLOSURE_MISSION.md`.

Meanwhile `openhands_loop.py` (started by step 1) is *also* driving the same conversation on its own
cadence for unattended/autonomous work — the two are not in conflict, they share the one conversation
and its persisted history; `run_turn.py` is just how you inject an ad-hoc instruction from the
keyboard.

## 3. The GPU handoff (by design — no failover)

`tools/qwen_supervisor.sh` owns Qwen's VRAM lifecycle, unchanged from how it already worked for
Hermes: **a local GPU research job and the Qwen server never co-reside.** When a local job is queued
or running, the supervisor unloads Qwen; when the local queue goes idle, it reloads Qwen automatically.
Neither `openhands_takeover.sh` nor `openhands_loop.py` load/unload the model themselves — they only
ever read state and wait, so there is exactly one decider (this is a deliberate 2026-09-08 change from
the original prototype, which called `qwen_serve.sh` itself; see the script headers).

**While Qwen is unloaded for a GPU job, OpenHands' next turn simply pauses** (the loop waits for
`qwen_up()` to go true again) and an ad-hoc `run_turn.py` call will refuse to start with a clear
"endpoint not reachable" message rather than hang or silently retry forever. **There is no automatic
failover to a different model** — this is intentional (per the owner's design), not a bug. Wait for
the run to finish (Qwen reloads within seconds of the queue draining) and try again, or check status
below.

## 4. Check status

```bash
bash tools/openhands_takeover.sh status
```

Reports: who's driving, whether the venv is built, whether the supervisor and `openhands_loop.py` are
running, and `qwen_serve.sh`'s own status (up/down, VRAM). Add `tail -f tools/openhands_proto/state/openhands_loop.log`
to watch the loop's own turn-by-turn log live.

## 5. Hand back to Claude

```bash
bash tools/openhands_takeover.sh off
```

Stops `openhands_loop.py`, clears the driver sentinel, unloads Qwen, and frees the GPU for research
runs. The supervisor daemon is left running (harmless — it goes inert with no driver sentinel set).

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `run_turn.py` refuses with "endpoint not reachable" | Qwen is unloaded for a GPU research job (by design, see §3) | `bash tools/qwen_serve.sh status` — wait for the run to finish, it reloads automatically |
| `openhands_takeover.sh on` says the venv isn't built | First-time setup skipped | Run the §0 build commands, then retry |
| `openhands_takeover.sh on` refuses with "HERMES_ACTIVE is set" | Hermes is (or was) the active driver | `bash tools/hermes_takeover.sh off`, then retry |
| Qwen never loads after `on` | The local GPU queue never goes idle, or you ran `on` from a worktree (see §1) | `bash tools/qwen_supervisor.sh status` to see the live verdict; re-run `on` from `/home/dant123/Projects/sim` |
| `openhands_loop.py` isn't running per `status` | It crashed on start | `tail -40 tools/openhands_proto/state/openhands_loop.log` |
| Two drivers seem to be fighting over Qwen | Both `hermes_takeover.sh on` and `openhands_takeover.sh on` were run | Run `off` on both, then start only the one you want |

## Status (2026-09-08) — what is verified vs. what is deferred

- **Verified live, 2026-09-06** (before this productization pass): the OpenHands SDK harness drove the
  local Qwen endpoint over two turns of one persisted conversation and correctly referenced its own
  prior turn — the one-continuous-session property works end-to-end on real hardware. See
  [`docs/2026-09-06-local-agent-stack-review.md`](2026-09-06-local-agent-stack-review.md).
  Verified offline (no GPU/network), 2026-09-08: `tools/openhands_proto/validate_offline.py`, 9/9,
  rebuilt venv (`openhands-sdk`/`openhands-tools` 1.45.0).
- **Deferred to the next GPU-clear moment**: a live smoke of the *productized* pieces built this
  session — `tools/openhands_takeover.sh on`, the generalized `tools/qwen_supervisor.sh` (now watching
  `OPENHANDS_ACTIVE`), and `openhands_loop.py`'s new supervisor-managed (non-self-managing) VRAM mode.
  These were only dry-tested (read-only status checks + isolated sentinel-file logic, no GPU/network
  calls) — see the exact commands below. Do this from the canonical checkout when the GPU is free:

```bash
# 1. Confirm nothing local is running, and the venv is built:
bash tools/qwen_supervisor.sh status
bash tools/openhands_takeover.sh status

# 2. Take over for real:
bash tools/openhands_takeover.sh on
tail -f tools/openhands_proto/state/openhands_loop.log     # watch Qwen load, then a turn fire

# 3. One ad-hoc turn on a real task:
tools/openhands_proto/.venv/bin/python tools/openhands_proto/run_turn.py \
    --prompt "What did OpenHands conclude in its last turn? Answer in one sentence, no tool calls."

# 4. Confirm the GPU handoff works with the NEW supervisor-owned path (not the old self-managed one):
#    enqueue a short local job in another terminal and watch qwen unload/reload around it, driven by
#    tools/qwen_supervisor.sh (check its log), not by openhands_loop.py itself:
bash tools/gpu_queue.sh add 'echo openhands-handoff-smoke && sleep 20'
tail -f research/queue/qwen_supervisor.log

# 5. Hand back:
bash tools/openhands_takeover.sh off
```

If all of 2-4 hold, OpenHands is fully live-verified on the productized path and can become the
default recommendation over Hermes without caveats.
