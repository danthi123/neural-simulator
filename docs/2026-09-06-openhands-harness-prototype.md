# OpenHands harness prototype — install, config, offload design (2026-09-06)

Reference/decision note (not a science finding — no seeds, no artifact). Owner-approved prototype:
evaluate **OpenHands** as a replacement for the in-house **Hermes** harness
(`tools/hermes/loop.py`), driven by the owner's decisive requirement — **ONE continuous, scrollable
session** (like Claude Code), not Hermes v2's fresh-session-per-turn design. Follows on from
[`docs/2026-09-06-local-agent-stack-review.md`](2026-09-06-local-agent-stack-review.md), which named
OpenHands as the lead candidate for exactly this reason.

**Verdict up front: promising, with the actively-maintained artifact being a different (leaner) thing
than "OpenHands" first suggests.** The heavyweight app + the `OpenHands-CLI` binary are not the right
target — see §1. The `openhands-sdk` Python library is: no Docker, no Node, installs in under a
minute into an isolated venv, and its `Conversation(persistence_dir=..., conversation_id=...)` +
`LLMSummarizingCondenser` genuinely deliver the one-continuous-session property Hermes cannot. All of
the SDK's own machinery (tool execution, persistence, condenser construction, agent construction) is
now verified working on this box; only the live model call itself is untested, because qwen is
offloaded to a research run right now (§5 gives the controller the exact commands to close that gap).

## 0. What was NOT touched

Nothing outside `tools/openhands_proto/` was created or modified. The install lives entirely in its
own Python 3.12 venv (`tools/openhands_proto/.venv/`, ~485 MB, built via `uv venv --python 3.12`
+ `uv pip install openhands-sdk openhands-tools`) — never the repo's main `.venv`. No systemd unit was
installed or enabled. `tools/qwen_serve.sh` was only ever called with `status` (read-only) during this
work; `up`/`down` were never invoked, and the live GPU job that was running throughout
(`research/queue/gpu.running`, a `_rank2_integrated_loop_webapp_thread_derisk` 6-seed run) was left
completely alone. Reversal is `rm -rf tools/openhands_proto/`.

## 1. What "OpenHands" means today (mid-2026) — a real landscape shift since Jan-2026 sources

A web-verified research round (2026-09-06) found the ecosystem has moved since most training-era
knowledge of "OpenHands" (né OpenDevin): the project split into three distinct things, and the one
most people mean by "OpenHands" is now the *least* suited to this task.

| Artifact | Current status (2026-09) | Fit for us |
|---|---|---|
| `openhands-sdk` + `openhands-tools` (PyPI, `pip install openhands-sdk openhands-tools`) | **Actively maintained.** Latest `openhands-sdk` 1.44.1 (28 Aug 2026), matched-version `openhands-tools`. Pure Python library: `LLM`, `Agent`, `Conversation`, `Tool` classes + a marketplace of tool implementations (terminal, file editor, grep, glob, task tracker, browser, apply-patch, …). No Docker requirement — "sandboxed workspaces in Docker/K8s" are an *optional* extra (`openhands-workspace`/`openhands-agent-server`), not the default path. | **This is the right target** — see §2. |
| `OpenHands-CLI` (`github.com/OpenHands/OpenHands-CLI`, `uv tool install openhands` / `pip install openhands`) | **No longer actively maintained.** The repo's own README tells users to migrate to "Agent Canvas." Still installs and still supports `--headless`/`--resume`, but is a dead end to build on. | Ruled out — do not invest here. |
| **Agent Canvas** (`github.com/OpenHands/openhands`, `npm install -g @openhands/agent-canvas`) | The new flagship (launched June 2026): a self-hosted, agent-agnostic "developer control center" that can drive OpenHands, Claude Code, Codex, or Gemini as interchangeable backends, each conversation isolated in its own git worktree, with Slack/GitHub automations. Actively developed (86k★, hundreds of open PRs/issues). Node/npm-based; Docker "optional but recommended" for sandboxing. | Not evaluated further here — heavier than the ask (a multi-agent visual workspace, not a single headless loop), and this box's system `node` is currently broken (`libada.so.3` missing — a pre-existing, unrelated system issue, not something this task touched or needs to fix for the SDK path). Worth a look later if the owner wants the "reusable local-automation platform beyond this project" angle Agent Canvas is explicitly built for. |

The owner's ask — a headless loop, one continuous session, driving a local OpenAI-compatible
endpoint, coexisting with our own VRAM supervisor — is squarely an `openhands-sdk` use case, not an
Agent Canvas one. Agent Canvas is a UI/orchestration layer *around* agents like OpenHands; adopting it
would mean solving today's problem (one continuous session) plus taking on a Node stack and a
multi-agent UI we don't need yet. **Recommendation: prototype and (if it holds) adopt `openhands-sdk`
directly; revisit Agent Canvas only if the owner separately wants the multi-agent visual workspace.**

## 2. Install (done, isolated, reversible)

```bash
cd tools/openhands_proto
uv venv --python 3.12 .venv          # uv fetches a standalone CPython 3.12.13 (system python3 is 3.14,
                                       # too new — openhands-sdk's pyproject pins Python 3.12; this
                                       # download does NOT touch system Python)
uv pip install --python .venv/bin/python -U openhands-sdk openhands-tools
```

Result: `openhands-sdk==1.44.1`, `openhands-tools==1.44.1`, ~180 dependencies (litellm, pydantic,
libtmux, rich, tiktoken, …), all inside `tools/openhands_proto/.venv/`. No Docker daemon interaction,
no `sudo`, no system package changes. `tmux` is not installed on this box; the terminal tool
auto-falls-back to a `subprocess`-backed session when `tmux` is unavailable (verified §4 — no need to
install `tmux`, which would have been the one plausible "system-level change" this task would have
had to stop and ask about).

Docker itself **is** already active on this host (the user is in the `docker` group; other unrelated
containers — a separate `open-webui`, a separate app stack — are already running) — but nothing in
this install path uses it, so the "stop and ask before a system-level change" bar was never reached.

## 3. Local-endpoint config

`tools/openhands_proto/agent_config.py` is the single factory both scripts below import from:

```python
llm = LLM(
    usage_id="qwen-local",
    model="openai/qwen-local",                    # litellm's custom-OpenAI-endpoint convention:
                                                     # "openai/<id>" — llama.cpp's llama-server does
                                                     # NOT exact-match this field when one model is
                                                     # loaded, so any string works; override via
                                                     # OPENHANDS_MODEL if a future multi-model backend
                                                     # (vLLM) needs an exact match.
    base_url="http://127.0.0.1:8033/v1",           # tools/qwen_serve.sh's default HOSTADDR:PORT
    api_key="local-no-key-required",               # llama.cpp needs no auth; litellm needs a non-empty string
    num_retries=3, retry_min_wait=5, retry_max_wait=30,   # a short safety-net, NOT the offload mechanism (see §4)
)
condenser = LLMSummarizingCondenser(llm=llm.model_copy(update={"usage_id": "condenser"}),
                                    max_size=120, keep_first=6)
agent = Agent(llm=llm, tools=[Tool(name=t) for t in
                              ("terminal", "file_editor", "task_tracker", "grep", "glob")],
             condenser=condenser)
conversation = Conversation(agent=agent, workspace=<repo root>,
                            persistence_dir="tools/openhands_proto/state/conversations",
                            conversation_id=<fixed uuid5, stable across restarts>)
```

Every value is env-overridable (`QWEN_BASE_URL`, `OPENHANDS_MODEL`, `OPENHANDS_WORKSPACE`,
`OPENHANDS_CONVO_ID`, the condenser sizes, the retry knobs) — see the module docstring/top for the
full list. The `conversation_id` is a **fixed** `uuid5` derived from a constant name, not a fresh
`uuid4()` each run: that single line is what makes every invocation resume the *same* persisted
session instead of starting cold, which is the entire point of this evaluation.

**Tolerating the endpoint being down (the owner's explicit design constraint).** `qwen_serve.sh`'s
endpoint is only up while the model is loaded — it comes down whenever a GPU research job needs the
card. Two independent layers handle this, mirroring how `tools/hermes/loop.py` already solves the
identical problem for Hermes:

1. **Construction never touches the network.** Building `LLM`/`Agent`/`Conversation`/the condenser
   is pure object construction — verified empirically (§4) with qwen down throughout. Only
   `conversation.run()` calls the network.
2. **The wrapper (not the SDK's retry budget) owns the wait.** `run_turn.py` health-checks
   `GET /health` before doing anything and refuses fast (exit 1, no hang, no retry storm) if the
   endpoint isn't up. `openhands_loop.py` is a continuous version of the same idea: it never fires a
   turn while `qwen_up()` is false or a GPU job is in flight. The `num_retries=3` on the `LLM` object
   is a small safety net for a sub-minute race (e.g. the health check passes seconds before qwen
   actually finishes loading), not the mechanism that survives a multi-hour experiment — that's
   architectural, not a timeout value.

## 4. Offload-wrapper design: how the supervisor wraps OpenHands unchanged

`tools/openhands_proto/openhands_loop.py` is a **design artifact + working script, not a live
service** (no systemd unit was installed — see §0). It ports the four gpu-handoff functions from
`tools/hermes/loop.py` (`gpu_busy`, `_running_job`, `dispatcher_alive`, `vram_handoff`,
`qwen_up`/`qwen_down_cmd`/`qwen_up_cmd`) essentially verbatim, because that logic is already hardened
against real incidents this project has hit (stale `gpu.running`, a dispatcher dying mid-job, a hung
`nvidia-smi` during a GPU-crash) and there is no reason to re-derive it for a second harness. The
**only** structural change from Hermes' loop: where `loop.py` fires a brand-new webui/gateway session
per turn (`fire_turn()`/`poll_run()`, losing all prior context by design), this loop calls
`conversation.send_message() + conversation.run()` on the **one** persisted `Conversation` object —
so the model actually sees its own prior turns (condensed once the history grows), not a
freshly-primed context every time.

```
while true:
    if not loop_active(): sleep; continue                 # own sentinel, never Hermes' HERMES_ACTIVE
    if gpu_busy(): vram_handoff(); continue                # identical invariant to loop.py: never reload
                                                             # qwen while any GPU job holds the card
    if not qwen_up(): qwen_up_cmd()                        # ports tools/qwen_serve.sh up
    if gpu_busy(): continue                                # a job may have queued during the load
    conversation.send_message(TURN_PROMPT); conversation.run()   # the ONE persisted session, not a fresh one
```

This is exactly the "OpenHands drives the model; the supervisor owns VRAM — they're orthogonal" split
the owner specified: `openhands-sdk` never touches `nvidia-smi`, `gpu_queue.sh`, or `qwen_serve.sh`
itself; the wrapper never touches the `Conversation` object's internals. Swapping harnesses later
(back to Hermes, or to something else) only means swapping what sits inside the `while gpu is free`
branch — the VRAM lifecycle code is untouched either way, which is the reuse the 2026-09-06 stack
review already called out ("Offload is ORTHOGONAL to the harness").

**⛔ Not to be run alongside `hermes-loop`.** Both `tools/qwen_serve.sh` and `tools/gpu_queue.sh` are
singleton, shared-repo resources — two independent processes both deciding when to load/unload the
one local model is the exact double-load race `qwen_serve.sh`'s own comments warn about. This
prototype's loop uses its own sentinel (`tools/openhands_proto/state/OPENHANDS_LOOP_ACTIVE`, distinct
from `research/queue/HERMES_ACTIVE`) precisely so it can never be *mistaken* for the live Hermes loop,
but that only prevents confusion, not the race — evaluating this live means stopping `hermes-loop`
first (`systemctl --user stop hermes-loop`), not running both.

**A real, separate gap this surfaced, out of scope to fix here.** `tools/gpu_queue.sh` resolves a
shared root across git worktrees (`git rev-parse --path-format=absolute --git-common-dir`, its own
header explains why: two worktrees must never run two dispatcher daemons against the one physical
3090). `tools/qwen_serve.sh` has **no equivalent resolution** — it derives its PID-file/log directory
from its own script path's parent only. Invoking a worktree's copy of `qwen_serve.sh` would track a
*different* `qwen_server.pid` than the canonical checkout's Hermes loop uses, which is a latent
double-launch risk if anything (an agent, a stray script) ever ran it from a worktree by accident.
`openhands_loop.py` works around this by always resolving `SERVE` through the same shared-root logic
`gpu_queue.sh` uses, but the underlying gap is in `qwen_serve.sh` itself and applies regardless of
which harness calls it. Flagged, not fixed, here (see the spawned follow-up task).

## 5. Validated without the GPU vs. the live test

**Validated (all pass — `tools/openhands_proto/validate_offline.py`, 9/9, reproducible any time,
qwen state irrelevant):**

- SDK imports (`openhands-sdk` 1.44.1, `openhands-tools` 1.44.1).
- `TerminalExecutor` runs a real shell command in this repo (`terminal_type="subprocess"`, no `tmux`
  needed) — confirms shell+git tool access works standalone, with no LLM involved.
- `FileEditorExecutor` reads a real file (`HERMES.md`) in this repo standalone.
- `LLM`/`Agent`/`LLMSummarizingCondenser`/`Conversation` all construct against the (currently-down)
  `http://127.0.0.1:8033/v1` endpoint with **zero network calls** at construction time.
- `Conversation` persistence: constructing with a fixed `(persistence_dir, conversation_id)` writes
  `base_state.json` capturing the full agent/LLM/condenser config; reconstructing fresh Python objects
  against the same id/dir resumes correctly (`ConversationState.create()`'s open-or-create semantics,
  confirmed by reading the persisted JSON, not just trusting the docs).
- `qwen_up()` correctly reports `False` right now (a real research run holds the GPU) — fails fast via
  `curl`-equivalent, no hang.
- Read-only GPU-queue introspection (`openhands_loop.py`'s ported functions) correctly resolves the
  shared queue root and reports the REAL live state: dispatcher alive, one job running
  (`_rank2_integrated_loop_webapp_thread_derisk`), queue depth 0 — proving the offload-awareness logic
  reads ground truth correctly without needing to actually call `qwen_serve.sh up/down`.

**NOT validated (needs the live model — qwen is offloaded to the research run above right now):**
an actual LLM call, tool-call parsing/execution through the agent loop (native tool calling vs.
prompt-based, whichever Qwen3.8-27B needs), the condenser actually firing and summarizing once history
grows, and the real offload/reload cycle end-to-end (kill mid-turn, does resume genuinely pick the
conversation back up).

### Exact commands for the controller once the GPU frees + qwen (or vLLM-Q4) is up

```bash
# 1. Confirm the endpoint is actually reachable (llama.cpp target, from tools/qwen_serve.sh, or a
#    future vLLM endpoint on the same port/convention):
curl -sf http://127.0.0.1:8033/health && echo UP

# 2. One real turn on a REAL repo task, single process, from the canonical checkout (not a worktree —
#    see the qwen_serve.sh gap in §4):
cd /home/dant123/Projects/sim
tools/openhands_proto/.venv/bin/python tools/openhands_proto/run_turn.py \
    --prompt "Read HERMES.md and research/coordination/live_state.md, then summarize the current \
frontier in 5 bullet points into /tmp/openhands_smoke_test.md. Do not modify any repo file."

# 3. Confirm the ONE-CONTINUOUS-SESSION property: run a SECOND turn with no prompt override (sends
#    agent_config.TURN_PROMPT) and confirm the transcript shows the agent referencing what it did in
#    turn 2, not starting cold — this is the actual claim under test, not just "it produced text":
tools/openhands_proto/.venv/bin/python tools/openhands_proto/run_turn.py \
    --prompt "What did you conclude in your previous turn? Answer in one sentence, make no other tool calls."

# 4. Inspect the persisted event history directly (confirms both turns are one conversation, not two):
ls tools/openhands_proto/state/conversations/*/events/ | wc -l   # should show >1 event group across both turns

# 5. (Only if 1-4 look right) pilot the offload cycle for real, WITH hermes-loop stopped first:
systemctl --user stop hermes-loop
touch tools/openhands_proto/state/OPENHANDS_LOOP_ACTIVE
tools/openhands_proto/.venv/bin/python tools/openhands_proto/openhands_loop.py
# ... enqueue a short GPU job in another terminal (bash tools/gpu_queue.sh add '<cheap cmd>'), watch
# the loop's log (tools/openhands_proto/state/openhands_loop.log) unload/reload qwen around it, then:
rm tools/openhands_proto/state/OPENHANDS_LOOP_ACTIVE      # stop the loop
systemctl --user start hermes-loop                        # restore the real driver
```

## 6. Honest fit assessment

**Delivers the one-continuous-session UX the owner asked for, mechanically — this is the strongest
finding here.** `Conversation(persistence_dir=..., conversation_id=...)` plus
`LLMSummarizingCondenser` is exactly the "linear-cost summarization + resume" pair the 2026-09-06
stack review named as the draw, and it is real, shipped, actively-maintained code (not a roadmap
promise) — verified by reading the persisted JSON, not by trusting the docs. Hermes v2's
fresh-session-per-turn design cannot produce this property by construction; OpenHands's can, and does,
on this box, offline.

**The gaps are real and worth naming plainly:**

- **The live model call is genuinely untested.** Everything short of the network hop passes; the
  network hop is the one thing that needed the GPU, which this task correctly did not touch. §5 hands
  the controller the exact next step — this is not "should work," it's "here is the one remaining
  check."
- **"OpenHands" the brand has moved past the artifact we want.** Anyone reading current OpenHands
  marketing lands on Agent Canvas (Node/npm, multi-agent UI) or the deprecated CLI, not the SDK. This
  is a discoverability trap the owner should know about before pointing anyone else at "OpenHands
  docs" — the actual integration point is `openhands-sdk`'s Python API, documented separately from
  the flagship product.
- **Tool-calling fidelity with Qwen3.8-27B at Q2/Q4 is unknown.** `openhands-sdk`'s default preset
  assumes reasonably strong native or prompted tool-calling; the 2026-09-06 stack review already
  flagged that no committed eval exists of the current Q2 model's dev-decision quality at all,
  independent of harness. A harness swap does not resolve that open question — it inherits it.
- **`qwen_serve.sh`'s worktree-unsafe path resolution (§4)** is a small, separate, real gap that
  applies to Hermes too, not something this prototype introduced — flagged via a background task
  rather than fixed inline, since fixing shared offload infra is outside this evaluation's footprint.

**Recommendation:** proceed to the live test in §5 next session (or whenever the GPU next frees
naturally — no need to preempt the running research job for this). If turns 2-3 in §5 confirm the
model genuinely references its own prior turn (not just "produces plausible text again"), and
tool-call parsing holds up on a real edit+commit task, this is a credible harness to migrate toward per
the 2026-09-06 stack review's step 2 ("prototype OpenHands on the local endpoint + wrap our offload
supervisor") — with the caveat that the actual cutover (retiring `hermes-loop`, the webui, the gateway)
is a separate, larger decision the owner should make deliberately once the live test lands, not a
mechanical follow-on from this prototype passing.
