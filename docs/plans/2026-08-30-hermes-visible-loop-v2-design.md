# Hermes visible autonomous research loop — v2 design (robust)

**Status:** design, pending owner review + deliberate implementation (2026-08-30). Supersedes the
v1 patch-pile (`tools/qwen_supervisor.sh` webui-drive + `tools/hermes/webui_continue.py`), which
worked in pieces but kept surfacing new failure modes because it was built incrementally around a
single ever-growing webui gateway-chat session.

## Goal (owner)

One place in the webui where Hermes drives the research **like Claude does** — reads state, starts
GPU runs, checks results, edits/commits, starts new runs — **visibly**, with Qwen auto-unloading /
reloading around GPU runs, and the owner able to watch and interject. Not headless.

## Failure modes observed in v1 (each a real, separate bug)

1. **Queue starvation.** SYSTEM-level (`sudo`) `gpu-queue-autofill`/`refill` timers (the four-day-queue
   campaign) kept the GPU queue full, so Qwen never reloaded for a turn. (User-level disable missed them.)
2. **Supervisor deadlock.** Qwen-loaded detection shelled to `nvidia-smi` (slow/flaky under GPU load)
   → skipped the unload → deadlock.
3. **Wrong session workspace.** `/api/session/new` ignores `HERMES_WEBUI_DEFAULT_WORKSPACE`; sessions
   defaulted to `~/workspace` → `read_file research/coordination/live_state.md` "File not found" → Hermes
   flailed. Also: even with the session workspace = repo, the **read_file tool's cwd ≠ repo** (terminal's
   cwd IS the repo). So relative reads are unreliable; absolute paths / terminal are required.
4. **Mid-turn Qwen unload cut the turn** (lost work). No reliable "turn is generating" signal:
   `/slots is_processing` is stuck true, `/metrics` is off, `pre_llm_call` hook doesn't fire for
   gateway runs, `/api/chat/stream/status` doesn't reflect gateway-run streaming.
5. **Stale-stream jams.** An interrupted/errored turn leaves an uncleanable `active_stream_id`; every
   `/api/chat/start` then 409s → session permanently jammed. "Skip on 409" avoids killing a *live*
   turn but then jams forever on a *dead* one — no local signal distinguishes the two.
6. **Cognitive-turn dead-end.** A turn that launches no GPU run left nothing to trigger the next.
7. **Context-length balloon (the decisive one).** A single long-lived session accumulates history +
   re-read files + large Q2 turns until the webui's auto-compaction fails
   (`Context length exceeded: max compression attempts (3) reached`) → the turn errors → (5) jams it.
   In ~2.5 h of "cycling" this produced **0 commits, 0 GPU runs** — alive but unproductive.
8. **Q2-at-xhigh is slow** — multi-minute, very large turns; this *amplifies* (7).

## Root insight

Driving an autonomous loop through **one ever-growing webui gateway-chat session** is the wrong
substrate: context accumulates (→7), interruptions jam it (→5), and there is no clean turn-active
signal (→4). The project's own principle already points the way: **durable state (`live_state.md` +
the repo), not conversation history, is ground truth.** So each turn should be **stateless and
re-anchored from durable state**, not dependent on a growing transcript.

## Proposed architecture (v2): stateless per-turn sessions, run-status–tracked

- **One turn = one fresh webui session**, created by the loop, titled e.g. `🤖 loop · turn N · HH:MM`.
  It re-reads `live_state.md` + the mission (absolute paths) and does ONE cycle: harvest the last
  run, decide, edit/commit, launch the next run, end. Because each session is short, it **never
  balloons** (kills 7), **never inherits a stale lock** (kills 5), and each turn's context is small
  (mitigates 8 — faster prompt processing).
- **Watchability:** the owner watches the *sequence* of turn-sessions in the sidebar (each = one
  visible cycle). This trades "one continuous conversation" for robustness. (Open decision below.)
- **Turn tracking (the reliable signal we lacked):** fire via the gateway **runs API** (`POST /v1/runs`
  returns a `run_id`); poll `GET /v1/runs/{run_id}` for terminal status. That is the clean "is the
  turn done?" signal — no `/slots`, no stream/status guessing. The loop fires the next turn only when
  the previous `run_id` is terminal (done/failed) — kills the cognitive dead-end (6) AND the
  turn-active ambiguity (4/5) without a stale-lock guessing game.
- **VRAM:** unchanged supervisor role — unload Qwen when the local GPU queue has a job (endpoint-detect,
  timeout-bounded), reload when idle. The queue-settle guard becomes unnecessary: a turn ends (run_id
  terminal) *before* the loop lets the queued GPU run take the card, so turns are never cut mid-flight.
- **The loop owns the GPU queue.** No competing autofill (kills 1) — the four-day-queue campaign is a
  separate mode, not co-run.
- **Context budget per turn:** the turn prompt is lean (a pointer to read live_state, not the whole
  mission inlined); rely on the agent reading what it needs. Absolute paths / terminal for file reads
  (works around 3).

## Loop (pseudocode)

```
while HERMES_ACTIVE and not GAME_MODE:
  wait_until(qwen_up and no_local_gpu_job)          # VRAM free for a turn
  sid  = new_session(workspace=REPO, title=...)     # fresh, bounded
  run  = POST /v1/runs {session_id: sid, input: TURN_PROMPT}   # or /api/chat/start; capture run_id
  wait_until(run terminal  OR  a GPU run got queued and settled)   # poll GET /v1/runs/{id}
  if a GPU run was launched:                          # let it have the card
     unload Qwen; wait_until(run done); reload Qwen
  # else purely cognitive turn — just continue
```

## Implementation plan (deliberate, not live-poked)

1. New `tools/hermes/loop.py` (single owner of the loop) replacing the supervisor's ad-hoc fire logic:
   run-status polling, fresh-session-per-turn, VRAM handoff. Supervisor keeps ONLY VRAM up/down.
2. Verify the runs-API path end-to-end on a scratch session (run_id lifecycle, terminal status).
3. Dry-run 3–5 cycles with a bounded turn budget, watched, before unattended.
4. Keep the four-day-queue campaign paused while the loop owns the GPU.

## Open decisions for the owner

- **Watchability:** per-turn sessions (max robustness, a sidebar *list* of cycles) vs a bounded
  session that starts fresh every K turns / at a context threshold (closer to "one conversation",
  slightly more complex). Recommendation: **per-turn** first (simplest + robust), revisit if the
  list is annoying.
- **Reasoning effort:** xhigh is faithful but slow (huge turns → context pressure). Consider `high`
  for the routine loop turns to cut turn size/latency; keep xhigh available.
```
