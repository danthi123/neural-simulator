# HERMES HANDOFF — running this project on local Qwen when Claude usage is out

This is the operating manual for **Hermes** (Nous Research's open agent, installed at `~/.hermes`) driving this
repo on a **local Qwen3.8-27B** brain, for when Claude usage is exhausted. Claude (in the Claude Desktop app) and
Hermes are **separate drivers — run ONE at a time**; `tools/hermes_takeover.sh` is the handoff switch.

## The model (per the sdkyuan card + DFlash2)
- **Target:** `sdkyuan/qwen3.8-27B-qat-q2_0-gguf` (8.76 GB, Q2_0, 27B).
- **Speculative drafter:** `HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF:Q2_K_S` (535 MiB, a DFlash2 block-diffusion
  drafter that speeds up the target). Both **auto-download** on first launch via llama.cpp `-hf`/`-hfd`.
- **Server:** `~/.unsloth/llama.cpp/llama-server` (already has `--spec-type draft-dflash`; no rebuild needed),
  OpenAI-compatible at `http://127.0.0.1:8033/v1`.

## The pieces (all in `tools/`)
| Script | What it does |
| --- | --- |
| `qwen_serve.sh {up\|down\|status\|restart}` | Launch/stop the target+DFlash2 server. First `up` triggers the ~9 GB download. |
| `qwen_supervisor.sh` (systemd `qwen-supervisor.service`) | The VRAM dance — see below. INERT unless `HERMES_ACTIVE`. |
| `hermes_takeover.sh {on\|off\|status}` | **The owner's one command.** `on` = Hermes drives; `off` = back to Claude. |
| `hermes_local_setup.sh` | One-time: point Hermes at the local endpoint (backs up your Hermes config). |
| `hermes_gpu_run.sh "<cmd>"` | The ONE way Hermes launches a **local GPU** job (see "How Hermes runs GPU work"). |

## The VRAM unload/reload design (the crux)
**Invariant: a local GPU job and the Qwen server never co-reside.** The supervisor (a systemd user service,
polling every 8 s) enforces it, and is **completely inert while Claude drives** (Qwen stays down, GPU free for
research). When `HERMES_ACTIVE` is set (via `hermes_takeover.sh on`):
- **Local GPU job queued/running** → unload Qwen (free the whole card for the run).
- **Local queue idle** → reload Qwen, then **nudge Hermes** (`hermes -z …`) to come back and read results + continue.
- **`GAME_MODE`** (from `tools/game.sh on`) → keep Qwen down (owner wants the GPU). Absolute priority.
- **Mini-PC pool runs never trigger it** — they're remote and don't contend with the local GPU.

So Hermes' own async loop is: launch a GPU job → end its turn → (Qwen unloads, job runs, Qwen reloads) → Hermes is
re-invoked to harvest. Same pattern Claude uses.

## How Hermes runs GPU work
Hermes must launch **local GPU** experiments only via:
```bash
bash tools/hermes_gpu_run.sh "SIM_BACKEND=cupy .venv/bin/python -m research.runners.X --seed 42 --out research/findings/raw/..."
```
It enqueues to the shared `gpu_queue`; the supervisor frees Qwen's VRAM so the job gets the whole card, then
reloads Qwen and re-invokes Hermes. **CPU / mini-PC pool work does not use this** (it never contends): use
`tools/sweep_pool.sh` as usual. Hermes must NOT run `SIM_BACKEND=cupy python …` directly (that would fight its own
brain for VRAM).

## FIRST-TIME SETUP (do these once, when the GPU is free — needs the download + a live launch)
1. `bash tools/hermes_takeover.sh on` — sets `HERMES_ACTIVE`, starts the supervisor, brings Qwen up (first run
   downloads ~9 GB; watch `research/queue/qwen_server.log`).
2. `bash tools/hermes_local_setup.sh` — points Hermes at `http://127.0.0.1:8033/v1` (or run `hermes setup` and
   enter that base URL + an `sk-…` key; Hermes auto-detects the model).
3. Smoke test: `hermes -z "print the current git SHA and the top of research/coordination/live_state.md"` — confirms
   Hermes is talking to local Qwen and can use its tools.
4. Confirm the VRAM dance: `bash tools/hermes_gpu_run.sh "SIM_BACKEND=cupy .venv/bin/python -c 'import cupy; print(cupy.zeros(3))'"`
   then watch `research/queue/qwen_supervisor.log` — Qwen should unload, the job run, Qwen reload.

## Pause / resume for gaming or a break (the important one)
**Closing Hermes is NOT a pause** — while `HERMES_ACTIVE` is set, the supervisor will happily reload Qwen and
nudge a fresh Hermes turn, so it would "spin back up" on you. To actually take a break, use the pause button —
either run it yourself or just tell Hermes ("pause for a break" / "I'm going to game") and it will run it:
```bash
bash tools/game.sh on      # frees the local GPU + CPU NOW: pauses local runs (current requeues, no loss),
                           #   unloads Qwen, and STAYS down — nothing auto-restarts until you resume.
bash tools/game.sh off     # resume: local runs pick back up; if Hermes is the driver, Qwen reloads within ~8s
bash tools/game.sh status  # paused or running?
```
Guarantees: `game.sh on` sets a **persistent** `GAME_MODE` sentinel — so a **reboot mid-break stays paused**, and
a reboot when you're **not** paused resumes development automatically (the services are enabled). The **mini-PC
pool keeps running** either way (it's remote, doesn't touch your local GPU/CPU). This is the same button whether
Claude or Hermes is driving.

## Takeover / hand-back
- **To Hermes** (Claude usage out): `bash tools/hermes_takeover.sh on` → then work in `hermes`. This now ALSO turns
  on autonomous mode (see below) — Hermes works hands-off from the moment it becomes the driver.
- **Back to Claude** (usage reset): `bash tools/hermes_takeover.sh off` → Qwen unloads, GPU frees for research,
  autonomous mode pauses; then resume Claude-side compute (`bash tools/gpu_queue.sh resume` if it was paused). Tell
  Claude "continue" — it re-anchors from `research/coordination/live_state.md` + `GAP_CLOSURE_MISSION.md` and
  judges/continues Hermes' work.

## AUTONOMOUS MODE — the default while Hermes drives

**Goal**: Hermes works overnight/hands-off, Claude-bypass-permissions style — the `approvals` system still prompts
for genuinely dangerous commands (never bypassed with `--yolo`), but ordinary dev commands never interrupt it, and
the 15-minute heartbeat cron IS the autonomous loop (it re-injects the parallelism/lanes/frontier audit and its own
prompt tells Hermes to act on it, never end a turn on a status report alone).

**The switch**: `bash tools/hermes_autonomous.sh {on|off|status}` — `on` ensures `hermes gateway` (the process that
hosts Hermes's built-in cron ticker — confirmed against the installed Hermes's own test suite: the ticker only runs
inside the gateway process) is installed (first time; a **user-level systemd service, no sudo**) and running, then
resumes the `sim-heartbeat` cron job; `off` pauses the job (gateway stays up, harmless idle); `status` shows both.
Already wired into the driver switches: `hermes_takeover.sh on/off` call it automatically, and `game.sh on/off`
pause/resume it around a gaming break (belt-and-suspenders alongside `sim_heartbeat.sh`'s own `HERMES_ACTIVE`/
`GAME_MODE` gate — a stray tick during a pause is a safe no-op either way).

**Queueing feedback without interrupting**: `bash tools/hermes_say.sh "<feedback>"` appends a timestamped line to
`research/coordination/.hermes_feedback_queue`; the `pre_llm_call` hook drains it and injects it into Hermes's next
turn's context automatically, exactly once. No `hermes -z` interrupt, no context switch mid-task.

**The desktop control panel** (`~/Desktop/hermes-sim.sh` — copy it there once from the repo's tracked original):

```bash
cp /home/dant123/Projects/sim/tools/hermes_desktop_control.sh ~/Desktop/hermes-sim.sh
chmod +x ~/Desktop/hermes-sim.sh
```

| Command | Does |
| --- | --- |
| `~/Desktop/hermes-sim.sh start` | **The one-command go-live**, safe from ANY starting state (idempotent): clears any earlier pause (`gpu_queue resume` + `GAME_MODE`/`GPU_PAUSE` off via `game.sh off`), hands the project to Hermes, confirms autonomous mode, brings Qwen up. Prints a clear ✓/✗ per step. |
| `~/Desktop/hermes-sim.sh stop` / `resume` | `game.sh on` / `off` — pause/resume for gaming or a break; pauses/resumes the autonomous cron too. |
| `~/Desktop/hermes-sim.sh handback` | `hermes_takeover.sh off` — back to Claude. |
| `~/Desktop/hermes-sim.sh status` | One screen: driver, Qwen, supervisor, autonomous cron, GAME_MODE. |
| `~/Desktop/hermes-sim.sh check` | **Post-reboot / post-system-update health gate** — run before trusting an overnight run after a CachyOS update or any reboot. Read-only, green/red per line (see below). |
| `~/Desktop/hermes-sim.sh say "<feedback>"` | `hermes_say.sh` — queue a note for Hermes without interrupting it. |
| `~/Desktop/hermes-sim.sh logs` | Tail the Qwen/supervisor/autonomous/cron logs. |

### Reboot-resilience — verified against the real installed services, not assumed

- **`qwen-supervisor.service`** — already a `systemctl --user` unit, `enabled` + `WantedBy=default.target`, and
  `loginctl show-user` on this box reports **`Linger=yes`** — confirmed directly, not inferred. Linger is what makes
  a user-level service start **at boot without a login**; without it a user unit only starts at the next interactive
  login. **Verdict: survives reboot automatically, no owner action.**
- **`hermes gateway install`** (no `--system`) is also a **user-level** systemd unit, and its own installer
  auto-enables linger if it's somehow off (verified by reading `hermes_cli/gateway.py`'s `_preflight_user_systemd`).
  Read directly from source: in a non-interactive context (or accepting the default prompt) it installs with
  `enable_on_startup=True` and starts immediately — i.e. plain `hermes gateway install` already behaves like
  `--start-now --start-on-login` by default; `hermes_autonomous.sh` passes those flags explicitly anyway, for
  certainty rather than reliance on a default. **Verdict: once installed, survives reboot automatically** (same
  linger mechanism as the supervisor).
- **The `sim-heartbeat` cron job** persists in Hermes's own state once created (`hermes-parity/apply_cron.sh`) —
  reboot does not remove it, only PAUSING it does (`hermes cron pause`).
- **So: which is true, "auto-resumes" or "run start once"?** Both, depending on what's ALREADY true when the
  reboot happens: **IF** `research/queue/HERMES_ACTIVE` is set, `GAME_MODE`/`GPU_PAUSE` are absent, the gateway was
  already installed, and the cron job was not paused — **autonomous work resumes automatically** with zero owner
  action (both services restart via linger, the supervisor's own poll loop re-evaluates state and reloads Qwen, the
  gateway's ticker resumes firing the already-scheduled job). **BUT** `GAME_MODE`/`GPU_PAUSE` are themselves
  **persistent sentinel files by design** (`tools/game.sh`'s own contract: "a reboot mid-break stays paused") — so
  a reboot that happens while paused for a test/break stays paused, correctly, until something clears it. **For a
  reboot you are about to trigger yourself** (e.g. after a system update), the reliable, idempotent move is: run
  `~/Desktop/hermes-sim.sh start` once, after the reboot — it clears any stale pause and re-confirms every piece
  regardless of what state it finds, so it is correct whether or not auto-resume already happened.
- **After the reboot**: the mini-PC **pool** keeps running (it's remote, on separate machines, entirely unaffected
  by this box rebooting). Any **local GPU** job that was mid-run when the box went down is **requeued intact** by
  `tools/gpu_queue.sh`'s own design (a killed/interrupted job re-runs from the front of the queue on restart) — at
  most that job's in-progress work is lost, never the job itself.

### Post-reboot / post-CachyOS-update health check

`bash tools/hermes_health_check.sh` (or `~/Desktop/hermes-sim.sh check`) — read-only, no GPU load, safe to run any
time. Checks, green/red: llama.cpp present **and still built with `--spec-type draft-dflash`** (a system package
update can silently replace it with a build lacking the flag — `qwen_serve.sh` already refuses to launch in that
case; this surfaces the same check proactively, before you try), the local target GGUF still present, `nvidia-smi`
responsive, `systemd` linger + `qwen-supervisor.service` + `hermes gateway` all active, the `pre_llm_call` hook
registered **and allowlisted** (`hermes hooks doctor` — a hook can be registered but not yet allowlisted if the
gateway has never run a session since the hooks were configured; the check tells the two states apart rather than
flagging the expected-pending case as a failure), the git pre-commit gate (`tools/hermes_parity_check.sh`), and —
only if Qwen is currently up — the local endpoint's reachability.

### Two-driver etiquette while autonomous mode is on

Hermes is the **sole active driver** during autonomous mode. If a Claude session is also open (e.g. for QA/review
while Claude usage isn't the blocker), it should stay **read-mostly** — read `live_state.md` / `GAP_CLOSURE_MISSION.md`
to see what Hermes did and judge it, but not take actions that would race Hermes's own. `hermes_takeover.sh on`
prints this reminder every time it hands over.

## Workflow parity (so Hermes works with the same discipline)
Hermes must obey the same non-negotiables as Claude — see **`CLAUDE.md`** + the CONSTRAINTS block in
`research/coordination/live_state.md` (brain-based-only · one-brain · no-defer · 6-seed · gates authoritative ·
commit BOTH remotes via `tools/push_both.sh` · cost-routing). The gate system is **automatic for any agent**: the
gates run as the git **pre-commit hook** (`tools/githooks/`), so Hermes' commits are gated exactly like Claude's.
The Claude-Code-specific layer (PostToolUse hooks, the heartbeat, skills) is translated to Hermes' hooks/cron/skills
in **`docs/HERMES_WORKFLOW_PARITY.md`**; verify it with **`bash tools/hermes_parity_check.sh`**.

## hermes-webui — watch AND drive the autonomous session from a browser (anywhere)
Installed at `~/hermes-webui` (nesquena/hermes-webui), runs as the reboot-persistent user service
`hermes-webui.service` (secret-free repo template: `tools/systemd/hermes-webui.service`). A live-streaming
(SSE) UI — thinking, tool cards, context usage — that reads `~/.hermes` so it uses the same local Qwen.
Manage: `~/hermes-webui/ctl.sh {start|stop|status|logs}` or `systemctl --user {status|restart} hermes-webui`.

**Reachable THREE ways (configured + verified 2026-08-30), one exposed surface, password-gated:**
- **Local** — http://127.0.0.1:8787
- **LAN** — `http://<this-desktop-LAN-IP>:8787` (currently `192.168.0.68:8787`; open it from a phone on the same wifi)
- **External** — https://hermes.dant123.com (homelab caddy reverse-proxies to this host:8787)

The webui binds `0.0.0.0:8787` and is the ONLY network-exposed surface. The gateway HTTP API
(`127.0.0.1:8642`) and Qwen (`127.0.0.1:8033`) stay **localhost-only** — never exposed. The login password
lives only in the LIVE unit (`~/.config/systemd/user/hermes-webui.service`, `HERMES_WEBUI_PASSWORD=`); change
it anytime in the webui **Settings**, or edit that line + `systemctl --user daemon-reload && restart`.

**Caddy snippet** (owner's homelab caddy, not in this repo — Caddy v2 handles SSE/WebSocket automatically):
```
hermes.dant123.com {
    reverse_proxy 192.168.0.68:8787
}
```
Caddy v2 passes the original `Host` upstream and sets `X-Forwarded-Proto`/`X-Forwarded-Host` by default; the
service trusts those (`HERMES_WEBUI_TRUST_FORWARDED_HOST/PROTO=1`) so the `hermes.dant123.com` origin passes
the CSRF gate and cookies are `Secure` over https — while plain-http LAN access still works (the Secure flag is
decided per-request). If caddy is set to rewrite the upstream Host, the explicit
`HERMES_WEBUI_ALLOWED_ORIGINS=https://hermes.dant123.com` allowlist still lets it through.

**Option C — the webui drives the gateway agent directly.** Chat routes through the gateway HTTP API
(`HERMES_WEBUI_CHAT_BACKEND=gateway`, `..._GATEWAY_BASE_URL=http://127.0.0.1:8642`,
`..._GATEWAY_USE_RUNS_API=true`), enabled by `API_SERVER_KEY` (+ `API_SERVER_HOST=127.0.0.1`) in
`~/.hermes/.env`. So sending a message in the webui starts a self-continuing gateway RUN on the same Qwen +
repo + `live_state.md` + memory — you watch it stream live (thinking, tool cards) and interject via steer, from
anywhere; the run lives on the gateway server, so it keeps going if you close the browser and you reattach to
watch. Verify: authenticate, then `curl -b <cookies> http://127.0.0.1:8787/api/health/agent` →
`gateway_chat.enabled: true` (end-to-end proven: a `POST /v1/runs` returned the agent's reply).

**OPERATING MODEL (owner, 2026-08-30) — the webui `/goal` loop drives, fully visible, controlled by chat. NO
headless cron.** The owner switches to Hermes whenever they want (not tied to Claude usage) by driving one webui
session:
- **Start working:** in a webui session, set a standing goal — `/goal Continuously advance the neural-simulator
  mission. Each turn read research/coordination/live_state.md and do the next concrete action from CURRENT STATE;
  commit via tools/push_both.sh. Never consider this complete; keep going until I say stop.` Hermes then works
  turn-after-turn AUTONOMOUSLY and VISIBLY (thinking + tool cards stream; "↻ Continuing toward goal (N/500)").
- **Control by chat:** `/goal clear` stops · `/goal resume` continues after a pause · just typing steers/interjects.
- **Hand back to Claude / pause:** tell it to stop (`/goal clear`); for the GPU too, `bash tools/game.sh on`.
- **Switch back to Hermes later:** in the same session, `/goal resume` (or set the goal again). Continuity is via
  DURABLE STATE (`live_state.md` + repo + memory), not the transcript — so "pick up where Claude left off" works
  because Hermes reads the current `live_state.md`; no session-import needed.

Config that makes this work (all set 2026-08-30): `HERMES_WEBUI_DEFAULT_WORKSPACE=/home/dant123/Projects/sim` (agent
operates IN the repo), `agent.reasoning_effort: xhigh` + `model.reasoning: xhigh`, `goals.max_turns: 500` (long
autonomous stretches before an auto-pause), all in `~/.hermes/config.yaml` / the webui unit. The `sim-heartbeat`
cron is **paused/retired** for this model (`bash tools/hermes_autonomous.sh off`) — its `deliver:local` turns were
never viewable in the webui anyway. Gateway + Qwen stay up so the webui is always ready.

**RESEARCH-LOOP MODE (owner, 2026-08-30) — autonomous GPU-interleaved research, VISIBLE + engageable in the
webui.** For runs that need the whole GPU (Qwen 20 GB XOR a sweep — never both on one card), the VRAM supervisor
(`tools/qwen_supervisor.sh`) runs the loop: `[Qwen up: Hermes harvests the last run → decides → edits → launches the
next via tools/hermes_gpu_run.sh → commits] → [Qwen down: the GPU run executes] → [Qwen reloads: the supervisor
fires the next turn]`. The between-runs turn is fired INTO one persistent webui conversation titled
**"🤖 Autonomous research loop"** (via `tools/hermes/webui_continue.py` → `/api/chat/start` on a reused session id
in `research/queue/.hermes_webui_session_id`), so you **watch every harvest/decide turn stream there and type into
the same conversation to steer** (falls back to headless `hermes -z` only if the webui is unreachable). Toggle the
visible path with `HERMES_CONTINUE_VIA_WEBUI=1` (default on).
- **Start it:** either restore the deferred research queue (`cat research/queue/gpu.queue.deferred.* >> research/queue/gpu.queue`)
  so the first run completes and fires the first visible harvest turn, OR open the "🤖 Autonomous research loop"
  session and type `Start the loop: read live_state.md, launch the next run, keep going`. It then self-sustains
  (each turn launches the next run; each completion fires the next turn).
- **Monitor:** open that session — turns stream in between runs. **Engage:** type in it; messages land during the
  "Qwen up" windows (Qwen is down mid-run, so it responds between runs). **Pause:** `bash tools/game.sh on` (frees
  the GPU immediately) or tell it to stop launching runs (the queue drains → the loop ends).
- Trade-off vs the dev `/goal` mode above: in research mode Qwen cycles (responsive between runs, not during); in
  dev mode Qwen is pinned (always responsive) but no local GPU sweeps run. Pick per session. The mini-PC **pool
  (CPU)** runs in parallel in either mode (never touches the GPU).
