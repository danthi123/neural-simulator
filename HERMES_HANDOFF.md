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

## Takeover / hand-back
- **To Hermes** (Claude usage out): `bash tools/hermes_takeover.sh on` → then work in `hermes`.
- **Back to Claude** (usage reset): `bash tools/hermes_takeover.sh off` → Qwen unloads, GPU frees for research;
  then resume Claude-side compute (`bash tools/gpu_queue.sh resume` if it was paused). Tell Claude "continue" — it
  re-anchors from `research/coordination/live_state.md` + `GAP_CLOSURE_MISSION.md` and judges/continues Hermes' work.

## Workflow parity (so Hermes works with the same discipline)
Hermes must obey the same non-negotiables as Claude — see **`CLAUDE.md`** + the CONSTRAINTS block in
`research/coordination/live_state.md` (brain-based-only · one-brain · no-defer · 6-seed · gates authoritative ·
commit BOTH remotes via `tools/push_both.sh` · cost-routing). The gate system is **automatic for any agent**: the
gates run as the git **pre-commit hook** (`tools/githooks/`), so Hermes' commits are gated exactly like Claude's.
The Claude-Code-specific layer (PostToolUse hooks, the heartbeat, skills) is translated to Hermes' hooks/cron/skills
in **`docs/HERMES_WORKFLOW_PARITY.md`**; verify it with **`bash tools/hermes_parity_check.sh`**.
