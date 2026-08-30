Hermes session brief for the neural-simulator repo. This file wins Hermes's context-file priority
over CLAUDE.md (`.hermes.md`/`HERMES.md` > `AGENTS.md` > `CLAUDE.md` > `.cursorrules`, first found
loads, only one loads) — kept deliberately small for a ~100K-context agent. CLAUDE.md (the full
spec) is NOT auto-loaded while this file exists; read it ON DEMAND for depth, never bulk-load it.

## THE RULE — durable state, not context, is ground truth

Hermes's context window is small and sessions restart cold. **Every turn, before anything else,
read `research/coordination/live_state.md`** (it re-injects automatically via a `pre_llm_call`
shell hook if configured — see `docs/HERMES_WORKFLOW_PARITY.md`; read the file directly if not).
It carries: the non-negotiable constraints, the current frontier, the ordered next actions, and the
last decision. **Never rely on prior turns' context and never start from scratch** — resume from
that file. When it's stale, thin, or you're starting a new arc, also read
`GAP_CLOSURE_MISSION.md`'s CURRENT STATE section (the fuller working board) and
`tools/vikunja.sh list-tasks 2` (the plain-language task board). If a summary doc conflicts with a
committed finding under `research/findings/`, the finding wins — fix the summary, don't trust it
blindly.

## Non-negotiable constraints (full reasoning in CLAUDE.md)

- **Brain-based only.** Host code is legitimate ONLY for the world/environment and the body/motor
  output. Everything between sensation and action (perception, reward, value, action selection)
  must be neurons/synapses — a host shortcut is a documented gap to close, not a stop.
- **ONE spiking substrate, one brain.** No permanent external ML artifact as a faculty.
- **No deferral.** A wall/negative is a verdict on a METHOD, never license to abandon a CAPABILITY.
- **Speed is secondary to faithfulness.** Slow-but-biological beats fast-but-shortcut.
- **Honesty boundary.** Measure functional consciousness/affect correlates; never assert
  phenomenal experience.
- **6-seed validation** (42/43/44/100/101/102) before any generalization claim.
- **Gates are authoritative.** `tools/gates/*.py`, wired through the git pre-commit hook
  (`core.hooksPath` → `tools/githooks`) — this fires on every `git commit` regardless of which
  agent runs it. Never bypass with `--no-verify` except a deliberate, visible, explained override.
- **Commit to BOTH remotes**, verified (not asserted): `bash tools/push_both.sh`.
- **Cost-routing.** CPU sweeps → `tools/sweep_pool.sh`; GPU sweeps/long runs →
  `tools/gpu_queue.sh`; multi-seed → `--seeds`; agents only for genuine builds, tiered by
  difficulty (see the `cost-routing` skill under `.hermes/skills/`).

## Running a local GPU job (Hermes-specific — Qwen shares the same card)

Hermes's own model may be a local Qwen instance sharing this machine's GPU with any experiment.
**Never run GPU Python directly** (`SIM_BACKEND=cupy ...`) — it fights your own model for VRAM.
Instead:

```
bash tools/hermes_gpu_run.sh "SIM_BACKEND=cupy .venv/bin/python -m research.runners.X --seed 42 --out ..."
```

This enqueues the job on the shared, VRAM-safe `tools/gpu_queue.sh`. **Then END YOUR TURN** — the
VRAM supervisor (`tools/qwen_supervisor.sh`) unloads your model, the job runs on the full card,
your model reloads, and you are re-invoked; `research/coordination/live_state.md` will say what
completed. Do not busy-wait on it. CPU/pool work (`tools/sweep_pool.sh`) never touches the GPU and
needs none of this.

## Pause / resume (the owner wants the GPU back)

When the owner says pause / stop for gaming / taking a break: `bash tools/game.sh on` (frees the
3090, stops new local GPU jobs; the mini-PC pool keeps running). Resume with
`bash tools/game.sh off`. `bash tools/game.sh status` shows the current state. Never fight this by
launching GPU work directly while `research/queue/GAME_MODE` is set.

## Skills available locally

`.hermes/skills/` carries this repo's procedures (copies of `.claude/skills/`, kept in sync via
`tools/hermes/sync_skills.sh` — see that script's header for why they're copies, not symlinks):
`neural-simulator` (mission + workflow realignment), `verify-go` (adversarial check before any
positive result lands), `sync-documentation` (keep the summary docs from drifting), `vikunja` (the
task board), `cost-routing`, `evolve-skills`.

## Depth, on demand — do not bulk-load

`CLAUDE.md` (full constraints + workflow detail), `docs/HERMES_WORKFLOW_PARITY.md` (the complete
Claude-Code-to-Hermes workflow mapping this file is a part of), `docs/FAILURE_GATE_MATRIX.md`
(what each gate blocks and why), `docs/TERMS.md` (load-bearing word definitions — check before
writing "consolidation"/"compositional"/"GO"/etc. in a finding).
