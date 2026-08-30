# Hermes parity — config snippets to apply

Everything here is a **file to review and apply by hand**; nothing in this directory was executed
against the owner's real `~/.hermes/` by the agent that produced it (that agent worked in an
isolated git worktree and was explicitly told not to touch the global Hermes home). Full reasoning
for every mapping lives in [`docs/HERMES_WORKFLOW_PARITY.md`](../docs/HERMES_WORKFLOW_PARITY.md);
this file is just the apply order.

## What's already live (nothing to do)

- The git pre-commit gate (`tools/gates/*.py` via `tools/githooks/pre-commit`) is wired through
  `git config core.hooksPath`, a repository-level setting shared by every worktree and every tool
  that calls plain `git commit` — Claude Code, Hermes, or a human at a terminal. **No action
  needed**; `tools/hermes_parity_check.sh` verifies it.
- `HERMES.md` (repo root) loads automatically into Hermes's system prompt whenever its working
  directory resolves into this repo (`agent/prompt_builder.py::build_context_files_prompt`,
  priority order `.hermes.md/HERMES.md → AGENTS.md → CLAUDE.md → .cursorrules`, first found wins,
  only one loads) — verified directly against the installed Hermes: with `HERMES.md` present the
  composed context-files prompt contains `## HERMES.md` and **no** `## CLAUDE.md` section at all.
  This is the centerpiece of Hermes parity here: Hermes's context window (~100K) is far smaller
  than Claude Code's, so `HERMES.md` is a small (~5KB, vs `CLAUDE.md`'s ~25KB) pointer whose one
  hard rule is "read `research/coordination/live_state.md` every turn, never rely on prior context,
  never start from scratch" — reliability comes from durable on-disk state Hermes RE-READS, not
  from anything carried in its context. `CLAUDE.md` stays available for on-demand deep reads.
  Point Hermes's `terminal.cwd` (or `--in`, or `--workdir` for cron) at `/home/dant123/Projects/sim`
  and it discovers both files. **No action needed** beyond that cwd pointing.
- The `pre_llm_call` shell hook (`tools/hermes/hook_live_state_context.py`) is Hermes's own
  documented `UserPromptSubmit` equivalent (its hooks doc names this explicitly) — it re-injects
  `research/coordination/live_state.md` as ephemeral context on every LLM call once the hooks
  snippet below is applied. `HERMES.md`'s manual-read rule is the fallback for when this hook is
  NOT yet configured (first run, or the owner chooses not to auto-accept hooks) — the two are
  complementary, not redundant: the hook makes the re-read automatic; the file's rule makes it
  survive even if the hook never fires.

## Apply order

1. **Back up** `~/.hermes/config.yaml` (it carries a lot of unrelated state already).
2. **Hooks** — merge `config.hooks.snippet.yaml`'s `hooks:` block (and `hooks_auto_accept: true`,
   if you accept that trust decision — see the file's own comment) into `~/.hermes/config.yaml`.
   Test each one before trusting it broadly:
   ```bash
   hermes hooks test pre_llm_call
   hermes hooks test pre_tool_call --for-tool terminal --payload-file <(echo '{"args":{"command":"pkill -f foo"}}')
   hermes hooks doctor      # exec bit / allowlist / mtime drift / JSON validity / timing
   ```
3. **Skills** — run `bash hermes-parity/skills_trust.sh` (one line: `hermes skills trust
   /home/dant123/Projects/sim`). Verify with `hermes skills list`.
4. **Heartbeat cron** — run `bash hermes-parity/apply_cron.sh`. It copies
   `scripts/sim_heartbeat.sh` to `~/.hermes/scripts/` (cron's `--script` is sandboxed to that
   directory — an absolute path is rejected) and creates the `sim-heartbeat` job. Verify with
   `hermes cron list` and, after one tick, `hermes cron runs sim-heartbeat`.

## Files in this directory

| File | Applies to | What it does |
|---|---|---|
| `config.hooks.snippet.yaml` | `~/.hermes/config.yaml` | 4 shell hooks: session-start live-state regen, per-turn live-state + advisory injection, post-edit doc-drift/regen, the self-matching-kill guard. |
| `scripts/sim_heartbeat.sh` | `~/.hermes/scripts/sim_heartbeat.sh` (copy) | Thin `exec` wrapper around the repo's own `tools/heartbeat_cmd.sh` — no duplicated logic. |
| `apply_cron.sh` | run once | Copies the heartbeat script + creates the `sim-heartbeat` cron job (15 min, agent mode). |
| `skills_trust.sh` | run once | `hermes skills trust /home/dant123/Projects/sim`. |

The actual hook logic lives in the repo proper, not here, so it stays under normal code review and
version control: [`tools/hermes/`](../tools/hermes/).
