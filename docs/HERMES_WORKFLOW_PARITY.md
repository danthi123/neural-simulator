# Hermes workflow parity

Maps every Claude-Code-specific workflow surface in this repo to its Hermes equivalent, so Hermes
(local agent, `~/.hermes`, CLI at `~/.local/bin/hermes`) can develop `neural-simulator` with the
same discipline — the gates, the durable-state discipline, the skills, the heartbeat, the tooling —
that Claude Code uses here, and can take over when Claude usage runs out.

**Status tags**: `[AUTOMATIC]` — already true, no action; `[CONFIG SNIPPET PROVIDED]` — a file under
`hermes-parity/` implements it, apply per `hermes-parity/README.md`; `[NEEDS OWNER STEP]` — a
one-time decision or command only the owner should make/run (touches global `~/.hermes/`).

Every claim below marked "verified" was checked against the REAL installed Hermes
(`~/.hermes/hermes-agent`, via its source and, where noted, `hermes hooks test` / a scratch
`HERMES_HOME` against the actual binary) — not inferred from `--help` text or docs alone. Where a
claim could not be verified this way, it says so.

---

## 0. THE CENTERPIECE: durable on-disk state, not context, is what Hermes must trust

**The reliability problem this section solves.** Hermes runs with roughly a 100K-token context
window; Claude Code here typically runs with roughly 1M. A workflow built assuming "the agent
remembers what happened earlier this session" degrades badly at 100K — sessions restart cold more
often, and a long research arc will not fit in context at all. Claude Code's own answer to this
(see `CLAUDE.md`'s "When Compacting" section and the `⟦LIVE-STATE⟧` mechanism below) is: **never
trust context to carry the frontier — trust a small file on disk that gets RE-READ every turn and
re-generated from ground truth on every landing.** That design transfers to Hermes directly, and
transfers BETTER, because Hermes has a documented, verified mechanism for exactly this.

### 0.1 `HERMES.md` — the lean file Hermes actually loads `[CONFIG SNIPPET PROVIDED — already in this worktree]`

Hermes discovers project context files in a strict priority order, **only one of which loads**:
`.hermes.md`/`HERMES.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules` (first found, walking from
cwd up to the git root, wins) — `agent/prompt_builder.py::build_context_files_prompt` /
`_find_hermes_md`. **Verified directly**: with `HERMES.md` present at the repo root, the composed
context-files prompt contains a `## HERMES.md` section and **no** `## CLAUDE.md` section at all —
confirmed by calling `build_context_files_prompt()` against this worktree with the real installed
Hermes. `CLAUDE.md` is 25,670 bytes; the `HERMES.md` this worktree adds is 4,793 bytes — a ~5×
reduction in what loads into every session's fixed budget, with **zero configuration**: file
presence alone decides it.

`HERMES.md` (repo root) is deliberately lean: the non-negotiable constraints in compressed form,
pointers to `CLAUDE.md` / `docs/FAILURE_GATE_MATRIX.md` / `docs/TERMS.md` for on-demand depth, the
GPU-sharing rule (§5 below), the pause/resume command, and — the one rule that matters most for
this section — **"every turn, before anything else, read `research/coordination/live_state.md`;
never rely on prior turns' context; never start from scratch."** This is the fallback path: it
holds even if the hook in §0.2 is not yet configured, because it is baked into what Hermes loads
every single session regardless of hook state.

**Verify**: `tools/hermes_parity_check.sh` checks `HERMES.md` exists, is reasonably small, mentions
`research/coordination/live_state.md`, and that no stray `.hermes.md` shadows it (the dotfile
variant is checked *before* `HERMES.md` in the same directory, so a leftover `.hermes.md` with
different content would silently win).

### 0.2 `pre_llm_call` — the automatic per-turn re-injection `[CONFIG SNIPPET PROVIDED]`

Hermes's shell-hook system supports exactly one event that can inject text into the running turn:
`pre_llm_call`. Hermes's own docs name it explicitly as the answer to this exact question —
*"Claude Code's `UserPromptSubmit` event is intentionally not a separate Hermes event —
`pre_llm_call` fires at the same place and already supports context injection. Use it here."*
(`website/docs/user-guide/features/hooks.md`, worked example 3, shipped with the install).

**A note on how this was scoped.** An earlier draft of this mapping was steered toward event names
`prompt_submit` / `user_message` / `turn_start` / `UserPromptSubmit`. None of those exist in this
Hermes install's `VALID_HOOKS` (`hermes_cli/plugins.py`) — confirmed by grepping the full source
tree (only unrelated internal symbols share fragments of those names: a CLI keybinding function,
a *memory-provider* interface method unrelated to shell hooks). `pre_llm_call` is what the shipped
documentation and the actual dispatcher (`agent/shell_hooks.py`) support, and it is what was built,
wired, and tested end-to-end below.

`tools/hermes/hook_live_state_context.py` is the hook script: it prints
`research/coordination/live_state.md` (regenerating it via `tools/live_state.py --emit` if
missing) as `{"context": "..."}`. **Verified end-to-end against the real binary**, not just read
from source: registered the exact snippet below in a throwaway `HERMES_HOME`, ran
`hermes --accept-hooks hooks test pre_llm_call`, and confirmed the dispatcher received and parsed
`{"context": "⟦LIVE-STATE⟧ re-injected each turn ...", ...}` — the live content of this repo's
actual `research/coordination/live_state.md` at test time.

```yaml
hooks:
  pre_llm_call:
    - command: "python3 /home/dant123/Projects/sim/tools/hermes/hook_live_state_context.py"
      timeout: 15
```

Apply via `hermes-parity/config.hooks.snippet.yaml` (full snippet, all 4 hooks) —
`hermes-parity/README.md` has the merge steps and the `hermes hooks test` / `hermes hooks doctor`
verification commands.

### 0.3 Why both layers, not just one

`HERMES.md`'s manual rule and the `pre_llm_call` hook are complementary, not redundant:

| | fires when | survives hook not yet configured |
|---|---|---|
| `HERMES.md` rule | loaded once, every session, from file presence alone | yes — it's the fallback |
| `pre_llm_call` hook | every LLM call, automatically | no — needs `hooks_auto_accept` or a TTY approval |

Together: even a completely fresh Hermes install with no hooks configured yet still gets told, in
its very first loaded context, "go read `live_state.md`, don't trust what you remember." Once the
hook is applied and allowlisted, the re-read becomes automatic and does not depend on the model
remembering to follow the instruction.

### 0.4 `on_session_start` — force a full regen at session boundaries `[CONFIG SNIPPET PROVIDED]`

Claude Code's `SessionStart` hook does two things: regenerate `live_state.md` from ground truth
(the board, the ledger, live-run status — never the lossy chat summary) and inject it. Hermes's
`on_session_start` can only do the first half — **verified**: its return value is never consumed
(`agent/conversation_loop.py:1049-1061` fires it fire-and-forget: `_invoke_hook("on_session_start",
...)` with no assignment of the result). So this hook exists purely for the regenerate side effect
(`python3 tools/live_state.py`, no `--emit`/`--fast`), keeping the file the `pre_llm_call` hook
reads fresh at the start of a brand-new session, not just on landings.

---

## 1. Git-level gates (`tools/gates/*.py`) `[AUTOMATIC]`

`tools/githooks/pre-commit` is wired via `git config core.hooksPath` = `tools/githooks` **in the
repository's shared `.git/config`** — not a per-worktree setting, not something living under
`.git/hooks/` (which is worktree-local and currently empty; `extensions.worktreeConfig` is unset,
so `core.hooksPath` is one setting shared by the canonical checkout and every worktree). This means
**any** `git commit` — from Claude Code, from Hermes running its `terminal` toolset, or from a
human typing `git commit` by hand — runs the same four gates: document structure (W1/W2), findings
claims traced to a cited artifact, biology bindings, and finding-status frontmatter, plus the
`tools/gates/*.py` auto-discovered registry underneath. **No Hermes-specific action needed** — this
was true before this task started, and `tools/hermes_parity_check.sh` verifies it stays true
(resolves `core.hooksPath`, checks the hook file is executable, `bash -n` syntax-checks it,
confirms the canonical `.venv/bin/python` the hook depends on exists). Verified by directly
executing `tools/githooks/pre-commit` (not through `git commit` — no commit was made) — exit 0,
clean run, no crash.

The only way to defeat this is `git commit --no-verify`, which is exactly as visible/deliberate for
Hermes as for Claude Code (it's a flag on the command Hermes's `terminal` tool would run) — no new
guard needed; `HERMES.md` names the gates as authoritative and `--no-verify` as a deliberate,
explained override only.

---

## 2. `.claude/settings.json` hooks → Hermes shell hooks `[CONFIG SNIPPET PROVIDED]`

`.claude/settings.json` wires four Python scripts under `.claude/hooks/`. Each has a Hermes port
under `tools/hermes/` (new files, this task) and a config entry in
`hermes-parity/config.hooks.snippet.yaml`.

| Claude Code hook | Event(s) | Hermes port | Event | Fidelity |
|---|---|---|---|---|
| `live_state_inject.py` (regen half) | `SessionStart` | `tools/live_state.py` (no wrapper needed — side effect only) | `on_session_start` | Full — see §0.4 |
| `live_state_inject.py` (read half) | `UserPromptSubmit` | `tools/hermes/hook_live_state_context.py` | `pre_llm_call` | Full — verified end-to-end, see §0.2 |
| `live_state_nudge.py` | `PostToolUse` (Write\|Edit) | `tools/hermes/hook_post_edit.py` (merged with the row below) | `post_tool_call` (matcher `write_file\|patch`) | Full for the regen side effect |
| `check_doc_drift.py` | `PostToolUse` (Write\|Edit) | `tools/hermes/hook_post_edit.py` (same file) | `post_tool_call` (matcher `write_file\|patch`) | Partial — see gap below |
| `block_self_matching_kill.py` | `PreToolUse` (Bash) | `tools/hermes/hook_block_dangerous_kill.py` (logic ported verbatim) | `pre_tool_call` (matcher `terminal`) | Full — verified end-to-end |

Tool-name translation (Claude Code → Hermes), used for every `matcher:` above — **verified against
`agent/display.py:458`'s primary-arg map**, not guessed: `Bash → terminal` (arg `command`),
`Write → write_file` (arg `path`), `Edit → patch` (arg `path`).

**The one real, honest gap: `post_tool_call` is observer-only in Hermes.**
`model_tools.py::_emit_post_tool_call_hook()` calls `invoke_hook("post_tool_call", ...)` and
discards the return value — confirmed by reading the call site; there is no `systemMessage`
channel for this event the way Claude Code's `PostToolUse` has one. So `hook_post_edit.py` cannot
show the agent anything directly when it detects a `docs/WRITING.md` violation or a
sync-documentation-worthy change. **Bridge, not a full fix**: it appends the advisory to
`research/coordination/.hermes_pending_advisory`, and `hook_live_state_context.py` (§0.2) drains
and injects that file's contents alongside `live_state.md` on the *next* `pre_llm_call`. **Verified
end-to-end**: fired `post_tool_call` for an edit to a `research/findings/*.md` path via
`hermes hooks test`, confirmed the advisory file was written, then fired `pre_llm_call` again and
confirmed the advisory appeared in its `{"context": ...}` output and the file was deleted (drained
exactly once). This is one LLM-call later than Claude Code's immediate `systemMessage`, not zero —
an honest, bounded degradation, not a silent drop.

`tools/hermes/hook_block_dangerous_kill.py` ports `.claude/hooks/block_self_matching_kill.py`'s
logic unchanged (same regexes, same heredoc/comment stripping, same "command position" requirement)
— only the wire adapter differs (`tool_name == "terminal"` instead of `"Bash"`). **Verified
end-to-end** with `hermes hooks test pre_tool_call --for-tool terminal` against a synthetic
`pkill -f my_runner.py` payload: exit code 2, `stderr` carrying the explanation, and the dispatcher
correctly translating that into `{"action": "block", "message": "..."}` — the same wire shape
Hermes uses in production.

Apply: `hermes-parity/config.hooks.snippet.yaml` → merge into `~/.hermes/config.yaml`. **Consent
note** `[NEEDS OWNER STEP / DECISION]`: shell hooks require either a first-use TTY approval or
`hooks_auto_accept: true` (or `--accept-hooks` / `HERMES_ACCEPT_HOOKS=1`) to register at all under
non-interactive use (cron, gateway). The snippet sets `hooks_auto_accept: true` with a comment
explaining the trust tradeoff (hooks run arbitrary code with the owner's full privileges) — this is
a real decision the owner should make deliberately, not something this task decided on their
behalf.

---

## 3. `.claude/skills/` → `.hermes/skills/` `[CONFIG SNIPPET PROVIDED]` + `[NEEDS OWNER STEP]`

Hermes's repo-local skill discovery looks for `./.hermes/skills/<name>/SKILL.md` or
`./.agents/skills/<name>/SKILL.md` at the project root — **not** `.claude/skills/`
(`agent/skill_utils.py`, confirmed against `tests/agent/test_project_skills.py`). The `SKILL.md`
**format is identical** (YAML frontmatter `name:`/`description:` + a markdown body) — no content
conversion needed.

**A real gap found and fixed empirically, not assumed away.** The first attempt made
`.hermes/skills/<name>/SKILL.md` a symlink back to `.claude/skills/<name>/SKILL.md` (avoids content
drift). Hermes scans every trusted project skill with `tools/skills_guard.py` before loading it; a
`"dangerous"` verdict silently excludes it (`agent/skill_utils.py::is_quarantined_project_skill`,
fail-closed). **A symlink whose target resolves outside the skill's own directory trips a CRITICAL
`"traversal"` finding by itself** — verified directly by calling the scanner: all 6 skills scored
`"dangerous"` as symlinks into `../../../.claude/skills/`. Converting to plain file copies of the
identical bytes fixed it: 4 of 6 scored `"safe"`, 2 (`neural-simulator`, `verify-go`) scored
`"caution"` (a `os.environ` mention inside a past-bug narrative reads as a potential env-dump to
the scanner) — **`"caution"` still loads**; only `"dangerous"` quarantines
(`is_quarantined_project_skill`'s own comment: *"'caution' loads (matches hub behavior for
prose-level keyword hits) — the quarantine is for high-confidence findings only"*). **Verified
end-to-end**: `hermes skills list --source local` against a scratch `HERMES_HOME` with this
repo trusted shows all 6 skills, `enabled`, `local` — after clearing Hermes's own project-skill
scan cache, which is content-hash-keyed but did not distinguish "symlink" from "copy" content
identity in this test (a real footgun for anyone iterating on this: `hermes hooks doctor`-style
staleness is not automatic here — the cache lives under `~/.hermes/cache/project_skill_scans/`,
outside the repo, invalidated by a bundle hash. If a skill scan looks stale, clear that directory).

Because they are copies, **not** symlinks, they will drift if `.claude/skills/<name>/SKILL.md`
changes later. `tools/hermes/sync_skills.sh` re-syncs them (`cmp`-checks first, only copies changed
files) — **not wired into a hook automatically** (see §6, known gaps).

Trust the repo (writes `skills.trusted_project_dirs` to `~/.hermes/config.yaml`, so this task did
not run it itself): `bash hermes-parity/skills_trust.sh`, i.e. `hermes skills trust
/home/dant123/Projects/sim`. Verify: `hermes skills list --source local`.

---

## 4. Heartbeat → `hermes cron` `[CONFIG SNIPPET PROVIDED]` + `[NEEDS OWNER STEP]`

Claude Code's heartbeat is a Monitor re-firing roughly every 15 minutes, running
`tools/heartbeat_cmd.sh` (GPU/proc state, ls-remote-verified unpushed-commit count,
`tools/workflow_check.sh`, `tools/parallel_audit.py`). Hermes's `cron create --script <name>
--workdir <dir>` does the schedule + prompt-injection half natively: **default mode (no
`--no-agent`) injects the script's stdout into Hermes's own prompt each run**, so Hermes actually
reasons over the audit and can act, matching `CLAUDE.md`'s *"⛔ UNDER-PARALLELIZED ... launch the
listed independent work ... BEFORE holding."* `--workdir` also injects `AGENTS.md`/`CLAUDE.md`
(and, per §0.1, would find `HERMES.md` first) from that directory for the job.

One real constraint, not a design choice: `cron create --script` resolves the path strictly under
`~/.hermes/scripts/` (`cron/scheduler.py:4342-4351`, `.resolve().relative_to(scripts_dir)` — an
absolute path elsewhere, or a symlink escaping that directory, is rejected). So
`hermes-parity/scripts/sim_heartbeat.sh` (a one-line `exec` of the repo's own
`tools/heartbeat_cmd.sh` — no duplicated logic) must be **copied**, not referenced in place, to
`~/.hermes/scripts/sim_heartbeat.sh`.

Apply: `bash hermes-parity/apply_cron.sh` (copies the script, then `hermes cron create "15m" "<the
react-to-this-audit prompt>" --name sim-heartbeat --script sim_heartbeat.sh --workdir
/home/dant123/Projects/sim`). Verify: `hermes cron list`, then after one tick `hermes cron runs
sim-heartbeat`.

---

## 5. GPU / local-model tooling (built by a separate effort; documented here so Hermes is pointed at it) `[AUTOMATIC]`

Out of scope to build (a separate process owns the model launcher + VRAM supervisor), but real,
already present in the canonical checkout, and referenced by `HERMES.md` so Hermes knows to use it
rather than running GPU Python directly and fighting its own model for VRAM:

- **`tools/hermes_gpu_run.sh "<cmd>"`** — the one way Hermes should launch a local GPU job:
  enqueues on `tools/gpu_queue.sh`, then Hermes should end its turn; the VRAM supervisor unloads
  Qwen, runs the job on the full card, reloads Qwen, and re-invokes Hermes.
  `research/coordination/live_state.md` reports what completed (its own header comment says so
  verbatim) — this is the same durable-anchor pattern as §0, applied by the sibling effort too.
- **`tools/qwen_supervisor.sh {__daemon|status}`** — the VRAM-aware auto load/unload, gated by the
  `research/queue/HERMES_ACTIVE` sentinel (master switch: Hermes is the driver) and
  `research/queue/GAME_MODE` (owner override, absolute priority).
- **`tools/qwen_serve.sh {up|down|status|restart}`** — launch/stop the local Qwen server itself;
  called automatically by the supervisor, rarely needed directly.
- **`tools/game.sh {on|off|status}`** — the owner's pause/resume for gaming or a break: `on` frees
  the GPU and stops new local jobs (mini-PC pool keeps running); `off` resumes, and is
  Hermes-aware (`off`'s own output branches on whether `research/queue/HERMES_ACTIVE` is set, to
  tell the owner whether Qwen will auto-reload or Claude needs a manual "continue").

These four were read directly from the canonical checkout (not this worktree — they don't exist in
any worktree yet, only `/home/dant123/Projects/sim/tools/`, confirming they're the sibling effort's
in-progress output) to make sure `HERMES.md`'s citations of them are accurate, not invented.

---

## 6. Load-bearing `tools/` Hermes should know about `[AUTOMATIC — Hermes inherits `terminal`]`

Hermes's `terminal` toolset runs bash, so it inherits every script below with no wiring. One-line
each, verified by reading the script:

| Tool | What it does |
|---|---|
| `tools/push_both.sh [branch]` | Push to both remotes (origin + gitea) and **verify** via `git ls-remote` (not just assert) — always use this instead of raw `git push`. |
| `tools/gpu_queue.sh` | Sequential, VRAM-contention-safe local GPU job queue; `pause --now`/`resume` for gaming (also driven by `tools/game.sh`). |
| `tools/sweep_pool.sh` | Dispatch CPU sweeps/tuning to the mini-PC pool (free, non-Claude compute). |
| `tools/rag/rag_search.py "<q>" N --corpus finding\|plan\|doc\|catalog\|kandel\|paper\|all` | The local RAG index — findings, the biology catalog, textbooks; check here before any external research. |
| `tools/vikunja.sh list-tasks 2` | The plain-language task board (Vikunja project 2) — read at session start, sync on a landing. |
| `tools/before_you_build.sh "<defect>"` | Corpus + existing-gate check before the first lever against any difficulty. |
| `tools/deep_research.sh "<wall>"` | Local + external literature research round at a repeated-lever wall (required after ≥3 findings in one lane in 3 days). |
| `tools/parallel_audit.py` | Detects under-parallelization: idle GPU/pool capacity next to ready board tasks. |
| `tools/lane_check.py` | Maps running/queued jobs to roadmap lanes; flags monoculture, an unserved crux, or no CPU lane. |
| `tools/check_docs.py` | The W1/W2 document-structure gate — same one the pre-commit hook runs; runnable standalone. |
| `tools/cost_audit.py` | Flags un-tiered/Opus-default agent scripts (the cost-routing enforcement `parallel_audit.py` calls each cycle). |
| `tools/heartbeat_cmd.sh` | The canonical heartbeat body (§4) — GPU/proc state, unpushed-commit check, `workflow_check.sh`, `parallel_audit.py`. |
| `tools/workflow_check.sh` | The composite parallelism/lanes/sources/cluster rule check that `heartbeat_cmd.sh` folds in. |

---

## 7. Known gaps — honest, not silently bridged

- **`post_tool_call` cannot inject a `systemMessage`.** Bridged one call later via a pending-
  advisory file (§2), not a full fix — a doc-drift nudge is one LLM turn behind Claude Code's
  immediate one.
- **Hooks need explicit consent** (`hooks_auto_accept: true` or per-hook TTY approval) to register
  under non-interactive use. This is a real trust decision (hooks run arbitrary code with full
  privileges) the owner needs to make, not something automatable from inside this task.
- **`.hermes/skills/` are copies, not symlinks, and will drift** from `.claude/skills/` if the
  Claude-side skill is edited later. `tools/hermes/sync_skills.sh` re-syncs them but is not wired
  into a hook or the pre-commit gate — a future improvement would be a `gates/*.py` check (mirrors
  the existing gate architecture) that blocks a commit touching `.claude/skills/*/SKILL.md` without
  a matching `.hermes/skills/*/SKILL.md` update, the same shape as `check_doc_drift.py`.
- **Skill content is written for Claude Code's own tool names and idioms** (e.g. "the Monitor
  tool", "the Agent tool", "TaskCreate") — Hermes will interpret these as natural-language intent
  (check on a background process, delegate a sub-task) rather than literal tool bindings, since
  skills are prose instructions, not programmatic tool schemas. Not a hard break, but not
  word-for-word fidelity either; worth revisiting if a specific skill's Hermes behavior looks off.
- **`hermes import-agent claude-code --dry-run`** (Hermes's own "migrate another agent's setup"
  command) was tried first, as the most direct-looking shortcut for this whole task. It reported
  *"Nothing to import from claude-code"* — it reads `~/.claude` (the GLOBAL Claude Code config
  directory: auth, cross-project settings), not a specific project's `CLAUDE.md`/`.claude/`, so it
  is not a substitute for the per-repo mapping this document does. Investigated and ruled out, not
  a gap in this task's coverage.
- **`GAP_CLOSURE_MISSION.md` (879 KB) is never auto-loaded by either agent** — Claude Code does not
  bulk-load it either; both read it on instruction (`CLAUDE.md`'s own text says to read it "FIRST,
  EVERY session", which Hermes will also see once it loads `HERMES.md`'s pointer, or `CLAUDE.md`
  itself on demand). No special Hermes handling needed; noted here only so it isn't mistaken for an
  oversight.

---

## Apply checklist (owner)

1. `bash tools/hermes_parity_check.sh` — confirms the git gate, `HERMES.md`, and the in-repo bridge
   scripts/skills are all present and correct *before* touching global `~/.hermes/` state.
2. Back up `~/.hermes/config.yaml`, then merge `hermes-parity/config.hooks.snippet.yaml`'s `hooks:`
   block (§2) — includes deciding on `hooks_auto_accept`.
3. `bash hermes-parity/skills_trust.sh` (§3).
4. `bash hermes-parity/apply_cron.sh` (§4).
5. Re-run `tools/hermes_parity_check.sh` — the `PENDING` lines from steps 2-4 should now read
   `LIVE`.
6. Point whatever invokes Hermes for this repo (`hermes chat`, a cron `--workdir`, a gateway
   profile) at `/home/dant123/Projects/sim` as its working directory — everything above depends on
   Hermes's cwd resolving into this checkout.
