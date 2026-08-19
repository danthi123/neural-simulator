---
type: plan
status: active
date: 2026-08-19
mechanism: codex-to-claude-transition
---
# Codex → Claude-only transition cleanup

**Owner directive (2026-08-19):** OpenAI Codex is retired from this project. Transition everything to be suited for
Claude-only development. Keep the `codex/` branch-name prefix (owner has "no issue with 'codex' as a name" — it is now
just the project's research-branch namespace, used by Claude too). **Do the actual deletions LATER — once the in-flight
agents finish and nothing is actively using the target branches/worktrees.** This doc is the durable checklist so the
cleanup happens correctly and the data-loss risks below are not forgotten across sessions.

## ⛔ The load-bearing risk — read before any git deletion
The primary checkout `~/Projects/sim` is sitting on **`codex/gap4-axon-capd-derisk`** (Codex's old working branch),
carrying **159 uncommitted + ~1,153 UNTRACKED files** as of 2026-08-19. Untracked files have **NO backup** (per the
Linux-migration state: sole source of truth = `~/Projects/sim` + origin). Commit `e4eec09f4` ("preserve codex-unique
research before branch reconciliation") banked a first tranche but explicitly flagged other items **DIVERGENT — for
careful merge to main, NOT blind-checkout**. So the reconciliation is UNFINISHED. **Never `git reset --hard`, never
`git checkout`/switch the primary off this branch, and never delete this branch until the untracked work is audited and
preserved.** (CRLF also makes ~5,217 files look modified — use `git diff --ignore-cr-at-eol`.)

## Preconditions for executing ANY step here
1. No live Claude agents (`ListAgents` empty) — this session had 5+ frontier agents on `codex/*` branches in
   `/tmp/.../scratchpad/wt-*` worktrees.
2. No decisive run in flight (the margin-clean mouth-readout GPU 6-seed was running 2026-08-19).
3. Re-verify each target is free at execution time — do not trust this snapshot.

## The checklist (deferred)

### 1. Retire `AGENTS.md` (safe; smallest step)
`AGENTS.md` (repo root, untracked, gitignored at `.gitignore:87`) is Codex's mirror of `CLAUDE.md` — verified 2026-08-19
to differ only in 6 trivial name-swap lines (header/footer self-references), **zero unique content**. Action: delete
`AGENTS.md`, then remove the `AGENTS.md` line from `.gitignore`. No content is lost (`CLAUDE.md` is the tracked source).

### 2. Reconcile the primary checkout off Codex's abandoned branch (the big one — needs care + owner)
Audit `codex/gap4-axon-capd-derisk`'s unique content vs `main` (`git diff --ignore-cr-at-eol main -- <path>` per file;
inventory the 1,153 untracked). Preserve anything unique with narrow commits onto `main` (the pattern e4eec09f4 used).
Handle the DIVERGENT items it flagged (`_gap4_dfc_plateau_credit`, `_curiosity_reward_omission_veto`,
`tools/gates/instrument_required`, `tools/gates/operating_point`, `_teacher_loop_neurogenesis`) by careful merge, not
blind checkout. Only then decide: keep developing on this branch (the name is fine) or move the primary to a fresh
Claude branch. **Confirm with owner before moving the primary checkout.**

### 3. Prune redundant `codex/*` branches (safe subset only)
Rule — delete a `codex/*` branch ONLY when ALL hold: (a) merged into `main` OR its unique work is confirmed-banked on
`main`; (b) NO worktree references it (`git worktree list`); (c) it is not a live agent branch. As of 2026-08-19 there
were ~80 `codex/*` branches; several merged ones are ALSO checked out in live worktrees (e.g. the just-landed
`codex/gnw-multistep-deliberation`), so "merged" alone is NOT sufficient — always cross-check the worktree list.
Recompute the set at execution time; never bulk-delete `codex/*`.

### 4. Prune dead worktrees
100+ worktrees exist under `.claude/worktrees/agent-*`, `.claude/worktrees/wf_*`, `sim-worktrees/*`, and the
session scratchpad. Run `git worktree prune` (removes those whose dirs are already gone), then explicitly
`git worktree remove` the finished-agent / finished-workflow ones. Keep: `gate-b-v2-clean` (the `main` consolidation
worktree) and any with unmerged unique work.

### 5. Sweep for other Codex-assuming config
Re-grep for `Codex`/`codex.ai` references in tracked config/tooling once 1–4 are done; none found beyond `AGENTS.md`
on 2026-08-19. Leave the `codex/` branch prefix as-is (owner keeps the name).

## Not in scope
Renaming branches (owner keeps `codex/`); touching `main`'s history; anything that risks the primary checkout's
untracked files before they are audited and preserved.
