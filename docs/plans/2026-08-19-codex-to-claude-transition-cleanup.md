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

## PROGRESS (2026-08-19) — safe bulk DONE; the scary part is resolved
- **AGENTS.md retired** — the tracked "Codex entry-point" on main (commit `2d9fa11f`) AND the primary's 24KB untracked
  mirror. Both gone.
- **The primary-checkout data-loss risk is RESOLVED.** The 1,153 untracked files were categorized: ~1,065 `.json`
  regenerable research outputs + scratch, and only **5 unique source files**. The 3 useful ones (GPU-crash training
  watchdog `tools/train.sh` + `tools/gpu_train_watchdog.sh` + `.service`) were **preserved to main** (commit
  `32106b0e`); the other 2 (`raw/_bounded_launcher.py`, `_throttle_daemon.py`) are superseded pre-migration scratch,
  left in place. **No unique work is stranded.** Since the owner keeps the `codex/` name, the primary can STAY on
  `codex/gap4-axon-capd-derisk` — no risky branch-switch needed. Step 2 above is effectively closed.
- **Branches:** 7 merged-into-main codex branches deleted (`git branch -d` from the main worktree so the merge check
  runs against main, not the primary's HEAD). ~71 codex branches remain, nearly all UNMERGED.
- **Worktrees:** 156 → 49 (107 removed): 26 clean stale + 81 harness-throwaway (`worktree-agent-*`/`worktree-wf_*`,
  force-removed with their branches — no research value, no live process). All done `--force`-free except the
  throwaway class; the 6 named dirty worktrees (wt-keystone/wt-moutheprop/wt-genwire/wt-dmnbasins/wt-rung2c/
  wall-value-critic-neural) were KEPT (genuine uncommitted work).

## COMPLETED via the worktree-cleanup-audit workflow (2026-08-19, wf_dcc9f8ff)
An 11-agent read-only audit inspected all 43 non-live worktrees and verdicted each SAFE_REMOVE (39) /
PRESERVE_THEN_REMOVE (4) / KEEP (0), accounting for CRLF noise and cherry-picked-but-graph-unmerged content. Result:
- **2 genuinely-unique runners preserved to main** (`6af523a0`): `_laneC_source_monitor_hetero_encoding_sweep.py`,
  `_emerge_wm_hybrid_scale_derisk.py`. (Two of the 4 "preserve" flags were false-ish: integration-5's finding is
  already on main under its real 2026-08-10 date — the inspector searched a truncated 08-18 name; wall-value-critic's
  finding is on main in a fuller 6-seed form. Trust-but-verify caught both.)
- **All 43 worktrees removed + their branches**, then a patch-equivalence pass deleted 25 more fully-on-main codex
  branch refs. **Final: 6 worktrees (the live set only), 16 codex branches (5 live/primary + 11 older refs).**

## What REMAINS (optional; harmless — zero data-loss risk since nothing is being deleted)
- **11 older codex branch REFS** (causal-composition, cross-region-one-brain, gap4-decolle/softhebb, i7-burndown1/2,
  relational-spatial-code, replay-consolidation-v3, research-escalation, rolegate-fbalign, v13-deterministic-baseline):
  patch-id-unique commits — could be genuinely-unique unmerged work OR modified cherry-picks. KEPT (refs only, no
  worktree, no disk cost). A future content-audit (same workflow, adapted to branch refs via `git diff main...<branch>`)
  can decide each. Owner keeps the `codex/` name, so there is no urgency.
- Run-safety rule stands: re-verify nothing live is using a target (`readlink /proc/<pid>/cwd`) before removing it.
