# HANDOFF (overwrite each session; <= 40 lines)

- **TASK:** Roll out the lean method (short CLAUDE.md + AGENTS.md link, 3 skills, action-time checks, Hermes and Pi
  on the same method), then resume the new core's next build: sleep consolidation into cortex weights, per
  `brain/research/2026-11-02-record-brief-consolidation-and-selection.md`.
- **TYPE:** ops (rollout), then build
- **DRIVER:** claude
- **STARTED:** 2026-11-02
- **TIMEBOX:** 2 days per problem (rollout by 2026-11-04)
- **DONE MEANS:** rollout merged on main and Hermes/Pi read the same core (AGENTS.md); then sleep consolidation
  writes hippocampal experiences into cortex weights, a consolidated fact is recalled after a restart without the
  hippocampus, and a lesion visibly changes a reply.
- **MILESTONE:** day 14 (2026-11-16), `brain/ARCHITECTURE.md` §4: sleep consolidation; high-use facts never wrong;
  one lesion visibly changes replies. (Day 7, 2026-11-09: hippocampus + restart + weekly test.)
- **STATE:** branch `research/lean-method-1102` (worktree `.claude/worktrees/lean-method`): new CLAUDE.md,
  AGENTS.md link, this file; old handoff at `docs/archive/HANDOFF-2026-09-29.md`; old CLAUDE.md appended to
  `docs/CLAUDE_RATIONALE_ARCHIVE.md`. The new core is on branch `brain-v2` (`/home/dant123/Projects/sim-brain-v2`),
  to be merged. Skills, hooks and the Hermes/Pi driver changes are built by other agents in this run.
- **LIVE RUNS:** Stage-2 retake on S (Hermes finishing; old track). None on W from this task.
- **NEXT STEP:** merge `brain-v2` + this branch, confirm Hermes and Pi load AGENTS.md, then start the
  consolidation build with the before-build skill.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
  Earlier the same day: "Isn't having a ton of gates and such part of what slowed down our original work?" ...
  "simply writing documentation doesn't guarantee you'll actually read it at the right times."
- **OPEN QUESTIONS:** `brain/PREREQUISITES.md` is written but UNTRACKED on `brain-v2`: commit it before or with the
  brain-v2 merge (the brain-prereq gate needs it). No `.hermes.md` on S (checked 11-02), so Hermes gets AGENTS.md.
