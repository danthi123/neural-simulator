# CLAUDE.md (AGENTS.md is a link to this file)

**Goal.** A functional digital brain you can talk with openly, that learns and grows. The new small core lives in
`brain/` (`brain/ARCHITECTURE.md`). The old track (`sim/`, `research/runners`) is frozen except the Stage-2 retake.
Pivot: `docs/plans/2026-11-02-pivot-functional-digital-brain-DECISION.md`.

**Progress** = `brain/ARCHITECTURE.md` §4 milestones met by their dates, and the weekly transcript
(`brain/eval/weekly.py` -> `brain/eval/transcripts/<date>.md`) better each week. Process work is overhead.

## Rules
- Brain-based: host code only for world, body, teacher, or a declared scaffold (`brain/SCAFFOLDS.md`).
- One shared substrate is the long-term target.
- A negative is a verdict on a METHOD, not the capability. Judge it against scale and biological prerequisites
  first (`brain/PREREQUISITES.md`), then try a new method.
- Six seeds (42/43/44/100/101/102) before any generalization claim.
- Never claim phenomenal experience; self-reports are functional read-outs.
- GPU/heavy jobs: via `tools/gpu_queue.sh`, under `tools/memcap.sh` once `tools/mem_ok.sh` allows.
- Never `--no-verify`, never `git reset --hard`. Push: `bash tools/push_both.sh <branch>`.
- Tier every agent model (haiku/sonnet; opus only for hard judgment).

## Router
| Task | Skill |
|---|---|
| build, probe, test, plan; call a wall; 2nd fix | before-build |
| commit/report a result, flip a default | land-result |
| GPU, pool, AWS, server, hooks, old track | ops |

## Where things are
- Current task + state: `HANDOFF.md`
- Design + milestones: `brain/ARCHITECTURE.md`
- Prerequisites index: `brain/PREREQUISITES.md`
- Record check: `bash tools/before_you_build.sh "<question>"`
- History: `docs/CLAUDE_RATIONALE_ARCHIVE.md`

## Session
One task per session. Update `HANDOFF.md` at the end. `/clear` between tasks.

## When compacting
Keep owner words verbatim, live runs, and `HANDOFF.md` state.
Drop everything else.
