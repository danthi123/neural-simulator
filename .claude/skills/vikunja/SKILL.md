---
name: vikunja
description: The neural-simulator project's human-readable task board (Vikunja at vikunja.dant123.com), mirroring the roadmap. Use it to (a) READ the live frontier/next-actions at session start or when choosing what to work on — a durable source of truth that survives context compaction, supplementing the RAG/findings record; and (b) SYNC on a landing — mark the landed rung done + add the next-rung(s), so the owner can monitor status/progress/tasks in the web app. Invoke when asked to "check tasks", "what's next", "update the board", "sync vikunja", or when a de-risk/integration lands.
---
# Vikunja — the roadmap task board (monitor + durable source-of-truth)

The owner monitors project status/progress via the Vikunja web app (**https://vikunja.dant123.com**); Claude keeps it current via `tools/vikunja.sh`. It is the ROADMAP-LEVEL board (one task per rung / de-risk / integration / milestone), NOT a per-commit log — the detailed record stays in `research/findings/` + the RAG index + `GAP_CLOSURE_MISSION.md`. Its job: let the owner see "what's done / in-progress / next" at a glance, and give Claude the next-actions durably (outside the context window).

## The helper (token never touches the repo)
`tools/vikunja.sh` reads the instance URL + token from `~/.claude-config/secrets/vikunja.json` (OUTSIDE the git repo — no committed file holds the secret). Default output is compact (context-thrifty); `--json` for raw.
```
tools/vikunja.sh list-projects
tools/vikunja.sh list-tasks all            # the whole board, grouped by lane (READ THIS at session start)
tools/vikunja.sh list-tasks <project_id>   # one lane
tools/vikunja.sh create-task <project_id> "title" ["desc"] [priority 0-5] [YYYY-MM-DD]
tools/vikunja.sh update-task <task_id> true|false     # mark done / reopen
tools/vikunja.sh set-desc <task_id> "description"
tools/vikunja.sh set-priority <task_id> <0-5>
tools/vikunja.sh create-project "title" [parent_id] ["description"]
```

## The structure (parent #2 "Neural-Simulator" + lane sub-projects)
| id | lane |
|----|------|
| 3 | GNW Workspace & Consciousness (the keystone) |
| 4 | Language & Mouth |
| 5 | Memory & Episodic |
| 6 | Perception |
| 7 | Affect & Self-Model |
| 8 | Integration → Production (the spine) |
| 9 | Learning & Credit (gap#4) |
| 10 | Roadmap & Milestones |

Priorities: 5 = north-star, 3-4 = live frontier / in-flight, 1-2 = mapped-but-later. A task's **description** carries the finding filename + commit SHA + one-line status, so a reader (or a future session) can jump to the detail.

## The workflow (lean — do NOT let it become a maintenance tax)
- **READ (session start / choosing next work):** `tools/vikunja.sh list-tasks all` → the open tasks by priority ARE the candidate next-actions. Cross-check against `GAP_CLOSURE_MISSION.md` CURRENT STATE + a RAG check (a Vikunja task is a POINTER, like any summary doc — if it conflicts with a finding, the finding wins and you fix the task).
- **SYNC (when a rung LANDS — a finding is committed that changes a lane's status):** in the SAME cycle as the doc-sync, (1) `update-task <id> true` on the landed task, (2) `create-task` the next-rung(s) it opened (title + a description with the finding path + the GO-gate + the named next lever), (3) adjust priorities if the frontier moved. This mirrors the `sync-documentation` forcing-function: a landing isn't done until the board reflects it. Keep granularity at the rung level (a 6-seed de-risk / an integration / a milestone), never per micro-commit.
- **A NEW frontier / owner steer** → add the task(s) under the right lane at the right priority so the next session sees it.

## Guardrails
- The board is a SUMMARY/pointer, not ground truth — the findings are. Never adopt a Vikunja "next" without the RAG check (drift #12 applies here too).
- Never echo or commit the token. It lives only in `~/.claude-config/secrets/vikunja.json`. If `tools/vikunja.sh` errors on a missing secrets file, recreate that file (owner supplies the token) — do not hardcode it anywhere in the repo.
- Creating/updating tasks on the owner's own board for this project is low-risk and does not need per-action confirmation; deleting a project/many tasks does — ask first.
