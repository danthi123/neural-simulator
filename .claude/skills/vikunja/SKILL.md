---
name: vikunja
description: Use when reading or updating the project's plain-language task board (Vikunja, vikunja.dant123.com) - choosing the next task, landing a result, or when asked "check tasks", "what's next", "update the board". Command reference for tools/vikunja.sh plus the board's two rules.
---
# vikunja: the plain-language task board

The owner's view of progress. One task per capability or milestone; the findings and HANDOFF.md hold the detail.

## Rules
1. ONE project: id **2** (Neural-Simulator). Never create sub-projects; areas are labels.
2. PLAIN LANGUAGE in titles and the first sentence (no GNW, BTSP, CA3, gap#N, SHAs, "6-seed", GO/NO-GO).
   Technical detail goes in a trailing `(Ref: mechanism, verdict, commit <sha>.)` in the description.
3. Find a task by CONTENT (`list-tasks 2`, match the title) before editing. A finding's "board: N" is not the
   Vikunja id: two closed tasks were overwritten that way (2026-08-30).
4. Done = on by default in production (or, for a research question, its finding landed). A de-risk that is not
   default-on stays open with the `de-risked` label.
5. Never echo or commit the token; it lives in `~/.claude-config/secrets/vikunja.json`.

## Labels
Areas: Focus & deliberation 1 · Speech & language 2 · Memory 3 · Vision 4 · Emotion & self-awareness 5 ·
Live brain (chat) 6 · Learning 7 · Big picture 8. Ladder: de-risked 10 · blocked 11 · epic 12.
Priority 0-5: 5 north-star, 3-4 live frontier, 1-2 later, 0 done/reference.

## Commands
```
tools/vikunja.sh list-tasks 2                       # the board
tools/vikunja.sh list-labels
tools/vikunja.sh create-task 2 "plain title" "plain lead. (Ref: ...)" [priority] [YYYY-MM-DD]
tools/vikunja.sh label-task <task_id> <label_id>     # every new task gets an area label
tools/vikunja.sh update-task <task_id> true|false    # done / reopen
tools/vikunja.sh set-desc <task_id> "..."   |  set-priority <task_id> <0-5>  |  set-due-date <task_id> <date>
```
On a landing: move the item (label or done), close any older task it delivered, add the next rung, and
append a dated line to `docs/.vikunja_sync`. Deleting tasks or projects needs the owner's OK.
