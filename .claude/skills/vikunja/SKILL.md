---
name: vikunja
description: The neural-simulator project's PLAIN-LANGUAGE task board (Vikunja at vikunja.dant123.com) — the owner's monitor + Claude's durable next-action source-of-truth that survives context compaction (supplements RAG/findings, does not replace them). AUTONOMOUS: read `tools/vikunja.sh list-tasks 2` at session start / when choosing work, and SYNC on every landing (mark done + add next-rungs) in the same cycle as the doc-sync — you do NOT need to be asked. Also invoke on "check tasks", "what's next", "update the board".
---
# Vikunja — the plain-language task board (autonomous: read at start, sync on landing)

The owner monitors project status/progress in the Vikunja web app (**https://vikunja.dant123.com**); Claude keeps it current via `tools/vikunja.sh`. This is a STANDING part of the autonomous workflow, not an on-request tool: **read it at session start, and sync it on every landing** — the same forcing-function as `sync-documentation` (a landing is not done until the board reflects it). It is roadmap-LEVEL (one task per capability / de-risk / integration / milestone); the detailed record stays in `research/findings/` + the RAG index + `GAP_CLOSURE_MISSION.md`.

## ⭐ TWO HARD RULES (the owner set these)
1. **ONE project.** Everything lives under the single **Neural-Simulator** project (id **2**). Do NOT create sub-projects. Differentiate areas with **labels** (below), never with separate projects.
2. **PLAIN LANGUAGE — no internal jargon in titles or the lead.** The owner must fully understand a task and its context WITHOUT knowing the project internals. BANNED from titles/leads: GNW, Rung-2d, STN, e-prop, BTSP, CA3, DMN, LC, n_ignited, "6-seed", "GO/NO-GO", commit SHAs, finding filenames, gap#N. Write what the ability MEANS ("The brain can halt a shaky thought before acting on it"), not the mechanism. **Technical detail is allowed — and encouraged — in the DESCRIPTION**, as a trailing `(Ref: <mechanism>, <verdict>, commit <sha>.)` line for Claude's own use. Title = plain; description = plain lead sentence, then the `(Ref: …)`.

## Structure
- Project **2 = Neural-Simulator** (the only project; parent of everything).
- **Labels** (areas — plain-language, applied to every task):
  `Focus & deliberation` (1) · `Speech & language` (2) · `Memory` (3) · `Vision` (4) · `Emotion & self-awareness` (5) · `Live brain (chat)` (6) · `Learning` (7) · `Big picture` (8).
- **Ladder-status labels** (make the production-integration LADDER visible so an open task shows HOW FAR it's come — a de-risked+wired-default-off faculty must NOT look identical to an untouched one): `de-risked` (10, GO at runner level, not yet production-default) · `blocked` (11, de-risked but production-wiring blocked upstream — e.g. no live consumer) · `epic` (12, mission-framing / a stage / the north-star — NEVER a checkbox). The authoritative per-faculty ladder is `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` (de_risked → wired → on_by_default → scaffold_retired); the label just SURFACES it on the board.
- **Priority** 0–5: 5 = the north-star, 3–4 = live frontier / in-flight, 1–2 = mapped-but-later, 0 = a done landing or a reference marker.
- **DONE means production-default, NOT de-risk-passed.** For a FACULTY/wiring/integration task, "done" = the ledger row reads wired=YES AND on_by_default=YES (scaffold_retired may still be a separate follow-on rung). A GO de-risk that is NOT yet wired-default-on is **NOT done** — it stays OPEN with the `de-risked` (or `blocked`) label. (This is the exact discipline the owner flagged 2026-08-19: ~40 default-off GO de-risks looked "done" while production stayed host. A pure research/negative-result task IS done when its finding lands.)

## The helper (token never touches the repo)
`tools/vikunja.sh` reads the URL+token from `~/.claude-config/secrets/vikunja.json` (OUTSIDE the git repo). Compact output by default; `--json` for raw.
```
tools/vikunja.sh list-tasks 2                 # THE BOARD — read this at session start
tools/vikunja.sh list-labels
tools/vikunja.sh create-task 2 "plain title" "plain lead. (Ref: …)" [priority 0-5] [YYYY-MM-DD]
tools/vikunja.sh label-task <task_id> <label_id>      # attach an area label (do this on every new task)
tools/vikunja.sh update-task <task_id> true|false     # mark done / reopen
tools/vikunja.sh set-desc <task_id> "…"  |  set-priority <task_id> <0-5>  |  set-due-date <task_id> <YYYY-MM-DD>
tools/vikunja.sh create-label "title" [hexcolor]  |  create-project …  |  delete-task <id>  |  delete-project <id>
```
Creating/updating tasks + labels on the owner's own board is low-risk (no per-action confirm needed). Deleting a project or many tasks is destructive — confirm first.

## The autonomous loop (do this without being asked)
- **READ (session start / choosing next work):** `tools/vikunja.sh list-tasks 2` → the open tasks by priority are candidate next-actions. It is a POINTER like any summary doc: RAG-check before adopting one, and if it conflicts with a finding, the FINDING wins and you fix the task (drift #12 applies here too).
- **⛔ BEFORE SCOPING A BUILD/AGENT for a faculty, check what's ALREADY SHIPPED — the board's DONE tasks AND `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` AND `main`'s current findings — not just the OPEN list.** Earned 2026-08-19: three build agents in one session (v1-lateral-inhibition, metacog-hedge, cross-position-pooling) were each scoped from a STALE premise (the open task / the primary checkout's old branch) for work ALREADY CONCLUDED on main; verify-first inside each agent caught it, but the wasted scoping is the drift. The board not reflecting shipped state is what lets you re-scope finished work — so reconcile it (below) AND read the done-state before launching.
- **SYNC (on a landing — a committed finding/integration that changes a capability's status), same cycle as the doc-sync:** (1) **advance the LADDER label** to match the ledger — a landed GO de-risk gets `de-risked`; when it becomes wired-default-on, DROP `de-risked` and `update-task <id> true`; if production is blocked, `blocked` + a one-line why in `set-desc`; (2) **RECONCILE, don't just append** — a landing often DELIVERS or SUBSUMES an older/overlapping task: search the board for it, and `update-task <id> true` with a `set-desc` note naming the delivering commit(s) (e.g. #74 was closed by #77+#85). Do NOT leave a shipped capability sitting open under a differently-worded old task; (3) `create-task` + `label-task` the genuinely-NEW next-rung(s) it opened — plain title, plain lead, `(Ref: …)`, right area + ladder label + priority; (4) adjust priorities if the frontier moved. Rung-level granularity only — never per micro-commit. **(5) STAMP the marker — this is now ENFORCED, not remembered:** update `docs/.vikunja_sync` (bump `last_reconciled` + `at_commit` + a one-line note of what changed) and STAGE it in the reconcile commit. The `vikunja-freshness` gate (`tools/gates/vikunja_freshness.py`, CLASS VF) BLOCKS a finding-adding commit once 6 findings have landed since that marker was last touched — so the board sync gets the same forcing-function the roadmap docs already have (`summary_doc_freshness`, CLASS SF). Stamping resets the budget; drifting a whole session (the 2026-08-20 failure this gate exists to prevent) can no longer happen silently.
- **NEW frontier / owner steer** → add the plain-language task(s) with the right label + priority so the next session sees it.

## Guardrails
- The board is a SUMMARY/pointer; the findings are ground truth. Never adopt a Vikunja "next" without the RAG check.
- Never echo or commit the token — it lives only in `~/.claude-config/secrets/vikunja.json`. If the script errors on a missing secrets file, recreate it (owner supplies the token); never hardcode it in the repo.
- Keep it plain (rule 2) and single-project (rule 1) on EVERY edit, or it stops being human-readable and the owner stops trusting it.
