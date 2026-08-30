---
name: evolve-skills
description: Periodically review what's WORKING and what's RECURRINGLY FAILING in this project's workflows, then make INCREMENTAL updates to the applicable skills (or a one-line CLAUDE.md pointer) so the workflows compound instead of re-learning the same lessons. Run when a process lapse RECURS (a problem the owner had to catch is a skill gap — highest-value trigger), at a natural session-end / pre-compaction inflection, or when the owner asks ("review/evolve our workflows"). Grounded in real evidence (this session's lapses + wins, recent findings/memories/commits/skills), honest (surface real recurring problems, not self-congratulation), incremental (smallest edit the evidence justifies), and LEAN (encode in on-demand skills, never bloat always-loaded CLAUDE.md/memory).
---

# Evolve Skills — continuous workflow improvement

The project's retrospective-that-acts. It periodically asks *"what's been working, what keeps going wrong?"* and turns the answer into concrete, incremental improvements to the skills that encode how we work — so the workflows compound over the project's life instead of the owner catching the same class of lapse repeatedly.

**Announce at start:** "Running evolve-skills: reviewing what worked + what recurred, then updating the applicable skills."

## When to run (the "reliably" part — no daemon, so these are the triggers)
- **A process lapse RECURS** — the HIGHEST-value trigger. The owner had to catch the same *class* of problem twice, or you notice you repeated a mistake. A caught lapse IS a skill/doc gap; evolve the skill so it can't recur (don't just patch this instance).
- **A natural session-end / pre-compaction inflection** — a substantial arc concluded; before the context turns over, bank the workflow lessons into the skills. (The CLAUDE.md "When Compacting" note points here.)
- **The owner asks** — "review our workflows / evolve the skills."
Do NOT run it every turn (that is churn) — only when there is genuine evidence to act on.

## Step 1 — GATHER the evidence (grounded, never from vibes)
- THE SESSION: which workflows/patterns were used? Which caught a real problem (a WIN)? Which failed, or the owner had to catch (a LAPSE)? Especially: a lapse that happened **more than once**.
- The record: `git log --oneline -30`, `ls -t research/findings/*.md | head -15`, the `feedback_*` memories (MEMORY.md index), and the existing skills (`ls .claude/skills/`).
- Be specific with evidence: *"adversarial-verify caught 2 over-claims (W3 immunity, P1.2 affect)"* (win) / *"idled ~2h twice on stalled subagent Monitors, owner caught both"* (lapse) — not "things went well."

## Step 2 — IDENTIFY (honest, ranked)
- **Recurring PROBLEMS** — a lapse that recurred, a mistake the owner caught, a workflow that was slow/wasteful. Each is a skill/doc GAP. Rank by cost × recurrence.
- **WORKING patterns** — a practice that reliably helped. Is it *encoded in a skill* (so it's reused), or only in your head this session?
- Distinguish a genuine recurring pattern from a one-off. Don't over-fit to a single incident — but a lapse the owner caught, or one that recurred, IS actionable.

## Step 3 — UPDATE the skills (incremental + lean)
- Each recurring PROBLEM → update the APPLICABLE existing skill to prevent it (add a check, a rule, a guard). Create a NEW skill only for a genuinely new recurring workflow with no home; prefer extending an existing skill.
- Each WORKING pattern not yet encoded → add it to the relevant skill.
- **LEAN:** encode in the on-demand SKILL; reference from CLAUDE.md with at most ONE line. Do NOT inline the rule into always-loaded CLAUDE.md / memory (the bloat the owner flagged 2026-07-24).
- **INCREMENTAL:** the smallest edit the evidence justifies; don't rewrite a working skill wholesale.
- Each update must be EARNED by evidence (a real lapse/win), not speculative "might help."

## Step 4 — REPORT
```
## evolve-skills — <date>
### Worked (reinforced / already encoded)
- [pattern → which skill encodes it]
### Recurred / lapsed (→ skill updated)
- [problem + evidence → the check/rule added to prevent it]
### Skills changed
- [skill: the specific edit]
### Left for the owner (judgment forks)
- [anything needing a human call, not auto-changed]
```
Commit the skill edits (this is workflow maintenance, low-risk); push both remotes.

## What this skill MUST NOT do
- Do NOT self-congratulate or invent lessons — every item is grounded in real session/record evidence.
- Do NOT churn — don't rewrite working skills; make the smallest edit the evidence justifies; don't create a skill for a one-off.
- Do NOT bloat always-loaded context (CLAUDE.md/memory) — encode in on-demand skills; CLAUDE.md gets at most a one-line pointer.
- Do NOT touch the science / findings / the mission's NON-NEGOTIABLES — this evolves WORKFLOW only, never the research verdicts or the brain-based-only / one-brain / no-defer / speed-secondary / honesty-boundary rules.

## Why this skill exists
2026-07-24: in one session the owner had to catch THREE process lapses — idle-stalls on dead subagent Monitors, deferring a named surpass (gap#4), and doc-drift (findings committed, board left stale). Each was a workflow gap a periodic retrospective would have caught + fixed proactively, instead of the owner catching each and me patching it once. This skill makes workflow-improvement a reliable, self-driven loop so the skills that encode how we work *compound*. It embodies the owner's 2026-07-24 steer: recurring workflows → on-demand skills → referenced leanly from CLAUDE.md, not context bloat.
