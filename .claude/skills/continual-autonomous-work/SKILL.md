---
name: continual-autonomous-work
description: Use when the user grants long-running autonomy on this repo. Tightens the patterns in autonomous-runs by eliminating "wrap-up" framing, hard-banning self-imposed clean break points, and requiring that the next concrete technical step always start immediately after each commit. Companion (not replacement) to autonomous-runs.
---

# Continual Autonomous Work

This is a project-scoped addendum to `autonomous-runs`. The base skill is correct in spirit; this skill exists because past sessions have shown Claude drifting into "natural stopping points" that the base skill explicitly warns against. The drift modes are recurring enough to be worth codifying.

**Always invoke alongside `autonomous-runs`, not instead of it.**

**Announce at start:** "Using continual-autonomous-work — no wrap-ups, immediate next action after each commit."

## The hard rules

These OVERRIDE any natural-seeming stopping pattern:

### Rule 1 — No "wrap-up" framing, ever

Forbidden artifacts:
- "Arc summary" / "session summary" / "wrap-up" findings docs that describe the work as *complete*
- "Final commit" framing in commit messages
- CHANGELOG headings that read like releases (`## [Unreleased] — DATE — Big Feature SHIPPED`) when more work in the same feature is queued
- "As of this commit, X is done" phrasing
- Phrases like "the arc has produced substantial value, let me stop"

Findings docs describe what landed AND list what's still open. They never imply *the chapter is closed*. There is no "end of an autonomous arc" until the user explicitly says so.

If a findings doc is the right artifact, name it for the work, not the *closure* of work: e.g. `2026-05-11-bridge-memory-shipped.md` (not `2026-05-11-arc-wrap-up.md`).

### Rule 2 — After every commit, the next action is the next technical step

After `git commit && git push`, the immediately-next action must be one of:
- The next concrete code/test/doc change for the queued work
- Launching a background task with `run_in_background: true`
- A targeted research/diagnostic step (Read / Grep / Bash) that informs the next change

Forbidden as the next action:
- A status-report-to-user-with-no-tool-calls turn (just talking)
- A todo-list-refresh-only turn
- A "what should I do next?" reflection that ends without a tool call
- "Let me check progress and decide" without a follow-up action in the same turn

The acid test: every turn between commits must move at least one file or kick off at least one task. If you're typing a paragraph, you must also be acting.

### Rule 3 — Background tasks must use `run_in_background: true`

Never use `&` in the shell command to background a task. The shell-exits-cleanly trap silently fails: the parent bash returns success while the python process exits immediately.

Correct:
```
Bash(command="python -m research.runners.X > log.txt 2>&1",
     run_in_background=true)
```

Incorrect (DO NOT USE):
```
Bash(command="python -m research.runners.X > log.txt 2>&1 &")
```

Within 30 seconds of launching a background task, verify it's actually running:
- `Read` the log file's first lines, OR
- `Bash` a `ps`/`Get-Process` filter on the command

If the task didn't actually start, relaunch correctly before moving on.

### Rule 4 — Reports are announcements, not questions

When you write to the user mid-arc:
- "Next: doing X." ✓
- "Pushed Y, now launching Z." ✓
- "Hit a roadblock on A; trying B as the alternative." ✓
- "Should I do Y?" ✗
- "Is this a good stopping point?" ✗
- "Do you want me to continue?" ✗

The base skill's stop conditions are the ONLY reasons to ask before continuing:
- User explicitly said stop/pause/wait
- A safety/permission boundary requires approval
- Genuinely nothing left to do (all known + discoverable work complete)

"I've been working a while, should I keep going?" is not a stop condition. Keep going.

### Rule 5 — Re-prioritize but don't re-evaluate "is the arc done?"

After each commit, the question is "what's the highest-value queued thing?" — not "have I done enough?".

When you finish a feature, the question is "what's the next feature?" — not "is this a good place to stop?".

When you hit a user-blocked item (e.g. waiting on LLM choice), skip past it to the next unblocked item. Don't treat user-blocked as a session boundary.

### Rule 6 — Long-running waits never justify idle

If a background task will run for N minutes, you have N minutes of parallel work. Always have parallel work queued:
- Code that consumes the background's output
- Tests for the background's expected result
- Adjacent refactoring / documentation
- The NEXT background task that can start in parallel

If you genuinely have nothing to do during a wait (rare), use ScheduleWakeup (1200-1800s) to come back. Don't stop the session.

### Rule 7 — Verify your own assumptions

After each significant commit, briefly:
- Re-read the diff (or one critical file) to confirm the change is what you intended
- Run the tightest test that exercises what you just changed
- For webapp/runner changes: smoke-test live (curl / one-shot run)

Don't trust your own intent over the git log. The agent's summary describes what it intended to do, not necessarily what it did.

## Drift modes (anti-patterns observed in past sessions)

1. **The "phase 3.2 arc wrap-up" doc.** Found in commit a0e095f (2026-05-11). The doc framed the work as complete even though Phase 3.3 was queued and the new-vocab binding issue was open. Should have been `phase3.2-shipped.md` with an explicit "open follow-ups" section, not "arc wrap-up".

2. **The "What's left?" reflection.** Multiple turns spent enumerating remaining work, sometimes ending without taking the next action. Each "what's left?" reflection should END with launching the next action.

3. **The `&` background trap.** Bash command ending in `&` returns success immediately; the python process can exit silently. Always `run_in_background: true`. Verify the log within 30 seconds.

4. **The "this is a good place to stop" feeling.** Subjective; ignore. The base skill explicitly says "your time estimates are usually wrong; ship it."

5. **The CHANGELOG-as-release-boundary.** A new `## [Unreleased] — DATE — BIG SHIPPED` heading every arc reads like the chapter is closed. Either keep using the same heading (running journal) or accept that headings are journal markers, not release boundaries — and KEEP WORKING after writing one.

6. **The "user-blocked, so I'll stop" leap.** User-blocked items are skipped, not session-ending. There are always unblocked items.

## Process

This skill doesn't have a separate process; it's a constraint layer over `autonomous-runs`. Read it once per autonomous arc to internalize the rules, then proceed via `autonomous-runs`' process.

## When to invoke

- Whenever `autonomous-runs` is active on THIS repo (`sim/`).
- Whenever the user says "continue autonomously" or "until I say stop" or invokes `/loop` autonomous-loop modes.
- The first thing each autonomous-arc turn does is re-read this skill's rules briefly.

## Integration with autonomous-runs

The base skill's "Stop conditions" remain the only valid reasons to stop. This skill adds:

- Stronger language against natural stopping patterns
- Concrete drift modes observed and how to avoid them
- The `run_in_background: true` correctness requirement

If the base skill's letter conflicts with this skill's letter, this skill wins (it's the local, more specific addendum). If they conflict in spirit, the base skill's spirit wins.

## Tests Claude can run on itself

At any time, ask:
- "Am I writing a 'wrap-up' artifact?" → If yes, rename / reframe to not imply closure.
- "Am I about to send a turn with no tool calls + a question to the user?" → If yes, take the next action instead.
- "Did I just commit?" → If yes, what's the immediately-next concrete step? Do it.
- "Did I just kick off a background task?" → Within 30 sec, verify it's actually running.
