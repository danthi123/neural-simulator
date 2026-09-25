# Local-LLM weekend runbook

**Who this is for:** a LOCAL model (Qwen3.8-27B, 64K context, run via `bash tools/local_llm/llm.sh claude`)
supervising this repo's compute while the owner is away and Claude usage is conserved. **Keep everything you
read SHORT** — you have far less context than Claude does. Read `tools/status.sh`'s output first, always; only
open a file below it when `status.sh` tells you something needs attention.

Start here every session: `bash tools/status.sh` (read-only, <=40 lines, ~10s). It shows the GPU queue, the
pool queue + per-node state, AWS spend vs cap, whether the local-llm unit is up, every registered battery's
rows-landed vs expected, and recent failures.

## What you MAY do

1. **Run `bash tools/status.sh`** as often as you like — it changes nothing.
2. **Harvest a READY battery.** `status.sh`'s Batteries section marks a row `READY` when its landed rows meet
   the expected count and nothing is currently running that mentions its name. Run its registered harvest
   command from `research/coordination/handoff_batteries.tsv` (or "Harvest recipes" below if that battery has
   one), read the output, then:
   - `git add` the raw result files + their `.prov.json` sidecars,
   - write ONE dated line on the board (`GAP_CLOSURE_MISSION.md` CURRENT STATE) naming the battery and its
     headline numbers — no interpretation beyond what the artifact says,
   - commit **through the normal gates** (`git commit`, never `--no-verify`) and push with
     `bash tools/push_both.sh <branch>` — commit straight to `main` for the raw-results-plus-board-line commit
     ONLY (this is data landing, not a claim), and only if your working tree IS `main` with nothing else staged.
3. **Draft a finding, on a branch.** Create `research/draft-<battery>-<date>`, write the finding from the
   battery's `finding_template`, commit it there, push it. **Do not merge it.** A draft finding states what the
   artifact shows; it does not carry a verdict (GO/NO-GO) — that is Claude's or the owner's call.
4. **Re-provision a revision** when `status.sh` or a dispatch log shows pool nodes pinned to a commit they don't
   have: `bash tools/pool_provision.sh --revision <sha> pool40 pool41 pool42`.
5. **Restart `pool-dispatch` / `pool-sync`** per the troubleshooting table below.
6. **Queue ONLY the pre-approved lines** listed under "Queue next" below — nothing else, ever.

## What you must NEVER do

- Design an experiment, write or amend a preregistration, or decide what to run next beyond "Queue next".
- Flip a default, merge any research branch, or touch `main` for anything except the harvest commit in step 2.
- Edit `sim/`, `webapp/`, or `tools/gates/`.
- Use `git commit --no-verify`, or any other gate bypass.
- Delete data — raw results, findings, or board history.
- Touch AWS beyond `tools/status.sh` / `tools/aws_budget.sh status` (no launch, stop, or terminate).
- Rule on a review, a GO/NO-GO verdict, or a retraction.
- Restart or touch anything GPU-side (`gpu-queue-dispatch`, `gpu-queue-refill`, `gpu_queue.sh`) beyond reading
  its status — the GPU units are system-level and out of scope for this runbook.

## Stop / escalate: write it under "NEEDS CLAUDE" below, then STOP that thread

- A pre-commit gate blocks a commit and the fix isn't in this runbook.
- A battery's landed numbers surprise you (a rate, a GO/NO-GO-shaped value, anything you did not expect).
- A job you restarted fails a second time.
- AWS spend is within $5 of the daily cap (`tools/status.sh`'s AWS line).
- Anything else this runbook doesn't cover.

Write the item as one line: date/time, what you saw, what you already tried. Do not guess further.

## Troubleshooting table (user-level units only)

| unit | check | restart |
|---|---|---|
| `pool-dispatch.service` | `systemctl --user status pool-dispatch` | `systemctl --user restart pool-dispatch` |
| `pool-sync.service` | `systemctl --user status pool-sync` | `systemctl --user restart pool-sync` |
| `local-llm` | `bash tools/local_llm/llm.sh status` | `bash tools/local_llm/llm.sh off && bash tools/local_llm/llm.sh on` |

If a restart doesn't clear the symptom within one `status.sh` cycle, it's a NEEDS CLAUDE item — do not iterate.

## Start-of-session checklist

1. `bash tools/status.sh` — read the whole thing before doing anything else.
2. Any battery `READY`? Harvest it (see "What you MAY do" #2), one at a time.
3. Any node/unit down that the troubleshooting table covers? Restart it once.
4. AWS spend near cap, or a job failed twice? -> NEEDS CLAUDE, then move on.
5. "Queue next" has an unstarted line and capacity is idle? Queue it — nothing else.
6. Nothing to do? Re-run `status.sh` after a break; do not invent work.
7. Anything you're unsure is in-scope: treat it as NEEDS CLAUDE, not a judgment call.
8. Never run two harvests of the same battery concurrently.
9. Every commit goes through the normal gates; if one blocks and the fix isn't obvious, NEEDS CLAUDE.
10. End of session: leave "NEEDS CLAUDE" and board state as the handoff — no separate summary needed.

## Harvest recipes

<!-- ORCHESTRATOR: fill in a battery-specific recipe here ONLY if its registered `harvest_cmd` in
     handoff_batteries.tsv (the generic `battery_status.py --harvest <name>` listing) isn't enough. -->

_(none registered yet — use each battery's `harvest_cmd` from `research/coordination/handoff_batteries.tsv`)_

## Queue next

<!-- ORCHESTRATOR: pre-approved queue lines go here, one per line, exact command text. The local model may
     queue ONLY what is listed here, verbatim. Empty = queue nothing. -->

_(none pre-approved yet)_

## NEEDS CLAUDE

<!-- Append one line per escalation: `- YYYY-MM-DD HH:MM  <what happened>  <what you already tried>` -->

_(empty)_
