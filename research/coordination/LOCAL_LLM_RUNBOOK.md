# Local-LLM weekend runbook

**Who this is for:** a LOCAL model (Qwen3.8-27B, 128K context since 2026-09-25, run via `bash tools/local_llm/llm.sh claude`)
supervising this repo's compute while the owner is away and Claude usage is conserved. **Keep everything you
read SHORT** — you have far less context than Claude does. Read `tools/status.sh`'s output first, always; only
open a file below it when `status.sh` tells you something needs attention.

The commands this runbook uses are pre-approved in `tools/local_llm/claude_local_settings.json` (loaded by `llm.sh claude`);
anything else asks for permission, and the explicit denies (--no-verify, force-push, merge, AWS, sim/ webapp/ tools/gates/
edits) are refused outright.

Resume the previous local session with `llm resume` (or `llm claude --continue`); plain `llm claude` starts a new one.

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
   have: `bash tools/pool_provision.sh --revision <sha> --isolated pool41 pool42` (ALWAYS `--isolated`: without it the
   revision overwrites the nodes' shared `~/derisk-pool/sim` tree, and the pinned `revisions/<sha>` dir the jobs need is never
   created). If it REFUSES as stale, that is a NEEDS CLAUDE item -- do not set POOL_PROVISION_ALLOW_STALE yourself.
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

Run these ONLY when `status.sh` marks the battery READY. Save the output JSON where shown, `git add` it with the raw
files, add ONE board line with the verdict string the command printed (copy it, do not interpret), commit, push.

- **fi** and **d6_n2000**: already harvested by Claude on 2026-09-25 (fe1066f64). Nothing to do.
- **settle_a2_wiring** (only at 18/18; prereg `research/findings/2026-09-24-affect-marker-settle-flip-criteria-AMENDMENT-PREREG.md`
  Amendment 2): `.venv/bin/python -m research.runners._affect_marker_settle_congruence --score-wiring --raw-dir research/findings/raw/_affect_marker_settle_congruence/wiring --seeds "42 43 44 100 101 102" --out research/findings/raw/_affect_marker_settle_congruence/wiring/verdict.json`
- **When a harvest commit is blocked by `verdict-preconditions`** (a verdict file without a `preconditions` block):
  unstage ONLY that verdict/aggregate file (`git reset <file>`), commit the per-seed raw files + board line, and add a
  NEEDS CLAUDE line naming the file. Never hand-write a preconditions block.
- **When blocked by `device-and-cost`** (a long run with no cost projection): do not edit the raw files; add a NEEDS CLAUDE
  line and leave that battery uncommitted.
- **arcc_awake_completion** (awake replay with pattern completion; prereg
  `research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md` Amendment 8 / Addendum 8a), only at 6/6:
  `.venv/bin/python -m research.runners._da_tag_capture_chat_probe --family arcc --aggregate research/findings/raw/_awake_replay_completion`.
  Commit the six seed files + board line; if the aggregate is blocked by verdict-preconditions, leave it (see below).
- **b2b_base: DO NOT HARVEST.** The three torn cells are being re-run on pool41/pool42 (queued 17:51); two seed-102 cells
  (discourse-register, episodic-memory) never produced a result and wait for Claude's ruling. It reads ~253/258; Claude scores it.

## Commit blocked by an idle-compute check (lane-starvation / compute-idle-persistent)

These checks refuse commits while pool/GPU capacity sits idle with ready work. Over the weekend capacity is idle BY
DESIGN ("Queue next" is empty). The honest exception, then retry the commit:

- If the owner is gaming (`bash tools/game.sh status` shows GAME_MODE on): write `research/queue/.lane_waiver` AND
  `research/queue/.parallel_compute_waiver`, each containing `CLASS: OWNER-PAUSE` and one line
  `reason=owner gaming (GAME_MODE on); weekend handoff, Queue next is empty`.
- Otherwise: write the same two files with `CLASS: NO-READY-WORK` and a `checked=` line naming what you looked at, e.g.
  `checked=<date time>: status.sh shows no READY battery beyond the one being committed; LOCAL_LLM_RUNBOOK Queue next is empty`.

Never write anything in a waiver that is not true. Waivers expire after 6 h; write a fresh one when needed. If the
check still refuses (its waiver budget is exhausted), that is a NEEDS CLAUDE item.

## Queue next

<!-- ORCHESTRATOR: pre-approved queue lines go here, one per line, exact command text. The local model may
     queue ONLY what is listed here, verbatim. Empty = queue nothing. -->

_(none — nothing reviewed is ready to queue this weekend; the pool and AWS stay idle and stop themselves. Claude adds
lines here after the Tuesday reset.)_

## Weekend draft work (owner-approved 2026-09-25) -- DRAFTS ONLY, never merged

Serves the adopted plan `docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md`. Rules: work on a branch
`research/draft-<topic>` (`git switch -c research/draft-<topic>`), write under `docs/drafts/`, commit through the gates,
push with `bash tools/push_both.sh research/draft-<topic>`, never merge, never touch `main` for this. Append to the draft
file after EVERY chunk of work, so nothing is lost if the session ends or compacts. Claude reviews the drafts after Tuesday.

1. **Host-decision seam map (plan section 1)** -> `docs/drafts/host-decision-seam-map.md`. Trace ONE default chat turn
   (`/api/brain-chat` in `webapp/server.py`) in order. For every place Python decides something from neural output, one
   table row: file:line, the operation, the neural input just before it, the decision made, the category (candidate
   generation / scoring / admission-filtering / winner selection / commitment / composition / routing), whether a
   neural signal already represents the choice (cite what shows it), and "unsure" wherever you are not certain. Work in
   chunks of ~300 lines of server.py; record facts only, no recommendations.
2. **Prior-work memos (plan sections 4-5)** -> `docs/drafts/credit-and-sleep-prior-work.md`. For the long-delay credit
   task and sleep transitive inference: run `bash tools/before_you_build.sh "<topic>"` and
   `.venv-rag/bin/python tools/rag/rag_search.py "<q>" 5 --corpus all`, OPEN each hit, and list what the project already
   has (eligibility traces, DA gating, sleep replay, transitive/relational memory runners and findings) with paths and
   one-line quotes. Do not design the experiment.
3. **Test-suite triage (parked item)** -> `docs/drafts/test-suite-triage.md`: run the resumable full-suite command from
   "Parked for Claude" (it takes 2+ hours; the owner approves the command), then group failures by error message.
   Mark `cudaErrorNoDevice` / GPU-hidden failures as ENVIRONMENT; list the rest with the first error line. No fixes.

## Parked for Claude (after the Tuesday reset) — do NOT work on these

- Prioritized memory (owner's top memory directive): the DESIGN is on main (e7c7a281f) with its open re-review issues listed at
  the top (1 MEDIUM, 3 LOW); next is a preregistration that closes them, then the build. The DA tag-capture + sleep-replay pair
  stays off until this lands (owner, 2026-09-25).
- B2b: (a) before AWS pool1 or pool2 next starts, run Steps 1a/1b of `research/coordination/b2b0924_reruns_commands.txt`
  (move the stale copies aside on those nodes); (b) seed 102 discourse-register + episodic-memory were dispatched twice
  (00:54 pool2, 13:15 pool1/pool2), left partial arm files and no lb.json -- decide under prereg A1.4 whether a further
  re-run is allowed; (c) fill `research/coordination/b2b0924_reruns.tsv` results when the three redo cells land, then score.
- Score the arcc awake-replay battery (6 seeds queued 18:05) once it lands; its aggregate needs a checked `preconditions` block.
- SlotBinder fast teach (`research/slotbinder-fast-teach-final` @ 06bf76489, touches sim/): NOT merged. Final re-review:
  the change wraps `cp_plasticity_rate_gain` in a cupy subclass on EVERY bridge (flag on or off), never run on cupy, the
  production default -- needs a flag-gated wrap + a cupy parity run; also stale '7.7-17x' title numbers and an
  uncalibrated AMENDMENT 3 pass criterion. Issue list in workflow wf_fb8debe7-bdc.
- Full test suite: the 2026-09-25 CPU-only run (`CUDA_VISIBLE_DEVICES=""`) exceeded its 2 h cap at ~97% with ~400 `F` marks and
  no summary, so failures are unclassified. Re-run so partial results survive, then sort GPU-environment failures from
  regressions against a pre-2026-09-25 baseline: `bash tools/memcap.sh 12 -- env SIM_NO_PROVENANCE=1 .venv/bin/python -m pytest -q
  -p no:cacheprovider --continue-on-collection-errors --junitxml=<scratch>/suite.xml -rfE tests` (with the GPU free, or hidden).
- B2c combined battery prereg (`research/b2c-paired-flip-prereg-fixround1`): open review issues.
- Tooling: claim-check round 8 (MEDIUM: more false blocks on block-scope derived markers) and the pool-stall detector (HIGH: no
  timeout on the live-node memory probe), issue lists in workflow wf_9944d7c2-aca; make `tools/gpu_queue.sh` and
  `tools/pool_autodispatch.sh` daemons immune to in-place edits of their own script; make `pool_provision.sh --revision` imply
  `--isolated` (FAILURE_LOG 2026-09-25).
- Small fixes: findings cite the git-ignored receipt `research/queue/.corpus_checks.jsonl` (if a commit in a worktree is
  blocked for it, copy that file from the main checkout into the worktree); the GPU auto-swap's two LOW review notes (validate
  the profile name before writing the restore marker; `|| true` on the test-only stop path).

## NEEDS CLAUDE

<!-- Append one line per escalation: `- YYYY-MM-DD HH:MM  <what happened>  <what you already tried>` -->

_(empty)_
