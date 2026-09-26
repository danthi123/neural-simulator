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

Never write anything in a waiver that is not true. Waivers expire after 6 h and there is a 6-hour budget of waiver
time per rolling 24 h (2026-09-25: the budget was used up at ~22:30 and frees gradually from ~16:00 Saturday).
- **Commits whose staged files are ALL Markdown (.md) are exempt from this check** -- drafts, worklogs, preregs, the
  seam map and board lines can always be committed.
- For a commit that includes code or raw results: write the honest waiver just before the commit and delete it right
  after, so it does not sit charging the budget. If the check still refuses because the budget is exhausted, keep
  working and commit Markdown only; commit the code/results later. Never bypass it.

## Queue next

<!-- ORCHESTRATOR: pre-approved queue lines go here, one per line, exact command text. The local model may
     queue ONLY what is listed here, verbatim. Empty = queue nothing. -->

_(none — nothing reviewed is ready to queue this weekend; the pool and AWS stay idle and stop themselves. Claude adds
lines here after the Tuesday reset.)_

## Pending commit from Claude (do this after ~16:00 Saturday 2026-09-26, when the waiver budget frees)

The local-llm fix below is already live in the working tree but could not be committed (code commit + exhausted waiver
budget). After 16:00 Saturday: write the honest NO-READY-WORK waiver (see the idle-compute section), then
`git add tools/local_llm/llm.sh tests/conftest.py tests/_stub_systemctl_inactive.sh tests/test_no_real_llm_unit.py`,
run `.venv/bin/python -m pytest -q tests/test_no_real_llm_unit.py tests/test_llm_gpu_autoswap.py tests/test_llm_session_resume.py`,
commit with the message `fix(local-llm): tests can no longer stop the real local model; llm on clears a failed unit;
llm claude continue means resume (Claude-authored 2026-09-26, committed by the local model)`, delete the waiver, and
push main. If anything differs from this description, stop and write NEEDS CLAUDE.

## Weekend work (owner-approved 2026-09-25): build on branches, run DEV seeds, Claude corrects after Tuesday

Serves `docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md`. The owner wants the local model to carry real
work, done so Claude can review, fix or discard it later. **Hard rules for everything in this section:**

- Each task on its own branch: `git switch -c research/draft-<topic>`; commit through the gates; push with
  `bash tools/push_both.sh research/draft-<topic>`; **never merge, never commit this work to `main`**.
- Keep `WORKLOG.md` at the top of the branch's draft folder (`docs/drafts/<topic>/`): each entry = date/time, what you
  did, commands you ran, results (paths), and anything you are UNSURE of. Append after every step, so a restart or
  compaction loses nothing. Mark guesses as guesses.
- **Seeds:** use ONLY dev seeds (7, and 1-15 for small dev sweeps). **Never** 42/43/44/100/101/102; those are gate
  seeds, run only under a preregistration Claude has reviewed. Label every dev result "DEV, not a verdict".
- Compute: CPU on the mini-PC pool (`tools/pool_queue.sh add`, jobs pinned to your pushed branch commit after
  `bash tools/pool_provision.sh --revision <sha> --isolated pool41 pool42`); small local runs only under
  `bash tools/memcap.sh 8 -- ...`. **No GPU jobs** (the GPU auto-swap would stop your own model mid-session).
- Code goes in NEW files under `research/runners/` and `tests/` (never edit `sim/`, `webapp/`, `tools/gates/`). To
  observe production code, import it and wrap/monkeypatch functions inside your runner; never change it.
- Stop and write NEEDS CLAUDE when: a gate blocks and the fix isn't obvious, a result looks like a GO/NO-GO, or a design
  choice would change what the experiment means.

Tasks, in priority order. Keep the pool busy: while pool jobs run, work on the next task.

1. **Host-decision seam map (plan section 1)** -> `docs/drafts/seam-map/host-decision-seam-map.md`. Trace ONE default
   chat turn (`/api/brain-chat` in `webapp/server.py`) in order, ~300 lines per chunk. One table row per place Python
   decides something from neural output: file:line, operation, neural input just before it, the decision, category
   (candidate generation / scoring / admission-filtering / winner selection / commitment / composition / routing),
   whether a neural signal already represents the choice (cite the evidence), and "unsure" where you are not certain.
   Facts only.
2. **Host-share instrument, first seams (plan section 2)** -> `research/runners/_draft_host_share_probe.py` + a test.
   After the map has 5+ rows, pick 1-2 winner-selection or commitment seams. Build a runner that drives a dev-seed chat
   turn (numpy, CPU) and, by wrapping the seam function, records (a) the host decision and the neural inputs
   (decoding), (b) the downstream result when the host decision is replaced by a neural-derived one (substitution),
   and (c) when it is flipped with the neural state held fixed (intervention). Dev seeds only; queue a small dev sweep.
3. **Long-delay credit task (plan section 4)** -> `research/runners/_draft_long_delay_credit.py` + test + a DRAFT
   prereg in `docs/drafts/long-delay-credit/`. First run the prior-work search
   (`bash tools/before_you_build.sh "long delay temporal credit eligibility trace"` and
   `.venv-rag/bin/python tools/rag/rag_search.py "<q>" 5 --corpus all`, opening each hit) and REUSE existing
   eligibility-trace / dopamine machinery. Implement the arms from the plan (variable delay, uninformative terminal
   state, final decoy cue, trace lesion, immediate-reward and shuffled-reward controls) on a small spiking network.
   Dev-seed smoke, then a small dev sweep on the pool.
4. **Sleep transitive inference (plan section 5)** -> DRAFT prereg + a runner skeleton that REUSES the sleep-replay
   capture machinery (`webapp/sleep_replay_capture.py`, `research/runners/_da_tag_capture_chat_probe.py`, read-only).
   Arms: normal replay, wake-only, order-scrambled replay, replay off; the novel-context test. Dev smoke only if it runs.
6. **Finish the public-docs refresh** (branch `docs-refresh-readme`, worktree `/home/dant123/Projects/sim-worktrees/docs-refresh`):
   fix every item in `docs/drafts/docs-refresh/ISSUES.md` (25 honesty-check issues: stale numbers, contradictions,
   overclaims, conflated mechanisms), re-checking each number against the ledger/findings; keep the honesty boundary;
   log in that folder's WORKLOG.md; commit there; never merge (Claude reviews and publishes after Tuesday).
5. **Test-suite triage** -> `docs/drafts/test-triage/`: the resumable full-suite command from "Parked for Claude"
   (2+ hours; start it on the pool or locally under memcap and do other tasks meanwhile), then group failures by error.
   `cudaErrorNoDevice`/GPU-hidden = ENVIRONMENT; list the rest with the first error line. No fixes.

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
- Local-llm host RAM: the unit runs under `MemoryMax=12G`; on 2026-09-25/26 it hit a 12G memory peak AND 12G swap peak
  (mmap'd weights + the default 8 GiB host prompt cache, `-cram`), which slows long sessions. Measure, then raise the cap or
  set `-cram` in the 128K profile.
- Small fixes: findings cite the git-ignored receipt `research/queue/.corpus_checks.jsonl` (if a commit in a worktree is
  blocked for it, copy that file from the main checkout into the worktree); the GPU auto-swap's two LOW review notes (validate
  the profile name before writing the restore marker; `|| true` on the test-only stop path).

## NEEDS CLAUDE

<!-- Append one line per escalation: `- YYYY-MM-DD HH:MM  <what happened>  <what you already tried>` -->

_(empty)_
