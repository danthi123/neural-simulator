# Local-LLM runbook v4 (2026-09-30; Hermes on the server; covers about 2026-10-01 to 2026-10-14)

**For** Hermes Agent (official image, `~/apps/hermes` on homelab-server), default profile, driving from
`/home/dant123/Projects/neural-simulator` (S, canonical home since 2026-09-30 —
`docs/plans/2026-09-30-home-to-server-MIGRATION.md`) via its local terminal backend; it talks to the local model
(Qwen3.8-27B, llama-swap `qwen38-27b-iq4nl-mtp-128k-q4`) served from the workstation at `http://192.168.0.68:8080/v1`.
The repo-driven `tools/local_llm/llm.sh claude` session is retired; the 2026-10-06 date in
`docs/plans/2026-09-29-local-model-transition-PLAN.md` is superseded by this move. Claude Code inspects occasionally
from the workstation checkout `/home/dant123/Projects/sim` (W, now a follower + compute node). One driver at a
time: if the owner says a Claude session is driving, run only `bash tools/status.sh`. Read this whole file at the
start of every session; after a compaction read only sections 2-3, your task in section 4 and `tail -n 25` of its
WORKLOG. Each task links a packet HANDOFF: read only the section you execute. The HANDOFF governs its commands and
tables; this runbook governs order, permissions, `max_age_h`, verdicts, NEEDS-CLAUDE and where results go.
Set in every Bash call that uses them: `P=/home/dant123/Projects/neural-simulator; PY=$P/.venv/bin/python;
WT=$P/.claude/worktrees`. **Work only on S (2026-10-01, replaces the 09-30 caveat).** Every packet branch is on S:
a local `research/<packet>` branch, else `origin/research/<packet>`. If `$WT/<packet>` is missing, create it with
`git worktree add $WT/<packet> research/<packet>` (or `git worktree add -b research/<packet> $WT/<packet>
origin/research/<packet>`). Never read, edit or commit in the workstation's checkout (`/home/dant123/Projects/sim`,
reached over ssh): commits there are invisible here and to the inspector. Queue GPU jobs only with `bash
tools/gpu_remote.sh add` from a clean, committed `$WT/<packet>` (section 3); a `/home/dant123/Projects/sim/...` path
in an older task line means the same packet under `$WT`. Committed files under `$P/research/findings/raw/...` arrive
from the pool and the workstation through `sim-sync.timer`. "#n" = the HANDOFF's NEEDS-CLAUDE item n.

## 1. Every session

1. `bash tools/status.sh` (its READY rows `b2b_base`, `arcc_awake_completion` are stale: scored 09-29).
2. Packet status (`running>0` = still running):
   ```bash
   cd /home/dant123/Projects/neural-simulator && bash -c 'for k in _long_delay_credit_bg/sweep _long_delay_credit_bg/gate _iw2/identity iw2-R- iw2-flip- _sleep_transitive_inference/parts raw/_host_share/gate _stage1_host_share/ _stage1_exit/gate _stage1_c1_category/gate _bg_live_seam/gate _iw1/identity iw1-R- iw1-flip- b2d0929/ _stage2_exit/dev; do q=$(grep -c -- "$k" research/queue/pool.queue); h=$(grep -c -- "$k" research/queue/pool.queue.held 2>/dev/null); r=$(for n in pool40 pool41 pool42 wspool; do ssh -n -o BatchMode=yes -o ConnectTimeout=5 $n "pgrep -fc -- $k" 2>/dev/null; done | awk "{s+=\$1} END {print s+0}"); echo "$k queued=$q held=${h:-0} running=$r"; done'
   ```
3. Read section 6 (RULINGS) and your open lines under NEEDS CLAUDE.
4. Work the FIRST task in section 4 whose "Start" holds and that no open NEEDS CLAUDE line blocks. After every step
   append to its WORKLOG: date/time, command, result paths, anything unsure (guesses marked). While pool jobs run,
   move on; never wait idle on one task.
5. Nothing ready: section 5, else stop. Never invent work; queue only what a task names.

## 2. Hard rules

- `$P` stays on `main` (live services run from it). Packet work: the packet's worktree. Your own builds:
  `$WT/draft-<topic>` on `research/draft-<topic>`, dev seeds only (7, 1-15).
- Never: merge, except your routine work through `tools/agent_merge.sh` (RULING `routine-merges`) and main into your own branch through `tools/agent_sync_branch.sh` (RULING `sync-branch`); `--no-verify`; a waiver file; a filler job; write a file a gate reads (`lane_builds.jsonl`, queue
  files, ledgers); delete data; touch AWS; edit `sim/`, `webapp/`, `tools/gates/`; flip a default on main; change a
  prereg, bar, arm, seed list, scorer or job line (except adding `max_age_h`). Gate seeds (42/43/44/100/101/102) run
  only through lines a HANDOFF registers. Scripted exceptions: the iw1/iw2 flip commit on the packet branch (owner
  ruling 09-29), the LDC Amendment A text, and T15's field rename and correction notes (sleep-TI ruling 2).
- DENIED to you: `git merge`, `git checkout`, `git switch`, `gpu_queue.sh add`. A step that needs one (a merge
  `tools/agent_merge.sh` refuses, `git checkout -- .`) is a STOP: NEEDS-CLAUDE. Instead of `git checkout -- <runs.jsonl>`: never
  `git add research/findings/raw/_provenance/runs.jsonl`. Never wrap a denied command in `bash -c` or a script.
- A command that asks for permission while the owner is away (always so in a headless resume): skip it, WORKLOG it,
  continue elsewhere.
- Never run `tests/test_game_pause_resume.py` (it kills live GPU jobs). One fix round per issue; a second failure is
  a STOP. S is not a compute node (RULING `compute-placement`); a short run here still goes under `bash
  tools/mem_ok.sh <gb>`, then `bash tools/memcap.sh <gb> -- ...`.
- Context: `grep -n` first, then Read at most 120 lines; never a whole source file or HANDOFF; never re-read a file
  you are writing (append; `tail -n 20`); pipe output through `head -40` / `cut -c1-200`.

## 3. Standing operations

**Pool (dispatch runs on S).** `sim-pool-dispatch.service` (host unit, user `sim`, not a user unit) dispatches
`pool.queue` from S to pool41/pool42; the queue files are the same `research/queue/pool.queue*` commands below, just
under S's `$P`. Every line you add carries `max_age_h=N` in its `--checked` reason (N = expected wait in hours, at most
240; without it the dispatcher skips the line after 12 h). If a HANDOFF line lacks it, append ` max_age_h=168`,
nothing else. The dispatcher is FIRST-FIT in file order: a node (15 GB, 2 reserved) takes the first fresh line whose
`mem_gb` fits, so 2 GB lines slip past 7-11 GB ones; heavy lines run one per node. An expired line stays in
`pool.queue` but never runs: when a packet's unrun lines are within 24 h of expiry (`bash tools/pool_queue.sh list`
shows ages), NEEDS-CLAUDE "[<task>] <n> lines expire <date>". Never re-add queued, held or dead lines.
**Pool canary (2026-09-30).** `pool_queue.sh add` refuses a gate-seed line until a gate-seed line of the same runner at
the same revision has passed on a pool node (`bash tools/pool_canary.sh list`; every requeue_*.tsv pin already has
one). A refusal naming the pool canary is a STOP: NEEDS-CLAUDE "[<task>] needs a pool canary at <sha>".
**Re-queue missing shards ONCE** (only when the task says; section 1 must show queued=0 running=0 for the key; the
block stops if any line at the file's revision is still queued or held, and re-adds only shards whose `--out` is absent):
```bash
cd /home/dant123/Projects/neural-simulator && F=<requeue file> bash -c 's=$(head -1 "$F" | grep -oE "revisions/[0-9a-f]{9}"); n=$(cat research/queue/pool.queue research/queue/pool.queue.held 2>/dev/null | grep -c "$s"); [ "$n" = 0 ] || { echo "STOP: $n lines at $s queued/held"; exit 1; }; m=0; while IFS=$'"'"'\t'"'"' read -r cmd why; do out=${cmd##*--out }; [ -f "$out" ] && continue; m=$((m+1)); bash tools/pool_queue.sh add "$cmd" --checked "$why" || echo "ADD-FAILED: $out"; done < "$F"; echo "re-queued $m"'
```

**GPU (runs on W via `gpu_remote.sh`).** The GPU queue and the local-llm server (llama-swap) live on the workstation
(W); from here queue a job with `bash tools/gpu_remote.sh add 'cd <clean committed checkout> && ...'` (gate seeds
also `--prereg <committed path>`) — it pushes your committed revision to W as `refs/server-jobs/<sha>`, builds a
sparse per-job worktree there and execs that worktree's own `gpu_add.sh`; you never `ssh` by hand. `add --rev <sha>
'<cmd>'` runs at a pinned older commit on a branch instead of HEAD (a gate pin; only where a HANDOFF says so). Status: `bash
tools/gpu_remote.sh status`; log: `bash tools/gpu_remote.sh log [N]`. Your queued jobs wait while your model is
up (it shares the 3090), so hand the card over at the END of the run that queued them: commit, WORKLOG "yielding
for <jobs>; resume at <task/step>", then `bash tools/gpu_remote.sh yield --max-hours N` (1-12; the jobs' expected
hours plus margin) as your last command (the model stops ~45 s later). The queue then runs the jobs; when it
drains, or N hours pass, W restarts llama-swap with the model unloaded. Your `sim-driver` wake gate starts
the next run within ~5 min of llama-swap answering, with "Woken: the workstation model server just came back" at
the top of the prompt. If the owner games mid-window, the window pauses and gets the time back. Results come back
through `sim-sync.timer`'s 15-min pull (`POOL_NODES=workstation bash tools/pool_sync.sh`), not directly from this
session: if a job's `--out` is not under `research/findings/raw/...` yet, do another task and harvest next run
rather than polling W.
`tools/local_llm/gpu_add.sh` / `llm.sh yield` run directly only when actually on W (e.g. Claude inspecting there).
**Model unreachable.** When the owner is gaming on W, `game on` stops llama-swap (and pauses W's GPU queue); the
wake gate then skips runs until `game off` restarts it (or resumes your GPU window). If a call still fails
mid-run with a connection error: do not retry in a loop; WORKLOG "model unreachable" and end the run. The same applies if
llama-swap is down for another reason (check `bash tools/gpu_remote.sh status`'s remote log, or ask Claude to check
`journalctl --user -u local-llm` on W). A cloud `fallback_model` can be added to Hermes's config for this case but is
not set up yet.

**Commits.** Pathspec only (never `-A` or `.`); `git diff --cached --name-only` lists only what you meant. Blocked
only by `compute_idle_persistent` / `lane_starvation`: retry every 30 min (max 6), then commit the Markdown part
(exempt) and NEEDS-CLAUDE; any other gate: STOP, NEEDS-CLAUDE with its message. Packet/draft branches stay on S
(RULING `push-credential`): never push them; wherever a task says "push", commit on the branch and name
`<branch> @ <sha>` in your RESULT line. On `main` commit ONLY this file: `git add
research/coordination/LOCAL_LLM_RUNBOOK.md` (the checkout has unrelated uncommitted files: the staged list must be
exactly this file), `git commit -m "runbook: <what>"` — then STOP: do not run `push_both.sh main` yourself.
`sim-sync.timer` (host unit on S, user `sim`, every 15 min) publishes `main` to both remotes via `sim-publish`;
do not check the remotes (RULING `private-repos`): a failed publish fails the unit, which alerts the owner.

**Verdict rule.** Every GO/NO-GO/PASS/FAIL you write is PROVISIONAL until Claude inspects it or records a matching
re-derivation with the registered scorer. Keep the HANDOFF's finding template; under the H1 add `**PROVISIONAL**:
written by the local model from the registered scorer's output; not yet inspected by Claude.`, and never remove it.
Numbers only from the scorer's output; words only the HANDOFF's fixed sentences.

**Results.** Per scored packet: the finding + raw files + scorer output committed and pushed on the packet branch;
the HANDOFF's board line in the WORKLOG and as a RESULT line under NEEDS CLAUDE. Where a HANDOFF says to edit
`GAP_CLOSURE_MISSION.md` or Vikunja, write the RESULT line instead (Claude syncs both).

**Units.** On S (host units, user `sim`): `systemctl status sim-pool-dispatch.service sim-sync.timer`; logs via
`journalctl -u sim-pool-dispatch` / `-u sim-sync`. You are not `sim` and cannot restart them (never `sudo`): a failed
or stopped unit is NEEDS-CLAUDE "[units] <unit> <state>". W's own `pool-dispatch`/`pool-sync` user units were
disabled at cutover and stay off; `memcap.sh` does not run inside Hermes's container (heavy local work: route to the
pool instead, per CLAUDE.md non-negotiable 10).
**Inspection.** When the owner says Claude will inspect, and after a session that wrote a RESULT: `mkdir -p _scratch
&& $PY tools/inspection_digest.py > _scratch/inspection_digest.md`, check your NEEDS CLAUDE lines appear in its
section 3, give the owner the path. Never `--mark` (Claude's).

## 4. Tasks, in expected landing order

At writing (09-30 16:55) Claude was re-queueing the lines held at 14:00 (`research/queue/pool.queue.held`), in
this order: LDC tail, iw2 D/R, host-share, BG seam, sleep-TI, Stage-1 (iw1, B2d, C1 expected next). Lines pinned to
`59edf3255`, `4f6d9d7fb` or `bb6705de6` are DEAD (worker gate-seed bug; replaced by
`research/queue/requeue_*_{f7f2bd169,c238ff325,6c2cb1362}.tsv`). The pool stays full to about 10-12.

**T0. Queue check** (every session). A key already queued at writing (not `.../gate` of LDC, not `-flip-`) that
shows queued=0 running=0 before its task's Start count: NEEDS-CLAUDE "[pool-hold] <keys>" once. Never move lines
between queue files.

**T1. Long-delay credit sweep -> Amendment A -> gate.** `$WT/long-delay-credit-spiking/docs/drafts/long-delay-credit/HANDOFF.md`
steps 4-6; WORKLOG `docs/drafts/long-delay-credit/SPIKING-WORKLOG.md`. Start:
`ls $P/research/findings/raw/_long_delay_credit_bg/sweep/ | grep -v prov | grep -c json` = 243, key queued/running 0.
First: step 4's `cp -r` + instrument command (dev seeds: no verdict). rc=0: step 5; a `chosen` other than `base`
needs a CreditParams edit that asks permission (refused or owner away: NEEDS-CLAUDE "[ldc-amendA] <tag>", stop).
Then step 3, and step 6 with ` max_age_h=168` after `mem_gb=2`. Stop: rc=2 (#4), a crash twice (#5), a gate (#2).

**T2. Integration wave 2, D/S/R.** `$WT/integration-wave2/research/packets/integration-wave2/HANDOFF.md` sections 5-8
(P=`573a9438f`); WORKLOG `research/packets/integration-wave2/WORKLOG.md` (create). Start:
`ls $P/research/findings/raw/_iw2/identity/573a943/*.json | grep -vc prov` = 5 and
`ls $P/research/findings/raw/_load_bearing/_shards/iw2-R-*-573a943/s*/*/lb.json | wc -l` = 36. First: section 5's D0
and D_p lines; its R lines only after RULING `iw2-R9`. Then the table and section 6 (flip commit), with
`neural-selfid` in FLIP only after RULING `iw2-selfid`. Write: `copy_results.sh`, verdicts, PROVISIONAL NO-GO drafts
the tables name, RESULT. Waves: T11.

**T3. Sleep transitive inference.** `$WT/sleep-transitive-inference/drafts/HANDOFF.md` steps 7-8 (1-6 done, gate at
`bfde9a195`). It is UNTRACKED: first `cd $WT/sleep-transitive-inference && git add drafts/HANDOFF.md && git commit
-m "docs(sleep-ti): execution packet"`, push. WORKLOG `WORKLOG-sleep-transitive-inference.md`. Start:
`ls $P/research/findings/raw/_sleep_transitive_inference/parts/seed*_*.json | wc -l` = 30. First: step 7 from `$P`, then
`mkdir -p $WT/sleep-transitive-inference/research/findings/raw/_sleep_transitive_inference && cp -a $P/research/findings/raw/_sleep_transitive_inference/. $WT/sleep-transitive-inference/research/findings/raw/_sleep_transitive_inference/`,
the step-8 finding on the branch, commit, push. Predicted STI NO-GO (STI7). Every outcome hands back (#8; U1/U2/U3/U7: #7).

**T4. Host-share baselines. DONE 2026-10-02** — BASELINE-VALID finding merged `c0679d474` (gate `f7f2bd169`); the
t4-fire-and-dev re-queue re-run re-scored 2026-10-11 reproduced it exactly (WORKLOG `6651c5ac1`), re-run cells not
committed per RULING host-share-clobber. The count=108 Start re-trips on restored data: do NOT re-score, sections 7-11
are closed (BASELINE-VALID #8 was filed 10-02).

**T5. BG live-seam retirement.** `$WT/selection-seam-need-reversal/docs/drafts/selection-seam/HANDOFF.md` steps 6-8
(gate `0bc62cc31`); WORKLOG `docs/drafts/selection-seam/WORKLOG.md`. Start:
`ls $P/research/findings/raw/_bg_live_seam/gate/s*/*.json | grep -vc prov` = 42. First (results land in `$P`):
`mkdir -p $WT/selection-seam-need-reversal/research/findings/raw/_bg_live_seam/gate && cp -a $P/research/findings/raw/_bg_live_seam/gate/. $WT/selection-seam-need-reversal/research/findings/raw/_bg_live_seam/gate/`, then step 6's score in the worktree. Missing seeds: step 5's line once, `SHA=0bc62cc318278839a77293358a6fa9ad9e36a516`,
` max_age_h=72` appended. Write: step 8 (today's date), RESULT; not GO: #5.

**T6. Stage-1 exit gate.** `$WT/stage1-exit/docs/drafts/stage1-exit/HANDOFF.md` sections 6-10 (GATE `c238ff325`);
WORKLOG `docs/drafts/stage1-exit/WORKLOG.md`. Start: section 6's count = 204; stalled 6 h below it: re-queue ONCE
with `F=research/queue/requeue_stage1_exit_gate_c238ff325.tsv` (never `stage1_exit_gate_jobs.tsv`). First: section 7.
Write: finding + section-8 sentences, RESULT; any PASS: #7. Exit due 10-24: under 204 on 10-17: NEEDS-CLAUDE
"[stage1-window] <n>/204".

**T7. Stage-1 C1, category inheritance.** `$WT/stage1-c1-category/research/packets/stage1-c1-category/HANDOFF.md`
sections 4-8 (GATE `6c2cb1362`; never steps 1-3, its own re-queue block); WORKLOG
`research/packets/stage1-c1-category/WORKLOG.md`. Start: section 4's count = 54. First: section 5. Write: finding,
section-6 sentences, RESULT. Nothing landed by 10-12: #10.

**T8. Integration wave 1, D/R.** `$WT/integration-wave1/research/packets/integration-wave1/HANDOFF.md` sections 5-8
(P=`049d4bb0b`); WORKLOG `research/packets/integration-wave1/WORKLOG.md` (create). Start:
`ls $P/research/findings/raw/_iw1/identity/049d4bb/*.json | grep -vc prov` = 5 and
`ls $P/research/findings/raw/_load_bearing/_shards/iw1-R-*-049d4bb/s*/*/lb.json | wc -l` = 18. First: section 5,
then the table and section 6; section 7 only after RULING `iw1-stepB`. Write: as T2.

**T9. B2d stub arm.** `$WT/b2d-mouth-renderer/research/packets/b2d-mouth-renderer/HANDOFF.md` sections 4-10
(P=`5818dc629`); WORKLOG `research/packets/b2d-mouth-renderer/WORKLOG.md` (create). EVERY session: section 4's
"Every later session" line + ledger commit; section 5's check (mouth lines queued by Claude: yield
`--max-hours 12`). Score (sections 7-9) when 258 stub cells have landed and the mouth arm has landed or RULING
`b2d-mouth` retires it. Write: section 10, RESULT, #2.

**T10. Long-delay credit gate.** T1's HANDOFF step 7 + outcome table. Start: 150 files in
`$P/research/findings/raw/_long_delay_credit_bg/gate/`. Write: failing gates, `mechanisms`, `signatures` in the
WORKLOG, the data commit, RESULT; the finding prose is Claude's (#7-#9).

**T11. Waves B/F (iw2, then iw1).** Each HANDOFF's section 7: 43 flip lines when fewer than 10 `iw2-flip`
(`iw1-flip`) lines are queued; log `waves.tsv`; score; decision table. Merging is Claude's (#1).

**While waiting (no pool; in order):**

**T12. Seam-map amendment.** `$WT/draft-seam-map`. Start: `grep -c "Amendment 2026-09-29"
research/findings/2026-09-27-tone-bg-nchannel-tonic-firing-wall.md` = 0. Append the quoted block of T5's HANDOFF
("Separate staged work", second bullet) without the leading `> `; Markdown commit; push `research/draft-seam-map`.

**T13. Conductance-overshoot guard.** Branch `research/conductance-overshoot-guard` (on S since 10-01; make
`$WT/conductance-overshoot-guard` from it). The first cupy run failed 3/4 in the test file, not the feature (NumPy
conversion of CuPy arrays); Claude fixed the test (9e8edbec1) and queued the cupy re-run on the workstation 10-01
12:40. Start: `bash tools/gpu_remote.sh log 400` shows a `DONE(rc=` line for `test_bounded_conductance_integration.py`
after 2026-10-01 12:40. Write its rc and pass count in the WORKLOG; rc=0: RESULT line "ready to merge"; rc!=0: STOP
line with the failing test names.

**T14. Stage-2 dev arms.** When 8 new files land in `$P/research/findings/raw/_stage2_exit/dev/s7/` (or their
`max_age_h=24` lines expire): INFO line. The gate is queued 11-03..11-10, never before 11-01
(`$WT/stage2-exit-gate/docs/drafts/stage2-exit/HANDOFF.md` section 1).

**T15. Ecker CA3 OU-noise field** (sleep-TI ruling 2). Nine `research/runners/_gap5_*` set `cfg.ou_noise_sigma_pa`,
which the engine never reads (it reads `ou_std_current_pA`, default 100 pA in `sim/config.py`): runs reported at
40 pA ran at 100. `git worktree add .claude/worktrees/draft-ecker-ou-field -b research/draft-ecker-ou-field main`;
`grep -n "ou_noise_sigma_pa\|ou_std_current_pA" research/runners/*.py`. A runner setting only the phantom field:
rename it to `ou_std_current_pA` (same value, a comment naming T15), `$PY -m py_compile` it. One setting both
(`_gap5_onebrain_capstone.py`): no edit, NEEDS-CLAUDE. Per runner `grep -ln "<stem>" research/findings/*.md`; table
(finding, runner, stated value, value run) in `docs/drafts/ecker-ou-field/AFFECTED.md`; append to each affected
finding a dated `Correction (PROVISIONAL, local model)` paragraph with both values, nothing re-interpreted. Commit,
push, no runs. Then NEEDS-CLAUDE "[ecker-ou] FAILURE_LOG row + gate (task_8f9171ae)".

**T16. chat-fact-fixes.** `$WT/chat-fact-fixes/research/packets/chat-fact-fixes/HANDOFF.md`: nothing unless RULING
`cff-reverify` (then its section 2).

**T17. Phantom-cfg audit.** `$WT/draft-phantom-cfg` on `research/draft-phantom-cfg` from `main`. Start: T15's RESULT
line exists and `docs/drafts/phantom-cfg/AFFECTED.md` does not. Do RULING `phantom-cfg-audit`; Markdown only.

**Session-2 packets (cloud Claude, 2026-10-01; branch `claude/session2-packets`).** T18-T20 move the three host
decisions RULING `iw2-stage1-credit` names into the spiking substrate. T21 then makes them the default paths. Start
each when `git -C $P log --oneline -1 -- tools/s2p_score.py` prints a commit (the owner landed the branch).
T18-T20 are independent: run them in parallel, since the pool takes all three.

**T18. fb-spiking-witness** (who saw the move: a spiking presence/absence flip-flop). `$WT/fb-spiking-witness`,
`docs/drafts/fb-spiking-witness/HANDOFF.md` sections 0-5 (it creates the branch from main); WORKLOG
`docs/drafts/fb-spiking-witness/WORKLOG.md` (create). Write: selection, O/R verdicts, finding, RESULT.

**T19. transitive-spiking-reach** (the chain walk and the reachability test as a spiking wave + goal detector).
`$WT/transitive-spiking-reach`, `docs/drafts/transitive-spiking-reach/HANDOFF.md` sections 0-5; WORKLOG
`docs/drafts/transitive-spiking-reach/WORKLOG.md`. Write: as T18.

**T20. word-role-parse** (the self-ID role map reads the words: a Hebbian word x position -> role parser).
`$WT/word-role-parse`, `docs/drafts/word-role-parse/HANDOFF.md` sections 0-5; WORKLOG
`docs/drafts/word-role-parse/WORKLOG.md`. Write: as T18.

**T21. s2p-flip** (the three spiking paths become their faculties' defaults; full Stage-1 credit).
`docs/drafts/s2p-flip/HANDOFF.md`. Start: the iw2 flips are on main (RULING `iw2-flip-hold`: after steps B and F
passed and Claude merged them), AND Claude has confirmed at least one of T18-T20 with O PASS + R PASS. Flip only the
confirmed ones. Due before the Stage-1 exit attempt (10-24): if not started by 10-17, NEEDS-CLAUDE "[s2p-window]".

**Session-3 packet (cloud Claude, 2026-10-02; branch `claude/session3-packets`).**

**T22. stage1-host-share** (who owns the decision at the Stage-1 C2/C3 seams, by intervention; T6 #10). `$WT/stage1-host-share`,
`docs/drafts/stage1-host-share/HANDOFF.md` sections 0-6 (it creates the branch from main); WORKLOG
`docs/drafts/stage1-host-share/WORKLOG.md` (create). Start: `git -C $P log --oneline -1 -- research/runners/stage1_host_share.py`
prints a commit. Independent of T6 and T18-T21 (its own 22 dev + 132 gate lines; FIFO puts them behind T6's gate). Write:
devcheck, verdict, finding, RESULT. Under 132 on 10-21: NEEDS-CLAUDE "[s1hs-window]".

**Session-4 GPU packets (cloud Claude, 2026-10-02; branch `claude/session4-gpu`).** The pool is the bottleneck and the
3090 sits idle. No queued pool task can move to the GPU (RULING `gpu-audit`), so these two give the GPU its own work.
Both queue from S with `bash tools/gpu_remote.sh add` and run one brain process at a time; T23 first (FIFO). Work
them alongside the pool tasks: they never wait on the pool, and the pool never waits on them.

**T23. gpu-b2d-mouth** (B2d's registered 18-job mouth arm, Qwen on cupy; about 12 GPU-h). `docs/drafts/gpu-b2d-mouth/HANDOFF.md`
sections 0-6, from `$WT/b2d-mouth-renderer` (T9's worktree); WORKLOG T9's. Start: `git -C $P log --oneline -1 --
docs/drafts/gpu-b2d-mouth/HANDOFF.md` prints a commit. It replaces T9's "mouth lines queued by Claude" wait and the
B2d HANDOFF's section 5 (RULING `b2d-mouth-gpu`). Write: the mouth score on the packet branch, RESULT; the B2d
finding stays T9's step 10.

**T24. gpu-s1-lived-cupy** (Stage-1 C3 `lived` on cupy, the gate's runner at `c238ff325`, informational; about 7-14
GPU-h). `docs/drafts/gpu-s1-lived-cupy/HANDOFF.md` sections 0-4; WORKLOG `docs/drafts/gpu-s1-lived-cupy/WORKLOG.md`
on `research/s1-lived-cupy` (create). Start: as T23, with `docs/drafts/gpu-s1-lived-cupy/make_jobs.sh`. Queue now;
aggregate and write up only after T6 is scored (RULING `s1-lived-cupy`). Write: devcheck, aggregate, the
informational finding, RESULT.

## 5. Idle: self-directed draft research (owner 2026-10-01; only when no section 4 task can move)

Test triage is done (RESULT 10-01). When nothing in section 4 can move, run ONE draft at a time:
1. Pick the first open next-item in `GAP_CLOSURE_MISSION.md` CURRENT STATE, else the master roadmap's "next to
   queue" (`docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md`), preferring what serves the next stage exit (Stage 1
   attempt 10-24, Stage 2 11-21, Stage 3 12-19) and integration-first (a faculty on by default whose lesion changes
   the reply). Write the choice and why in `docs/drafts/<topic>/WORKLOG.md`.
2. Research the record first: `bash tools/before_you_build.sh "<defect>"` and `.venv-rag/bin/python
   tools/rag/rag_search.py "<question>" 5 --corpus all`; read the sources it points to.
3. Plan with one MiMo Pro delegate: `docs/drafts/<topic>/PLAN.md` with the question, the biology it follows, dev seeds
   only (7, 1-15), and an outcome-to-reading table written BEFORE any run. A second delegate must try to refute the
   plan against CLAUDE.md's non-negotiables (brain-based only, one brain, no host shortcut, honest measures); revise
   until it approves.
4. Build in `$WT/draft-<topic>` on `research/draft-<topic>`: new `research/runners/_<topic>_*.py` copies; never edit
   `sim/`, a registered runner, a scorer or a gate. Run it per RULING `compute-placement`.
5. Report as a PROVISIONAL finding (`research/findings/<date>-<topic>.md`: dev seeds only, what it does NOT show),
   check it per RULING `verify-go`, merge per RULING `routine-merges`, then NEEDS-CLAUDE "[draft-<topic>] RESULT: <one
   line>; proposes <next step>" so the weekly review can promote it. Gate seeds, preregs, scorers, default flips and
   `sim/` stay Claude's; a draft never claims a GO.

## 6. RULINGS (Claude writes `- <KEY>: <ruling> (<date>)`; you follow it)

- push-credential: by design there is no push credential on S and `sim-publish` publishes `main` only. Packet and
  draft branches live in this checkout (ZFS-snapshotted and backed up); do not run `push_both.sh`. After committing on a
  branch, name `<branch> @ <sha>` in the RESULT line; Claude reviews and merges on S. GPU jobs need no push
  (`gpu_remote.sh` sends the revision to W itself). (2026-09-30)
- private-repos: GitHub and Gitea are private and S has no credential, so never `git fetch`, `pull` or `ls-remote` a
  remote from here. Every pushed `research/*` branch was imported as `origin/research/<name>` (2026-09-30); later
  branches are created here. Start a packet/draft worktree from `origin/research/<name>` or the local branch.
  (2026-09-30)
- permissions: the seeded item was about Claude Code's workstation allow list, which Hermes does not use. Under
  Hermes, scheduled runs may use ssh, rsync, `source`, file edits, `bash -c`/`-lc`, pytest and git without a
  prompt; `python -c`/`-e` one-liners are still refused: write the snippet to `_scratch/<name>.py` and run that.
  (2026-09-30)

- T1-LDC: Step 4 (rc=2) is an instrument failure, not a verdict: the outcome DA never signed (prereg 7.3), so no
  credit gate ran. No Amendment A, no CreditParams edit, no gate seed, never re-run Step 4. Bank the sweep once, then
  run Step 4a (outcome-timing pilot, your draft build, dev seeds) exactly as the LDC HANDOFF says
  (research/long-delay-credit-spiking @ bab667eb7, which also fixes both scorers' `.prov.json` glob). After the Step 4a
  RESULT, T1 waits for `## Step 4b` in that HANDOFF (Claude). (2026-10-01)
- iw2-R9: shared prereg Amendment 1 is committed: a capability cell whose LESION reply still states the content reads
  DEFINED `lesion-kept`, an R FAIL. It lives on the LOCAL branch research/integration-wave2 (@ 758519b3f): start the T2
  worktree from it, never from origin/research/integration-wave2; score R only when
  `grep -c lesion-kept tools/iw2_score.py` >= 1 there. (2026-10-01)
- iw2-selfid: ACCEPT. In T2 section 6, neural-selfid goes in FLIP only if its section-5 row reads D0 PASS,
  D_neural-selfid DIFFERS-IN-SCOPE and R_si PASS, and in its identity on-arm file at the flip pin the "what are you?"
  and "who are you?" turns each read `abstained` true or `recalled_svo[0]` == "brain". Otherwise flip the others,
  leave it out and write NEEDS-CLAUDE "[iw2-selfid] turn <n>". The Stage-1 C2 "it" self-alias risk is accepted
  (Claude checks B1's anaphora rows at merge). Say in its finding's Deviations that these questions now abstain.
  (2026-10-01)
- iw1-stepB: fixed. `pool_queue.sh add` now checks only the flags after the runner's own `-m` (main), so guarded iw1
  lines queue. Identity lines pin PYTHONHASHSEED=0 on the LOCAL branch research/integration-wave1 (@ 2c6ae7591): start
  the T8 worktree from it. The 5 landed P-pin identity files stand (no re-run). Proceed with T8 section 7. (2026-10-01)
- b2d-mouth: keep and run the registered 18-job mouth arm (a dev-seed prediction is not a gate result). Claude queues
  it on the GPU queue (`gpu_queue.sh add` is denied to you); when its lines are queued, yield per T9 section 5. Score T9
  once the 258 stub cells and the 18 mouth cells have landed. (2026-10-01)
- c1-lane: no reorder, no AWS. The pool clears the Stage-1/C1 lines inside their 168 h expiry and the 10-24 exit.
  Only if T6's 10-17 tripwire or T7's 10-12 tripwire fires, write its NEEDS-CLAUDE line (Claude decides AWS). (2026-10-01)
- T12-seam-map: done. Amendment a6aff478c is correct and stays on research/draft-seam-map in this checkout
  (push-credential: nothing to push; a draft branch is not merged to main by this task). T12 is closed. (2026-10-01)
- ecker-ou: the 8 renames on research/draft-ecker-ou-field are correct, but AFFECTED.md is incomplete: runners that
  import an affected builder ran under the same bug. On that branch (T15's correction-note exception): list every
  runner that imports `_gap5_ecker_adex_ca3_replay_derisk`, `_gap5_ecker_adex_ca3_stdp_band_derisk` or
  `_gap5_ecker_reactivates_REAL_d5_derisk` (`git grep -l`), the findings that ran each, stated vs run OU value, and
  add the same PROVISIONAL note to each finding (at least: braun-gapcoding, btsp-directional-write,
  conduction-delay-directional-replay, ecker-adex-store, graded-recall-instrument and reverse-edge 2026-08-27
  learn-through-use, forward-band-homeostatic-scaling-ltu 09-16, read-noise-floor K18 09-17). Also append to
  AFFECTED.md: "_gap5_onebrain_capstone.py's ou_noise_sigma_pa=40.0 is dead code behind an always-false hasattr
  guard and was not an intended value (it would reverse the line's own 'kill the composer's OU noise' intent); no
  finding needs a correction." No re-runs. RESULT; then Claude merges and decides re-runs. The FAILURE_LOG row and
  gate PF (`tools/gates/cfg_phantom_field.py`) are on main. (2026-10-01)
- phantom-cfg-audit (task T17, after T15): run `.venv/bin/python tools/gates/cfg_phantom_field.py` on main. For each
  flagged runner record the field written, its value, the real field and its default (sim/config.py), whether the
  value differs from the default, and the findings that ran the runner. Write docs/drafts/phantom-cfg/AFFECTED.md on
  research/draft-phantom-cfg; no code edit, no re-run; RESULT. List first `_emerge87_reservoir_coresident_derisk.py`
  (a non-default value dropped; a GO finding cites it); most others are `cfg.dt` where the field is `dt_ms`.
  (2026-10-01)

- models: since 2026-10-01 (owner) you run on MiMo v2.6 Flash in the cloud (OpenRouter, data_collection=deny) and
  the workstation's local model is your automatic fallback, so gaming and GPU windows no longer stop you; still yield
  the GPU for GPU jobs as section 3 says. Delegated subagents run on MiMo v2.6 Pro, the stronger judge: for a design
  proposal, a verdict on results, or a judgment you would otherwise raise as NEEDS-CLAUDE ASK, delegate the reasoning
  to one subagent (give it the task's files and these rules), then act on its answer and quote it in the WORKLOG.
  What section 2 reserves (merge, gates, preregs, scorers, bars, `gpu_queue.sh add`) still goes to NEEDS-CLAUDE.
  Every call costs money now: one task per run, no idle exploration. (2026-10-01)

- server-load: the server's CPU reaches 100 C and throttles, and long test runs in this container are the biggest load
  (homelab, 2026-10-01). Run any test run longer than a few minutes under `nice -n 10`, one at a time, and never start a
  full suite while another is running; prefer the test files a change touches over the whole suite. (2026-10-01)

- work-on-S: you committed T3 (f18da9f8f) and T13 (91608ce47) in the workstation's checkout because this runbook's
  header and T13 text told you to; that was the runbook's error, not yours. Both branches are now on S; the header now
  says work only on S. Judgment that section 2 does not reserve (a verdict on results, a design proposal, an ASK) goes
  to one MiMo Pro delegate per RULING `models`, not to NEEDS-CLAUDE. (2026-10-01)

- T1-stop2: main is merged into `research/long-delay-credit-spiking` (0b935bec7), so its biology gate has the fix.
  Your 488 staged sweep files were unstaged for the merge and re-staged byte-identical (checked); commit the bank as
  the HANDOFF says, then continue to Step 4a. (2026-10-01)

- T0-identity: the 2 missing iw2 identity lines (`--arm default`, `--arm off`, revision 573a943) were claimed just
  before the 09-30 20:31 dispatcher restart, which killed them; Claude re-queued both at the queue head 10-01 12:50.
  T2 starts when 5/5 identity files are in `_iw2/identity/573a943/`. (2026-10-01)

- verify-go: (owner-approved 2026-10-01) the project's `verify-go` skill is now one of your skills (read-only; the
  other project skills are disabled for you). Before you write any line that reports a PASS, a GO, or that something
  works, load it (`skill_view verify-go`) and give the result to TWO independent MiMo Pro delegates, each told to run
  the skill's checks and try to refute the claim; quote both answers in the WORKLOG. If either finds a problem or
  they disagree, write a STOP line instead of the PASS. Where the skill names a Claude-only tool, do the check by
  hand and say so. A PASS still goes to Claude to confirm. (2026-10-01)

- owner-merges: the owner now opens the Claude sessions for merges and rulings, prompted by the report's "Claude
  check-in" line (due once a NEEDS-CLAUDE line is 72 h old). Make every NEEDS-CLAUDE line self-contained so it can be
  pasted as is: the branch @ sha, what to merge or rule, and what it unblocks. (2026-10-01)

- compute-placement: (owner 2026-10-01) S is the project's home, not a compute node. On S run only checks that finish
  in about 10 minutes (a smoke test, one reproduction cell, a scorer). Anything longer, and every grid or sweep, goes
  out: CPU work to the mini-PC pool, GPU work to the workstation (`gpu_remote.sh add`). Pool lines run at a revision
  already provisioned on the nodes (`cd ~/derisk-pool/revisions/<sha>`); every line must start with that pin, because
  pool40 and wspool have no unpinned checkout and an unpinned line fails there. You can queue lines for a provisioned
  revision yourself (`bash tools/pool_queue.sh add '<line>' --checked '<task>: <why> ; mem_gb=<gb> ; max_age_h=48'`,
  dev seeds only for drafts). To provision a new revision yourself (server-side, owner-approved 2026-10-01): commit it
  on a `research/*` branch that you created from `main` (for an older branch, run `bash tools/agent_sync_branch.sh research/<branch>` first), then create an EMPTY regular file named for the full 40-hex sha: `mkdir -p
  $P/research/queue/provision.req && touch $P/research/queue/provision.req/$(git rev-parse <branch>)`. Read the result
  in `/opt/host-status/sim-provision/<sha>.json` (`state`, `reason`, `hint`; the server checks every 5 minutes). `ok`
  or `already`: queue your pool lines. `queued`, `waiting`, `running` or `deferred`, or `failed` with a
  `next_retry_utc`: wait, the server retries by itself. `refused`: read `reason`/`hint` and fix it if it is yours (for
  example, the sha is not on a `research/*` branch). NEEDS-CLAUDE when the result is `blocked`, `refused` over
  `tools/pool/provisioning/`, `failed` with no `next_retry_utc`, or when no result file appears
  within 15 minutes. GPU work still goes to the workstation (`bash tools/gpu_remote.sh add '<line>'`). Never fall back
  to S. Step 4a: Claude stopped the S grid at the owner's request (its script is now
  `/opt/data/cache/scratch/grid_driver.sh.moved-to-pool`; do not restore it), merged main into the draft branch
  (e63b22f6d, docs only, code unchanged), provisioned that revision and queued all 24 cells at the pool head. Results
  arrive in `$P/research/findings/raw/_ldc_outcome_timing/` through `sim-sync.timer`; score them from there when 24/24
  are in. (2026-10-01)

- board: (owner 2026-10-01) Keep the owner's Vikunja board current with the vikunja skill: plain-language titles, the
  one project, roadmap-level tasks only. Read it when you choose work; when a landing changes what a capability can
  do, update or close its task the same cycle. Then append one line `## YYYY-MM-DD  <what you reconciled>` at the END
  of `docs/.vikunja_sync` on your branch before `agent_merge.sh` (the vikunja-freshness gate refuses a findings merge
  after 4 findings without it). Never delete tasks or projects, and never rewrite earlier lines of that file.
  (2026-10-01)

- t18-credit: (owner 2026-10-02) T18's read-out mode, where a host threshold read gates the belief write, earns
  PARTIAL Stage-1 credit only, never FULL: the host still decides when the belief is written. If read-out is the
  operating point T18 selects, report it as PARTIAL and start the fully synaptic write as the next method (no-defer).
  (2026-10-02)

- host-share-packet: (owner 2026-10-02; REPLACED by the cloud Claude session 3, 2026-10-02) The fifth package, a host-share
  measurement at the Stage-1 seams (T6 NEEDS-CLAUDE #10), was BUILT EARLY with the owner's approval in session 3 instead
  of at the 10-08 review: runbook task T22, `docs/drafts/stage1-host-share/HANDOFF.md`. Run it as T22 says once the
  owner lands `claude/session3-packets`; still never build or edit it yourself. (2026-10-02)

- aws-capacity: (owner 2026-10-02) If T6's 10-17 tripwire fires (the pool cannot finish the Stage-1 gate in time),
  write NEEDS-CLAUDE with the queue numbers. Claude decides on AWS at the review, within the owner's cap of about
  $50/day with auto-stop. Never start AWS yourself. (2026-10-02)

- iw2-provision: (2026-10-02) Never run `tools/pool_provision.sh` on S: it needs the workstation's key and rsync,
  which S does not have. Provision a revision only through the server, per RULING `compute-placement`: `touch
  $P/research/queue/provision.req/<full 40-hex sha>` for a commit on a `research/*` branch, then read
  `/opt/host-status/sim-provision/<sha>.json`. This overrides any HANDOFF line that says to run `pool_provision.sh` or
  `gpu_remote.sh provision`. A `refused` with "does not contain published main" means: run `bash
  tools/agent_sync_branch.sh research/<branch>` first, then request the new head at once. UPDATE 00:40Z: flip F
  1fb805e9e IS provisioned on pool41 and pool42 (owner-approved one-off;
  `/opt/host-status/sim-provision/1fb805e9e0bedcce684a47388d95848daf1008a2.json` reads ok). Do NOT sync or re-pin flip
  F: run steps B and F at 1fb805e9e as the HANDOFF says. The server will soon also accept a revision when main's lead
  over it is docs-only, or when it contains a main published within the last 60 minutes (results then carry
  `behind_main`). handled: [iw2-provision] (2026-10-02)

- cloud-claude: (owner 2026-10-01) Claude may also work from cloud sessions (docs/CLOUD_CLAUDE_SESSIONS.md); the owner
  lands their `claude/<topic>` branches on main. Their rulings read like any other. When a ruling says `handled:
  [key], ...`, delete those NEEDS-CLAUDE lines yourself. Keep each NEEDS line self-contained (branch @ sha, what to
  rule, what it unblocks): a cloud session sees only GitHub, where your `research/*` branches are published.
  (2026-10-01)

- sync-branch: (owner 2026-10-01) When your branch carries an old copy of something main has since fixed (a gate fails
  on a tool main already fixed, provisioning is `refused` for published main, or the GPU queue's stale-revision guard
  refuses), bring main into it yourself: in `$P`, `bash tools/agent_sync_branch.sh research/<branch>`. It only ever
  merges main INTO a research branch, runs every gate on the result, keeps your staged files staged, takes main's
  runbook and keeps both sides' FAILURE_LOG rows. Exit 2 means it refused: NEEDS-CLAUDE with its message. Never edit
  `tools/agent_*`. (2026-10-01)

- results-without-pass: (owner 2026-10-01) A RESULT that is not a PASS or GO (a NO-GO, an UNDEFINED, an audit, a
  triage) does not wait for Claude. Write it up as a new PROVISIONAL finding, merge it per RULING `routine-merges`,
  then take the next method it points to and keep going (no-defer: a NO-GO retires a method, never the capability; run
  `before_you_build` first). Only what section 2 reserves waits for Claude (a prereg, a bar, a scorer, a seed list,
  `sim/` code, a default flip, an edit to an existing finding, confirming a PASS): put that on ONE NEEDS-CLAUDE line
  and move on to other work. (2026-10-01)

- routine-merges: (owner 2026-10-01) you merge your own ROUTINE work into main yourself; Claude keeps the rest. The
  only way: in `$P` on main, `bash tools/agent_merge.sh research/<branch>`. It merges only new results files,
  worklogs, draft docs (not HANDOFFs or preregs), new tests, new findings that say PROVISIONAL and new `_`-prefixed
  draft runners, and it runs every gate on the merge. Anything else, any deletion, or an edit to an existing finding,
  test, runner, scorer, HANDOFF, prereg, `sim/` or `tools/` is refused, and stays Claude's. Before running it, give
  `git diff main...research/<branch>` and the task's HANDOFF section to TWO independent MiMo Pro delegates, each told
  to find a reason not to merge; merge only if both approve, and quote both in the WORKLOG. A new finding stays
  PROVISIONAL until a Claude check-in confirms it. A refusal or a failed gate: NEEDS-CLAUDE "[<task>] merge
  research/<branch> @ <sha>: <the refusal>; unblocks <what>". Never edit `tools/agent_merge*` or run `git merge`
  yourself. When it refuses only for something on your own branch, fix that on the branch and run it again: a new
  finding without PROVISIONAL, add the word; a `research/coordination/` file, commit those lines on main directly,
  then on the branch `git checkout main -- <file>` and commit. Draft data files (`.txt`, `.csv`, `.tsv`, `.json` under
  `docs/drafts/`) and run-ledger lines that only append (`research/findings/raw/_provenance/*.jsonl`) are routine.
  `research/draft-phantom-cfg` already passes the policy: merge it. (2026-10-01)

- iw2-R-pre: (cloud Claude, branch claude/session1-backlog) FIXED. `tools/iw2_score.py` now writes a
  `tools.verdict.Verdict` preconditions block beside every R, B and A verdict (shared prereg Amendment 2). Step-R
  cells also need the packet flag = "1" in every sidecar. Claude re-derived R from the 36 committed shards:
  - fb, tr and si each read PASS: capability 6/6, control 6/6.
  - The per-seed results equal your unamended run.
  - The verdicts are committed at `research/findings/raw/_iw2/verdicts/573a943/R_{fb,tr,si}.json`.
  Your three staged R files are superseded. Move them aside (`mkdir -p _scratch/iw2_R_unamended && mv` them there;
  never delete them). After the owner lands the branch, run `bash tools/agent_sync_branch.sh
  research/integration-wave2`. Then T2 continues at section 6 with
  `FLIP=false-belief-chat,transitive-chat,neural-selfid`. iw2-selfid's turn condition holds at P: turns 14-15 read
  `abstained` true. Before the section-6 commit, update the board task for these three chat capabilities (vikunja
  skill) and stage `research/coordination/board_sync.json` with the flip commit. The board-sync gate refuses a ledger
  change without it; any other gate is a STOP. Provision F per RULING `compute-placement`, then section 7.
  (2026-10-01)
- iw2-verify-go: (cloud Claude) The step-R PASSes hold. Two independent skeptics reviewed them.
  - The instrument/reproducibility lens could not refute: all 219 sidecars are pinned and numpy, the flag is "1",
    the six seeds are distinct, and the controls are not vacuous.
  - The host-ownership lens bounds what may be claimed. The finding (HANDOFF section 11) must use these sentences:
  - false-belief-chat: "The spiking belief store carries the belief that the reply reads. Who saw the move is decided
    by the host witnessing parse, and that decision gates the write. The lesion forces the gate open."
  - transitive-chat: "The workspace's spiking ignition decides when the chase stops, and the reply depends on it. The
    next node is the composer's read, and the reachability test is host code."
  - neural-selfid: "This is the content-selection parser, the same positional role map, reaching self questions. The
    real change is that the host router no longer answers them."
  Count neural-selfid as a reach of the content-selection mechanism, not as a new mechanism. The false-belief chat
  stays a sub-rung of `w5-affective-tom`; it gets no cognitive-ToM row while the witnessing decision is host-owned.
  These rule iw2 HANDOFF NEEDS-CLAUDE items 2, 7 and 8. Next methods (no-defer, drafts for session 2):
  - a spiking presence/witness population driving the `witness_other` gate;
  - the composer unbind spikes driving the workspace slots, not a host loop.
  (2026-10-01)
- ecker-ou-merge: (cloud Claude) T15 is merged on claude/session1-backlog. The 8 correction notes are confirmed: the
  engine reads only `ou_std_current_pA` (default 100 pA, `sim/config.py:370`). AFFECTED.md gains two 07-25 findings
  whose scratchpad producers are gone, so their noise level is unverifiable; their notes are unchanged. Re-runs:
  - The GOs (07-25 x3, 08-20 x2) need none: they held at 2.5x the stated noise, and the notes give the run value.
  - The 08-20 "does NOT reactivate via soma recurrence" negative: excess noise could cut either way, so re-check
    it at 40 pA only if that route reopens.
  - Re-run the 09-17 K18 read-noise floor first, because noise is the quantity it measures. Run it at 40 pA once
    the rename is on main, as pool lines at a provisioned main revision. The prereg and its exact lines, the s42
    smoke first, are in `research/findings/2026-10-01-gap5-read-noise-floor-ou40-remeasure-PREREGISTRATION.md`.
    Name that prereg in `--checked`.
  - The 09-16 forward-band and 08-27 LTU NO-GOs wait on its result.
  RESULT: a PROVISIONAL finding comparing sigma_read at 40 pA against 100 pA. (2026-10-01)
- T17-phantom-cfg: (cloud Claude) merged. Fix: `cfg.enable_stp` maps to `enable_short_term_plasticity`. Findings in
  AFFECTED.md are grep mentions, which over-include (T15 showed that mentioning a runner is not running it). Scripted
  exception, as for T15:
  - For each runner in the headline table (non-default dropped), add a dated `Correction (PROVISIONAL, local model)`
    note (stated vs run value, nothing re-interpreted).
  - Add it only to findings whose frontmatter `runner:` or whose command names that runner as the producer.
  - Skip the 8 `ou_noise_sigma_pa` runners (T15 covers them) and AUTONOMOUS_STATE.
  - Commit on `research/draft-phantom-cfg`; RESULT.
  No runner edits and no re-runs. `cfg_phantom_field` blocks a flagged runner on its next touch, which fixes the
  default-equal `cfg.dt` lines with no behavior change. (2026-10-01)
- T3-sleep-ti: (cloud Claude) merged; the finding is inspected. The grader re-derives `aggregate.json`
  byte-identical, and its selftest passes. STI0-STI6 hold on the 4 defined seeds, so the night does integrate.
  STI7 fails 4/4 (registered) and STI3b fails 3/4 (not predicted). The family word stays UNDEFINED (U6 on 42 and
  101), but it cannot reach GO: this is a NO-GO for the method. Do not write a standalone U6 prereg or re-run: neither
  can change the outcome. Next (prereg section 10, your section-5 draft, dev seeds 8-11 only):
  - Add the symmetric CA3-CA3 rule (Mishra 2016) as a new arm in a `_sti_symmetric_*` copy of the runner.
  - Use the within-class weight permutation (equal pre-night weight classes) as its STI3 control from the start.
  - Report PAB and PBD retention, DI and the control's validity.
  RESULT, then Claude writes the next family prereg. Sleep downscaling is the second companion. (2026-10-01)
- triage: (cloud Claude) merged as a report.
  - The 28 REAL failures are old July demo consoles (emerge58/60/61/66: composer D 128 vs 256), v14 stage B, and 8
    `*_no_harm` tests that pin a past arc's files. None is on the production path.
  - POLLUTION (33) is test isolation: `cpd_v1.NOUN_VOCAB = ...` at import time. The fix (monkeypatch and restore)
    is an edit to existing tests, so it is a work package for a later Claude session.
  - Retiring the `*_no_harm` pins is the owner's call. Nothing for you to run. (2026-10-01)
- handled: [iw2-R-pre], [T3], [ecker-ou], [T17], [triage] (2026-10-01)
- T17-merge: (cloud Claude) research/draft-phantom-cfg @ 64899914f (round-1 and round-2 refuter fixes) is merged on
  claude/session1-backlog. Two corrections at merge:
  - The `cfg.enable_stp` heading names the real field `enable_short_term_plasticity`, not `enable_stdp`.
  - The `enable_stp` row IS a drop. The same block disables STDP separately (line 568), so `enable_stp` meant
    short-term plasticity, and STP stayed on. The count is 20 rows / 18 runners.
  Nothing for you to run. The T17-phantom-cfg ruling above stands; the 06-11 cortex decorrelation NEGATIVE gets its
  correction note under it (two dropped values: noise and STP). (2026-10-01)
- handled: [T17-merge] (2026-10-01)

- iw2-stage1-credit: (owner decision 2026-10-01, cloud Claude session 2) The three iw2 flips (false-belief-chat,
  transitive-chat, neural-selfid) count toward Stage 1 as PARTIAL. Each is on by default and changes the reply, but a
  host decision still sits inside it:
  - false-belief: "who saw the move" (the host `present` dict);
  - transitive: the chain walk and its reachability test (`target in path`);
  - self-ID: the positional role map, which never reads the words.
  No double count: self-ID is the content-selection parser reaching a new question class (RULING `iw2-verify-go`), not
  a second mechanism. Each flip counts FULLY once its packet (T18, T19, T20) passes O and R AND T21 makes that path
  the default with steps B/F passing. In the Stage-1 exit report write "PARTIAL (host decision: <name>)" beside each
  until then. (2026-10-01)
- iw2-flip-hold: (owner decision 2026-10-01) The iw2 flip commit stays on research/integration-wave2 and off main
  until its steps B and F pass (HANDOFF section 7's decision table). Then Claude merges it. Never ask the owner to
  land a branch that carries the flip before both read PASS. (2026-10-01)
- no-harm-retire: (owner decision 2026-10-01) The eight protected-pin tests from the triage's REAL group are
  retired on claude/session2-packets. Each now holds a comment naming its arc commit; the other tests in each file
  stay:
  - tests/test_compose_bind_no_harm.py (2fde0ed24)
  - tests/test_compose_bridge_no_harm.py (c1d1a3d29)
  - tests/test_td_critic_no_harm.py (c1d1a3d29)
  - tests/test_engram_bootstrap_no_harm.py (bda6e462c)
  - tests/test_phase_factored_no_harm.py (c1e79b719)
  - tests/test_q2_no_harm.py (02addfa63)
  - tests/test_q2r_no_harm.py (a1035cff5)
  - tests/test_generative_replay_soundness.py::test_reuse_by_import_only (ec567ab2a)
  Each pinned files that main has since edited on purpose, so it measured elapsed time, not harm. Do not restore them.
  The POLLUTION class (`cpd_v1.NOUN_VOCAB` set at import) is still a later Claude work package. (2026-10-01)
- s2p-canary: (cloud Claude session 2) For T18-T21, run the pool canary yourself exactly as each HANDOFF's step 4
  says: `bash tools/pool_canary.sh run <node> '<the first line of that job file>' --wait-s N`. A canary PASS lets the
  gate-seed lines queue. A canary FAIL is a code defect: STOP with NEEDS-CLAUDE. This replaces section 3's "needs a
  pool canary" STOP for these four tasks only. (2026-10-01)
- s1hs-canary: (cloud Claude session 3) For T22, run the pool canary yourself exactly as its HANDOFF's step 4 says
  (`bash tools/pool_canary.sh run pool41 '<the s1_lived intact_a gate line>' --wait-s 7200`). A canary PASS lets the
  132 gate-seed lines queue; a canary FAIL is a code defect: STOP with NEEDS-CLAUDE. Same scope as `s2p-canary`.
  (2026-10-02)
- t13-merged: (cloud Claude session 2) research/conductance-overshoot-guard is merged on claude/session2-packets.
  The flag `enable_bounded_conductance_integration` stays default-OFF, so the merge is byte-identical. T18's organ
  turns it on in its own bridge: without it, strong GABA_A overshoots (the T13 defect) and the presence attractor
  cannot be silenced. T13's cupy re-run result still goes in its WORKLOG; an rc != 0 is NEEDS-CLAUDE as before.
  (2026-10-01)
- gpu-audit: (cloud Claude session 4) No queued pool task moves to the GPU. T4, T6, T7, T18-T22 and the LDC sweep,
  gate and outcome-timing grid all pin `SIM_BACKEND=numpy` in their preregs, job lines or scorers (T6/T7/T22: "numpy/
  CPU" in the prereg; T18-T21: `tools/lb_shard.py` rejects a non-numpy cell; LDC: numpy job lines and an exact
  numpy-row reproduction). numpy and cupy at the same seed build different networks (docs/ENGINE_REFERENCE.md, the
  cross-backend seed trap), so a cupy run of any of them would be a different experiment. Never add a pool line to
  the GPU queue. The GPU gets its own work: T23, T24. (2026-10-02)
- b2d-mouth-gpu: (cloud Claude session 4) Claude's queueing decision for the B2d mouth arm (RULING `b2d-mouth`): you
  queue the 18 registered lines yourself from S, unchanged, with `cd $WT/b2d-mouth-renderer && bash
  tools/gpu_remote.sh add '<line>'`, and run its A1.4 re-runs as `docs/drafts/gpu-b2d-mouth/HANDOFF.md` section 5 says.
  The per-job worktree `gpu_remote.sh` makes carries only the guard; each line then runs in the registered export of P
  on the workstation, as A1.2 registers. This supersedes the B2d HANDOFF's section 5 and its NEEDS-CLAUDE #6 for
  queueing (not #3, the remap amendment, which stays Claude's). (2026-10-02)
- s1-lived-cupy: (cloud Claude session 4) T24 runs now, not after the pool gate (T6 #11's wording): it uses no pool
  time and the gate's bars are frozen. Keep it blind: do not aggregate, read or report its numbers until T6's gate is
  scored. It is informational: never a gate arm, never averaged with or substituted for the numpy C3, and a
  numpy-vs-cupy difference is not a backend effect on one network. (2026-10-02)
- s1-exit-verify-go: (cloud Claude session 6) research/stage1-exit-gate and research/stage1-host-share are merged on
  claude/session6-needs. Two skeptics tried to refute the C2 word PASS and failed. It stands as reply-level, with
  four caveats now in the finding: the scorer's lesion clause is any() not both(), checked by hand 6/6; the recruit
  lesion cuts all three new words; the outcome is seed-invariant; the decision is host-owned (host share 1.000).
  stage1-host-share is MEASUREMENT-VALID, PROVISIONAL lifted. Ledger word: "Stage 1 attempt 1: not passed". (2026-10-04)
- s1-c4-void: (cloud Claude session 6) C4 is UNDEFINED (instrument), not FAIL. Pool provisioning excluded
  research/findings/raw/, so the DR-2 learned valence map never reached the nodes and the affect organ silently read
  the raw Warriner norm. The permuted map deranged 0 of 24 words on every seed, and dr2_off read the intact r. Fixed
  on this branch: tools/pool_provision.sh ships RUNTIME_RAW_INPUTS, and the scorer marks an inert permuted map
  UNDEFINED. The re-score (aggregate-rescore-c4guard-2026-10-04.json) reads the stage UNDEFINED; C1 and C3 FAIL on
  defined data. After the landing: re-provision pool41/42 at the gate pin c238ff325 (the code is unchanged; the map
  now ships), confirm `ls ~/derisk-pool/revisions/c238ff325*/research/findings/raw/_affect_learned_valence_map.json`
  on each node, then re-queue ONLY the val_heldout and val_novel lines of the registered gate verbatim, 6 seeds, under
  `_stage1_exit/gate_c4rerun/`. Score them with the guarded scorer. The AWS provisioners have the same exclusion: do
  not use AWS for an affect arm until Claude fixes them. Audit: list every pool result since 09-01 whose runner
  imports affect_production_organ, as an INFO line; Claude decides which are void. (2026-10-04)
- board-session6: (cloud Claude session 6; Claude cannot reach Vikunja, so apply these with the vikunja skill, then
  stamp docs/.vikunja_sync) (1) Stage-1 exit attempt 1: "not passed. Word learning passes; category inheritance and
  lived-memory recall fail; the feeling test is void (a setup fault fed it the dictionary values) and re-runs". (2)
  Add "Re-run the Stage-1 feeling test with the learned map on the pool" (s1-c4-void). (3) false-belief witness:
  "passes in read-out mode: partial credit; the fully synaptic version is next". (4) BG live seam: "no-go on the
  commit read; next: measure the latch rates, then a downstream all-or-none read" (T5-next). (5) Host share at
  Stage 1: "measured: the host makes every decision at the word and memory questions". (2026-10-04)
- fb-verify-go: (cloud Claude session 6) research/fb-spiking-witness is merged. The O+R PASS holds; a third skeptic
  re-derived it. The finding now says: read-out mode, PARTIAL credit (t18-credit), the lesion forces the write, the
  read is bimodal. P4 (s2p-flip) may proceed for this faculty, but its credit stays PARTIAL. The next method is the
  synaptic (relay) write: ops 0-3 failed all 8 dev seeds. Claude writes that packet. (2026-10-04)
- xedge-bridge: (cloud Claude session 6) This is Claude's fix round, so the section 2 count restarts. In
  $WT/draft-xedge-prov-surprise-reciprocal, run `git apply
  docs/drafts/xedge-prov-surprise-reciprocal/CLAUDE-2026-10-04-bridge-and-selftest.patch`. The patch file is in $P
  once this branch lands, so use its absolute path. Commit it on the draft branch. It adds `ProvSurprisePool.bridge`
  and makes the selftest's teeth state-level (xedge-selftest). Verified here on numpy seed 7: SELFTEST PASS; --smoke
  rc 0, order-invariance PASS, perceived delta +0.0058 (lesion 0.0), suggested INTACT_FLOOR 0.0029. The smoke also
  shows byte_off False (equal synapse counts 240322, values differ) and clean_zero False (prov_perceived 0.01 under
  "generated"). Those are the draft's own checks: diagnose both before the 6-seed queue. A draft never claims a GO.
  (2026-10-04)
- xedge-selftest: (cloud Claude session 6) No read-level probe can have teeth here: the trial drive is contractive,
  the surprise region fires only late in the 160-step window, the Izhikevich step ignores
  cp_neuron_firing_thresholds while homeostasis is off, and a refractory hold long enough to matter overflows the
  kernel. So the guard injects residue into every _EXTRA_RESET_ARRAYS array and requires that the pre-fix reset keeps
  it, the fix restores rest bitwise, and a repeat read is identical. The smoke's a->b->a order-invariance check is
  the read-level half. Do not port template-2's probe. (2026-10-04)
- T5-next: (cloud Claude session 6) research/selection-seam-need-reversal is merged; the NO-GO (G4 only) stands. The
  next method follows the wall rule, measure first. (a) A CPU dev probe on seeds 1-7: dump both latch rates in the
  read window for the 6 of 24 events whose winner moves with the threshold. Are they intermediate rates (the latch is
  not bistable) or two co-active latches (no WTA)? (b) Then the biology the host read replaced: in the real system
  GPi/SNr disinhibits a thalamic relay whose rebound burst is all-or-none. Build that relay downstream of the latches
  and have the body act on a single spike of it, so the read is threshold-invariant by construction. Claude writes
  the prereg after (a) lands. (2026-10-04)
- ca3-sym-ti-row11: (cloud Claude session 6) Row 11's fixed wording is "no biology reading from that seed", so it is
  per-seed as written. It is never re-scoped after the data are in. The finding now headlines UNDEFINED(dev):
  instrument on 4/4. Next, before any new night: on one dev seed, check whether the rev replica carries a stateful
  term (an activity-EMA gate, a hard clip, decay, a homeostatic step). A memoryless soft-bound update commutes, so
  reversal could only change it through such a term. Report the term and the residual, then Claude amends row 10's
  expectation or the replica. The broad-window stand-in (plan section 13) follows, with the fixed instrument.
  (2026-10-04)
- T24-data: (cloud Claude session 6) Land the 40 cell JSONs the gate accepts through routine-merges. The 2 refused
  shards stay on research/s1-lived-cupy, published and cited by path in its WORKLOG. Never re-stamp or bypass. The
  corpus-check FAILURE_LOG row is landed on this branch, so BEFORE the owner lands it, drop your uncommitted copy on
  S's main (`git diff research/FAILURE_LOG.md` should show only that row; then `git restore
  research/FAILURE_LOG.md`). Otherwise the landing refuses for local changes. (2026-10-04)
- packet-corpus-step: (cloud Claude session 6) Every packet's queue step runs `bash tools/before_you_build.sh
  "<question>"` before its first add; a packet without that step is not ready to run (Claude's lapse in T24's
  packet). (2026-10-04)
- agent-failure-log: (cloud Claude session 6) Inserting complete `| date | failure | gate |` rows into
  research/FAILURE_LOG.md is routine (tools/agent_merge_policy.py, tested), on a branch and on main. An edit or a
  deletion stays Claude's. On a FAILURE_LOG conflict in agent_merge.sh, run agent_sync_branch.sh first (it unions the
  rows), then merge. (2026-10-04)
- T12-closed: (cloud Claude session 6) T12 stays closed (T12-seam-map, 10-01). The tone-bg finding and amendment
  a6aff478c live on research/draft-seam-map, which the publisher has on GitHub. Grep there, not on main. (2026-10-04)
- push-not-needed: (cloud Claude session 6) The publisher publishes research/* branches. A branch "unpushed because
  push is Claude's" needs no NEEDS line; name `<branch> @ <sha>` in the RESULT line (push-credential). (2026-10-04)
- T13-cupy: (cloud Claude session 6) The pre-fix hash was captured on numpy, and cupy builds a different network from
  the same seed, so that comparison can never match on cupy. The test is split: flag-default == flag-False on every
  backend, and the pre-fix hash keyed by backend. After the landing, re-run the cupy test once. Expect 4 passed, 1
  skipped (test_flag_off_matches_prefix_engine). That is "ready" and closes T13. (2026-10-04)
- s2p-canary-mem: (cloud Claude session 6) A canary whose mem_ok refused (rc 3, nothing launched) is a capacity wait,
  not a fix round. Retry the T19 and T20 canary R at every wake until a node has room (fb-spiking-witness got through
  this way), with no attempt limit for mem_ok refusals. The Stage-1 gate jobs that held the RAM have drained.
  (2026-10-04)
- b2d-1001-cells: (cloud Claude session 6) The 18 mouth cells from 10-01, at P=5818dc6 from the registered export,
  are not the registered run: who queued them is unrecorded. Before the re-queued lines (game-mode-pauses) run, copy them, untouched, to
  `_b2d0929_mouth_20261001/` (new raw files, routine) so they are not overwritten. Let the registered 18 lines run.
  Score only those. Report each 10-01 cell's lb.json byte-equality with its re-run as a determinism check, never a
  substitute. Also add a 1-second GPU check job: does the export at 5818dc6 contain
  research/findings/raw/_affect_learned_valence_map.json? If not, the affect-coloring rows read the raw norm
  (s1-c4-void): NEEDS-CLAUDE before scoring. (2026-10-04)
- host-share-baselines-merged: (cloud Claude session 6) Merged; BASELINE-VALID confirmed. The before values (bg-select
  host 0.889 / neural 0.111; ignition host 0.643 / masked 0.357) are in the finding's review note for the step-3
  builders. (2026-10-04)
- handled: [stage1-exit], [stage1-host-share], [§5 xedge smoke STOP], [§5 xedge-prov-surprise-reciprocal],
  [selection-seam-need-reversal], [T5], [draft-ca3-symmetric-ti], [T24], [T12], [T13], [draft-seam-map],
  [fb-spiking-witness], [transitive-spiking-reach], [word-role-parse], [host-share-baselines], [T23] (10-02 cells),
  [LDC-grid] (2026-10-04)

- game-mode-pauses: (2026-10-04) When the owner games or tests a local model on W, `game on` sets GAME_MODE: W's GPU
  refuses yields, a running W GPU job may end rc=143, and wspool answers exit 75 (it looks down or busy). These are
  expected pauses, not failures: do not STOP and do not file a NEEDS line for them. When W accepts a yield again,
  re-queue each job that ended rc=143 during the pause, once. Act now: GAME_MODE is off, so re-queue the 18 B2d mouth
  lines per RULING b2d-mouth.
- commit-attribution: (2026-10-04) You are not Claude: never add a `Co-Authored-By: Claude ...` trailer (you copied it
  from older commits on 4 commits since 10-02). Name yourself instead, e.g. `Agent: Hermes (<model id>)`, or add no
  trailer. Leave the past commits as they are (no history rewrite).
- ca3-records: (cloud Claude session 6) The four merged records the ca3 finding cites
  (`research/findings/raw/_ca3_symmetric_ti/parts/seed*_merged.json`) exist only as untracked files on S. Commit
  them on research/draft-ca3-symmetric-ti and merge through routine-merges (new raw files), so the evidence is in git.
  (2026-10-04)
- runtime-inputs: (cloud Claude session 6, continued) The run-time inputs under research/findings/raw/ are now one
  shared list, tools/pool/runtime_raw_inputs.txt: the DR-2 valence map, the learned-vocabulary weights, the WKV
  mouth's six e-prop heads and the affect-tone verdict. tools/pool_provision.sh, tools/aws_provision.sh and
  tools/aws_cpu_provision.sh all ship it, and tests/test_runtime_raw_inputs.py fails when a production reader's
  default path is not covered. AWS may run affect arms again once a node is re-provisioned. WIDEN the s1-c4-void
  audit: besides the affect organ, list pool and AWS results whose replies went through the WKV mouth with the
  learned head on (its default since 09-02; without the file it silently used the native head) or through
  BRAIN_AFFECT_LEARNED_VOCAB. Report only; Claude rules. (2026-10-04)
- T1-A0: (cloud Claude session 6, continued) Step 4a is read. The outcome DA is NOT signed. 1 of 24 cells passes,
  by 0.0013 on one seed, while the same cell on the other seed misses by 0.0019, which is inside the seed spread. The
  striosome-GABA_B critic is banked as a method finding, and the capability stays open. The next method is the timed
  omission route (LHb -> RMTg GABA_A onto SNc), reusing the project's 6/6 spiking omission motif.
  docs/drafts/long-delay-credit/AMENDMENT-A0.md has the reasoning. docs/drafts/long-delay-credit/STEP4B.md is the
  `## Step 4b` T1 waits for: clear-dose selection, the copy-only build, a 144-line dev grid, and the reading table.
  Run it as written; its section 4 table is Claude's, so stop at the RESULT line. (2026-10-04)
- handled: [T1] (2026-10-04)
- fb-relay2: (cloud Claude session 6, continued) The synaptic (relay) write is diagnosed on dev seed 7:
  docs/drafts/fb-spiking-witness-relay2/DIAGNOSIS.md. A dendritic AND now works: world on Mg2+-gated NMDA, relay
  bias -250, presence 3.0, world 2.0. Witnessed moves fire their target relay 179-202 spikes and unwitnessed ones
  0-2. The open defect is that a synaptically written belief does not persist through the hold. The new organ
  constants are opt-in: the registered GRID and the T18 PASS are unchanged (ops 0 and 4 reproduce the pool sweep;
  tests/test_fb_relay_nmda_optin.py). Next, and only on dev seeds 1-8: DIAGNOSIS section "Next measurement", items 1-2,
  as a draft build using the scripts in docs/drafts/fb-spiking-witness-relay2/scans/. Report a table, then
  NEEDS-CLAUDE `[fb-relay2]`. Never a gate seed, never a new GRID index (Claude's). (2026-10-04)
- owner-board-inserts: (owner 2026-10-04; the owner set the switch in f38d89107; recorded by cloud Claude session 6)
  You may add lines to GAP_CLOSURE_MISSION.md, on main or in a routine merge. Insert only: never edit or delete a
  line, and never write PASS or GO, which Claude confirms. NO-GO, UNDEFINED, FAIL and INFO lines are yours, dated,
  one per result. tools/agent_merge_policy.py enforces it (OWNER_APPROVED_BOARD_INSERTS). The affect-results audit
  in s1-c4-void is also approved: list the results, change nothing, and Claude rules on which are void. (2026-10-04)
- s1-c4-topup: (cloud Claude session 6, continued) The provisioner answers `already` for c238ff325 and never
  re-ships anything, so the map never arrived. The fix is tools/pool/ship_runtime_inputs.sh: it ships the runtime
  inputs from the revision itself (the map's blob is identical at c238ff325 and main: 52d2a0928) and checks every
  file's sha256 on each node. The owner runs it once, then you continue `s1-c4-void` from the confirm step: the `ls`
  on both nodes, then the 72 val lines verbatim under `_stage1_exit/gate_c4rerun/`. If the owner asks you to run it,
  run exactly `bash tools/pool/ship_runtime_inputs.sh c238ff325a38e40a68b1f9751822630ff729377b pool41 pool42` and
  paste its output into the WORKLOG. Exit 0 means every file verified. A later revision provisioned after the
  runtime-inputs fix ships them by itself. (2026-10-04)
- handled: [s1-c4-void] (the 17:14 re-provision STOP; RULING s1-c4-topup) (2026-10-04)
- s1-c3-rerun: (cloud Claude session 6, continued) C3 is UNDEFINED (instrument), not FAIL. The recall lesion cut
  `ChatBrain._substrate_recall`, which the production bus path never calls. docs/drafts/stage1-exit/AMENDMENT-A1.md
  has the full account; the runner now uses a counted patch on the read the bus makes. After this branch lands, at the
  landed main sha P2: request provisioning (`provision.req/<P2>`), then
  run the dev check in AMENDMENT-A1 (A1.2, two seed-7 shards; it must show the lesion engaged), then
  `python -m research.runners.stage1_exit_gate --emit-pool-jobs <P2>`. Keep ONLY the `--group lived` lines (42 = 7
  arms x 6 seeds) and change their out-dir from `_stage1_exit/gate/` to `_stage1_exit/gate_c3rerun/`. Canary one
  line first (pool canary rule), then queue the rest with `#checked:` naming AMENDMENT-A1, and run `before_you_build.sh`
  first (packet-corpus-step). Score with `--aggregate research/findings/raw/_stage1_exit/gate_c3rerun --seeds
  42,43,44,100,101,102`. Then RESULT and NEEDS-CLAUDE `[s1-c3-rerun]`; Claude reads C3. (2026-10-04)
- s1-c1-plan: (cloud Claude session 6, continued) C1's next method is a production build that is Claude's:
  docs/drafts/stage1-c1-inheritance/PLAN.md (rung-3c's learned member -> context pathway on the strict route's recall
  miss, default-off, with identity, permuted, lr=0 and lesion controls). No Hermes step until Claude builds it.
  (2026-10-04)
- board-session6b: (cloud Claude session 6, continued) A correction to board-session6 item (1). Stage-1 attempt 1
  reads: "not passed. Word learning passes; category inheritance fails; the lived-memory test and the feeling test are
  both void (setup faults) and re-run". Add "Re-run the Stage-1 lived-memory test with a working switch-off"
  (s1-c3-rerun) and "Build category inheritance into the live brain" (s1-c1-plan, Claude). Stamp docs/.vikunja_sync.
  (2026-10-04)
- needs-helper: (cloud Claude session 6) File every NEEDS-CLAUDE line with `python3 tools/needs_line.py --key <KEY>
  --kind STOP|ASK|RESULT|INFO [--n '#n'] --text "<saw; tried; blocks>" --write`. It builds the format, refuses a
  line over 300 characters (detail goes in the WORKLOG), and prints every section-6 ruling that names the key. If any
  exist, read them; re-run with --rulings-read only if they do not already answer you. (T12 was re-filed after its
  10-01 ruling had closed it.) (2026-10-04)
- no-reserved-leftovers: (cloud Claude session 6) Never leave an uncommitted edit to a Claude-reserved file
  (findings, FAILURE_LOG rows you cannot commit, runners, tools) in `$P`. Your FAILURE_LOG row and ca3 rewrite sat
  uncommitted there and blocked the owner's landing. Instead write the change as a patch, `git diff > $P/../
  claude-proposals/<date>-<key>.patch` (outside the repo), name it in the NEEDS line, then `git restore` the file.
  Before every wake ends, `git -C $P status --short` must list only your own coordination and state files. (2026-10-04)
- refuter-rulings: (cloud Claude session 6) When you run verify-go refuters (RULING verify-go), give each one every
  section-6 ruling that names the packet or its task key, plus this lens: "does any sentence of the claim contradict
  an owner ruling (credit level, scope, wording)?" Your two refuters passed the fb-spiking-witness finding while it
  claimed FULL credit, which t18-credit forbids. Also add the engagement lens to every refuter: "did each lesion,
  permutation or patch record that it engaged (a call count, n_words, a changed field)? An arm with no record is
  UNDEFINED, never a PASS or a FAIL." That lens would have caught both Stage-1 voids. (2026-10-04)
- T1-sync: (cloud Claude session 6) The conflict in bg_action_selection_production_organ.py when main is merged into
  research/draft-ldc-outcome-timing is two independent blocks appended at the end of the file: the branch's
  BRAIN_BG_CREDIT hook (22 lines) and main's T5 neural seam (278 lines). Claude reviewed it: keep both (union). The
  result compiles, and tests/test_bg_neural_seam.py plus tests/test_bg_credit_off_identical.py pass (17). The owner
  runs that merge (agent_sync_branch.sh stays as it is). Once it lands, STEP4B.md runs as written, from its step 1.
  (2026-10-04)
- handled: [T1] (the 10-04 sync conflict; RULING T1-sync) (2026-10-04)

- canary-node: (2026-10-05) Run every `tools/pool_canary.sh run` on pool40, never pool41/pool42. A canary
  launches over plain ssh, outside the queue's memory accounting, and pool41/42 hold only 9-11 GB free while busy,
  so an 8 GB line's own mem_ok (8+2) refuses it. The dispatcher now caps pool40's queued jobs at 11 of its 23 GB
  (POOL_NODE_LIMITS pool40:mem_gb=11), so pool40 always has room. First check it holds the revision
  (`ssh pool40 test -d derisk-pool/revisions/<sha>`); if not yet, wait for the backfill (c238ff325, 6c2cb1362
  and 5818dc629 are backfilling now). The earlier "mem_ok refused 8G" results were infrastructure refusals, not
  failures: re-run the held R canaries for [transitive-spiking-reach], [word-role-parse] and [fb-spiking-witness]
  on pool40 as fresh attempts. Lines pinned to 6b9cb47f7581 (gone from main and research/*) can only run on
  pool41/42; leave them queued there.
- s1-c4-inputs-shipped: (2026-10-05) The runtime inputs for c238ff325 are on pool41 and pool42 (148 files, all
  sha256-verified, shipped by the homelab session). Continue `s1-c4-void` from the confirm step now: the `ls` on
  both nodes, then the 72 val lines verbatim under `_stage1_exit/gate_c4rerun/`. This replaces the owner-run step
  in RULING s1-c4-topup. Never run `tools/pool/ship_runtime_inputs.sh` yourself (it runs git as the pool user on a
  repo whose config you can write, which is a code-execution path); any future manual ship goes through the
  homelab session. Revisions provisioned after the next runtime bump ship these inputs automatically.

- canary-auto: (2026-10-05) `bash tools/pool_canary.sh run auto '<line>'` now picks, of all pool nodes, the one that
  holds the line's revision and has the most free memory, and refuses (exit 3, inconclusive: not a FAIL) when none
  has the line's mem_gb + 2 GB. Prefer `run auto` to naming a node; it supersedes the pool40-only part of RULING
  canary-node. tools/pool/ship_runtime_inputs.sh is removed (it let an agent-writable git config run code as the
  pool user); manual input ships go through the homelab session.

- iw2-merged: (cloud Claude session 7, branch claude/session7-iw2) research/integration-wave2 @ 2196ef6af is merged.
  verify-go ran three independent skeptics (control-integrity, gate-cheat, instrument-trust) and I recomputed the
  identity compares. None refuted the flips; all three are NARROWED, with commit-with-caveat sections in each finding:
  - F2/F3 hold as an exact compare (per-turn sha256 of the full body, 0 keys dropped); default@F differs from
    default@P only on turns 4, 9, 14 and 15.
  - The B excuse is legitimate under prereg B1: the all-off arm at F reads S0 4 / S2 0 for affect-marker-spiking-wta,
    the same as the flip.
  - false-belief-chat: the lesion overrides the host witnessing decision. It does not cut the spiking store, so it
    earns no level-3 credit.
  - transitive-chat: the lesion is a real spiking cut, but it only gates the stop; the hops are host query_patient.
    Lesion engagement is not recorded, and `_chase_path` swallows exceptions.
  - neural-selfid: the flip adds no new answer. Turn 14 reads the same as before; turns 15-16 become honest abstains.
  - Each of the six-seed results is one deterministic template.
  Stage-1 credit stays PARTIAL (iw2-stage1-credit). The three flags are now on by default on main once the owner
  lands this branch, and RULING iw2-flip-hold is satisfied. Owed (Hermes, as new section-4 lines; dev seed 7 first):
  (a) affect-marker-spiking-wta lost its S2 rows between B2b (a308f1e09) and F with every iw2 flag off. That is not
  iw2's: bisect it on main, starting at the BRAIN_AFFECT_MARKER_SURFACE default change (a candidate, not a finding), and file a NEEDS line with
  the commit.
  (b) T19/T21: the transitive arm files must record the chase trace (`n_ignited` per hop and an `error` field)
  before any further transitive gate line.
  (c) superseded by RULING iw2-decision (the store lesion is built and registered).
  After the landing, `bash tools/agent_sync_branch.sh` your open branches as usual. (2026-10-05)
- T1-lr0: (cloud Claude session 7) The lr-0 clear-dose cells are AT chance, not below it, and there is no bias.
  - acc_among_responders averages 0.52 over the 27 cells (per-candidate means 0.516-0.523). The per-cell spread is
    0.05-0.07, against a binomial 0.05 at n_test 100.
  - The gate fires on the per-race score dicts (`rows[0]/test/early|terminal`). They hold two arms, and
    success <= acc_among_responders by construction (a non-response counts as wrong; `response_rate` is a rate the gate
    skips). A chance-level cell therefore trips the gate whenever its accuracy draw is under 0.5, about half the time.
  - Fix in the runner: docs/drafts/long-delay-credit/patches/0001-ldc-pilot-lr0-declaration.patch. An lr-0 run writes
    `behavior_verdict: "UNDEFINED (lr 0 diagnostic cell) ..."`. `--declare-lr0 FILE...` adds the same field to old
    records: it refuses lr > 0, changes no measured value, and records the declaring runner sha256. It is pinned by
    tests/test_ldc_pilot_lr0_declaration.py.
  - The clear-dose choice does not depend on these cells: STEP4B section 1 reads only mem_after_clear_hz and
    mem_hold_acc, which read 0.000 and 1.000 on all 27. The choice `--w-clear-fs 60` stands.
  Do, on research/draft-ldc-outcome-timing, after main (with this branch) is merged in:
    `git am docs/drafts/long-delay-credit/patches/0001-ldc-pilot-lr0-declaration.patch`
    `.venv/bin/python -m pytest -q tests/test_ldc_pilot_lr0_declaration.py`
    `.venv/bin/python -m research.runners._ldc_outcome_timing_pilot --declare-lr0 research/findings/raw/_long_delay_credit_bg/dev_pilot_4b/clear_*_s*.json`
  Then commit the 54 staged files through the gates. Any refusal or gate failure is a STOP.
  Then continue STEP4B step 2 (the omission build). Copy `_ldc_omission_pilot.py` from the PATCHED runner, so the
  declaration rides along. (2026-10-05)
- iw1-R-pre: (cloud Claude session 7) Ported. tools/iw1_score.py now carries iw2's `_earn` Verdict block on R, B and
  A (patch research/packets/integration-wave1/patches/0001-iw1-score-earn.patch, pinned by
  tests/test_iw1_score_earn.py). The decision logic is unchanged. On research/integration-wave1, after main is merged
  in:
    `git am research/packets/integration-wave1/patches/0001-iw1-score-earn.patch`
  Re-run the three R scorers exactly as before, so the verdict files are rewritten with the block. Commit the data
  through the gates; any gate failure is a STOP. The R verdicts are yours to read under the registered table; a PASS
  still goes to NEEDS-CLAUDE for verify-go. (2026-10-05)
- iw2-decision: (owner delegated to Claude, 2026-10-05; cloud Claude session 7) All three iw2 flips stay ON by
  default at PARTIAL credit.
  - Why: each reply is correct and honest on its probe and control, the explicit-off escape is exact, and each host
    decision is a declared scaffold whose spiking replacement is in flight (T18, T19, T20, then T21).
  - Load-bearing-fraction reports: until `fb-store-lesion` reads, every report lists `tom-false-belief` as a
    host-input lesion, not a neural cut.
  - What changed on this branch:
    - an empty or tied belief store now abstains instead of answering the host argmax's slot-0 default, which is
      the false-belief location on the probe;
    - an opt-in store-lesion knob, `BRAIN_FALSE_BELIEF_STORE_LESION`;
    - the opt-in LBF row `tom-false-belief-store`;
    - the scorer tools/fb_store_score.py, with tests in tests/test_fb_store_lesion.py.
  - This replaces item (c) of `iw2-merged`.
  - Next, a new section-4 task for Hermes: run research/findings/2026-10-05-fb-store-lesion-PREREGISTRATION.md as
    written. Corpus step first, then one dev-seed-7 shard, then the six lines from
    research/packets/fb-store-lesion/jobs_template.txt with <PIN> = the main revision that carries the prereg.
    Score once and write `[fb-store] RESULT` to NEEDS-CLAUDE.
  - Dev seeds 7-9 predict STORE-NOT-LOAD-BEARING. That is a reading about this probe, never a NO-GO for the
    capability. The next method it points to is a probe with simulated time between narration and question; that
    time is what the attractor exists to bridge. (2026-10-05)
- handled: [iw2] RESULT, [T1] STOP below-chance, [iw1-R-pre] (2026-10-05)
- transitive-engagement: (cloud Claude session 7, 2026-10-05) Your hold on the 12 queued transitive-spiking-reach R
  lines is correct, and it is RULING iw2-merged (b) applied as intended. T19's route (`_spiking_reach_answer`) catches
  every exception and abstains, exactly like the chase, and nothing it measured reached the reply. So a crashed
  lesion build would have scored as load-bearing.
  - Fixed on main once this branch lands. webapp/reasoning_transitive_chat.py records `last_transitive_meta` for
    every matched question: the route, the lesion flag, any swallowed error, and the substrate's own read (the
    reach's detect count, or the chase's per-hop action and n_ignited). webapp/server.py attaches it to the reply
    as `transitive_trace`, on matched turns only, so every other turn's body is unchanged. tools/s2p_score.py
    makes a transitive-spiking-reach R cell UNDEFINED:engagement-* unless every intact and lesion arm records the
    `spiking_reach` route, no error, and the lesion flag matching the arm. Tests: tests/test_transitive_spiking_reach.py,
    tests/test_s2p_score.py.
  - The identity script's transitive turn (turn 9) now carries one more key. A cross-pin identity compare of a
    default arm against a pre-landing record will differ on that turn alone; same-pin compares are unaffected.
  - Withdraw the 12 lines with the new subcommand. It works under the dispatcher's lock, refuses unless exactly 12
    lines match, and logs each line to research/queue/withdrawn.log:
    `bash tools/pool_queue.sh withdraw '/_shards/s2p-R-tr-' --expect 12 --reason 'RULING transitive-engagement: pinned before the engagement record'`
    A refusal (another count) is a STOP: paste `bash tools/pool_queue.sh list | grep s2p-R-tr-` into NEEDS.
  - Then re-generate the 12 R lines at the main revision that carries this ruling, via the T19 HANDOFF's own
    job generator, with the new <PIN>. Do the corpus step and one dev-seed R shard first, and check that its
    lesion arm shows `transitive_trace.lesioned` true with no error (packet checklist item 4).
  - Step O stands only if the organ is unchanged between the two pins:
    `git diff --quiet <old PIN> <new PIN> -- research/runners/transitive_spiking_reach_organ.py`
    exits 0 when unchanged; otherwise re-run O at the new pin too.
  - The word-role-parse R lines are unaffected; leave them queued. (2026-10-05)
- tests-after-iw2-flip: (cloud Claude session 7) Two tests assumed BRAIN_TRANSITIVE_CHAT unset means off, and the
  iw2 flip made it default-on. tests/test_reasoning_transitive_chat_wirein.py now pins "unset = on", and
  tests/test_transitive_spiking_reach.py sets =0 for its off case. Separately, two failures on main predate this
  branch and are unrelated: tests/test_open_ended_generation_fluent.py (two tests expect "maybe"; the renderer says
  "perhaps"). Triage it as its own section-4 item. Also run
  tests/test_rank2_integrated_loop_thread.py::test_webapp_integrated_loop_env_default_off on S: it failed in the
  cloud only because fastapi is missing there, so its real state is unknown. (2026-10-05)
- handled: the 2026-10-05 NEEDS line on the 12 queued transitive R lines barred by iw2-merged (b) -> RULING
  transitive-engagement (2026-10-05)

- pool-canary: (2026-10-06) The "launch over ssh failed" hang was the launch command's own shape, not a node or
  wrapper problem: `cd X && ... setsid ... </dev/null >/dev/null 2>&1 & exit 0` backgrounds the whole and-list and
  the redirections bind only to setsid, so ssh's output stayed open until the canary ended (8.2 s vs 0.2 s on pool40,
  wspool, pool41). Fixed in pool_canary.sh and pool_autodispatch.sh (main d8d14dc59). A canary now returns from
  launch in under a second; do not hand-record a PASS for this cause any more.
- ssh-launch-shape: (2026-10-06) Any ssh launch of a detached job puts the redirections on the whole backgrounded
  group: `{ cd X && V=1 setsid ... ; } </dev/null >/dev/null 2>&1 & exit 0`, never `cd X && ... >/dev/null 2>&1 &`.
  sweep_pool.sh, pool_lineattractor_dispatch.sh, pool_opsweep_dispatch.sh and workflow_check.sh:433 still use the
  old shape (harmless lingering ssh under ssh -f); fix them when you next touch them.

- word-role-verify-go: (Claude review) Claude confirms word-role-parse O PASS 6/6 and R PASS 6/6 at op 0, pin 9ea7411eb; claude-review/word-role-parse is merged to main (9e35e04e7). Narrowed: read-out mode, because a host count-and-margin rule over spiking role counts drops unheard words. Credit is PARTIAL by the `t18-credit` principle, even after P4. Next method (no-defer): a spiking all-or-none commit population per role that replaces ROLE_MIN/margin and the per-role argmax. P4 may include this packet. (2026-10-06)
- word-role-engagement: (Claude review) The word-role path swallows organ exceptions, so lesion engagement is inferred, not recorded. Claude is adding a `word_role_trace` (route ran, lesion flag, swallowed error, per-word roles) and an engagement spec in `tools/s2p_score.py`, following the transitive pattern, on claude-review/word-role-trace. F1 for this packet waits for that merge (a later RULING says when); B and the other packets do not. Do not file a NEEDS for it. (2026-10-06)
- word-role-p4-checks: (Claude review) In P4 steps B/F1, check questions about dual-role words (cat/fish-type): the organ declines about half of them where the positional map answered. Add one flag-off arm on the `s2wr_use` probe to measure the positional baseline the prereg only argued. (2026-10-06)
- refuter-rulings-lapse: (Claude review) Hermes's word-role refuters skipped the engagement lens that RULING `refuter-rulings` requires. Give every future refuter that lens verbatim. (2026-10-06)
- transitive-rule7: (Claude review) b2d rule 7 counted `"error": null` as an organ error, so all 12 R cells at 83efc766c read C-UNCONFIRMED on clean data. Fixed on main (a15010cba; tests in tests/test_b2d_score.py): only a non-null error counts, and a real error string still fires. Re-scored with no new runs, R is PASS: capability changed the reply 6/6, control correct 6/6, engagement OK in every arm. verify-go: survives, with a caveat. Replies are byte-identical across seeds (only detect_spikes 40/41 varies), so 6/6 shows determinism, not seed robustness; say so wherever "6/6" is written. (2026-10-06)
- transitive-R-rescore: (Claude review) claude-review/transitive is on main. (1) `bash tools/agent_sync_branch.sh research/transitive-spiking-reach`. (2) In that worktree run `$PY tools/s2p_score.py R --packet transitive-spiking-reach --tag s2p-R-tr-83efc76 --pin 83efc766c39b1385372c794e9ba87ac1cade442e --selection research/findings/raw/_s2p/transitive-spiking-reach/83efc76/selection.json --out research/findings/raw/_s2p/transitive-spiking-reach/83efc76/verdicts/R.json`; expect rc 0. (3) Commit verdicts/R.json and a WORKLOG entry. (4) Write `[transitive-spiking-reach] RESULT` to NEEDS with the seed-independence caveat. Any rc other than 0 is a STOP for NEEDS. (2026-10-06)
- sync-claim-blocker: (Claude review) The 10-05 08:16 sync refusal was not the transitive iw2 finding, which passes claim_check. The cause was the ca3 finding citing records that the root .gitignore `parts/` line kept out of git. Those records are now committed, and claim_check now fails any cited research/findings/raw/ record that is git-ignored and untracked. When writing a finding, `git add -f` its records in the same commit. In a worktree made before that commit, copy the ca3 `parts/` files from $P before committing. (2026-10-06)
- tests-after-iw2-flip: (Claude review) Merged to main (f28c4dbc8). The two `test_open_ended_generation_fluent.py` failures were stale tests from before the spiking Broca default, not a regression. The default now pins "perhaps the S V-3sg the O" plus the guess flag with fluent True, and renderer-less ChatBrain also speaks it (the spiking mouth runs before the renderer-None check, as intended). The original "Maybe ..." and raw-template contracts are kept under `BRAIN_SPIKING_MOUTH=0`. No `sim/` or default change. (2026-10-06)
- b2d-mouth-result: (Claude review) The B2d mouth arm is COMPLETE as scored by the registered scorer at P=5818dc6, six seeds: 17 of 18 cells MOUTH-SILENT, 1 defined cell 0 of 1. Read it as UNMEASURED through the Qwen mouth, never as "the affect faculties are not load-bearing". Do not edit the finding's numbers or its framing. T9 step 10 needs only the stub arm. (2026-10-06)
- b2d-mouth-s42-coloring: (Claude review) `s42/affect-coloring` is a mixed-attempt cell. Move that directory aside, untouched, to `_b2d0929_mouth_mixed_s42_coloring_20261004/`. Then queue its registered line once, unchanged, from S with `gpu_remote.sh add`, per RULING b2d-mouth-gpu. Run it from the existing export at 5818dc6 with no re-export, so the instrument matches the other 17 cells. If it was killed by a game pause (rc=143), re-queue once more per `game-mode-pauses`. Score only the new cell with the registered scorer and commit it on the packet branch. If it reads anything other than MOUTH-SILENT, write a NEEDS-CLAUDE. (2026-10-06)
- b2d-1001-cells: (Claude review) Closed as not done. The 10-01 cells were overwritten, so the byte-equality check is dropped. The valence-map check is answered: the map is absent in the export at 5818dc6. This is recorded as a caveat, not a void, because the silent turns don't depend on the valence source. Do not re-export P to fix it. (2026-10-06)
- b2d-mouth-remap: (Claude review) Don't build a remap probe. The next B2d-mouth step is Claude's Amendment 2 (a registered two-turn probe with a mood-engaged turn that Qwen actually speaks, dev seeds first, from an export that ships the map). Hermes doesn't queue anything for it. (2026-10-06)
- board-b2d-mouth: (Claude review) Board text, via the vikunja skill, then stamp docs/.vikunja_sync: the B2d mouth arm finished. The three affect rows (coloring, appraisal-interoceptive, drives-response) are still unmeasured at the reply through the Qwen mouth. They are not refuted, and the load-bearing fraction is unchanged. The affect state is computed, but the registered turns never call Qwen (template, abstention or echo replies; recalls use the spiking mouth). Next step: Claude writes a remap amendment. One cell is being re-run cleanly, and the B2d verdict waits for the stub arm. (2026-10-06)
- xedge-read-drive: (Claude review) PLAN §4 governs: the read drives the mismatch trial + one content exemplar only, ctx lines silent; §7.1 means that drive spans the whole window. Runner fixed on claude-review/xedge (c2b420bdd, test pinned). The round-3 smoke is void for calibration. In the draft worktree run `git merge --ff-only claude-review/xedge` (it fast-forwards; anything else is a STOP for NEEDS), then provision. Re-run the seed-7 smoke: check §8 (a)-(e), then freeze INTACT_FLOOR = 0.5 x the new delta (sign +1 required; first freeze). CLEAN_ZERO_BAR re-declared once (PLAN §9): off-diagonals <= 0.02 and |asym| <= 0.005. Then queue the 6 dev seeds. (2026-10-06)
- fb-store-verdict: (Claude review) STORE-NOT-LOAD-BEARING 6/6 accepted as honest. The cut engaged in the substrate: the belief rate fell from 0.333 to 0.017-0.081 on every seed while slot 0 stayed the unique winner. The abstain path is tested. Probe-scoped: with no time between narration and question, the host-gated write's residue carries the reply; tom-false-belief stays a host-input lesion in LBF reports. Next: the simulated-time probe on T18/T21. Its delay is pre-registered from dev seeds as the time for the lesioned residue to reach 0. Not a NO-GO. (2026-10-06)
- fb-relay2-query-bias: (Claude review) Item 3 (mechanism pick) is HELD. `_query` drops the relay bias, so the relays fire on presence alone and eviction kills the written slot (8/8 dev seeds; WORKLOG 2026-10-06 review). claude-review/fb-relay2 is merged to main (ff2521766). Re-run item 1 traces with QUERY_RELAY_BIAS=True on dev seeds 1-8 for false_belief (A,0,L,1) and two_unseen_moves, write -> hold -> query. Report belief slot spikes, mean V and g_i, plus presence, as a table, then NEEDS-CLAUDE [fb-relay2]. Never a gate seed or a GRID edit. Registered ops 0-3 FAILs were measured with this defect. (2026-10-06)
- iw1-sync: (Claude review) Merged main into research/integration-wave1 on claude-review/iw1 (tip 3c06a1c8f). Fast-forward research/integration-wave1 to it (`git merge --ff-only claude-review/iw1` in that worktree); do not merge it into main. Before the fast-forward, clear the uncommitted R copies and the copy_results.sh edit in the integration-wave1 worktree: they are byte-identical to what is committed (check with `git diff --stat` against the branch first; any difference is a STOP for NEEDS). Only FAILURE_LOG conflicted (union). Patch 0001 is applied and R is re-scored and committed. (2026-10-06)
- iw1-R: (Claude review) verify-go done, three skeptics, nothing refuted. R d6 PASS (survives). R swm PASS NARROWED: value-only, the stated margin changes, the prediction does not. R wmf PASS NARROWED: the referent preference is fixed per seed and ignores the question; with it on, 2 of 4 sessions abstain where the default answers. The flip list stays as registered: d6-hebbian-write,wm-focus-bind,xedge-surprise-worldmodel. Proceed to section 6. Each flip finding carries its WORKLOG narrowing word for word. The owner may still hold wm-focus-bind out of the default; a later RULING would say so. (2026-10-06)
- s1-c3c4-read: (Claude review) Merged to main (c3d1269ae). C3 PASS 6/6 at P2, narrowed (interface lesion; host-owned selection; episodic engagement unscored). C4 FAIL as registered (permuted clause 4/6); the reply follows the learned map, and an amended control is owed before attempt 2 (Claude, needs owner sign-off). Board, via the vikunja skill, then stamp docs/.vikunja_sync: Stage-1 attempt 1 "not passed. Word learning and lived memory pass (host code still decides); category inheritance fails; the feeling test fails on a badly designed control that is being redesigned". Close the two "Re-run the Stage-1 ..." tasks. Add "Redesign the feeling test's scrambled-values control (Claude)". (2026-10-06)
- s1-c4-audit: (Claude review) Ruled in AUDIT-affect-pool-results.md, "Claude ruling 2026-10-06". Void = attempt 1's gate/ C4 cells only. Pool/AWS affect load-bearing rows before 10-05 stand with a caveat. The audit is closed; no further listing. (2026-10-06)
- pool-hold-c1: (Claude review) The 54 TSV lines were dispatched (pool.running records them from 10-04 21:12; that file is a dispatch log, not live state). 52 landed at 6c2cb1362. Missing: s42 lesion_lateral_learn and s42 lesion_lateral_read, sent to wspool 10-04 23:42 with no output. The HANDOFF guard cannot pass on S (env.sh P is the workstation path; pool.running never reads 0), so do not run the block. First confirm both out files are absent in $P and pgrep shows no stage1_c1_category on any node. Then re-add exactly those 2 TSV lines verbatim (field 2 as --checked). If still missing after landing: score anyway (UNDEFINED INCOMPLETE), RESULT. (2026-10-06)
- stage2-dev-smoke: (Claude review) Yes, re-queue, but not the expired lines: they are pinned to 6b9cb47f7, which predates the runtime-inputs fix and is gone from main. Run agent_sync_branch.sh on research/stage2-exit-gate (a conflict means NEEDS) and provision the synced tip. Run before_you_build. Emit at that tip, seed 7, out _stage2_exit/dev/s7/, mem_gb=9/memcap 10, max_age_h=168: the 7 lesion/informational arms plus intact_a of ask, inner, conv and infer (11 lines). Canary one with `pool_canary.sh run auto`. Queue by 10-13; INFO when landed; the gate window is unchanged (11-03..11-10). (2026-10-06)

- word-role-engagement-done: (Claude review) Supersedes the word-role-engagement wait. claude-review/word-role-trace (c6fb9ddc7) is merged to main: the reply carries `word_role_trace` on every turn where the word-role parse ran, and `tools/s2p_score.py` makes a word-role-parse R cell UNDEFINED:engagement-* unless every arm records it with no error and the matching lesion flag. P4's F1 lines for word-role-parse may proceed at a revision containing c6fb9ddc7; check one dev-seed lesion arm first (`word_role_trace.lesioned` true, `error` null). The 9ea7411 R data now re-scores UNDEFINED:engagement-no-record by design: do not re-score or re-run it; its confirmed PASS stands with its caveat. Run `tests/test_webapp_server.py` once on the pool. (2026-10-06)
- T1-A0.1: (Claude review) Engagement bar (a) was unreachable by construction: a 50 ms reward in a 250 ms window caps the ratio at 1.25. That is an instrument error (Claude's, in STEP4B 10-04), not a negative. docs/drafts/long-delay-credit/AMENDMENT-A0.1.md now reads it at the reward time (`lhb_engagement_a_rewardwin.pass`; `null` = UNDEFINED); factor 2 and bar (b) unchanged. r5 FAILS it in smokes (1.69), so the build is not done. Section 4 rows that name bar (a) read the same field. (2026-10-06)
- T1-provision: (Claude review) (1) In $WT/draft-ldc-outcome-timing: clean the tree (commit or `git restore` research/findings/raw/_provenance/runs.jsonl). Then `git merge --ff-only claude-review/ldc` (63907ce8d; anything but a fast-forward is a STOP). (2) Main has moved past c3d1269ae, so run `bash tools/agent_sync_branch.sh research/draft-ldc-outcome-timing`; a conflict is a STOP. (3) `touch $P/research/queue/provision.req/$(git rev-parse research/draft-ldc-outcome-timing)`, then read /opt/host-status/sim-provision/<sha>.json. Use that sha as the pin for everything below. (2026-10-06)
- T1-round4: (Claude review) Seed 7 only, same line shape as rounds 1-3 (credit_immediate, onset 150/outcome 250, w_outc_strio 30, w-clear-fs 60, n-train 200). Weights are outc/rus/lhbinh/rmtg/snc. Queue: r5 re-run 30/27/24/20/3, r11 30/40/36/20/3, r9 30/54/48/20/3, r10 30/54/96/20/3. The r5 re-run must reproduce every field of the old r5 record (U, R, I, whole-window LHb rates, success), else STOP. Freeze the first of r11, r9, r10 that passes U, R, I, membrane_ok, void False, amended (a) and (b) (off U min 0.4982). Put the table in the WORKLOG and commit the round 1-4 records. If none passes: STOP, `[T1] 4b tune`. (2026-10-06)
- T1-grid: (Claude review) After the freeze, queue STEP4B section 3 (144 lines) at the same pin with the frozen constants and `#checked:` naming STEP4B.md + AMENDMENT-A0.1.md. Then write PILOT-4B.md with `lhb_engagement_a_rewardwin` per cell and `[T1] RESULT`, and stop. Claude will check `void_floor_ceiling` on the route-off cells when reading the result. (2026-10-06)

- iw1-wmf-hold: (Claude, owner-delegated 2026-10-06) Hold wm-focus-bind OUT of the iw1 flip. It passes R as registered, but on its own probe flag-on abstains on 2 of 4 sessions the default answers; flipping a faculty that makes replies worse fails the integration goal. This is a decision not to flip, recorded as a deviation from the registered flip list; no verdict changes. Section 6 flips d6-hebbian-write and xedge-surprise-worldmodel only; both flip findings and the WORKLOG quote this ruling. Next method for wm-focus-bind (no-defer): the question's cue drives the referent competition (spiking cue-match input to the focus pool), so the held referent depends on the question. Claude writes that packet; Hermes does not build it ahead. (2026-10-06)
- word-role-credit: (Claude, owner-delegated 2026-10-06) word-role-parse credit stays PARTIAL. FULL requires the spiking all-or-none commit population (RULING word-role-verify-go) to replace ROLE_MIN/margin and the per-role argmax. (2026-10-06)
- s1-c4-control: (Claude, owner-delegated 2026-10-06) Registered here, before any attempt-2 run, as an outcome-blind design fix: the C4 scrambled-values control draws each seed's permutation by rejection sampling from a fixed RNG (seed = the brain seed) until |r(scrambled values, human ratings)| <= 0.10, and records the drawn permutation and its r in the run output. Every other C4 bar is unchanged (intact r >= 0.55, scrambled r < 0.20 on all 6 seeds, lesion silences the read). Also report, not gated, the within-sign-group r. Hermes builds the draw on the stage1-exit branch with a test (a permutation with |r| > 0.10 is never used); Claude reviews it before attempt 2. (2026-10-06)
- stage1-calendar: (Claude, owner-delegated 2026-10-06) Attempt 1 (10-04) is the Stage-1 exit attempt on the calendar (attempts run even when expected to fail; it ran: not passed). No attempt 2 before 10-24: it would fail C1 by construction. Attempt 2 runs once C1's next method (docs/drafts/stage1-c1-inheritance/PLAN.md) has a dev-seed candidate and the s1-c4-control draw is reviewed, and no later than the Stage-2 gate window (11-03..11-10), co-scheduled with it. (2026-10-06)

- verify-go-lens-block: (Claude, main 87bf8c53f) Every refuter prompt you write for verify-go ends with the block between REFUTER-LENS-BEGIN and END in .claude/skills/verify-go/SKILL.md, copied verbatim, after the section-6 rulings naming the packet. It covers engagement, answer-rate and where-ran. A refuter that cannot answer a lens from recorded data reports it UNDEFINED, and the claim does not survive on that lens. A refuter prompt without the block voids the PASS it supports. (2026-10-06)
- where-ran-audit: (Claude) Run `python tools/where_ran.py <result paths>` in every audit or finding that cites results, and quote its host summary. If results span host classes (pool, workstation, aws), state which class each cited number came from and whether validity depends on it (e.g. inputs the provisioner does not ship under research/findings/raw/). A result whose host reads UNKNOWN is not usable as evidence. (2026-10-06)
- flip-hurts-answers: (Claude) Before any default flip, run tools/answer_rate.py (prefixed on the packet's on_/off_ shard dirs, or paired --on-tag flip --off-tag alloff); for iw2-style flips pass --off-tag to iw2_score.py B. If flags contains FLIP-HURTS-ANSWERS, file a NEEDS line before flipping; the flip waits for a Claude ruling. UNMEASURED means not checked: queue the flag-off cells (the alloff run) first. Reported beside the registered verdict; changes no PASS or FAIL. Checked on iw1: wm-focus-bind FLIP-HURTS-ANSWERS 6/6 (held out per iw1-wmf-hold); run it on d6-hebbian-write and xedge-surprise-worldmodel before section 6. (2026-10-06)
- clean-worktree-before-merge-ask: (Claude) Before an ASK or RESULT asks Claude to merge or confirm a branch, its worktree must be clean (`git status --porcelain` empty); tools/needs_line.py now refuses the line and prints the files. Commit them, or `git restore`/`git clean` what you do not need; for a Claude-reserved file write a patch outside the repo per RULING no-reserved-leftovers. `--dirty-ok "<reason>"` is for state files you cannot commit and puts the reason in the line. (2026-10-06)
- claude-brief: (Claude) `python tools/claude_brief.py` (read-only) shows what Claude sees for each open NEEDS line: branch, ahead/behind, dirty worktree, conflicts, missing cited paths. Run it before filing NEEDS and fix MISSING cites and dirty worktrees first. Claude runs it at the start of each check-in. (2026-10-06)
- s1-branch-hygiene: (Claude) research/stage1-exit-gate's worktree holds uncommitted research/findings/raw/_stage1_exit/gpu_lived_cupy/, research/queue/stage1_exit_report.md and stage1_exit_report2.md: commit the data (git add -f if ignored), move the two queue reports out of the repo, then leave the tree clean. The branch is ~420 behind main with conflicts in docs/drafts/stage1-exit/WORKLOG.md and tests/test_stage1_exit_gate.py: Claude merges main into it when approving the C4 draw (s1-c4-control is under review). (2026-10-06)

- bar-sanity-line: (Claude, main 90a45e633) Every new or amended prereg or amendment carries a `bar-sanity:` line: either `bar-sanity: <committed record path>` whose verdict is OK, or `bar-sanity: N/A <reason, >= 20 chars>` when the file registers no bar. Run `python tools/bar_sanity.py <spec> --out <record.json>` BEFORE registering; a perfect brain must pass >= 95% of draws and a null brain <= 5%. `gates/bar_sanity_line` blocks the commit without the line. Docs: docs/BUILD_LANE_CHECKLIST.md "Bar-sanity". Existing preregs are grandfathered until edited. (2026-10-06)
- bar-sanity-stop: (Claude) UNREACHABLE (a perfect brain fails) or TOOTHLESS (a null brain passes) is a STOP for NEEDS: do not register the bar or queue its run. A spec whose perfect brain is easy or whose null brain is hard is reviewed like the bar itself. Re-run the CLI after any scorer edit (a changed scorer sha makes the record stale). (2026-10-06)
- s1-c4-control-sanity: (Claude, owner-delegated) At SCRAMBLE_R_MAX = 0.10 a realistic perfect brain (reads follow the learned values at r 0.93-0.96) passes the six-seed scrambled clause only 0.943 (record docs/drafts/stage1-exit/bar_sanity_record_c4_constrained_0p10.json): UNREACHABLE. The constant becomes 0.05 (about 0.98), nothing else changes. Claude is making the edit, the main merge and the OK record on claude-review/stage1-sync; a later RULING says when to fast-forward research/stage1-exit-gate. The attempt-2 amendment cites that record on its `bar-sanity:` line and the old control (0.461) only as `bar-sanity-before:`. The draw itself (bc8f804b9) is approved. (2026-10-06)
- t1-bar-sanity: (Claude) On research/draft-ldc-outcome-timing: `bash tools/agent_sync_branch.sh research/draft-ldc-outcome-timing` (it needs tools/bar_sanity.py from main), then `git merge --ff-only claude-review/bar-sanity-t1` if it fast-forwards, else `git merge --no-edit claude-review/bar-sanity-t1` only if clean (a conflict is a STOP). It adds the T1 specs/records and the `bar-sanity:` line on AMENDMENT-A0.1.md: amended reward-window bar perfect 1.000, null 0.000; the registered whole-window bar is UNREACHABLE and stays cited only as `bar-sanity-before:`. A0.1 stands unchanged; T1-round4 proceeds. (2026-10-06)

- stage1-sync: (Claude) claude-review/stage1-sync (4625d662a) = research/stage1-exit-gate (incl. your hygiene commit c96da2ce5) + main, SCRAMBLE_R_MAX = 0.05, bar-sanity record docs/drafts/stage1-exit/bar_sanity_record_c4_constrained.json OK (perfect 0.987, nulls 0.000, n=1000), stage1 tests 14/14. Once your research/stage1-exit-gate tree is clean, `git merge --ff-only claude-review/stage1-sync` there (never rebase); if the branch moved and it does not fast-forward, file NEEDS instead. The attempt-2 amendment cites that record on its `bar-sanity:` line. Attempt 2 timing per RULING stage1-calendar. (2026-10-06)

- b2d-amend2: (Claude review, main c5a06fb3b) Amendment 2 is registered (B2d PREREGISTRATION A2.1-A2.13; route map in research/packets/b2d-mouth-renderer/QWEN-ROUTE-MAP.md). Run research/packets/b2d-mouth-renderer/AMEND2-HANDOFF.md top to bottom as T23's successor, all GPU work via `gpu_remote.sh add` per RULING b2d-mouth-gpu and game-mode-pauses (~14 GPU-hours). Steps: pin P2 = main, export P2 with the valence map and check `export-ok-with-map`, 30 dev lines (seeds 1-8), `tools/b2d_amend2.py dev`, then the six gate `route` arms and `routecheck`, then the 48 remaining gate arms, then `gate`. STOP for NEEDS at NO-CANDIDATE, LESION-SMOKE-FAILED, ROUTE-FAILED (route-mismatch/silent) or any `needs_claude` line. Never read or report dev lesion replies; never write the finding. (2026-10-06)
- b2d-mouth-remap-retired: (Claude review) Supersedes RULING b2d-mouth-remap: the dev-7 remap (`emo` then "what does the dog chase") is retired, because the spiking recall mouth speaks its second turn. RULING b2d-mouth-s42-coloring still stands (one clean re-run of that cell for the original arm's record). (2026-10-06)
- b2d-amend2-trace: (Claude review) `BRAIN_MOUTH_TRACE` (default off) is observability only. Set it only in Amendment 2 arms; never count it as a production flag or flip it. (2026-10-06)

- spiking-commit-a1: (Claude, main fa3c0acf0) Packet word-role-commit (default OFF): a spiking commit stage (exclusive gate + self-holding latch) replaces ROLE_MIN, the margin and the per-role argmax. Run docs/drafts/spiking-commit/HANDOFF.md top to bottom: worktree research/word-role-commit, pin, tests, provision.req, dev sweep (8 pool lines), select, step O (6 lines, canary first), score O; only on O PASS, step R (12 shards). After the R canary, check the probe turn's word_role_trace (route word_role_commit, error null, commit_lesioned true in the lesion arm, false intact) before queueing; then score R. STOP + NEEDS on any test failure, select NONE or UNDEFINED twice, canary rc != 0, trace mismatch, O not PASS, R engagement-UNDEFINED or R FAIL. Verdicts PROVISIONAL. (2026-10-06)
- spiking-commit-design: (Claude) docs/drafts/spiking-commit/PLAN.md is the reusable decision stage wherever host code picks between spiking alternatives: exclusive gate + self-holding latch on the same bridge, read as which latch turned on, with a registered threshold-invariance bar. Order: A1 word-role-parse, then A2 wm-focus-bind (the question cue biases the referent competition), then A3 C3 fact selection and the d6 write threshold. Claude writes A2 after A1 step O; Hermes builds none of A2/A3 ahead. (2026-10-06)
- word-role-credit-path: (Claude) word-role-parse stays PARTIAL. FULL = word-role-commit O PASS and R PASS, then a flip packet making BRAIN_WORD_ROLE_PARSE and BRAIN_WORD_ROLE_COMMIT default on whose F1 passes with no FLIP-HURTS-ANSWERS. (2026-10-06)
- spiking-commit-owner-calls: (Claude, owner-delegated) First-come role assignment is accepted (incremental parsing commits early, as in garden-path effects), but per RULING word-role-p4-checks report dual-role-word questions in R; if the commit route declines or misassigns more of them than the count rule did, file NEEDS. The declared residuals (hand-set weights incl. the frozen W_RG, the latch-sparing inter-word reset, the organ on its own bridge) are a named credit boundary: FULL credit means spiking decides the role, never learned or one-substrate; every finding states them. (2026-10-06)

- stage2-bar-sanity: (Claude review) All six Stage-2 gate bars (C1-C5 and the stage verdict) read bar-sanity OK (n=1000, perfect 1.000, every null 0.000); records on claude-review/stage-bar-sanity-s2 (b7f8c1b41), which already contains a clean merge of main. In the research/stage2-exit-gate worktree: FIRST `git merge --ff-only claude-review/stage-bar-sanity-s2` (anything but a fast-forward is a STOP for NEEDS), THEN `bash tools/agent_sync_branch.sh research/stage2-exit-gate`. GATE_SHA, PREREG_SHA and the window are unchanged unless a later RULING amends C4/C5 (Claude is checking whether a host lookup can pass them). (2026-10-06)
- stage1-sync-ff-target: (Claude review) Supersedes the fast-forward target in RULING stage1-sync: fast-forward research/stage1-exit-gate to claude-review/stage-bar-sanity-s1 (f491ba99d), which is stage1-sync plus the C1/C2/C3 bar-sanity specs and records. Same rules: clean tree first, --ff-only, never rebase, NEEDS if it does not fast-forward. Stage-1 C2 `word` and C3 `lived` read OK and attempt-1 data re-score PASS 6/6 under the current runner. (2026-10-06)
- stage1-c1-lesion: (Claude review) The registered Stage-1 C1 bar is UNREACHABLE for the PLAN's pathway build: the counted lesion lesion_draw cuts the generative draw, which that route never touches, so every seed reads UNDEFINED (INCOMPLETE). Cite docs/drafts/stage1-exit/bar_sanity_record_s1_c1_registered.json only as `bar-sanity-before:`. Amendment A2 (Claude writes it with the C1 build) registers the pathway-cut patch `lesion_pathway` (call counter; applied = calls > 0) as C1's counted lesion and makes lesion_draw informational; correctness, permuted-tracking and six-seed bars are unchanged; A2 cites bar_sanity_record_s1_c1_fixed_pathway_lesion.json (OK). Do not queue C1 gate lines under the old lesion. (2026-10-06)

- c1-inheritance: (Claude, main 71772babe) The 09-30 C1 wiring (BRAIN_CATEGORY_INHERIT, default off) is on main; its own gate scores PASS 6/6 narrowed (finding research/findings/2026-10-06-stage1-c1-category-inheritance-RESULT.md; two skeptics, none refuted). Run docs/drafts/stage1-c1-inheritance/HANDOFF.md top to bottom: commit the 52 gate shards (git add -f if ignored); pin main; tests; provision; canary (seed 1 on cat_disc must give a flagged worm guess), then the other 95 lines; score with research/runners/c1_flip_dev_check.py; NEEDS [c1-flipdev] RESULT. STOP per the HANDOFF. Never flip, edit code or use gate seeds. (2026-10-06)
- s1-c1-plan: (Claude) SUPERSEDED: the 10-04 plan duplicated the 09-30 build and is retired; its replacement is in the same file. Order: flip dev check (Hermes), flip (Claude), attempt-2 C1 amendment counting lesion_inherit (Claude, before any attempt-2 data), attempt 2. Board: "Build category inheritance into the live brain" becomes "Check category inheritance is safe to switch on". (2026-10-06)
- stage1-c1-lesion-name: (Claude) Corrects RULING stage1-c1-lesion: the counted C1 lesion is `lesion_inherit` (the 09-30 organ's cut, engaged only if `category_inherit.read` records lesion "all" and lesion_held true), per docs/drafts/stage1-c1-inheritance/ATTEMPT2-C1-AMENDMENT-TEXT.md; `lesion_pathway` was a placeholder name. Before A2 is committed, Claude re-runs the C1 bar-sanity spec against lesion_inherit's engagement rule and cites that record. (2026-10-06)
- stage1-calendar-c1: (Claude) Amends RULING stage1-calendar: C1 now has a candidate (wiring PASS 6/6). Attempt 2 waits only for the flip dev PASS, the flip, and the attempt-2 amendment (C1 lesion arm + C4 control). It may run before the Stage-2 window and no later than 11-03..11-10. (2026-10-06)
- pool-hold-c1-closed: (Claude) Supersedes RULING pool-hold-c1: CLOSED. C1 was scored without the two s42 lateral arms (informational; verdict unaffected). Do not re-add them; if they land anyway, commit them with the shards, no re-score. (2026-10-06)
- c1-overextension: (Claude, owner-delegated) A default-on category-inheritance faculty may volunteer guesses flagged as not taught when categories blur (the bat case: "what does the canary swim" guesses lake), as normal learner over-extension within the honesty boundary. The flip commit reports that rate; the hop-attenuation fix (PLAN §5.1) opens the same day as Claude's packet. (2026-10-06)

- c1-hop-first: (owner 2026-10-06, "would be nice to fix it since we have the time to") Supersedes RULING c1-overextension: the hop-attenuation fix (PLAN §5.1, the bat case) lands BEFORE the C1 flip. From RULING c1-inheritance's HANDOFF do step 1 only (commit the 52 gate shards); queue nothing else for C1 until Claude merges claude-review/c1-hop and a later RULING gives the updated HANDOFF (the flip dev check then runs with the fix, and the C1 gate PASS is re-measured with it). (2026-10-06)

- c1-inheritance-tip2: (Claude, main e822a59f4) claude-review/c1-inheritance @ 41e00380f is on main: it adds install_patch("lesion_inherit_pathway") (call counter) to stage1_exit_gate.py for Amendment A2, and brings the stage1-sync runner (SCRAMBLE_R_MAX=0.05) and the s1 bar-sanity specs/records to main. Any C1 "flip first" wording in the agent's notes is superseded by RULING c1-hop-first: the hop fix lands before the flip, and the flip is Claude's after the flip dev check passes WITH the fix. The A2 lesion name is fixed when Claude writes A2. (2026-10-06)

- stage2-c4c5-amend: (Claude review) docs/drafts/stage2-exit/AMENDMENT-A3.md (on claude-review/stage2-c4c5 @ bd5c95bb4) hardens C4 (the ladder lesion must change the NAMED category on all four feel turns, as prereg 4.4 already says) and C5 (the idle fact after the permuted exposure must differ from the intact one). Outcome-blind (no gate data); no worker, arm, turn or verdict rule changes. The host-lookup and fixed-fact brains are now bar-sanity NULLs (0.000); a ladder-gated lookup and a host FIFO remain residuals for the host-share instrument (prereg section 5, NEEDS #10). C5 strictness accepted by Claude (owner-delegated). (2026-10-06)
- stage2-c4c5-ff: (Claude review) Supersedes the fast-forward step of RULING stage2-bar-sanity. In $P/.claude/worktrees/stage2-exit-gate: `git merge --ff-only claude-review/stage2-c4c5` (it contains stage-bar-sanity-s2 and your 319be09ce; anything but a fast-forward is a STOP for NEEDS), then `bash tools/agent_sync_branch.sh research/stage2-exit-gate`, re-run `pytest tests/test_stage2_exit_gate.py` and the runner selftest. PREREG_SHA stays 24dac499e. Pin GATE_SHA at a revision containing AMENDMENT-A3.md (check `git cat-file -e $SHA:docs/drafts/stage2-exit/AMENDMENT-A3.md`). (2026-10-06)
- stage2-c4c5-dev: (Claude review) No dev line is re-run: the existing seed-7 shards (incl. the 11 at pool revision f2aaabc8a) re-score under the amended aggregate; dev capability reads and NEEDS #15-#17 stand (dev seed 7 still lacks C4/C5 capability, as before). Cite only the `_A3` bar-sanity records for Stage 2; the un-suffixed ones are stale by scorer hash. Queue the dev smoke per RULING stage2-dev-smoke at the synced tip. (2026-10-06)

- gpu-utilization: (owner 2026-10-06: GPU fine "as long as it gets properly utilized going forward. We do have AWS available as needed") Queue every GPU line the moment its HANDOFF allows (B2d Amendment 2, C1 hop re-measure, C4/C5 dev, word-role-commit if GPU), via `gpu_remote.sh add`; never leave registered GPU work unqueued while W's queue is empty. Ask for AWS with NEEDS [gpu-capacity] (state the backlog in GPU-hours and the dated deadline) when either holds: GPU work has waited more than 24 h behind GAME_MODE, or the queued GPU backlog exceeds about 24 GPU-hours with a deadline within 7 days. Claude starts AWS within the owner's ~$50/day cap with auto-stop. Never start AWS yourself (RULING aws-capacity stands). (2026-10-06)

- s2-c5-build: (Claude review, main merge of claude-review/s2-c5-build @ 369561ce4) The idle-replay organ (default OFF, BRAIN_IDLE_REPLAY: one spiking assembly per stored fact, recency x DA-strength excitability, race + spiking commit latch, decoded on the brain's own memory bridge) is on main. Run docs/drafts/idle-replay/HANDOFF.md top to bottom: sync research/stage2-exit-gate with main (conflict = NEEDS), tests, provision, 8 organ lines + 34 reply lines + 1 regression-battery line (dev seeds 1-8 only, canary one reply line), queued by 10-13. Score with --score / --score-reply, commit the data, file one `[s2-c5-build] INFO` line. STOP with NEEDS on any HANDOFF §4 condition. Never flip a default, never touch GATE_SHA. (2026-10-06)
- s2-c5-15: (Claude) Stage-2 NEEDS #15 (C5 voices a demo-lexicon fact) is answered by a build, not a protocol change. The gate picks the organ up only after Claude flips it on main and the gate HANDOFF step 3 merges main and re-pins GATE_SHA. Claude reads the 8-seed dev data for the ~2.5%-per-draw second-newest miss (about 14% risk of one miss across 6 gate seeds) and may tighten the recency weight on DEV seeds before the flip; never after gate data. Moving the between-turn wander onto this store is a later packet, not before the gate. (2026-10-06)

- s2-c4-build: (Claude review, main merge of claude-review/s2-c4-build @ 7031916fa) The C4 emotion-category organ (default OFF; emotion channels decided by the spiking commit stage, read out through `affect_out`) is on main. Follow docs/drafts/stage2-c4-build/HANDOFF.md: sync research/stage2-exit-gate with main, tests + selftest, reproduce the organ battery (commit dev_select_repro.json), provision the synced tip, queue the 56 dev-seed handler lines by 10-10, score (emo group only) and `--flip-check` by 10-15. Dev seeds only; no GRID edit, no flip. Each HANDOFF §4 STOP is a NEEDS line. Run the organ-battery reproduce on the workstation or pool, not on S. (2026-10-06)
- s2-c4-flip: (Claude) The default flip is Claude's, after verify-go on clean dev data: emo PASS at dev 8/8 and a flip check with no FLIP-HURTS-ANSWERS / FLIP-CHANGES-NONFEEL / FEEL-NOT-APPEND-ONLY flag and nothing unmeasured; target 10-20. Then re-run the five dev-seed-7 intact_a arms at the flip tip; Claude pins GATE_SHA at a revision with the flip, AMENDMENT-A3, the runner and its tests, before 11-01. (2026-10-06)
- s2-c4-reading: (Claude, owner-delegated) Two interpretations the C4 result must state, and verify-go reviews: (1) the emotion channels extend the affect state beyond the valence ladder (which cannot separate fear from anger), and the registered ladder lesion now also cuts their read-out; (2) the trigger map and labelling curriculum are a hand-written seed (like the Warriner norms), and the teacher's emotion labels overlap the gate lexicon by design. A pass reads PASS, narrowed: spiking decides the category; the vocabulary and triggers are host-written. (2026-10-06)

- c1-hop: (Claude review, main merge of claude-review/c1-hop @ 7c00b9dc8) Hop attenuation in the category-inheritance organ (BRAIN_CATEGORY_INHERIT_HOP, default off; off is byte-identical to the 09-30 organ): the frame-to-frame pathway carries half its current and the read lasts 30 steps. Organ dev 16/16 on dev seeds 1-16 (bridge canary swim abstains, canary eat keeps worm), IN-SAMPLE: the 30-step read was the only grid setting passing all 16, so the gate_hop re-measure on the six gate seeds is the real test. Margins are thin (worm down to 1.6x floor, permuted bug 1.25x). (2026-10-06)
- c1-hop-handoff: (Claude) Supersedes RULING c1-hop-first's hold. Order: (1) commit the 52 C1 gate shards if not done; (2) run the updated docs/drafts/stage1-c1-inheritance/HANDOFF.md steps 2-5: flip-dev canary first, then the 54 gate_hop lines (step 3b) and the other 95 flip-dev lines; score both and write NEEDS [c1-gatehop] and [c1-flipdev]. Obey its STOP list. Pool lines only; nothing on S beyond scoring. (2026-10-06)
- c1-gatehop: (Claude) The 09-30 C1 PASS 6/6 is the no-hop record only. The PASS that the flip and Stage-1 attempt 2 rely on is the gate_hop verdict (prereg 09-30 section 13, Amendment 1: seeds 42/43/44/100/101/102, all 9 arms, --aggregate --hop). If it is not PASS after one re-run of missing cells: no flip, rework on dev seeds (next method: an in-substrate end to the retrieval cycle instead of the 30-step host read), never retune on gate seeds. (2026-10-06)
- c1-flip-config: (Claude) The flip (after gatehop PASS and flipdev PASS) turns on both BRAIN_CATEGORY_INHERIT and BRAIN_CATEGORY_INHERIT_HOP; HOP=0 restores the 09-30 organ. Amendment A2 counts `lesion_inherit_pathway` (one name for arm and patch), renames that arm key in the s1 bar-sanity spec/common file, re-runs those records, and requires the C1 read to record hop. (2026-10-06)
- c1-bat-guess: (Claude, owner-delegated) "What does the bat swim" -> lake stays allowed as a flagged guess: it is a one-hop inference through the bat's own taught "eat bug", the same inference C1 makes. Category competition (PLAN §5.6) and the in-substrate retrieval end are later rungs, not flip blockers. Thin margins are accepted for the re-measure. (2026-10-06)

- c1-hop-not-yet: (Claude) Correction to RULINGS c1-hop and c1-hop-handoff: claude-review/c1-hop is NOT on main yet (the merge was refused by claim_check on its finding; being fixed). Until a later RULING says it is merged, do only HANDOFF step 1 (commit the 52 C1 gate shards); HANDOFF step 0 must print READY before steps 2-5. (2026-10-06)

- c1-hop-merged: (Claude) claude-review/c1-hop @ dececb722 IS now on main (finding fixed for claim_check; scratch reads committed as hop_scratch_reads.json). Supersedes RULING c1-hop-not-yet: run the updated docs/drafts/stage1-c1-inheritance/HANDOFF.md steps 2-5 per RULING c1-hop-handoff (step 0 should print READY). (2026-10-06)

- iw1-ff-go: (Claude, answers NEEDS [iw1-sync] 16:09) The branch versions are authoritative: the three local R_{d6,swm,wmf}.json are the pre-rescore copies, superseded by the branch's re-score with preconditions. In the integration-wave1 worktree, move those three to /opt/data/cache/aside/iw1-pre-rescore/ (kept for the record), delete the 526 byte-identical uncommitted copies and restore copy_results.sh, confirm `git status --porcelain` is empty, then `git merge --ff-only claude-review/iw1` (anything but a fast-forward is a STOP for NEEDS). Then section 6 per RULINGS iw1-R, iw1-wmf-hold and flip-hurts-answers: run tools/answer_rate.py on d6-hebbian-write and xedge-surprise-worldmodel first; flip only those two. (2026-10-06)
- stage2-gate-timing: (Claude, on NEEDS [stage2-dev-smoke] 16:32) Score the 11 seed-7 dev shards with the amended aggregate and record each criterion's dev read in the WORKLOG; that closes the dev smoke. Queue NO gate lines before the window opens 11-03 (HANDOFF section 1). The GATE_SHA pinned at 4b330c082 is provisional: at HANDOFF step 3 in the window, merge main and re-pin so GATE_SHA contains the C4 and C5 default flips (RULINGS s2-c4-flip, s2-c5-15), then file NEEDS [stage2-gate-pin] and wait for Claude's confirming RULING before section 5 queues the 186 gate jobs. (2026-10-06)
- a2-no-push: (Claude, answers NEEDS [A2] 17:27) Skip push_both in AMEND2-HANDOFF step 0: it is a stale instruction. `tools/gpu_remote.sh add` delivers the commit to the workstation itself over its own ssh connection (refs/server-jobs/<sha>), so no GitHub/Gitea push is needed. Treat "PIN_P2 committed (e5bc97eb8), tests 21 pass" as step 0 done and run step 1 from the research/b2d-amend2 worktree. Never run push_both from the hermes container. (2026-10-06)
- iw1-ff: (Claude, on NEEDS [iw1-ff] 04:50) Your diagnosis was right: the FF target lacked your two WORKLOG marker commits. Claude merged research/integration-wave1 into claude-review/iw1, now at fef5bf63a (WORKLOG union-resolved), and research/integration-wave1 is an ancestor of it. FF research/integration-wave1 --ff-only to fef5bf63a, then run RULING iw1-ff-go unchanged: set the stale R_*.json aside, delete identical copies, restore copy_results.sh, answer_rate on d6 and xedge, flip only those two. (2026-10-07)
- stage1-ff: (Claude, on NEEDS [stage1-ff] 03:52) Same cause. Claude merged research/stage1-exit-gate (your marker 4e78bc0ff) into claude-review/stage-bar-sanity-s1, now at 7a785f68f. FF research/stage1-exit-gate --ff-only to 7a785f68f; this replaces f491ba99d in RULING stage1-sync-ff-target. General rule: when a Claude FF target lacks only your WORKLOG/marker commits, STOP is still correct (never merge yourself); Claude re-targets. (2026-10-07)
- lane-builds: (Claude, on NEEDS [lane-builds] 23:34) Ratified after the fact: the waiver and the three regs were truthful and both have expired; Claude deleted the expired .lane_waiver; the expired rows stay as inert history. Standing permission from now: the driver MAY write research/queue/.lane_waiver (never lane_builds.jsonl) only when lane_check.py shows no ready line for an idle lane, with CLASS NO-READY-WORK and the lane_check evidence as the reason, and must file one `[lane-waiver] INFO` NEEDS line in the same wake. Never to cover a lane that has a ready line. (2026-10-07)
- s2-c5-battery: (Claude, on NEEDS [s2-c5-build] 21:53) Correct STOP: the ~28 GB numpy battery fits no pool node. Claude is running the exact HANDOFF line at rev bc66ccaa3 on the workstation (memcap 32, started 05:32). When it ends, the JSON + prov + log land in /opt/data/cache/c5-battery/ (or a FAILED file with the log tail). On JSON: copy it to research/findings/raw/_regression_battery/ in your stage2-exit-gate worktree, score per HANDOFF section 5, commit the data, file the `[s2-c5-build] INFO` line. On FAILED: file NEEDS with the tail. Do not re-queue it anywhere. (2026-10-07)
- driver-venv: (Claude, on NEEDS [driver-venv] 22:03) Claude restored dearpygui 2.3.1 in the driver .venv (declared at requirements.txt:11); the tool_health auto-tune probe should read healthy at the next run. (2026-10-07)
- iw1-make-jobs: (Claude, on NEEDS [iw1-make-jobs] 07:11) Approved: commit the fix on research/integration-wave1 yourself, exactly one line: make_jobs.sh:76 becomes `$( [ "$STEP" = bisect ] && printf '_%s' "$PKTS" || true )` (under set -e the false test fails the assignment; output is identical). Then regenerate nothing: the off-cell files you made are byte-identical to what the fixed script emits; confirm that with one diff and note it in the WORKLOG. lane-waiver and T14 INFO lines: read, no action. (2026-10-07)
- transitive-reach-read: (Claude, on RESULT [transitive-spiking-reach] 10:32) Accepted as NARROWED: what is shown is determinism, not robustness across seeds, and the answer rate is UNMEASURED. Do not flip. Queue its answer-rate line next, and file a RESULT when it lands. (2026-10-07)
- transitive-engagement-withdraw: (Claude, on NEEDS [transitive-engagement] 11:30) Confirmed superseded: the 12 lines already ran 10-05 and R rescored PASS, so skip the withdraw and the regeneration and proceed with T19. (2026-10-07)
- answer-rate-off: (Claude, on NEEDS [answer-rate-off] 15:45) Take the off-side rate from each faculty's paired lesion arm (lesion_d6_hebbian_write, lesion_xedge_surprise_worldmodel), which runs the same probes and registers lbf_rows. If answer_rate.py cannot read a lesion arm as off-side, add an --off-arm option on research/integration-wave1 with a test, and label the result "off = lesion arm" in the WORKLOG. The d6/xedge flips wait for this read. (2026-10-07)
- step4b-below-chance: (Claude, on NEEDS [step4b-outputs] 18:47) The off and cut arms are controls designed to fail. Declare below_chance on their cells with reason "pilot control arm, failure expected by design", then commit the raw grid. If any below-chance cell is in an ON arm, list it UNDECLARED and file NEEDS. Claude reads PILOT-4B at the next review. (2026-10-07)
- t24-finding-fix: (Claude, on NEEDS [t24-finding-fix] 20:29) Approved: you make the correction in 2026-10-04-stage1-lived-cupy-replication-INFORMATIONAL.md. Replace the C3 sentence so it says that numpy C3 failed only on the void lesion_recall clause, which is corrected to UNDEFINED, and that C3 re-measured PASS 6/6 @7851dd220. Add a dated "Corrected 2026-10-07" line, and commit through agent_merge.sh. (2026-10-07)
- c1-requeue-board: (Claude, on INFO [c1-requeue] 23:28) Approved: delete the stale "Owed: re-queue s42 lateral cells" item from GAP_CLOSURE_MISSION.md CURRENT STATE, citing RULING pool-hold-c1-closed. affect-results-audit: read; Claude rules which results are void at the next review, and nothing is voided before that. (2026-10-07) — DONE 2026-10-08 (Hermes): the s42 clause is deleted from the CURRENT STATE "Owed" line (still-live items kept), the ruling cited inline; commit accompanies this line.
- s2-c5-battery-aws: (Claude, amends RULING s2-c5-battery) The workstation run is void: 18 h in, still on the first arm, and the worktree lacked data/corpus/tinystories.txt. Re-run started 10-08 03:51Z on AWS r7i.4xlarge at rev bc66ccaa3 with the data file present, the two arms running concurrently through the runner's own compare/output. Results land in /opt/data/cache/c5-battery/ as before (battery + arm_on/arm_off JSON, plus PROVENANCE.txt to cite in the commit message), or as FAILED. Claude's watcher terminates the instance. (2026-10-08)
- iw1-flip-go: (Claude, on RESULT [answer-rate-off] 02:41) The off side, read from the lesion arm, is OK 6/6 for both d6 and xedge with no FLIP-HURTS-ANSWERS, so the condition in RULING iw1-ff-go is met. Flip d6-hebbian-write and xedge-surprise-worldmodel default-ON as that ruling specifies (those two only, wm-focus-bind stays held), then file a RESULT with the flip commit. (2026-10-08)
- t24-landed: (Claude, on NEEDS [t24-finding-fix] 23:57) Claude merged research/s1-lived-cupy @ 23ff6bb94 to main (9ea2f9947). Policy stays as is: an edit to an existing finding lands through Claude. (2026-10-08)
- step4b-on-cells: (Claude, on NEEDS [step4b-on-cells] 23:43) Commit the 27 ON-arm cells declared below_chance with the reason "pilot ON-arm cell below chance: a measured result, pending Claude's PILOT-4B read, not a pass", so the record is complete. Do not tune anything against them. Claude reads PILOT-4B, including these cells, at the next review. (2026-10-08)
- c5-battery-void: (Claude, amends s2-c5-battery-aws) The AWS battery ended rc=0, but the ON arm (BRAIN_IDLE_REPLAY=1) wrote nothing and left no traceback (arms_built.on=false); a memory kill is most likely. Its 34 "regressed" are an artifact of the missing arm; see VOID.txt in /opt/data/cache/c5-battery/. The OFF arm completed in under 3 h, while the workstation's ON arm was still running after 18 h at 28 GB. So idle-replay ON appears to blow up time and memory in the integrated battery, which blocks the C5 flip. (2026-10-08)
- c5-on-arm-diagnose: (Claude) Next C5 step, before any flip: find which battery turns blow up with idle-replay ON. Queue the worker per probe group (--subset with that group's labels, BRAIN_IDLE_REPLAY=1, numpy, rev bc66ccaa3) under memcap 10 on the pool, recording peak RSS and wall time per group. Groups that exceed the cap are the finding. Commit the OFF arm JSON plus the per-group table, and file a RESULT naming the turns and the growth. Never tune the organ against the battery. (2026-10-08)
- tool-health-auto-tune: (driver note, 2026-10-08 06:41) The auto-tune ROTTED probe's root cause: `.venv` lacks `cupy-cuda12x` (declared requirements.txt:5) since the 09-30 server migration; `neural-simulator.py:146` imports cupy unconditionally, so the probe dies there — the dearpygui import fixed by RULING driver-venv now passes. Both pip and uv installs are BLOCKED by the package-threat-intelligence scan in unattended mode (3 attempts). One interactive command fixes it on the server: `uv pip install --python .venv/bin/python cupy-cuda12x` (1TB disk free; cupy imports lazily, no GPU needed).
- c1-board-merged: (Claude, on NEEDS [c1-requeue-board] 05:27) Merged research/c1-requeue-board @ 9daff3add to main (35df0bbda). (2026-10-08)
- tool-health-cupy: (Claude, on NEEDS [tool-health-auto-tune] 06:43) Claude installed cupy-cuda12x 14.2.0 in the driver .venv (requirements.txt:5). The container has no GPU: if any test or gate now fails because cupy imports but finds no device, file NEEDS with the test name. Do not uninstall it. (2026-10-08)
- c1-flipdev-diagnose: (Claude, on RESULTs [c1-gatehop] 09:41 + [c1-flipdev] 10:14) gate_hop PASS 6/6 while flipdev is 0/8 because the category_inherit stamp is missing, which looks like a missing reply stamp rather than a missing behaviour. Take one dev seed and one failing turn (cat_told/t_bird_eat), run it flag-ON with the per-turn trace, and report three things: did the inherit pathway fire (decision fields), did it change the reply, and where the stamp should have been written. File a RESULT. No flip, and no change to the flipdev scorer. (2026-10-08)
- ack-1008: (Claude) Read: s1-c3-rerun (C3 PASS at P2), flip-commit-blocked (re-apply the iw1 flip at the first wake after the waiver budget frees, as you planned), c5-on-arm-diagnose wspool rc=127 (requeue without GNU time is fine). File the c5 RESULT when all 6 groups have ended. (2026-10-08)
- pilot-4b-read: (Claude, STEP4B section 4 reading of PILOT-4B, dev seeds 8-15) (1) A, the LHb engagement clause, is causal: ON passes 45/48, cut fails 48/48, off is UNDEFINED 48/48 as designed. (2) U separates route-on from route-off (about 0.47 vs 0.498) but not ON from cut (difference under 0.002), and its 0.47 bar sits on the ON mean, so U is marginal. Run tools/bar_sanity.py on U before step 5. (3) R and I pass in every arm, so they are sanity bars, not evidence. (4) Behaviour: success is about 0.16-0.59 in all three arms, with no gap between ON and cut. The 27 below-chance ON cells belong to this: the mechanism engages but does not yet move choice. (2026-10-08)
- pilot-4b-next: (Claude) Verdict: the mechanism engages; no behavioural credit yet. Do not start step 5 as a GO attempt. Next, on dev seeds only: (a) a per-seed paired table of ON minus cut success, plus the ON void_floor_ceiling rate (3-6 of 8 ON cells are v1, so name what saturates); (b) check whether the omission dip reaches striatal plasticity inside the eligibility window at onset 30 and 150 (trace span vs onset), as a per-trial read. File one RESULT. This is a method diagnosis, not a closure. (2026-10-08)
- affect-audit-rule: (Claude, on INFO [affect-results-audit]) "Imports the organ" is too broad to void anything. A result is VOID only if (a) it ran on a pool or AWS node whose revision dir lacked research/findings/raw/_affect_learned_valence_map.json (provisioned before the RUNTIME_RAW_INPUTS fix shipped), AND (b) its claim reads the affect valence: affect-*, appraisal, b2d mood, onebrain-affect verify, stage1 C4 lines (already handled), and LBF/host-share rows for affect faculties only. (2026-10-08)
- affect-audit-not-void: (Claude) NOT void: regression-battery ON-vs-OFF comparisons for non-affect flags (both arms ran the same map-less organ), LBF and host-share rows for non-affect faculties, and the long-delay-credit runs. Produce the narrowed list under RESULTS.md section "Void per RULING affect-audit-rule" (runner, count, date range, any finding that cites them), commit it, and file a RESULT. Claude confirms it and registers any affected finding in docs/RETRACTED.md. (2026-10-08)
- c1-stamp-fix: (Claude, on RESULT [c1-flipdev-diagnose] 14:07) Two separate defects. (1) Instrument: the repair early return at server.py:6265 skips the category_inherit stamp. On research/stage1-exit-gate, set the stamp on every return path from the same fired flag, so it is present exactly when the pathway fired and absent otherwise, with a test covering both paths. Then file NEEDS for Claude's merge. (2) Real miss: on t_bird_eat the pathway did not fire (salmon in slot3, fish in slot4). Never force it. (2026-10-08)
- c1-flipdev-rerun: (Claude) After the stamp fix merges, re-run flipdev (8 dev seeds, unchanged scorer) and report two numbers: turns where the pathway fired vs not, and the pass count. If the misses cluster on slot collisions like t_bird_eat, name that pattern in the RESULT. It is the next C1 method question, not a scorer change. (2026-10-08)
- affect-audit-confirmed: (Claude, on RESULT [affect-audit-not-void] 15:16) Confirmed: 671 of 6680 rows VOID as listed, including the reward-value judgment and the subsets. PREREGs stay as written. On branch research/affect-audit-retract, register in docs/RETRACTED.md: (i) 2026-09-25 affect-learned-vocabulary-eval-NOGO, which is VOID because the map was absent, so status is UNMEASURED; (ii) the affect-pool verify verdicts in 2026-09-23 onebrain-affect-ladder...NOT-ALL-GO; (iii) only the Gate-B valence-sign claim of 2026-09-24 open-ended-gated-turn-seed7. Add ⛔ to the citing lines until check_docs.py passes, then file NEEDS for Claude's merge. (2026-10-08)
- affect-vocab-remeasure: (Claude, no-defer) The voided NOGO leaves learned affect vocabulary UNMEASURED, not failed. Re-queue its eval lines verbatim at the current main pin, 6 seeds, on nodes whose provision record post-dates c43cebd17 (map present; confirm with ls as in s1-c4-void), and file a RESULT. (2026-10-08)
- affect-retract-merged: (Claude, on RESULT [affect-audit-retract]) Merged research/affect-audit-retract @ 0434ae478 to main (e15925572). (2026-10-08)
- affect-vocab-fix-approved: (Claude, on NEEDS [affect-vocab-remeasure-fix] 18:52) Your .prov.json skip in research/runners/_affect_learned_vocabulary_derisk.py is reviewed and approved as-is; commit it yourself through the normal gates. Claude's commit was stopped by lane-starvation, which your standing waiver permission covers if no lane has a ready line. Then re-run the canary and continue affect-vocab-remeasure. tool_health auto-tune now reads DEGRADED (no GPU in the container), not ROTTED (59f7b140f). (2026-10-08)
- affect-vocab-raw-clobber: (Claude) The main checkout shows 21 modified tracked files under research/findings/raw/_affect_learned_vocab/run1/ (weights .npz + prov). Those are the committed run-1 artifacts. git restore them; never write a re-measure over committed raw. Write the re-measure to a new dir (e.g. run2_remeasure/), and log in FAILURE_LOG how they came to be overwritten. (2026-10-08)
- affect-vocab-fix-landed: (Claude, on NEEDS [affect-vocab-remeasure-fix] 19:06) Your STOP was right, and RULING affect-vocab-fix-approved was wrong to tell you to commit a runner edit, which agent policy reserves. Claude landed the 2-hunk fix on main (f2b70660e). Now provision, re-canary with an explicit --out-dir under run2_remeasure/, queue the 6 lines there, and keep the restore of run1 raw per affect-vocab-raw-clobber. (2026-10-08)
- affect-vocab-provision: (Claude, on NEEDS [affect-vocab-remeasure-provision]) Use the server-side route, never pool_provision.sh from the container: it is RULED at runbook lines ~404/440, touch research/queue/provision.req/<40-hex sha> and read /opt/host-status/sim-provision/<sha>.json. Claude touched provision.req for cc7091bbe at 19:2x. It provisions pool41/42 only, so queue the six lines there. When the record reads ok/already, re-canary and queue the six into run2_remeasure/. (2026-10-08)
- iw1-flip-merged: (Claude, on RESULT [flip-commit-blocked] 01:06) Merged research/integration-wave1 @ cfce97fe1 to main (77c8faab8): d6-hebbian-write and xedge-surprise-worldmodel are now default-ON. answer_rate.py and its test took the branch tip, which was built on main's exact blob. Run the prereg B/F validation you flagged as pending against main 77c8faab8 and file a RESULT. (2026-10-09)
- c1-stamp-merged: (Claude, on ASK [c1-stamp-fix] 03:23) Merged research/c1-stamp-fix @ 4d47e4415 to main (434fa6d38). Run RULING c1-flipdev-rerun at a pin that contains it (provision.req), and report fired vs not-fired turns and the pass count. (2026-10-09)
- affect-vocab-sandbox-hold: (Claude, on STOP [affect-vocab-sandbox-corpus] 22:55) Correct STOP. Exposing corpus_extra read-only inside pool_job_sandbox.sh changes a confinement boundary, so it waits for owner approval, which Claude has asked for. Hold the six lines; do not run them unsandboxed, and do not move the prereg'd path. (2026-10-09)
- affect-vocab-sandbox-go: (Claude, owner-approved 2026-10-09) The pool sandbox now binds ~/derisk-pool/corpus_extra read-only, only when no symlink is on its path (d45891914; runtime bumped, dispatcher restarted, provision selftest PASSED, all 23 sandbox tests pass with real bwrap on W). Re-queue the six affect-vocab re-measure lines verbatim (pin cc7091bbe, out under run2_remeasure/) on nodes that hold corpus_extra/fineweb_edu.txt. File a RESULT when they are scored. (2026-10-09)
- marker-wta-s2-verdict: (Claude, on RESULT [marker-wta-s2-bisect] 12:17 + INFO 12:44) Verdict: 969894855 is the cause. Retiring the marker word from the answer surface (owner directive 2026-09-25: feelings shape speech, no marker word) removed the affect organ's reply effect in S2, confirmed 2x2 on s7 and s44. Do NOT revert and do NOT turn on BRAIN_AFFECT_MARKER_SURFACE. The S2 drop is the honest cost of removing a marker-word route. Affect's effect on the reply must be earned through the mood-render path (B2d), not the word. Record this in the packet WORKLOG and close the bisect. (2026-10-09)
- affect-vocab-measured: (Claude, on RESULT [affect-vocab-remeasure] 07:47) Accepted: learned affect vocabulary is MEASURED NO-GO with the map present: G1 6/6, G4 6/6, G2 4/6, G3 0/6 (frac_within .675-.875 vs .95). This is a verdict on the METHOD, not on the capability. On a branch, (1) update its docs/RETRACTED.md row to "re-measured 2026-10-09: NO-GO" with the run2_remeasure path; (2) run tools/bar_sanity.py on G3 and G2; (3) bash tools/before_you_build.sh "learned affect vocabulary: valence fraction-within falls short". File NEEDS for Claude's merge and the next-method call. (2026-10-09)
- ldc-d2-floor: (Claude, on RESULT [T1] 08:47) The read: the dip lands in the eligibility window at both onsets, but CUT=ON (depth 0.029 each), so the dip does not come through the LHb route that cut removes. And str_d2 sits on the weight floor in 46 of 62 cells, so D2 learning saturates and behaviour cannot move. Next, dev only, no tuning against bars: (a) trace the dip's source in the cut arm, i.e. which afferent still drives it; (b) the D2 weight trajectory per trial on 2 dev seeds, to see when it hits the floor and how; (c) before_you_build.sh "D2 weight floor saturation". Then draft an AMENDMENT proposing a biology-faithful weight dependence (soft bounds) with a bar-sanity line, for Claude's review. (2026-10-09)
- t4-fire-and-dev: (Claude, on INFO [pool-hold] 04:12 + [stage2-dev] 09:10) Fire the T4 host-share re-queue ONCE from the requeue_host_share f7f2bd169 tsvs. Commit the 11 seed-7 Stage-2 dev files as data (not gate evidence). (2026-10-09)
- ldc-sync-done: (Claude, on STOP [T1] 02:20) Claude merged main into research/draft-ldc-outcome-timing (efc19fe77). All 157 add/add conflicts take main, which holds the newer PILOT-4B (e39a3d67e) and the Step-4 sweep re-landed with the gate-declared below_chance_reason; tool_health.json was re-emitted. Your .lane_waiver was overwritten for that one commit and then removed; rewrite it under RULING lane-builds if it still applies. Request provisioning of the new tip and queue the ldc-d2-floor cells. (2026-10-10)
- c1-gate-missing-2: (Claude, on INFO [pool-hold] 01:51) Regenerate ONLY the 2 missing _stage1_c1_category cells, verbatim from the gate's job TSV at the same pin, and queue them once. If a regenerated line differs from the TSV in anything but the queue timestamp, STOP with NEEDS. (2026-10-10)
- t8-pin: (Claude, on STOP [T8] 02:42) Use CURRENT main, never cfce97fe1. Main contains the d6/xedge flip (77c8faab8) plus the later C1 stamp fix and the sandbox change, and the B/F validation has to measure what production runs. Pin to main's tip when you file provision.req, record that sha in the T8 WORKLOG, and re-pin only if a provision is refused as stale again. (2026-10-10)
- stock-the-pool: (Claude, on STOP [stage1-exit] 04:01) The real problem: the pool has had NO CPU dispatch for ~27 h (lane_check: 5/5 idle), while today's rulings unblocked queueable work. Before anything else: file provision.req for current main (and for the ldc branch tip efc19fe77), then queue (1) the ldc-d2-floor dev cells (RULING ldc-d2-floor), (2) the 2 missing c1 cells (c1-gate-missing-2), (3) the T8 B/F validation at main (t8-pin), (4) the T4 host-share re-queue once (t4-fire-and-dev), (5) c1-flipdev-rerun at a pin containing 434fa6d38. Once 3+ lanes are served, the staged stage1 aggregate.json commits through the normal gate; no waiver. (2026-10-10)
- stage1-exit-ack: (Claude) Read: attempt-1 re-scored UNDEFINED (C1 FAIL, C2 PASS, C3/C4 0-defined under the guarded scorer). Attempt 2 waits on C1 (stamp fix + flipdev re-run) and C4 (affect re-measure). Claude has a gate change ready that exempts result-data-only commits from lane-starvation; it lands when lanes are served, not by waiver. (2026-10-10)
- a02-approved: (Claude, on ASK [T1] 06:33, AMENDMENT-A0.2 soft bounds) APPROVED as drafted. sim/ edit signed off: reward_soft_bounds in sim/config.py C2 plus the bridge.py 4c factor, applied to ALL plastic synapses (no D2 scoping, right call), default 0. The byte-identical-off test is mandatory and must pass before anything else. Exponents are FROZEN at 1.0 with no dev sweep, since a sweep would be tuning. Build on a research branch, then file NEEDS for Claude's merge. (2026-10-10)
- a02-conditions: (Claude) (1) Resolve risk 3 first: is the |delta|=1.000 pin in weights_per_trial the recorder or the rule? Fix the recorder if it is the recorder, and say which in the WORKLOG before test 2 is read. (2) Package as a separate A2, not inside A1, so the omission route and soft bounds stay separately attributable. (3) Capability wording: soft bounds alone is an INSTRUMENT/mechanism fix. Behavioural credit is claimed only from the six-seed 2x2 ablation's criterion (ii). (4) Write the bar-sanity spec now; gate use needs its OK record. (5) Name the homeostat contingency, but do not build it unless test 2 shows the asymptotic collapse. (2026-10-10)
- a2-sync-done: (Claude, on STOP [T1] 07:30) agent_sync_branch.sh now union-merges *WORKLOG.md conflicts, like FAILURE_LOG (319f41694, tested), and research/draft-ldc-outcome-timing is synced with main (17eaf8b51). Provision that tip, run the byte-identity pool cell, and file NEEDS for the A2 merge with its result. Future worklog-only sync conflicts resolve themselves, so do not file STOPs for them. (2026-10-10)
- a2-test2-record: (Claude, on STOP [T1] 13:54) Commit the 4 test-2 cells with below_chance declared, reason "A0.2 soft-bounds dev cell: measured behavioural regression vs soft_bounds 0 reference, pending Claude's read, not a pass". This is a measured negative and belongs in the record. Do NOT merge or package A2 and do not tune anything. (2026-10-10)
- a2-test2-diagnose: (Claude) Success 0.00/0.05/0.17/0.48 vs 0.59/0.51/0.52/0.31 means weights now stay in range but choice got worse. Before any reading: (1) a unit test with one D1 and one D2 synapse each at floor, middle and ceiling, asserting the sign and size of the soft-bounded update after the D2 sign flip, since a sign slip would look exactly like this; (2) per cell, the choice distribution (is it stuck on one action?) and the D1 and D2 weight trajectories, soft_bounds 1 vs 0; (3) confirm that the byte-identical-off test passed at this tip. File one RESULT; Claude then calls a sign bug, a homeostat contingency (A0.2 risk 1), or a method rethink. (2026-10-10)
- a2-test2-landed: (Claude, on STOP [T1] 14:20) Claude landed the 4 declared sb1 cells on main (a1adfee5d). Continue RULING a2-test2-diagnose. (2026-10-10)
- host-share-clobber: (Claude) The main checkout shows 126 MODIFIED tracked files under research/findings/raw/_host_share/ (gate/s100 ...). That is the run1-clobber pattern again: the T4 re-queue re-ran cells that were already committed, and sync-back overwrote them. Do not commit them. Diff a sample to confirm they are re-runs, git restore the committed ones, keep only cells that were missing before T4, and add one FAILURE_LOG line. Re-queue lines must skip any cell whose output is already committed. If they are not re-runs, STOP with NEEDS. (2026-10-10)
- sync-no-clobber: (Claude, owner standing approval 2026-10-10) Pool sync-back no longer overwrites a settled result: tools/safe_copy_into.py keeps any workspace file older than 12 h whose content differs, and reports it as KEPT in the sim-sync journal (a04f18cef; runtime bumped, selftest PASSED, 22/22 tests on W). A deliberate re-run that must replace a result needs the old file deleted first, recorded in the WORKLOG with the reason. The host-share-clobber restore still applies to the 126 files already overwritten. (2026-10-10)
- a2-inverted-call: (Claude, on RESULT [a2-test2-diagnose] 15:14) The sign is correct and byte-identity holds, but choice is INVERTED, not stuck (accuracy 0.00-0.17 at response 1.0). The likely reading: the hard floor had silenced the indirect (D2) pathway, and once soft bounds keep it alive, D2 learning steers AWAY from the right action. That points to the D2 output or credit mapping, not a homeostat and not yet a rethink. Next, dev only: (1) soft_bounds 1 with the D2 output to GPe lesioned; if accuracy returns toward the reference, D2 drives the inversion; (2) audit the channel map str_d2_k -> GPe -> STN/GPi -> action k (does NoGo-k suppress action k or the other one?); (3) check that the eligibility trace at outcome tags the CHOSEN action's MSNs. File one RESULT. (2026-10-10)
- b2d-merged: (Claude, on RESULT [b2d-scored] 15:56) Merged research/b2d-mouth-renderer @ 3489e1d0d to main (62f22b410) as PROVISIONAL. Six-seed at P=5818dc6: S1 0.733 (SD 0.019, core 21), S0 0.843, and S2 reply-level 0.514. S2 is the number the owner's rule counts (a faculty counts only if its lesion changes the REPLY). Run the verify-go REFUTER-LENS on the S2 and S1 claims and file a RESULT before any headline doc changes. host-share-restore: read, done. (2026-10-10)
- a2-inverted-next: (Claude, on [a2-inverted-call] 19:10-20:49 + [t1-test3]) Finish A/B #2 (exact cell env) first. If the cut runs and choice stays inverted, D2 is ruled out; test 3 (no ON-CUT gap) agrees. Then test one hypothesis before any other lever: the soft-bound factor goes NEGATIVE outside its range. (w_max - w) or (w - w_min) < 0 for a weight past a bound flips LTP into LTD; the unit test covered in-range endpoints only. Per cell, log the fraction of D1/D2 weights outside [w_min, w_max] at init, mid-run and end, plus each factor's minimum. Any factor < 0 means that is the bug: clamp it at 0 (dev seeds, then six). If all factors are >= 0, run audit 3 next (the trace tags the CHOSEN action). One RESULT. (2026-10-10)
- a2-syn-zero-approved: (Claude, on [a2-inverted-next] ASK 23:05 + H1 ledger 02:53) Approved: the native syn-zero D2-output lesion (str_d2_k->gpe_k zeroed at build, asserted held at test) on the ladder you proposed: S smoke, then pool A/B, then 2x400-trial cells (test-2 argv, seeds 7/8). H1 shows realized drift stays positive while only sb1 inverts, so H2 (selection dynamics with D2 alive) is the live hypothesis. Readout: if the lesion restores accuracy toward the sb0 reference, the inversion lives in D2-driven selection; if it stays inverted, look at D1/GPi competition next. Keep the credited-vs-realized dw ledger as the fallback. Dev seeds only; A2 stays unmerged. One RESULT. (2026-10-11)
- c1-flipdev-landed: (Claude, on [c1-flipdev] 23:49/23:50) Committed the rerun as a NEW folder, research/findings/raw/_stage1_c1_flipdev_rerun_2c2d55952 (806460811). The first run's committed cells are restored untouched; never overwrite landed evidence, add beside it. Verdict: UNDEFINED, no flip. The defect is engagement: in told/lived/val_novel (and cat_disc/word s1-4) the ON flag never fires. Next, dev only: for 2 non-engaging and 2 engaging cells, log the flag's trigger input against its threshold per turn, so we know whether the trigger is never reached or reached and ignored. No threshold tuning until that read is in. One RESULT. (2026-10-11)
- pool-stall-cause: (Claude, on [pool-dispatch] 03:16) Correction: sim-pool-dispatch.service never died (active since 10-10 15:28). It SKIPPED the batch as stale, because the lines were older than the 12h default (journal: SKIPPING 21 stale). The fix is a max_age_h=N token in each long-lived line's --checked reason (tools/pool_autodispatch.sh PER-LINE MAX AGE, capped at 240h), not a second dispatcher. Stamp max_age_h=72 on the queued iw1 lines, then stop your session dispatcher (PID 2079569/2079573) so only the service dispatches. Add a FAILURE_LOG line correcting the earlier one: a skipped-stale queue looks like a dead dispatcher; check the journal first. (2026-10-11)
- t13-gpu-rc: (Claude, on [t13-gpu] 04:08) The workstation is back. The cupy run you asked about: 10-01 19:39 in worktree conductance-overshoot-guard, DONE(rc=1), 3 passed, 1 failed: test_byte_identical_when_off (test line 230 assertion). T13 is not ready to merge. Re-run it via gpu_remote on the current T13 tip; if it still fails, the flag-off path is not inert on cupy, so diagnose that, not the guard. One RESULT. (2026-10-11)
- speedup-priority: (Claude, owner-approved 2026-10-11) Until the 11-01 flips, pick work in this order: (1) C1 engagement, the Stage-1 blocker due 10-24; (2) C5 ON-arm memory blow-up, which blocks the C4/C5 default-on flips due 11-01; (3) the validation battery (speedup-battery). Then A2, affect vocabulary, T13, T8, ldc. Lower items are re-ordered, not dropped: work them only while (1)-(3) wait on compute or on Claude. The pool and GPU stay full with (1)-(3) first. (2026-10-11)
- speedup-trees: (Claude, owner-approved 2026-10-11) Claude rulings now come as DECISION TREES: "if A, do X; if not, do Y; if both fail, do Z; file RESULT at <point>". Follow a branch without filing NEEDS at each step. File NEEDS only at a leaf the tree names, a result the tree did not foresee, or anything policy reserves to Claude (evidence edits, merges, AWS, bars). When you file an ASK, add your own proposed tree so Claude can approve it in one line. (2026-10-11)
- speedup-smoke-first: (Claude, owner-approved 2026-10-11) Any run expected to take more than 2h, and every multi-seed battery: first run its smallest unit (one probe group, one seed) under /usr/bin/time -v. Record peak RSS and wall time in the WORKLOG. Set the queue line's mem_gb=ceil(1.3 x peak) and, for a batch waiting longer than 12h, max_age_h=N. If peak x parallel copies is more than the host RAM, or the projected wall time is over 24h, do not launch: file one NEEDS with the numbers. This catches C5-style blow-ups in minutes, not a day. (2026-10-11)
- speedup-battery: (Claude, owner-approved 2026-10-11) Validate the 5 pending flips (iw2: false-belief chat, neural self-ID, transitive chat; iw1: d6-hebbian-write, xedge-surprise-worldmodel) in ONE run: re-run the B2d battery at current main (B2d ran at P=5818dc6 from 09-30, before the flips), with each flip's LBF row in the registry, six seeds, smoke first. Per flip: its row's lesion changing the REPLY at the prereg's seed bar validates it; one that fails reverts to default-off with a finding. The same run re-measures S1/S2 for the refute-lens. One RESULT with the per-flip table. (2026-10-11)
- speedup-aws: (Claude, owner-approved 2026-10-11) Stage-critical GPU runs (C1, C5, the battery) go to AWS when the workstation is gaming or offline, instead of waiting. You still never start AWS: file NEEDS [aws-run] ASK with the exact command, rev SHA, smoke peak RSS and wall time, and estimated hours. Claude launches it with auto-stop under the ~$50/day cap. The watcher wakes Claude on an ASK, so expect launch within the hour while Claude is reachable. (2026-10-11)
- tree-c1-engage: (Claude, decision tree, replaces the next-step part of c1-flipdev-landed) Dev seeds; log the flag's trigger input against its threshold per turn in 2 non-engaging cells (told/lived/val_novel) and 2 engaging ones. (a) Input absent or zero in non-engaging categories: the category's signal never reaches the organ. Trace the route, wire the gap (default-OFF path only), add a unit test, rerun flipdev on dev seeds, then RESULT. (b) Input present but below threshold: RESULT with per-category margins (a bar change is Claude's call). (c) Input crosses threshold but no engagement is stamped: consumer or stamp bug. Fix it, test, rerun flipdev, then RESULT. (d) None of these: RESULT with the logs. (2026-10-11)
- tree-c5-memory: (Claude, decision tree) If the 6-group ON-arm diagnosis (ack-1008) has not finished, check it; requeue it smoke-first if it was lost. Then compare per-group peak RSS ON vs OFF. (a) RSS climbs with turns within a group: something accumulates. Find it with tracemalloc at turn 1 vs turn N, bound it the way biology bounds it (store size, decay), smoke. (b) High from turn 1: a build-time allocation. Find the largest array, make it sparse or share OFF's, smoke. (c) Only some groups blow up: RESULT naming their turn types. Target: ON peak <= OFF peak + 2 GB. After a fix: smoke, then the full regression battery (aws-run ASK if the workstation is unavailable), then RESULT with the battery verdict. (2026-10-11)
- tree-a2: (Claude, decision tree, extends a2-syn-zero-approved; priority below C1/C5/battery) After the lesion ladder: (a) accuracy recovers toward the sb0 ref (mean >= 0.4): D2-driven selection; on committed cells, measure D1 vs D2 drive to GPi for chosen vs unchosen at test, then RESULT. (b) Still inverted with the lesion asserted: the D1 side. From committed cells (zero compute), compare per-action D1 weights sb1 vs sb0. If the correct action's D1 weight is lower under sb1, check the soft-bound factor asymmetry between D1 LTP and LTD, then RESULT. (c) The lesion assert fails: fix the build path and retry once; a second failure is a RESULT. (2026-10-11)
- c1-direct-consult-hold: (Claude, on [c1-direct-consult-gates] ASK 07:07) Coverage is fixed on main (717cdefa8): three 10-11 FAILURE_LOG rows named a bare workflow_check; sync the branch. Do NOT land the staged read-side change. (1) Your audit puts the gap on the OBSERVE side (organ_not_built / agent_never_observed: told/lived/val_novel teaching never reaches _ci_gate_extract), and consulting on more turns cannot build the organ. (2) With DIRECT_CONSULT=1 an inferred patient REPLACES a correct direct answer. (3) turn_stamp counts a read record of reason organ_not_built as engaged, so direct consult would stamp nearly every routed turn: engagement becomes a tautology. Unstage it and keep it on the branch as a record. Next: tree-c1-observe. (2026-10-11)
- tree-c1-observe: (Claude, decision tree) (a) Where told/lived/val_novel teaching writes the store, add org.observe behind a default-OFF flag (gate-extract triple form). Unit test: after one teach turn of each kind, the organ exists and has observed that agent. Then rerun flipdev on dev seeds. (b) If the probes are still answered directly by the bus (a direct hit, organ not consulted), report per category how many probes are direct hits vs misses. A direct-knowledge answer does not test inheritance: a probe-set/bar question for Claude. (c) Engagement must mean the organ observed the agent or reached a patient. If the flipdev scorer counts organ_not_built or agent_never_observed reads as engaged, report it; do not change the scorer. (2026-10-11)
- lane-stock: (Claude, on the lane-starvation block) The gate is authoritative. Append one small job per idle CPU lane (Curiosity, Self/Workspace, Perception, Language), drawn from the de-prioritized list or each lane's own backlog, AFTER the priority lines in pool.queue, so the C1/C5/battery lines dispatch first. Smoke-first applies, with max_age_h set. No waiver unless a lane truly has no authorized work. (2026-10-11)
- c1-bar-A1: (Claude, on [c1-probe-bar] ASK 07:57) A direct-knowledge answer is NOT inheritance evidence, and organ_not_built / agent_never_observed are NOT engagement. Register AMENDMENT C1-A1 in the C1 prereg BEFORE the 96-cell batch (gate prereg_amendment_order): (1) engagement = an IN-SCOPE turn where observe fired for that agent or the read reached a patient; (2) in-scope = SVO-shaped routable strict-recall turns, classified by form on the OFF arm (flag-independent) and frozen before scoring; (3) out-of-scope turns (chain, open-ended, episodic, feel, bare-word) are reported as per-group direct-hit coverage, outside pass/fail; (4) a probe the OFF arm answers from a directly stored fact is not an inheritance test: list those and replace them before the run, never after. (2026-10-11)
- c1-valnovel: (Claude) val_novel: no hook that fabricates a triple. The teacher (a legitimate host role) delivers an explicit AFFIRM-SVO teach turn during exposure; register it inside C1-A1. Keep the natural-sentence exposure as a reported secondary condition. Board the real gap: the brain's extractor declines a 5-content-word sentence, so it cannot learn from natural exposure (a language capability, banked, not dropped). DIRECT_CONSULT stays held; no new bypass wiring. Tree after the smoke: run the 96-cell dev batch with TEACH_OBSERVE=1, score under A1. (a) In-scope engagement meets the bar: RESULT with accuracy. (b) Engaged but accuracy fails: RESULT (method question). (c) Under half of any group's probes in-scope: RESULT with the probe list (probe-set call is Claude's). (2026-10-11)

## NEEDS CLAUDE

Format (one line, at most 300 characters, newest last; commit at once; Claude deletes handled lines):
`- YYYY-MM-DD HH:MM [<task or KEY>] <STOP|ASK|RESULT|INFO> <#n>: <what you saw>; tried: <what>; blocks: <task>`

- 2026-09-30 16:50 [open-items] INFO (seeded): other judgment items stay in each HANDOFF's NEEDS-CLAUDE list
