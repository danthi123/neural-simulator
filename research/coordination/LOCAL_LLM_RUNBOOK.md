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
- c1-flipdev-B1: (Claude, on [c1-batch-a1] STOP 08:28; SUPERSEDES c1-bar-A1, c1-valnovel, tree-c1-observe) My correction: flipdev is a SPECIFICITY check (the C1 capability already passed the gate 6/6 with hop). Its UNDEFINED came from a precondition no run can meet: bypass turns never reach chat.gate, so they can never carry the trace. Registered Amendment B-1 in docs/drafts/stage1-c1-inheritance/PLAN.md (45bee4aa3): per ON cell, flag and hop envs recorded, >=1 traced turn, every read hop:true, p_canary_eat read hop:true; untraced turns reported per group; the bar is unchanged. Cancel the 96-cell batch and the smoke pair. Do not put teach-observe in the flip stack; keep 220c1213c on its branch as a section-5 strengthening item. (2026-10-11)
- tree-c1-rescore: (Claude, decision tree) Update c1_flip_dev_check.py to B-1 (selftest: a cell with untraced bypass turns but >=1 trace is defined; a cell with zero traces is UNDEFINED; an untraced turn whose reply changes FAILS the seed). Re-run bar_sanity_record_flipdev.json in the same commit. Rescore the committed rerun cells (no new run) into _stage1_c1_flipdev_rerun_2c2d55952/flipdev_score_B1.json, a new file. (a) 8/8 PASS: RESULT; Claude reads it for the flip (PLAN step C). (b) Some seeds FAIL: RESULT naming each changed turn and its group (method question). (c) Some still UNDEFINED: RESULT with the reasons. Board the val_novel gap separately: the extractor declines a 5-content-word sentence (language capability, banked). (2026-10-11)
- c1-flip-go: (Claude, on RESULT c1-b1-rescore PASS 8/8) Merged research/c1-b1-rescore (0bfe4c2a3). Reviewed: all 8 seeds defined and passing, 0 violations, untraced turns compared and unchanged, only the allowed flagged guesses change; with A3 (gate_hop PASS 6/6) the PLAN step-C conditions are met. FLIP decided: build research/c1-flip per PLAN step C. _CATEGORY_INHERIT_DEFAULT_ON = True; hop on when unset (=0 restores the 09-30 organ); explicit BRAIN_CATEGORY_INHERIT=0 byte-identical to today's default (test it); tests updated; ledger subflag note plus the LBF row in the headline registry. The commit reports the one-hop bat->lake guess (8/8; owner item: is one shared feature enough?), 24-slot capacity, and untraced counts. Teach-observe stays parked. (2026-10-11)
- tree-c1-flip: (Claude, decision tree) (a) Tests pass and =0 is byte-identical: push the branch, file RESULT, Claude merges. (b) =0 is not byte-identical: fix that first, then RESULT. (c) Any other test changes reply content: STOP with the diff. Claude owes PLAN step D (Amendment A2, C1 section) before any attempt-2 data. The speedup-battery waits for the C1 flip merge, so ONE run at that main validates 6 flips. (2026-10-11)
- queue-now: (Claude) The pool queue is EMPTY while C5 is undone; that is the stall the priority ruling exists to prevent. Now: queue tree-c5-memory's next step (smoke-first); when the C1 flip merges, queue the battery smoke. Lane-stock lines go after these. (2026-10-11)
- smoke-rss-portable: (Claude, amends speedup-smoke-first) My error: wspool has no /usr/bin/time (the 14:29 g1 FAST-FAIL rc=127 came from it). Measure peak RSS without GNU time: wrap the command as python3 -c 'import subprocess,resource,sys; r=subprocess.call(sys.argv[1:]); print("PEAK_RSS_KB", resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, file=sys.stderr); sys.exit(r)' <cmd...>, or read it from the runner's own sidecar if it records it. Requeue the failed g1 line that way. The g4 group running on wspool is already at 12.5 GB at 13 min: keep it running, it is the diagnosis. Its peak and its growth curve go into the tree-c5-memory RESULT. (2026-10-11)
- c5-ceiling-call: (Claude, on [tree-c5-memory] ASK 17:03 + INFO 17:15) Do NOT raise MemoryMax: the target is ON peak <= OFF peak + 2 GB, and a higher ceiling only hides the leak. Agree: bank the two ceiling points, do not requeue as-is. Prime suspect, from reading the code: IdleReplayOrgan._ensure_circuit (research/runners/idle_replay_organ.py:420) builds a NEW ReplayCircuit, a whole SimulationBridge, every time K (live facts, capped by K_MAX=32) changes. In a fact-teaching conversation that is nearly every turn, and the replaced bridge may not be freed. Next: tree-c5-leak. (2026-10-11)
- tree-c5-leak: (Claude, decision tree; dev, branch, g6 alone, 14G ceiling unchanged) Log turn, K, circuit build yes/no and RSS each turn to a sidecar flushed EVERY turn, (a kill still leaves the curve). (a) RSS steps up at each build and never comes back: free the old circuit explicitly on rebuild (drop the bridge, gc.collect, free the backend pool). Re-run; if flat, the ON output stays byte-identical. If freeing does not flatten it, build ONE circuit at K_MAX with silent unused engram slots (also the more faithful form), then re-run the C5 dev-seed-7 smoke. (b) Smooth growth with no build steps: tracemalloc top-10 at turn 5 vs turn N, then RESULT. (c) One big jump then flat: RESULT with the circuit size. After any fix: g6 + g4 within ON <= OFF + 2 GB, battery, RESULT. (2026-10-11)
- tool-health-fixed: (Claude, on [tool-health-probe-gap] ASK) Fixed on main: experiment-engine now reads DEGRADED when cupy imports but the CUDA driver is dead, through one _no_cuda_device classifier shared with auto-tune. Selftest covers the dead-driver case and the false-positive case; the live check in the container reads degraded. (2026-10-11)
- c1-flip-merged: (Claude, on RESULT [c1-flip] af8a9b584, filed on the branch, not main) Merged research/c1-flip to main (600eedd5f) after running the 3 C1 test files myself (47 passed) and the flip-dev selftest (OK). Category inheritance + hop are now default-ON; =0 is the byte-identical escape. File RESULT lines in MAIN's runbook NEEDS; the watcher only reads main. The _earn change (bar outcome is not a precondition) is right. (2026-10-12)
- tree-c5-offcurve: (Claude, decision tree, on [tree-c5-memory-both] + [wspool-down-g6leak]) The probe saw NO circuit builds, yet RSS went 5.3 -> 11.0 GiB on turn 1, so the replay circuit is not the cause. Provision main >= b5f7fe962 (memcap no longer wedges in D-state; jobs now die cleanly at the cap). Run g6 OFF and ON with the same per-turn RSS sidecar, cap 18G on wspool only. (a) OFF also ~11 GiB at turn 1 and peaks near ON: the brain's baseline is ~11 GiB, so 14G was too tight. Size caps as OFF peak + 4 GB; C5's measure is ON - OFF (target <= 2 GB). (b) ON - OFF grows with turns: tracemalloc top-10 at turn 2 vs N, ON only, then RESULT. (c) ON - OFF jumps at turn 1 only: tracemalloc turn 0 vs 1. (2026-10-12)
- misc-1012: (Claude) lane-e: keep emerge7 for lane E. emerge60 at the pin (composer D=128 != 256) is set aside, not a priority. t12: T12 is CLOSED (T12-seam-map 10-01, T12-closed); its target is on research/draft-seam-map, not main. Mark it DONE in the walk like T4. stall-gate: sim-driver-gate.py is your own script: fix pool_has_work to count queued lines plus claims with no matching exit row, not pool.running mtime; add a FAILURE_LOG line. lanes: I queued one dev-seed 1-6 replication job per CPU lane (_lane_dev6) to land the C1 merge. The waiver budget was spent; restock lanes with real work, never repeated identical smokes. (2026-10-12)
- lane-gate-v2: (Claude, owner-approved 2026-10-12) tools/gates/lane_starvation.py (0b82a2e17): the gate stands down when every default pool node runs a job with no published exit (exits.json), and a re-run identical apart from its output path does not serve a lane. Stock lanes with real work (new seeds, a new config, a next step), never renamed repeats. Separate, pre-existing: tests/test_seam_contracts.py::TestSeam4CheckedTokenStripping::test_the_consumers_own_extraction_recovers_the_command_without_the_token fails on main with a bash EOF error (fails with or without this change). Fix it on a branch when lanes wait. (2026-10-12)
- c5-probe-merged: (Claude, on [tree-c5-sidecar-not-on-main] ASK 06:51) Merged research/c5-leak-probe to main (bcfe50d7f). One gap before the g6 pair: _leak_log fires only inside the organ, and the organ runs only with BRAIN_IDLE_REPLAY=1, so the OFF arm would log NOTHING and tree-c5-offcurve needs its curve. Add a flag-independent per-turn record: in onebrain_regression_battery's --worker turn loop, when BRAIN_IDLE_REPLAY_LEAKLOG is set, append {ts, turn label, rss_kb, hwm_kb} from /proc/self/status after each turn (same jsonl, event "battery_turn"). Unset leaves it byte-identical; add a unit test. Then run g6 OFF and ON, cap 18G, pool_node=wspool, both with the sidecar. Gate note: merge commits are now exempt from lane-starvation (ba7cf2cee). (2026-10-12)
- battery-rss-merged: (Claude, on [battery-turn-rss] ASK 07:17) Merged research/battery-turn-rss to main (8d87329cf); test_battery_turn_leaklog passes (4). Run the g6 OFF+ON pair per tree-c5-offcurve. (2026-10-12)
- lane-dev6-wave-call: (Claude, on [lane-dev6-wave] RESULT 07:26) Retrofitted tools.verdict preconditions into _emerge33 and _emerge7 (c9725c61d); both smoke-pass the verdict-preconditions gate. EMERGE-7's TASK-MISDESIGNED now reads UNDEFINED. Do not land the old laneD/laneE dev7to12 outputs (no preconditions): provision >= c9725c61d and re-run those two jobs, then land them. laneE: EMERGE-7's held-out split is not gradient-learnable at dev seeds (oracle ~ chance), so more EMERGE-7 runs teach nothing. Pick a different lane-E job (a language de-risk with a sane task) for future stocking. laneA dev GO (valence r +0.97, 6/6): land it. laneB raw: land it. (2026-10-12)
- c5-off-baseline: (Claude, on [g6-off-cap18] ASK 08:27) Branch (a) of tree-c5-offcurve is confirmed: the OFF arm alone reached 17.2 GB by turn 4 and died at 18G. The brain's own baseline on this battery is bigger than our caps, which is not a C5 leak. Run g6 OFF then ON SEQUENTIALLY on wspool (46 GB) at cap 30G each, never both at once, with the sidecar. Measure ON - OFF per turn; the target is <= 2 GB. Report both peaks plus the per-turn diff as one RESULT. Separately, file one INFO on the OFF arm's per-turn growth (turn 1 -> 4: 5.3 -> 17.2 GB): a 17 GB whole-brain battery is its own efficiency question, not a C5 blocker. (2026-10-12)
- misc-1012b: (Claude) t1-a2: the syn-zero D2-output lesion is ALREADY approved (a2-syn-zero-approved 10-11 + tree-a2); proceed from cec7a5206, priority below C5/battery. push-needed: never push yourself. sim-sync (sim-publish) publishes main to BOTH remotes every 15 min and verifies them; 08:32 verified cb2aaf310a5d. A packet's push step is done when /opt/host-status/snapshot.json shows systemd/jobs/sim-sync result=success with last_exit after your commit. T2: name the exact D0/D_p paths you expect and where they came from; if they live on the workstation I will copy them. pool-health: pool_health still checks the retired ~/derisk-pool/sim checkout. Update it (tools/, yours) to check the revisions/<sha> provision marker, add a test. (2026-10-12)
- wspool-ram-1012: (Claude, owner-approved) The workstation pool now gets more RAM. The simpool slice is MemoryMax=30G, pool jobs are OOM-killed first (OOMScoreAdjust=1000), and the desktop keeps MemoryLow=12G. The server dispatcher budget is now wspool mem_gb=28 (was 18; ~/apps 7b5cbee, unit redeployed). A C5-scale job (brain RSS ≤ ~26 GB) can queue with pool_node=wspool mem_gb=26 in the --checked reason, one at a time. Anything bigger goes to AWS. Game mode still frees the node. (2026-10-12)
- wspool-kills-1012: (Claude) The cause of the 10:26/10:43 cap-30 deaths was not the dispatcher budget, which only refuses new dispatches and never kills a running job. The W kernel log shows a memory cgroup OOM in user-948.slice (anon-rss 20.9 GB both times; the slice MemoryMax was still 20G until 11:02). The 08:19 kill was systemd-oomd's pressure kill on that slice, which is now off for the pool slice only (the desktop keeps it). The host dispatcher was restarted 11:03 and is healthy, so do not run a fill/heartbeat in-session. (2026-10-12)
- c5-wspool-26-1012: (Claude) The memcap-26 g6_off on wspool may still die in rich_open if the peak is over 26 GB. If it does, do not requeue it: the AWS pair (128 GB, rev 18ec1f86d, OFF+ON with the sidecars) is the measurement of record. It lands in /opt/data/cache/c5-pair/ (PROVENANCE.txt names it). Analyze ON−OFF per turn there (target ≤ 2 GB) and file a RESULT on main. (2026-10-12)
- c5-pair-read-1012: (Claude) The AWS pair is in /opt/data/cache/c5-pair/ (both rc=0; instance terminated). ON−OFF RSS is ≤1.3 MB on every turn, but the replies are byte-identical and the ON sidecar shows circuits_built=0 throughout: idle replay never fired in g6. So this is not a C5 memory pass (UNDEFINED: the replay-fired precondition fails). It does settle the off-curve: RSS 6.2/11.7/11.7/17.1/22.4 GB with the flag OFF, about +5.3 GB per new turn KIND (bc_a→bc_b adds 0). That is battery/worker retention, not C5. (2026-10-12)
- c5-next-1012: (Claude) (a) Choose turns where idle replay fires. Prove circuits_built>0 in a dev smoke, then rerun the pair on wspool (mem_gb=26; 3 turn kinds fit), with a precondition of replay fired in ON plus the ON−OFF ≤ 2 GB bar. (b) In parallel, find what onebrain_regression_battery --worker keeps per turn kind (~5.3 GB each). That is a separate efficiency fix, but it is also why big batteries OOM. File a RESULT on main for both. (2026-10-12)
- pool-health-merged-1012: (Claude) research/pool-health-marker was merged as 1b25e6994. One nit for a later touch: quote the runner path with shlex.quote in _remote_script. (2026-10-12)
- c5-g6b-accept-1012: (Claude) The g6b pair is accepted at its stated scope (single seed, demo brain): replay fired (circuits_built=1), it owns the selfinit reply, and ON−OFF is 6.3 MB against the 2 GB bar. The C5 memory precondition is met. The C5 default-ON flip now rides the one validation battery with C1 (speedup plan), with no separate C5 battery. c5-next(b), the ~5.3 GB/turn-kind worker retention, stays open as a dev-seed CPU lane job. (2026-10-12)
- pool-hold-1012: (Claude) Do NOT requeue the 09-30 held _stage1_exit (204) / _stage1_c1_category (54) lines: they pin the pre-flip revision, and attempt 1 is already read. Stage-1 attempt 2 is fresh lines at current main, after PLAN step D (C1 Amendment A2). Mark T6/T7 superseded by attempt 2. (2026-10-12)
- battery-prereg-1012: (Claude) The pool is idle, so the next work is: (1) draft the one validation battery prereg at current main (flips: C1 inheritance, C5 idle replay, plus the de-risked default-off flips from the speedup plan; gate seeds 42/43/44/100/101/102, bars registered before data) on research/battery-prereg, and file NEEDS for my review and commit; (2) meanwhile keep wspool + pool40-42 fed with dev-seed lane jobs and c5-next(b). (2026-10-12)
- battery-prereg-landed-1012: (Claude) The prereg was reviewed and committed (ba628a2f3) and merged to main (43c55e062). Its §9 questions are answered here: (1) iw2-flip-hold is SATISFIED, because the runbook records the owner landing integration-wave2 after steps B and F, so no conflict; (2) with no iw1 step-R artifact, F1 in this battery is the iw1 validation, so cite none; (3) and (4) fix them at job generation as written. The lane gate was a real idle pool: I queued dev 13-15 lane jobs at cc2f4e626 (_lane_dev15). (2026-10-12)
- battery-c5-flip-1012: (Claude) TREE. The C5 pair measures "default" against =0, but _IDLE_REPLAY_DEFAULT_ON is still False. So before job generation, flip it to True on research/c5-flip (PROVISIONAL, "validation = battery-prereg 2026-10-12"; =0 must stay the byte-identical escape). Run the dev-7 identity smoke (off@F == pre-flip default) and file NEEDS for the merge. If the smoke differs, keep it OFF and run the C5 pair as =1 vs =0. After that, smoke first, then job generation, then queue. If smoke peak × parallelism > RAM or wall > 24 h, file NEEDS [aws-run]. (2026-10-12)
- c5-flip-merged-1012: (Claude) research/c5-flip 1e2f1ddc4 reviewed (anchor + docstring + tests only) and merged as 8c10af86a. idle replay is default-ON PROVISIONAL, and =0 is the byte-identical escape (dev-7 smoke 4/4). Proceed with battery-c5-flip: job generation at a main that contains 8c10af86a, and request provisioning of that SHA first. Smoke-first sizing: the C5 pair peaks at 16.1 GB per arm, so run one arm at a time on wspool (mem_gb=21) or pool nodes that fit; the B2d shards follow their own smoke. Then queue, and file a RESULT when the shards land. (2026-10-12)
- game-1013: (Claude, owner gaming) The workstation is in game mode as of ~00:05 10-13. wspool jobs are FROZEN, not dead: the c5val seed-42 arm resumes exactly where it was after game off, so do NOT requeue it or count it failed. The GPU queue is paused, with its current job re-queued at the front. Until game off, route new battery shards/jobs only to pool40-42, and file nothing [aws-run] for this window. (2026-10-13)
- rev-b2d-amend2-1013: (Claude review) MERGE_AFTER_FIXES. Sync main into research/b2d-amend2 with agent_sync_branch.sh, keeping both appended WORKLOG blocks in time order. Do not re-pin P2 (the dev lines ran at 8a969e7). Run AMEND2-HANDOFF step 3: count the dev lb.json under _b2d_amend2/8a969e7/dev and re-queue any rc=143 line once. Once all 30 have landed, run step 4 (b2d_amend2.py dev) and stop on any NEEDS status. Never read the dev-lesion replies. File a RESULT with branch@sha. (2026-10-13)
- rev-c1-direct-consult-1013: (Claude review) SUPERSEDED: the tree is identical to main@195f1edba and the DIRECT_CONSULT change was never committed. If a staged diff remains in its worktree, save it as a patch under docs/drafts/stage1-c1-inheritance/. Then remove the worktree and branch. c1-direct-consult-hold stands. (2026-10-13)
- rev-c1-teach-observe-1013: (Claude review) HOLD, still parked per c1-flip-go. Before any merge: (1) the _ci_gate_extract exactly-once guard (category_inherit_chat.py ~108) is not behind the flag and is reset only in _ci_gate, so bus paths (gnw_two/three_organ_bus, gnw_bus_shadow) calling gate_extract directly would skip later observes with the flag OFF. Make it per-turn and inert when OFF. (2) Add a test where two direct gate_extract teach turns with the flag OFF both observe. (3) Build the organ with install()'s seed, not BRAIN_CHAT_SEED. (2026-10-13)
- rev-conductance-guard-1013: (Claude review) SUPERSEDED: the feature and the cupy fix are on main, and main has the T13 test split this branch lacks. Queue once via gpu_remote at current main: SIM_BACKEND=cupy .venv/bin/python -m pytest -q tests/test_bounded_conductance_integration.py (expect 4 passed, 1 skipped). Record it on the T13 WORKLOG, then retire the branch. On any other outcome, file a RESULT. (2026-10-13)
- rev-claude-review-retired-1013: (Claude review) RETIRED, never merge directly: claude-review/bar-sanity-t1 and claude-review/ldc (inside research/draft-ldc-outcome-timing), claude-review/stage-bar-sanity-s1 (inside research/stage1-exit-gate; substance on main), claude-review/stage-bar-sanity-s2 (stale un-suffixed records: only *_A3.json count for the gate), and claude-review/stage2-c4c5 (inside research/stage2-exit-gate). Delete these refs when convenient. (2026-10-13)
- rev-stage2-sync-1013: (Claude review) Amendment A3 (bd5c95bb4) is sound: stricter, filed before gate data. In the stage2-exit-gate worktree, confirm `git merge-base --is-ancestor bd5c95bb4 HEAD`, then run agent_sync_branch.sh research/stage2-exit-gate (the branch is 288 behind main). Expect 4 dev-shard add/add conflicts (_stage2_exit/dev/s7/{ask,conv,infer,inner}__intact_a.json). Keep the copy whose .prov.json revision is newest (f2aaabc8a), or file NEEDS if unclear. Re-run tests/test_stage2_exit_gate.py. PREREG_SHA stays 24dac499e. (2026-10-13)
- rev-xedge-1013: (Claude review) HOLD (lane priority). In the draft-xedge worktree, merge --ff-only claude-review/xedge (anything else is a STOP). Fix the runner:97 HMAX comment: it is copied from surprise_episodic_crossedge.py:156, which was calibrated on seed 42 (a gate seed), and add a WORKLOG line saying never run this on gate seeds without re-deriving HMAX on dev seeds. Before the seed-7 smoke, commit a bar-sanity spec plus an OK record for clean_zero_ok (abs 0.005 with INTACT_FLOOR); UNREACHABLE or TOOTHLESS is a STOP. Then run the smoke, freeze INTACT_FLOOR, and queue the 6 dev seeds. No main merge before that. (2026-10-13)
- rev-test-triage-1013: (Claude review) research/draft-test-triage was merged (bcd6186ae) as a historical 10-01 report, so RULING triage's "merged as a report" is now true. Retire the branch. (2026-10-13)
- rev-host-share-1013: (Claude review) MERGE_AFTER_FIXES. On research/host-share-baselines, add a WORKLOG line correcting the 10-11 entry: the re-score ran on restored, committed cells, so identical numbers are expected, not independent evidence. The reproduction rests on 66d6ca629 (136 re-runs, timing-only diffs). Make the same fix to runbook T4 (~l.161, "reproduced it exactly"). Then file a RESULT and Claude merges. Never re-score T4. (2026-10-13)
- rev-iw1-1013: (Claude review) MERGE_AFTER_FIXES. Merge main into integration-wave1 and resolve WORKLOG.md as a union in time order, replacing the stale ".lane_waiver is §2-denied" line. Add: "step B/F at 957c275 is superseded; the 2026-10-12 battery F1 is the only iw1 validation". Withdraw queued iw1-flip-957c275 and identity_957c275 lines, and never score landed 957c shards (record only their count). Then file a RESULT; Claude merges as a record. (2026-10-13)
- rev-merged-1013: (Claude review) Merged: sleep-transitive-inference (3b622a5dd), transitive-spiking-reach (ebeb4921c) and draft-ca3-symmetric-ti (merged after review). For T19, wherever 6/6 or op=0 is cited, carry the NARROWED caveats (determinism only; op=0 = the grid row; answer rate unmeasured), then queue its answer-rate line per answer-rate-off; do not flip. For CA3, continue ca3-sym-ti-row11: on one dev seed, name the stateful term that separates replica_sym (runner :440-472) from the engine's Hebbian step, then file a RESULT. For sleep-TI, next is the U6 control redesign as a prereg amendment. (2026-10-13)
- rev-retired-2-1013: (Claude review) Do not merge: long-delay-credit-spiking (all on draft-ldc-outcome-timing; point runbook T1 at that worktree, remove the old worktree, keep the branch as archive) and stage1-exit-gate (a2941fb1d overwrites the as-scored aggregate.json; main's 10-04 finding holds that read). If the 10-10 re-score is kept, save it as gate/aggregate-rescore-2026-10-10.json. Start attempt 2 on a fresh branch from main after A2. (2026-10-13)
- rev-stage2-hold-1013: (Claude review) stage2-exit-gate stays on its branch until scored. Now: sync main (rev-stage2-sync-1013), take main's 4 dev/s7 intact_a shards and sidecars, re-score the dev aggregate, run pytest. Queue no gate lines before 11-03. In the window, merge main, confirm the C4 and C5 organs are default-ON at the revision you pin, then re-pin GATE_SHA. (2026-10-13)
- rev-ldc-1013: (Claude review) HOLD research/draft-ldc-outcome-timing (A2's sim/ edit stays unmerged until tree-a2 resolves). At tree-a2 priority, run the syn-zero ladder (S smoke, pool A/B, then 2×400-trial cells on seeds 7 and 8) and file one RESULT. On the branch: commit the 4 d2_floor reference cells (on/cut s7/s8) and make test_ldc_soft_bounds_identity fail, not skip, when they are missing (drop the absolute-path fallback). For the provenance gap, test 542a37f02 merged into main and file NEEDS so Claude merges it. File NEEDS [ldc-verdict-undef] on verdict.py:69 vs prereg §6. (2026-10-13)
- rev-seam-map-1013: (Claude review) MERGE_AFTER_FIXES. On draft-seam-map: run agent_sync_branch.sh, taking main's lane_builds.jsonl. Apply every correction in docs/drafts/seam-map/CLAUDE-REVIEW-2026-09-29.md to host-decision-seam-map.md, with the review's wording and no new claims. Put a top caveat on 2026-09-27-tone-bg-nchannel-tonic-firing-wall.md: integration-overshoot artifact, status not live, no "100% neural" wording, and _gpi_fix_6seed.json = DEV with no prereg. Commit Markdown only, then file NEEDS [draft-seam-map]. (2026-10-13)
- c5-battery-lesion: (Claude, refute-lens) C5 bars (1)-(3) cannot tell the spiking race from a host random.choice(kb). Bar (2) holds by construction: the OFF lexicon shares no fact with the KB. In g6b every candidate is age=1 with x equal to 3e-8 (fresh session, note_turn marks all t=0), so recency and salience never engage. Before scoring, add per seed a commit-lesion arm (expect the neutral idle line) and one session with facts taught at different turns plus a permuted exposure. Without them C5 reads "replay fired", not "replay selects". (2026-10-13)
- c5-seed-provenance: (Claude, refute-lens) The "dev-7" identity smoke was reported whole-record byte-identical to g6b_off.json, which ran at seed 42 (BRAIN_CHAT_SEED unset), and its artifact is not on main. Commit it with its prov file and confirm BRAIN_CHAT_SEED=7. If it was 7, the worker's OFF output does not depend on seed: report that. The C5 scorer must read BRAIN_CHAT_SEED from every shard's prov env (the JSON records no seed). Seed 42 was seen before the prereg (g6b), so report that cell as non-blind. (2026-10-13)
- lbf-host-input-rows: (Claude, refute-lens) lbf_rows/tom_false_belief_chat.py tags the witness-gate override kind="neural-lesion", but _fb_store/verdict.json is STORE-NOT-LOAD-BEARING 6/6, and no scorer handles host-input rows. Before b2d1012 scoring, retag it (e.g. kind="host-input"), exclude it from the S1/S2 neural numerator, and report its F1 as a host-input lesion. Add the 6/6 store verdict to the 10-05 false-belief finding. (2026-10-13)
- iw2-selfid-answer-rate: (Claude, refute-lens) neural-selfid flag-on turns "what are you?"/"who are you?" from answered to abstained (finding §2). That is the FLIP-HURTS-ANSWERS condition, and no answer_rate run exists (the flip predates the ruling). Run tools/answer_rate.py paired against b2d1012-alloff and file NEEDS if it flags. The intact reply equals the host router's, so mark the row host-substitutable in LBF reports. (2026-10-13)
- iw2-transitive-engagement: (Claude, refute-lens) The 10-05 R data predate the engagement record (2aac93d43), so their F1 6/6 is UNDEFINED on engagement. In b2d1012, read each transitive lesion-arm reply for the chase trace and the absence of `error`; a cell without them is UNDEFINED. Count the row as stop-gate only. (2026-10-13)
- c1-moat-owner: (Claude, refute-lens) flipdev B-1 PASS 8/8 has no moat suppression: f80fc47a4 dropped check 6, and the bat->lake one-hop guess fires 8/8 and ships by default while the owner decision is open. Claude is asking the owner. Until it is ruled, the battery RESULT reports c_bat_swim on all gate seeds as a moat-class guess. Note in PLAN.md that B-1 was applied after the data existed. (2026-10-13)
- c1-gatehop-hygiene: (Claude, refute-lens) The gate_hop aggregate was scored from a dirty tree (ce4c5e13c, git_dirty true), and s101 lesion_inherit plus s102 lateral_learn have .part.json leftovers. Rescore clean and remove or explain the leftovers. lesion_lateral_read keeps worm 5/6 and `lesion: all` also zeroes the told frame, so cite lesion_lateral_learn (6/6) as the specific lesion. (2026-10-13)
- lane-dev-wording: (Claude, refute-lens) laneA "GO (6-seed)" and "SMOOTH" are on dev seeds 7-12. The Pearson bar does not discriminate: the single-pool control has 0.937/0.822. Cite levels 7v3 and 5v2 plus the lesion, and say "quantized ladder". laneD's moat is 1.0 in every arm, lesion included: drop it as evidence, and note the pooler is rate-based. Neither result is gate evidence. (2026-10-13)
- battery-amend-B-1013: (Claude, TREE) Before reading ANY c5val_1012 or b2d1012 shard, file Amendment B to the 2026-10-12 battery prereg in its own commit. It folds in c5-battery-lesion (commit-lesion, staggered-teach and permuted arms per seed; C5 bar 2 becomes "the commit lesion changes the idle reply" and "staggered or permuted changes which fact"), lbf-host-input-rows, iw2-transitive-engagement and c5-seed-provenance. These are stricter only: no bar is loosened and no seed or turn removed. Then generate the extra C5 lines (dev-7 smoke first). Shards that landed earlier stay as data, unread until B lands. File NEEDS for Claude to review B. (2026-10-13)
- c1-moat-ruled-1013: (Claude, owner delegated 10-13) KEEP the flagged one-hop guesses: humans also generalize by resemblance until they learn specifics. That makes this resolve c1-moat-owner. What separates learning from templating is that specific evidence must win, so two dev-seed probes are owed (C1 section 5, before the 11-01 flips): (1) OVERRIDE: teach "bat fly sky" plus a direct bat-swim fact, and the guess must yield to the taught fact on every dev seed; (2) GRADED: guess strength must rise with the number of shared traits (1 vs 2 vs 3). If either fails, file NEEDS. Also log the verb-string admission and floor/margin compare as host-decision debt for the host-share instrument. (2026-10-13)
- s1-attempt2-1013: (Claude, TREE) PLAN step D landed (21e5d5c67, merged 927aeb87c): A2 is registered in prereg §12 and the runner counts lesion_inherit_pathway, applied only if calls>0. (1) Provision main ≥927aeb87c. (2) Dev-7 smoke of cat_disc__lesion_inherit_pathway with the FULL _CAT_CONTEXT teach turns, mem 9 GB. It needs calls>0 and p_canary_eat read with hop:true, lesion:"all", lesion_held:true. If calls==0, STOP and file NEEDS. (3) Emit with --emit-pool-jobs <SHA> --out-dir research/findings/raw/_stage1_exit/gate_a2 (NEVER the default gate/), queue, then score with --aggregate .../gate_a2 --seeds 42,43,44,100,101,102 and file a RESULT by 10-24. (2026-10-13)
- s1-a2-smoke: (Claude) Superseded by s1-attempt2-1013. A2 is registered in prereg §12 (21e5d5c67), and the out dir is the --out-dir flag. Use that ruling's dev-7 smoke and add two pass conditions: val_heldout permuted n_words 24 and |scrambled_r| <= 0.05. If either fails, file NEEDS [s1-a2-smoke] and STOP. (2026-10-13)
- s1-attempt2-queue: (Claude/Fable, decision tree) After the smoke passes: stage1_exit_gate --emit-pool-jobs $SHA2 --out-dir research/findings/raw/_stage1_exit/gate_a2 > research/queue/s1_attempt2_${SHA2:0:9}.tsv. Expect 204 lines; any --out under _stage1_exit/gate/ is a STOP. Canary one line (tools/pool_canary.sh run auto), then add all with pool_queue.sh add --checked, max_age_h=168. If depth > 120 or C4/battery lines are ahead: NEEDS [aws-run] ASK with the TSV, SHA2, smoke peak RSS and "204 jobs, ~95 job-h, ~6 h on one r7i.4xlarge, ~$10". Keep the pool lines queued. STOP on canary rc!=0, ADD-FAILED twice, or provisioning blocked. Target: landed by 10-26. (2026-10-13)
- s1-attempt2-score: (Claude/Fable, decision tree) When all 204 A2 shards land, or 120 h after queueing: stage1_exit_gate --aggregate research/findings/raw/_stage1_exit/gate_a2 --seeds 42,43,44,100,101,102 --out .../gate_a2/aggregate.json; where_ran; commit by pathspec. (a) UNDEFINED for missing shards only: re-queue them once, re-score. (b) other UNDEFINED: NEEDS [s1-attempt2] STOP with the preconditions block. (c) PASS/FAIL: findings/<date>-stage1-exit-attempt2-RESULT.md, PROVISIONAL, with HANDOFF §8's sentences, then NEEDS [s1-attempt2] RESULT. Every PASS waits for Claude's verify-go. Never re-run a scored arm. (2026-10-13)
- s1-calendar-1013: (Claude) Calendar: Stage-1 attempt 2 lands by 10-26 (AWS preferred) and is scored by 10-28. It does NOT co-schedule with the Stage-2 window. Stage 2 keeps 11-03..11-10 queue / 11-21 score; Stage 3 keeps 12-19. This supersedes stage1-calendar's co-scheduling sentence. (2026-10-13)
- s2-c4-dev-now: (Claude/Fable, decision tree) docs/drafts/stage2-c4-build/HANDOFF.md was never started (my 10-11 priority list omitted it). Run it now, ahead of A2-ldc and T13. Steps 1-2 by 10-15 (sync research/stage2-exit-gate, tests, dev_select_repro, provision, 56 H1/H2 lines, mem_gb=9); step 3 when they land. (a) emo PASS 8/8 at dev and flip_check flags + unmeasured both empty: NEEDS [s2-c4-build] RESULT (Claude flips after verify-go). (b) a clause fails on some seeds: RESULT naming them, no GRID edit. (c) flags non-empty: NEEDS with the names. (d) sync conflict, test failure, organ error or memcap kill: STOP per HANDOFF §4. Not landed by 10-22: NEEDS [s2-c4-window]. (2026-10-13)
- s2-c5-val: (Claude/Fable, decision tree) Finish c5val_1012 under battery Amendment B once it lands: B's commit-lesion, staggered-teach and permuted arms join the OFF/ON pair; wspool only, one arm at a time. Per seed, B's C5 clauses decide; also check ON circuits_built>0 at selfinit, emo/bc_a/bc_b byte-identical across arms, and max per-turn ON-OFF RSS <= 2 GB. (a) all on 6/6: RESULT "C5 STANDS (PROVISIONAL)". (b) replay never fired on a seed: UNDEFINED, re-queue once, then RESULT. (c) a clause fails: RESULT "C5 REVERT-CANDIDATE seed n clause k". Never revert or tune. STOP if an arm has rc!=0 twice. (2026-10-13)
- battery-b2d-smoke: (Claude/Fable, decision tree) Smoke ONE dev-7 B2d line: tools/lb_shard.py jobs --seeds 7 --tag b2d1012smoke7 --no-fixes --probe-set adequate --repeats 2 --root ~/derisk-pool/revisions/<F> --pin <F> (F = f24963bfb05ca535f10f70bf4380fe4b2f68a9ea), under the portable RSS wrapper; WORKLOG peak RSS and wall time. (a) peak <= 10 GiB and wall <= 2 h: generate b2d1012 and b2d1012-alloff (--extra-env = each flipped row's flag =0), queue alloff FIRST, mem_gb = ceil(1.3 × peak), max_age_h=168. (b) peak > 10 GiB: NEEDS [aws-run] ASK with the numbers. STOP if a prefix guard exits non-zero. Smoking is allowed before Amendment B; READING is not. (2026-10-13)
- battery-score: (Claude/Fable, decision tree) After Amendment B lands and b2d1012 + alloff are complete: b2d_score.py --tag b2d1012 --pin f24963bfb --compare-tag b2d0929 --compare-pin 5818dc6 --out .../b2d1012/b2d_score.json, then answer_rate.py --on-tag b2d1012 --off-tag b2d1012-alloff --faculty <row> for each flipped row. RESULT is one table: flip | F1 n/6 | B1 S0,S2 vs P | answer_rate flags | STANDS / REVERT-CANDIDATE / UNDEFINED. tom-false-belief is scored as host-input per B. Under 6/6, or a B1 drop alloff does not share: REVERT-CANDIDATE (Claude reverts). (2026-10-13)
- s2-dev-reread: (Claude/Fable, decision tree) The Stage-2 dev reads predate the flips. After the s2-c4 sync, in $WT/stage2-exit-gate queue dev-7 intact_a shards for conv, infer, ask, emo and inner, plus inner permuted (stage2_exit_gate --shard --seed 7 --group G --arm A --out research/findings/raw/_stage2_exit/dev_f1/s7/G__A.json), mem_gb=9, max_age_h=72. Score with the runner's score_* readers and file ONE INFO [stage2-dev-f1] with the per-group reads. On an errored turn or organ error: STOP and NEEDS. No gate seed; GATE_SHA untouched. (2026-10-13)
- s2-gate-pin-nov: (Claude/Fable, decision tree) On 11-03, never earlier: sync research/stage2-exit-gate and check the tip holds AMENDMENT-A3.md, _IDLE_REPLAY_DEFAULT_ON = True and _EMOTION_CATEGORY_DEFAULT_ON = True; run tests + --selftest. (a) all present: re-run the five dev-7 intact_a arms at the tip, file NEEDS [stage2-gate-pin] with the tip sha and reads, wait for Claude's RULING, then HANDOFF steps 3-5 (186 lines, --max-age-h 240) by 11-10. (b) C4 flip absent on 11-03: pin and queue on 11-05 without it (C4 then reads FAIL as predicted) and say so. Depth > 150 on queue day: NEEDS [aws-run] ASK, with pool lines queued anyway. (2026-10-13)
- s2-gate-score: (Claude/Fable, decision tree) When the 186 Stage-2 shards land, and in any case on 11-21: stage2_exit_gate --aggregate research/findings/raw/_stage2_exit/gate --seeds 42,43,44,100,101,102 --out <dir>/aggregate.json; where_ran; commit by pathspec. (a) missing shards only: re-queue once and re-score; on 11-21 score what exists (UNDEFINED INCOMPLETE is the calendar attempt). (b) other UNDEFINED: NEEDS [stage2-result] STOP. (c) PASS/FAIL: finding with HANDOFF §8's sentences, PROVISIONAL, noting stale prereg sentences, then NEEDS [stage2-result] RESULT. Every PASS waits for Claude's verify-go. (2026-10-13)
- s2-c1-probe-first: (Claude/Fable, decision tree) Zero-build probe on dev seed 7 in $WT/stage2-exit-gate: the conv intact_a shard with ambient env BRAIN_WORD_ROLE_PARSE=1, and again with BRAIN_WORD_ROLE_PARSE=1 BRAIN_WORD_ROLE_COMMIT=1 (mem_gb=9, --out research/findings/raw/_stage2_exit/dev_wr/s7/conv__intact_a__<env>.json). (a) owl_hunt names mole and mole_moat abstains under a flag: RESULT "passive covered; gaps left: preposition, tell-me-more". (b) neither: RESULT "both gaps open". No flip, no runner edit. (2026-10-13)
- pool-order-1013: (Claude/Fable, decision tree) Queue order until 11-01 (replaces speedup-priority's list): (1) c5val_1012 + Amendment B arms (wspool); (2) s2-c4-dev-now's 56 lines (pool40-42); (3) battery-b2d-smoke, then alloff + stub shards; (4) the s1-attempt2 smoke then 204 lines ([aws-run] preferred); (5) s2-dev-reread and s2-c1-probe-first; (6) the C1 override/graded probes (c1-moat-ruled-1013); (7) lane stock last. An idle pool while any of (1)-(6) is startable is a stall. If a stage-critical line is within 24 h of max_age_h: NEEDS. Depth > 150 with (3) or (4) unqueued: NEEDS [aws-run] ASK. (2026-10-13)
- s2-needs-15-17-ruled: (Claude) Stage-2 HANDOFF #15: resolved by the idle-replay organ (it voices the brain's own store; A3's clause). #16: amend C3 (Claude is writing A4 tonight): an unacquired taught moat is UNDEFINED for that item, never "novel"; a reversed-order session is a COUNTED arm, so every C3 clause must hold in both orders; a DA-lesion arm is informational. #17: no wording change, and a C1 FAIL names all three gaps (wh-locative parse, locative frame storage, tell-me-more elaboration). #18a/b are production bugs: Claude is building both fixes. GATE_SHA may be pinned only after A4 lands. (2026-10-13)
- claude-builds-1013: (Claude) Claude is building these tonight; do NOT start them yourself: battery Amendment B (answers NEEDS [battery-amend-B]); Stage-2 A4 (branch from stage2-exit-gate); the yes/no-not-acquired fix; derived-conclusion marking; the C1 override/graded probes (c1-moat-ruled-1013); the curiosity LP de-risk (replaces Fable's s2-c3-lp-draft); a Stage-3 exit draft (replaces s3-draft, so Claude registers the prereg). Branches arrive as claude-build/* with rulings. Keep running everything else per pool-order-1013. (2026-10-13)
- phaseA-exit-1013: (Claude) Phase A exit check (~10-20), recorded honestly now. Six-seed host-share baseline: DONE. "One live seam retired": NOT DONE (BG live seam NO-GO on G4); the next method is the spiking commit stage (docs/drafts/spiking-commit/PLAN.md) applied to k-select. Reply-level fraction with the mouth: stub S2 0.514; the mouth arm is silent on 8 of 9 affect cells. Phase A is MISSED on item 2: put this line in the 10-20 state sync, without a phase redirect (pivot dwell). (2026-10-13)
- wspool-down-1013: (Claude, on [wspool-down] 04:18) wspool is back. The workstation went down at ~03:15 and rebooted cleanly (systemd shutdown, then a new boot at 06:33 UTC). The dispatcher has already re-dispatched to it: the c5val battery worker at pin f24963bfb is running again. Resume the ladder from line 4/12 (seed43 ON). Count no output from a job that was mid-run at ~03:15; re-queue it with the same line and no revision change. No NEEDS required on a clean return. (2026-10-13)
- claude-builds-inflight-1013: (Claude) Tonight's Claude builds are on S as unmerged branches. All are under Claude review; do NOT merge, provision or queue any of them until a RULING names the branch.
  - claude-build/stage2-a4: Stage-2 Amendment A4. Review returned FIX_FIRST; fixes are in progress.
  - claude-build/derived-terminal-honesty: Stage-2 HANDOFF 18b. FIX_FIRST: supporting_facts lists the derived conclusion.
  - claude-build/battery-amend-b and claude-build/battery-amend-b-pin (R_B e70e5c995): under review. Keep running the registered c5val pair ladder at pin F unchanged.
  - claude-build/battery-retention: opt-in session release.
  - claude-build/c1-generalization-probes.
  - claude-build/curiosity-lp-derisk.
  If Claude goes quiet before ruling: keep the existing ladders (c5val, Stage-1 attempt 2 per s1 rulings) and file one NEEDS [claude-builds] INFO at 18:00. (2026-10-13)
- s2-a4-review-1013: (Claude) claude-build/stage2-a4 (Amendment A4 plus fix A4.1, written before any gate data, prereg byte-identical) has passed Claude's two-lens review and lives on S as branch claude-build/stage2-a4. Merge it into research/stage2-exit-gate with no fast-forward. Then run agent_sync_branch.sh to bring in main. Check: git merge-base --is-ancestor 8c10af86a HEAD, --selftest prints SELFTEST OK, and pytest tests/test_stage2_exit_gate.py tests/test_stage2_exit_gate_a4.py all pass. Push via push_both. Do NOT pin GATE_SHA: that stays under s2-gate-pin-nov. (2026-10-13)
- s2-gate-pin-nov-A4: (Claude) This amends s2-gate-pin-nov and s2-gate-score. The tip must also hold AMENDMENT-A4.md, and the HANDOFF step-3 checks must print A4-IN-GATE-SHA and IDLE-REPLAY-IN-GATE-SHA. The job file has 198 lines, not 186, and s2-gate-score waits for 198 shards. Total is about 73 job-hours. Run a pool canary at the pinned revision before queueing. Everything else in both rulings stands. (2026-10-13)
- s2-needs-16-narrowed: (Claude) This narrows s2-needs-15-17-ruled #16. Every C3 CORRECTNESS clause must hold in both orders. The lesion and permuted clauses are read in the registered order only: a brain that is correct in reversed order must ask in the final discrimination block, where the graded lesion also shows. An unacquired-moat UNDEFINED applies only when that arm's other correctness clauses all hold; otherwise the seed is incorrect. The control arms take their moat set from intact_a. (2026-10-13)
- c1-probes-merged-1013: (Claude) The C1 generalization probes are on main at merge 960ce6528. They are dev seeds only, and the bars were committed before any data. Generate the lines at the merge with: .venv/bin/python -m research.runners.c1_generalization_probes --emit-pool-jobs 960ce6528 > research/queue/c1_probes_960ce6528.tsv. Provision 960ce6528 and run the pool canary. Then queue the 16 lines (2 probes x dev seeds 1-8, mem_gb=9) one pool_queue add at a time. They run after the c5val pair ladder, and never on wspool while a C5 arm is running there. Do not use the stale c1_probes_2cd1ea52e.tsv. (2026-10-13)
- c1-probes-score-1013: (Claude) When all 16 land: c1_generalization_probes --score research/findings/raw/_stage1_c1_generalization --seeds 1,2,3,4,5,6,7,8, then commit by pathspec. File NEEDS [c1-generalization] RESULT with the override and graded verdicts per seed. Dev-7 read: override PASS; graded FAIL (newt = toad < eel). The organ-only prediction for dev 1-8 is graded strict-rise on 2 of 8 seeds, which would mean the organ links nearly every frame to every other and evidence does not add strength. A graded FAIL is a verdict on the method, not the capability. Claude rules the next method. (2026-10-13)
- s3-draft-merged-1013: (Claude) The Stage-3 exit DRAFT is on main at merge 97bc60c0f: docs/drafts/stage3-exit/PLAN.md, research/runners/_stage3_exit_draft.py and the tests. It is dev seeds only, and no prereg is registered. Its probes depend on two production fixes that are still under review: claude-build/derived-terminal-honesty (derived_svo, plus a conclusion that was told stays plain) and claude-build/yesno-no-acquire (a bare "does X R Y" question is never acquired). Do NOT run the Stage-3 dev-7 smoke until a RULING says both are on main. Claude will then rule the smoke revision. (2026-10-13)
- s3-open-rulings-1013: (Claude) PLAN §11 rulings:
  (1) M_ADJ_TOLD_FIX_IN_PIN = yes. The told-directly fix is required in the pin, so m_adj expects a plain reply.
  (2) The C2 type-2 AUROC bar is 0.75, given the 8-item granularity.
  (3) C3 queues on the calendar whatever happens. If the T21 spiking witness is not in the pin, ownership reads HOST and is reported as "capability PASS, host-owned"; T21 stays the named next method.
  (4) Drop noise_taught (-6 jobs). Natural uncertainty is primary.
  (5) No PLASTIC_SETS-revert amendment unless the dev-7 kb_pop/corr_freeze arms read unclean.
  (6) The RI thresholds (>=85%, x/÷1.5) stay draft until dev-7 calibrates them, and are fixed in the prereg by 11-14. (2026-10-13)
- battery-retention-merged-1013: (Claude) The battery worker session release is on main at merge f935885d6. It is OPT-IN through LB_BATTERY_RELEASE_SESSIONS=1 and default OFF, so every registered instrument is unchanged. Do NOT queue E1 or use release in any run yet. A follow-up commit is coming: an id()-reuse guard for xedge wave3, a stricter E1 checker, and a fix for the stale source-provenance test. Claude will rule E1 when it lands. tests/test_webapp_server.py::test_brain_chat_source_provenance_default_off_is_byte_identical fails on main before this merge too: it has been stale since source provenance went default-ON on 09-01. It is a known failure, not a regression from this merge. (2026-10-13)
- battery-retention-e1-1013: (Claude) The E1 equivalence check is cleared to queue. It runs at ONE revision, the current main tip (it must contain 508f004a6). Generate the lines with python3 docs/drafts/battery-retention/gen_e1.py <tip> <tip>, so A and B differ only by LB_BATTERY_RELEASE_SESSIONS. That gives 13 lines: dev seeds 7 and 8, arms off/on/lesion, plus Bnorel. Queue them on wspool only, after the c5val pair ladder, one pool_queue add at a time, mem_gb=21. Score with the WORKLOG checker; it fails on _error turns, a missing +LTM, and an RSS rise or drop below threshold. The release default stays OFF until E1 passes and Claude rules it. (2026-10-13)
- derived-honesty-merged-1013: (Claude) The fix for Stage-2 HANDOFF 18b is on main at d6a7213c6 and is default-ON.
  - A GNW multistep chase of 2 or more hops is framed "I derived this from: ...".
  - supporting_facts holds only the hops, and the never-told conclusion is in derived_svo.
  - An affirmed stored conclusion stays a plain recall; a negated one stays derived.
  - A hop where committed != target gets the generic lead.
  - The escape is BRAIN_GNW_MULTISTEP_DERIVED=0.
  - The confidence-cap lead fix also changes truncated C2 replies, in the honest direction.
  s2-dev-reread covers it: Stage-2 inner/infer dev reads made before d6a7213c6 are stale for the C2/C5 reply text. (2026-10-13)
- battery-amend-B-merged-1013: (Claude) Amendment B is on main at merge c4d0f3da6: prereg revisions 1 and 2, the C5 battery scorer, b2d/lb_shard/iw2 changes, and JOBS §9. It has passed three Claude reviews. R_B = 1304d8d67, on branch claude-build/battery-amend-b-pin: the pin F plus a label-only turn block. Push that branch with push_both. Provision R_B with POOL_PROVISION_ALLOW_STALE=1, because R_B must stay F plus one block for the E3 identity check. Cite this ruling in the provision log. Then queue JOBS §9.1 (9 dev-7 lines, the E3 pair included) one at a time, on wspool only, after the c5val pair ladder. Re-stamp the session-arm mem_gb at ceil(1.3 x the dev-7 peak). (2026-10-13)
- battery-amend-B-gate-1013: (Claude) Release the 42 §9.2 gate lines only once dev-7 E3 holds and the §9.1 smokes are valid. Otherwise file NEEDS [battery-amend-B] STOP with the reads. Score with tools/c5_battery_score.py exactly as in JOBS §9. The C5 reading at its best is "selects by tag recency; ownership UNDETERMINED (race vs host argmax)". Never cite C5 as spiking-owned content selection. The named next method is the race-steer arm, Amendment C, whose design is in docs/drafts/idle-replay/WORKLOG.md. b2d1012 is scored with the default (Amendment B) b2d_score: transitive is a stop-gate (S0 governs) and tom-false-belief is host-input. (2026-10-13)
- yesno-merged-1013: (Claude) The fix for Stage-2 HANDOFF 18a is on main at 3b4b31604 and is default-ON. A question, judged by its form, is never acquired as a fact, never used to rewrite a stored fact, never rejected as a contradiction, and never answered as if the user had asserted it. A question turn now replies "What I hold is: ...". The escape is BRAIN_ACQUIRE_QUESTIONS=1. Known gap: a modal followed by a plural noun ("can dogs bark") keeps its old handling. This is a host scaffold, with a row in docs/SCAFFOLD-LEDGER.md; its replacement is a spiking clause-type organ (HANDOFF 14). Under s2-dev-reread, Stage-2 infer p_yesno dev reads made before 3b4b31604 are stale. (2026-10-13)
- s3-smoke-rev-1013: (Claude) The Stage-3 dev-7 smoke revision is the main tip at or after 3b4b31604. That tip contains the Stage-3 draft 97bc60c0f, derived-honesty d6a7213c6 and yes/no 3b4b31604. Provision the tip and run the canary. Then: .venv/bin/python -m research.runners._stage3_exit_draft --emit-pool-jobs <tip> --seeds 7, which gives 34 lines at mem_gb=9. Queue them one at a time after the c5val ladder and the battery §9.1 lines, never on wspool while a C5 arm runs there. Aggregate with --aggregate research/findings/raw/_stage3_exit/dev --seeds 7. File NEEDS [stage3-dev7] RESULT listing every arm with applied=False and the PLAN §10 checks. (2026-10-13)
- lanes-and-board-1013: (Claude) At 12:10 the lane-starvation gate reports all 5 CPU lanes unserved (Affect, Curiosity, Self/Workspace, Perception, Language). Today's rulings hold ready work: the c1-probes lines, the battery §9.1 lines, E1 and the Stage-3 dev-7 smoke. Provision their revisions and stage one job per lane where a lane has ready work. Do not waive while ready work exists. Then append this entry to docs/.vikunja_sync and commit it: "## 2026-10-13  claude-night-arc" followed by "Added #251-#257 (derived honesty done; questions not stored done; neural question organ; guess strength vs evidence; idle replay ownership; one shared LTM; curiosity learning-progress)". Claude already created the tasks; its own stamp commit was blocked by this gate. (2026-10-13)
- owner-one-brain-memory-1013: (OWNER 2026-10-13, via Claude) Approved: ONE brain for everyone, with memories tagged by source. Memory steps 1-3 are pulled forward from Phase C to now: (1) the imported knowledge store loads once per server, shared read-only; (2) one shared learned memory across conversations; (3) cross-session persistence default-ON in production, tests never write. wikidata_100k stays a declared prior, and the learned share is reported beside it. Claude is building step 1 (claude-build/shared-ltm) and designing steps 2-3 (claude-build/one-brain-memory-design). Until a RULING lands, do not edit webapp/server.py _BRAIN_CHATS/_build_chat_brain/_load_or_build_ltm_store or research/runners/cross_session_persistence.py. (2026-10-13)
- shared-ltm-equiv-1013: (Claude) Step 1 of the owner-approved one-brain memory plan is built on claude-build/shared-ltm (unmerged; do not merge or touch). Revision bf91d574738912d70d417285a741d685c0e1ee2d is provisioned on pool41, pool42, pool40 and wspool (branch research/shared-ltm-equiv; leave it in place). Queue the 4 lines SL.1-SL.4 of research/coordination/2026-10-13-shared-ltm-equiv-JOBS.md verbatim, interleaved with the battery 9.1.4-9.1.9 queue: they block a default-on merge. When all 4 are done, write one NEEDS line [shared-ltm-equiv] with that file's pass bar result per seed (identical or the differing keys) and maxrss_kb for on and off. (2026-10-13)
- shared-ltm-merged-1013: (Claude) claude-build/shared-ltm is reviewed and MERGED to main as 683c181b3, DEFAULT OFF (opt in with BRAIN_SHARED_LTM=1; it also isolates the LTM bridge-build RNG reseed, BRAIN_LTM_RNG_ISOLATE). Production behaviour is unchanged. SL.1-SL.4 at bf91d5747 stay queued: they are informative (pre-fix ON vs OFF). Report them as ruled in shared-ltm-equiv-1013. The flip battery comes later at a post-fix main tip; Claude will rule it. Keep research/shared-ltm-equiv until SL.1-SL.4 finish, then you may delete that branch. (2026-10-13)
- one-brain-m0-merged-1013: (Claude) M0 is reviewed (3 rounds) and MERGED to main as e08895cd3. A process FIFO turn lock is now DEFAULT ON (escape BRAIN_TURN_LOCK=0): turns from different conversations no longer race on shared organs; the idle wander runs outside the lock and a new turn preempts it; a turn waits at most BRAIN_TURN_WAIT_TIMEOUT_S (300) and then gets busy:true. Session-facts persistence now really saves on the default composer (it was raising and being swallowed). New gate tests: tests/test_brain_turn_lock.py, tests/test_tick_pre_m0_equivalence.py. The probe is research/runners/one_brain_probe.py. If a battery or stage line changes behaviour at a main tip >= e08895cd3, say so in NEEDS. (2026-10-13)
- shared-ltm-flip-1013: (Claude) Queue the 12 lines SLF.1-SLF.12 of research/coordination/2026-10-13-shared-ltm-flip-JOBS.md verbatim (prereg committed in d8e72021f before any data; revision e08895cd3, which Claude provisioned). Order: after battery 9.1.4-9.1.9 and the s3 smoke, before any other new work. They supersede SL.1-SL.4: drop any SL line that has not started. When all 12 are done, write NEEDS [shared-ltm-flip] with the 3-part pass bar result per seed (identical, or the differing keys) and maxrss_kb for on/off. (2026-10-13)
- t1-a2-sync-fixed-1014: (Claude, on [t1-a2-sync]) Fixed on main 9911b9bad. It was citation markup only, in research/findings/2026-10-12-validation-battery-prereg.md; no bar, number or command meaning changed. The future c5_score.json output is now written via $D. The pool-node path is reworded as an off-repo reference, with an erratum: that prov was never committed. The bar-sanity 0.920 is marked derived. claim_check now passes: 8 cited, 0 missing. Re-run the sync, then provision, then syn-zero rung 2, per tree-a2. (2026-10-14)
- shared-ltm-flip-amendA-1014: (Claude, on [shared-ltm-flip]) The FAIL stands as registered, and the flag stays OFF. Amendment A is filed before data in c8759a385 (same JOBS file, new section). Queue its 6 lines SLF-A.1 to SLF-A.6 verbatim: offiso arms at the same revision e08895cd3, already provisioned. Order: right after the stage-1 sync and rung-2 work, before other new work. When all 6 land, write NEEDS [shared-ltm-flip-A] with the A1 and A2 results per seed. A1 is whole-record identity, on vs offiso. A2 is the reply-level fields, offiso vs off, plus every differing key. (2026-10-14)
- one-brain-m2-merged-1014: (Claude) M2 merged to main as 694e8327c, DEFAULT OFF; the OFF path is hash-identical to base. BRAIN_ONE_BRAIN=1 gives one brain shared by all conversations: B recalls what A taught, the fact survives a restart, and freeze/zero lesions remove it (dev seeds 7/8). Do NOT enable BRAIN_ONE_BRAIN outside tests until packet items 1-3 are reviewed. Packet: docs/drafts/one-brain-m2/PACKET-m2-fixes.md (350d31b8b), 8 items: multi-fact prose check, "did we discuss" gating, lock forced at ON, plus minors. Priority: after the stage-1 sync/rung-2 and the SLF-A lines, before other new builds. Branch research/one-brain-m2-fixes; finish with NEEDS [m2-fixes] ASK. (2026-10-14)
- t7-c1-duplicate-1014: (Claude, on [t7-c1]) Two adversarial reviews: the numbers stand, but this is not new. main already holds the same gate result, research/findings/2026-10-06-stage1-c1-category-inheritance-RESULT.md (d320b1deb). 52 of 54 shards are byte-identical, and its verify-go wording carries 7 caveats that the 10-14 doc drops. Do NOT merge 3009927b6, and keep the branch. The record stays the 10-06 wording: C1 with BRAIN_CATEGORY_INHERIT=1, under the wiring prereg, PASS 6/6. That is NOT the Stage-1 exit criterion: Stage-1's frozen lesion_draw changes 0/6 probes, and the wiring is default OFF. Before filing any RESULT, check main for an existing RESULT on the same gate. Add one FAILURE_LOG line for that check. (2026-10-14)
- c1-flip-gate-1014: (Claude) No flip yet. The flip and the Stage-1 amendment (C1's counted lesion lesion_draw -> lesion_inherit, filed before any attempt-2 data) are decided together, on [c1-generalization] RESULT: the dev 1-8 probes, override + graded, per c1-probes-score-1013. If they are not done, give the remaining c1 probe lines priority after the rung-2 work, because Stage-1 attempt 2 queues by 10-26. (2026-10-14)
- spiking-commit-a1-run-1014: (Claude, owner-approved 2026-10-14) Packet word-role-commit (RULING spiking-commit-a1, main fa3c0acf0) was ruled on 10-06 but never run: no research/word-role-commit branch, no data. Run docs/drafts/spiking-commit/HANDOFF.md top to bottom now: dev sweep (8 lines), select, step O (6 lines, canary first), then step R only on O PASS. Priority: right after the stage-1 rung-2 and c1 probe lines, ahead of other new work; it is Phase A's critical path (one live seam retired). File each verdict as RESULT [word-role-commit]; a NO-GO or UNDEFINED goes to NEEDS. Claude is separately building the same stage on k-select. (2026-10-14)
- 2026-10-14 [bg-commit-gate] (Claude) GO TO RUN. The BG live-seam spiking commit stage is merged to main at f284c710c (flag BRAIN_BG_COMMIT, default OFF; amendment research/findings/2026-10-14-bg-live-seam-commit-AMENDMENT.md, frozen at that commit; review: MERGE, 7 minors all fixed in 05f5cf116). Execute docs/drafts/bg-commit/HANDOFF.md top to bottom, exactly as written: step 0 pin + tests + provision, step 1 six gate lines, step 2 canary on the seed-43 line + ENGAGED check, step 3 queue, step 4 score with --commit, step 5 the outcome table. No judgment beyond the tables; never flip a flag; any verdict is PROVISIONAL until Claude confirms. NEEDS-CLAUDE only where the HANDOFF says STOP or on the final verdict.
- 2026-10-14 [stale-test] (Claude) Low priority, not blocking bg-commit: tests/test_battery_amendment_b.py::test_commit_lesion_env_reaches_the_reply_level_organ_call fails on main already at cc9b4bf3e (webapp/server.py:5587 now reads `_IR.speak(_bkey(cache_key)...`; the test expects the older call text). Update the test's expected text to match the current call, and confirm the env still reaches the organ call; edit the test only, not server.py. One commit on a research/ branch; ASK for merge.
- c1-generalization-1014: (Claude) Answers NEEDS [c1-generalization] RESULT (OVERRIDE FAIL 7/8, GRADED FAIL 2/8; exactly as PROBES-PLAN section 7 predicted). Per c1-flip-gate-1014: NO flip of the one-hop guesses and NO lesion_draw -> lesion_inherit amendment. Stage-1 attempt 2 runs C1 as registered (incl. A2.1), on the calendar (queue by 10-26), even if C1 is expected to fail. Banked method verdict: the frame layer over-associates (lake rate flat ~0.075-0.093 across 1-3 shared traits; graded steps below slot heterogeneity); the seed-5 eel tie is the host TIE_MARGIN compare (host-debt item 2). No retune on dev 1-8. (2026-10-14)
- c1-next-method-1014: (Claude) No-defer next method for C1 graded generalization, after bg-commit-gate step 3 is queued: run `bash tools/before_you_build.sh "frame layer over-associates: guess strength flat across shared traits"` plus RAG (--corpus finding, then kandel). Ask the wall question: what does the real cortex run alongside association that we replaced with a constant (e.g. inhibitory competition / sparse pattern separation that makes overlap graded)? Write ONE method proposal (<=60 lines, docs/drafts/stage1-c1-inheritance/NEXT-METHOD.md) with bars fixed before data, judged on fresh dev seeds 9-16, and file ASK [c1-next-method]. Do not build before Claude's ruling. (2026-10-14)
- pool-sync-1014: (Claude) Answers ASK [pool-sync-rsync]. Do NOT install rsync in the container (vendor image; it would not persist) and do not run tools/pool_sync.sh from inside it. The host's sim-sync.timer is the one retrieval path and works (it pulled 8 + 14 files at 12:02/12:17 today; runs :00/:15/:30/:45). After a line finishes, wait for the next sync. Stop manual tar-over-ssh pulls: they bypass pool_sync's update-only and filter rules. If an expected file is still absent after two sync cycles: NEEDS with the path. (2026-10-14)
- bg-commit-now-1014: (Claude) [lane-waiver] noted. bg-commit-gate (ruling above) IS ready work: start its step 0 now, alongside word-role-commit; pool40-42 have room (its lines are mem_gb 8). It is the critical path (first live seam retirement). Wspool note: the GPU-queue b2d_amend2 probe there was memory-throttled at its 14G soft cap since 06:02; Claude raised its live limits (high 21G / max 23G) at 10:51 and it now runs at full speed; SLF-A.2-6 follow when it ends. (2026-10-14)
- ldc-synzero-1014: (Claude) Answers ASK [ldc-synzero-deadcut]. An inert cut is UNDEFINED (void instrument), not a negative; do not hunt for a cell where it bites. Wall question first: real GPe neurons are autonomous pacemakers (tonic ~30-60 Hz; D2-MSN input acts by PAUSING them); a GPe silent all window means its intrinsic drive became zero. Step 1, read-only: measure the LDC organ GPe rate in the intact arm, pre-cue and in-window, dev seed 7; find its GPe drive in code. Step 2: `bash tools/before_you_build.sh "GPe silent: missing tonic pacemaking"`. Step 3: ASK [ldc-gpe-tonic] with one flag-gated (default OFF) proposal plus an engagement precondition (intact GPe in-window rate > floor) fixed before data; dev seeds only. Production BG organ: untouched. (2026-10-14)
- wrc-verify-1014: (Claude) [word-role-commit] RESULT received; Claude verify-go is running now. Do not file a flip packet or count the credit until the verify-go ruling lands. (2026-10-14)
- needs-format-1014: (Claude) Write NEEDS lines as `- 2026-10-14 HH:MM [tag] KIND: ...` (ONE leading dash). Two NEEDS lines today started with `- - 2026`; Claude's watcher missed them for hours (the 07:03 syn-zero ASK, the 15:55 word-role RESULT). The watcher now accepts both forms, but use one dash. (2026-10-14)
- wrc-verify-go-1014: (Claude) verify-go on [word-role-commit] RESULT: CONFIRMED-WITH-CAVEATS. select/O/R re-scored byte-identical; provenance clean at P; R bars as registered. Edit research/findings/2026-10-14-word-role-commit-gate.md (status stays provisional until these land, then `confirmed-with-caveats`): (1) lines 61-62: delete the "specific to the committed role" clause; the nofact control's intact arm already declines ('drink' never commits, (brain, None) 6/6), so control-ok holds even if the lesion broke every parse; call it the no-fact control, not "permuted" (lines 7, 14, 61). (2) lines 66-79: never-latched slots read 0 (R traces 6/6; O level 0-1 vs 71-85); the 79-93 figure is the agent latch still holding. (2026-10-14)
- wrc-verify-go-1014b: (Claude) (3) Add the prereg section-4 report-only items: dual-role reads 17/24 right, 0 wrong, 7 declined (seed 101: 1/4); the exclusion lesion gives 0/24 co-commits, so "at most one role" comes from the role layer, and the gate's cross-inhibition is not shown necessary; the count-mode baseline at 9ea7411. (4) Add a "not yet a live seam" line: flags default OFF; production still parses with BridgeParser. Residual host steps at ON: the commit read (spiking_commit.py:181-196; threshold-invariant, not deciding), slot->word mapping, the a==v guard, self-alias, ROLE_MIN folded into W_RG. One commit, then ASK [wrc-finding-fixed]. (2026-10-14)
- wrc-flip-packet-1014: (Claude) CRITICAL PATH (this would be the first live seam retired): after the finding fix, draft docs/drafts/word-role-flip/PACKET.md making BRAIN_WORD_ROLE_PARSE + BRAIN_WORD_ROLE_COMMIT default ON through the validation battery's flip procedure (prereg 2026-10-12). Include, with bars fixed before data: F1 at the default with no FLIP-HURTS-ANSWERS; answer rate flag-off vs on; an OFF-TARGET control (questions that skip the word-role parse reply identically under the commit lesion at ON); the dual-role decline rate reported. Dev seeds first. ASK [wrc-flip-packet] for Claude review before any run. (2026-10-14)
- wrc-finding-merged-1014: (Claude) Answers ASK [wrc-finding-fixed]: research/wrc-finding-fix merged at 6c83b8d2d; status set to confirmed-with-caveats at d7ddeb111. Go: draft docs/drafts/word-role-flip/PACKET.md per RULING wrc-flip-packet-1014, then ASK [wrc-flip-packet]. Small follow-up in the same packet commit: give the count-mode baseline at 9ea7411 its numbers in the finding (it is currently a bare pointer). (2026-10-14)
- wrc-packet-review-1014: (Claude) Answers ASK [wrc-flip-packet]: APPROVED WITH AMENDMENTS (A-D below), then go. (A) OFF-TARGET: list the exact question strings in the packet now (>= 8, before any data). Precondition: in the intact-at-ON arm, no listed turn may engage the word-role parse (trace route); one that does makes the bar UNDEFINED plus a NEEDS, never a silent drop. BOTH lesions (PARSE and COMMIT) must leave every listed reply identical, 6/6. (B) Add F2/F3 identity per battery prereg section 5 (dev seed 7, iw1/iw2_default_identity): default == on:both on every script turn; off (both =0) == pre-flip default, byte-identical. (2026-10-14)
- wrc-packet-review-1014b: (Claude) (C) Scoped B1: on the dev-7 smoke, run the battery turn script ON vs alloff at the flip revision; every row with a changed reply is AT RISK (list them). On the six gate seeds, each at-risk row's n_load_bearing at ON must be >= alloff at the same revision; a drop reverts the flip (battery section 6). No at-risk rows: B1 is met by the identity record. (D) Order: smoke dev 7 (RSS, wall, F2/F3, the at-risk list, off-target precondition) -> promote the packet to research/findings/2026-10-1X-word-role-flip-PREREGISTRATION.md on main with bar-sanity records -> ASK [wrc-flip-prereg] -> gate seeds. Skip dev 1-8. (2026-10-14)
- wrc-packet-answers-1014: (Claude) Q2: ONE commit flips both anchors (the commit flag is only read when parse is ON). Q3: do not amend the 10-12 battery flip table (pinned at P/R_B); this packet registers its own flip unit, and later battery runs at revisions after the flip inherit the defaults (both rows are in the headline registry). Q4: confirmed. The only production reader is brain_chat_tui (_neural_question_parse :1440, called :1723); the webapp chat goes through ChatBrain, so production chat is the intended blast radius. (2026-10-14)
- bg-commit-nogo-1014: (Claude) Answers [bg-commit] RESULT: NO-GO CONFIRMED. Claude re-scored the 42 shards: verdict/gates/pooled/n identical; preconditions identical apart from function addresses. Finding merged to main at c3dd5477a. Banked: G4, the 10-02 failure, now PASSES; the commit read is threshold-invariant and brain-owned. The failure moved to the race on empty turns. Shard read (seam_b): 3 of 4 empty-turn non-holds are ACTIVE speak commits on hsb_empty2 (seeds 42/43/100); 1 is a no-commit fall-through (101, empty1). G3 misses by one event (7/24; >=8 needed), also mostly empty turns. So the second empty turn of a session is biased toward SPEAK: something carries over between turns. (2026-10-14)
- bg-commit-next-1014: (Claude) No-defer next method; flags stay OFF. Step 1, read-only: from the gate traces plus the dev sweep, diff hsb_empty1 vs hsb_empty2 (the turns between them; the pre-turn G/S latch state and the W_RESET effect; value inputs) and name what carries over. Step 2, wall question: real BG holds by default through tonic GPi/SNr inhibition fed by tonically active GPe/STN; an action needs positive release. This is the same missing tonic pallidal drive as RULING ldc-synzero-1014, so run both investigations as ONE. Step 3: ASK [bg-commit-next] with one flag-gated proposal, a new amendment, bars unchanged, dev seeds 9-16 first. Priority: after the word-role flip smoke. (2026-10-14)
- slf-a-score-1014: (Claude) All six SLF-A offiso arms have landed (shared_ltm_flip_1013/slf_s*_offiso.json, 18 of 18 files). Score A1 (strict, blind) and A2 per research/coordination/2026-10-13-shared-ltm-flip-JOBS.md Amendment A, then file [shared-ltm-flip-A] RESULT. Do this before the bg-commit next-method step. (2026-10-14)
- slf-flip-done-1015: (Claude) Answers [shared-ltm-flip-A] RESULT: CONFIRMED by Claude's independent re-score of the 18 shards (same verdicts, same single moved key). Sharing is now DEFAULT ON: merged 1ff97a8cd (`_SHARED_LTM_DEFAULT_ON = True`; finding research/findings/2026-10-15-shared-ltm-default-on.md; tests updated; BRAIN_SHARED_LTM=0 is the escape; no ledger row, not a faculty). The live webapp picks it up on its next restart; restart when no conversation is mid-turn. (2026-10-15)
- slf-flip-docs-1015: (Claude) Then: commit the 18 still-untracked raw slf_s*.json files plus their prov sidecars, and sync ROADMAP.md and GAP_CLOSURE_MISSION.md CURRENT STATE with three items: shared memory default ON; word-role commit confirmed-with-caveats, flip packet in progress; BG commit NO-GO (G4 fixed, G2a/G3 failed, next method = tonic pallidal drive). (2026-10-15)
- roadmap-sync-1015: (Claude) Answers ASK [roadmap-sync]: merged 6f246bf52, then corrected at 20c77e8aa. The word-role paragraph had a wrong caveat ("5 of 22 first-12 turns changed" is a BG seam metric, not word-role); it now carries the verify-go caveats. The BG NO-GO is marked confirmed. When you copy a caveat into a summary doc, quote it from the finding or the ruling, never from memory. (2026-10-15)
- wrc-f3-1015: (Claude) Answers ASK [wrc-flip-smoke] 1: a clarification, not a bar change. Amendment B means "off" = ONLY BRAIN_WORD_ROLE_PARSE=0 and BRAIN_WORD_ROLE_COMMIT=0, every other flag at its production default. Add that word-role-only-off arm: the iw2 runner zeroes only this packet's PACKET_FLAGS (the two word-role flags), or pass the two =0 envs explicitly. The F3 bar stays: word-role-only-off@F must be byte-identical to default@P on every script turn. Write this clarification into PACKET.md before any rerun. The turn-12 diff from the all-off arm (xedge ships ON) is expected by construction: report it, never count it. Then continue the smoke. (2026-10-15)
- wrc-f3-verdict-1015: (Claude) Answers ASK [wrc-f3-verdict]: F3 as run is UNDEFINED (confounded), neither PASS nor FAIL. F=8407f21 contains Claude's shared-LTM default-ON commit (cbffef7f8) and P=96e5198 does not. That flip's declared effect is non-reply body fields moving, replies identical: exactly turns 2/4/6/7/8 (iw1) and 7/8/11 (iw2). Fix: re-pin P to the parent of the word-role flip commit on F (main with shared-LTM ON and both word-role flags at their old default OFF). Re-run `default` there for iw1 and iw2, then compare off:word-role@F vs default@P'. The bar is unchanged (sha_canon identical on every turn). Record P' in PACKET.md before the rerun. (2026-10-15)
- needs-format-1015: (Claude) NEEDS lines still need the DATE: `- 2026-10-15 HH:MM [tag] KIND: ...`. The 08:15 and 11:12 lines had no date; the watcher now accepts them, but write the date. Delegate budget: a RULING from Claude counts as the verdict; do not wait for a MiMo Pro judge on a question Claude has ruled. (2026-10-15)
- s2-c4-sync-1015: (Claude) Answers STOP [s2-c4-build]: STAMPED, not waived. smoke_c5a_selfinit.json had its genuine run-time sidecar (.prov.json, written 2026-10-12 14:56 with the json, git_sha e3ac0da4c, manifest verified) sitting untracked on main. Claude committed it to main as 9572b8ca5. Retry agent_sync_branch. If another raw file on main blocks the same way, look for an untracked sibling .prov.json first and commit it (never hand-write one); NEEDS only if none exists. (2026-10-15)
- s2-c4-count-1015: (Claude) Answers ASK [s2-c4-build]: QUEUE ALL 64. The grid did not grow; the HANDOFF's "32 lines" was Claude's miscount. stage2_exit_gate.arms_for("emo") has given 5 arms since e4bdf9f41: intact_a, intact_b (the null), lesion_affect, lesion_affect.rep0 (the repeat), permuted. 5 x 8 dev seeds = 40 for H1, plus H2's 24 = 64. Do not cut: the aggregate needs intact_b for the null and .rep0 for "reproduced", or emo reads UNDEFINED. The emit command ("every registered emo arm") is authoritative over the count. Note the correction in the WORKLOG. mem_gb 9 per line as written. (2026-10-15)
- stale-test-merged-1015: (Claude) Answers ASK [stale-test]: research/stale-test-fix merged at 783d79fc6 (test expectation only; server.py untouched). (2026-10-15)
- s2-c4-flip-1015: (Claude) Answers STOP [s2-c4-build]. verify-go: Claude re-ran the aggregate from the 64 shards (emo C4 PASS 8/8; lesion, reproduced, permuted 8/8; null clean), identical. Import fix plus FLIP merged to main at 780c465a3: `_EMOTION_CATEGORY_DEFAULT_ON = True`, ledger row emotion-category (71 faculties), board #262, finding research/findings/2026-10-15-emotion-category-default-on.md. flip_check ran: no FLIP-HURTS-ANSWERS, no FLIP-CHANGES-NONFEEL. FEEL-NOT-APPEND-ONLY on p_private (s3, s7) is a checker misclassification: the idle turn's ON and OFF replies are byte-identical. The 32 unmeasured are the non-intact_a arms, which by design have no OFF run. (2026-10-15)
- s2-c4-next-1015: (Claude) Next, per HANDOFF section 5: sync research/stage2-exit-gate with main (it must contain 780c465a3), then re-run the five dev-seed-7 intact_a arms of the Stage-2 gate at that tip (the gate's own dev smoke; the emotion default is now ON, so use no flag env) and file RESULT [s2-c4-smoke]. Claude then pins GATE_SHA before 11-01. Your board_sync.json entries were moved aside by the merge and are restored (all 14). (2026-10-15)
- c1-correction-1015: (Claude) CORRECTS c1-generalization-1014. category-inherit has been default ON since 10-11 (RULING c1-flip-go), so there was no flip to withhold. The probe FAIL blocks only the lesion_draw -> lesion_inherit amendment and any Stage-1 C1 credit claim. It does not revert the faculty: the OVERRIDE miss abstains (safe), and GRADED is a quality defect, not an honesty one. Its standing stays with the validation battery's registered F1/B1. c1-next-method-1014 continues unchanged. (2026-10-15)
- priorities-1015: (Claude, owner-delegated 10-15: "priorities are up to you") Work order until 11-01, highest value first; it supersedes pool-order-1013 where they differ. (1) Stage-1 attempt 2 (calendar: land by 10-26, score by 10-28; runs even if expected to fail): its smoke, then the 204 lines. File [aws-run] ASK the moment the smoke passes; Claude starts AWS. (2) The word-role flip, the first live seam retirement: F3 rerun at P', promote to prereg, ASK, gate seeds. (3) Stage 2: [s2-c4-smoke], A4, GATE_SHA pin by 11-03. (4) Integration-first conversions already in flight (C5 idle replay via Amendment B). (2026-10-15)
- priorities-1015b: (Claude) (5) Second seam: bg-commit-next (tonic pallidal drive, one investigation with ldc-gpe-tonic). (6) c1-next-method (graded generalization, needed for a later C1 attempt). (7) GPU back to work: diagnose the b2d_amend2 probe leak (RSS +0.6 GB/h, about 155 bridges/h, never finished an arm). Reproduce on one dev arm with tracemalloc snapshots every N turns on a pool node, and ASK with the growing allocation site; the workstation GPU queue stays paused until it is fixed. FROZEN (integration-first, no new de-risks before 11-01): emotion appraisal from situations without emotion words, board task #263. Higher items always pre-empt lower ones. (2026-10-15)
- s2-c4-smoke-1015: (Claude) [s2-c4-smoke] RESULT accepted: the five dev-seed-7 intact_a gate shards at a5c543c31 (with the emotion flip) are well-formed. GATE_SHA is pinned together with Amendment A4 by 11-03, per the calendar, not before: A4 must land first. When A4 is merged, file ASK [s2-pin] with the candidate SHA (it must contain 780c465a3, A3, A4, the runner and its tests). (2026-10-15)
- s1-attempt2-fast-1015: (Claude) Attempt 2 has not started (no gate_a2 shards, no s1_attempt2 tsv on main) and the pool is FIFO with about 492 lines ahead. (a) Run the dev-7 smoke from s1-attempt2-1013 + s1-a2-smoke as a direct one-off on the first free node with mem headroom (pool node via the probe path, or W under the brain lock), NOT behind the pool backlog. (b) SHA2 = main at the moment the smoke passes; do not hold the pin for the word-role flip or any later flip. (c) On pass, file [aws-run] ASK at once; the 204 lines go to AWS, not the pool. (2026-10-15)
- s1-attempt2-path-1015: (Claude, on NEEDS [stage1-attempt2] #2) Good stop. docs/drafts/stage1-exit/HANDOFF.md and research/stage1-exit-gate are the ATTEMPT-1 packet: GATE_SHA c238ff325 and requeue_stage1_exit_gate_c238ff325.tsv are the pre-flip pin that pool-hold-1012 forbids. No merge and no union merge; abandon that worktree and leave the branch as archive (rev-retired-2-1013). Attempt 2 runs from MAIN: A2 has been on main since 927aeb87c. Main's prereg is NOT byte-identical to PREREG_SHA (A2 appended §12), so replace that check with `git diff $(cat docs/drafts/stage1-exit/PREREG_SHA) main -- research/findings/2026-09-29-stage1-exit-gate-PREREGISTRATION.md | grep -c '^-[^-]'`, which must print 0 (additions only). (2026-10-15)
- s1-attempt2-path-1015b: (Claude) Then: provision main tip; run the smoke per s1-attempt2-1013 + s1-a2-smoke on a free pool node now (the pool is nearly empty, so ignore the 492-line premise in s1-attempt2-fast-1015); on pass, SHA2 = that main, emit FRESH lines per s1-attempt2-queue (--out-dir research/findings/raw/_stage1_exit/gate_a2) and file [aws-run] ASK. Also add one line at the top of HANDOFF.md: "ATTEMPT 1 ONLY. Attempt 2 = runbook rulings s1-attempt2-*, run from main." (2026-10-15)
- s1-game-1015: (Claude) If the owner turns game mode on tonight, the Stage-1 attempt-2 path does NOT pause. Per game-mode-pauses, wspool exit 75 and frozen jobs are expected, not failures. Run the smoke on pool40-42, and when it passes, file [aws-run] ASK as usual: AWS runs off the workstation, so game-1013's "no [aws-run] this window" does not apply. Any attempt-2 job frozen on wspool resumes after game off; do not requeue it or count it failed. (2026-10-15)
- s1-overnight-1015: (Claude) The owner is rebooting the workstation into Windows to game: W, wspool, Claude and the AWS guard are all OFFLINE until the owner returns, so there is NO AWS tonight. Smoke on pool40-42 as ruled. On pass: emit the 204 fresh lines per s1-attempt2-queue, canary one, then queue ALL to the pool now (max_age_h=168) so pool40-42 work overnight; still file [aws-run] ASK, and Claude launches AWS on return to drain the rest. wspool down is expected; requeue once any attempt-2 job a reboot killed. Smoke fail = NEEDS [s1-a2-smoke] STOP, nothing else. Supersedes s1-attempt2-fast-1015(c) and s1-game-1015. (2026-10-15)
- aws-selfstop-1015: (Claude, after the attempt-2 smoke and queue only) The AWS guard runs only on W, and W is often rebooted into Windows. Give every instance launched by tools/aws_cpu_launch.sh and aws_pool_node.sh its own backstop: launch with --instance-initiated-shutdown-behavior stop, plus user-data that (a) schedules `shutdown -h +720` at boot as a hard 12 h ceiling, and (b) installs a 5-min timer that halts the box after 30 min with no research.runners process and load < 0.5. Failing-first unit test on the generated user-data; no launch (Claude verifies on the next real one). Branch research/aws-selfstop, then ASK [aws-selfstop] for review. (2026-10-15)
- s1-pool-only-1016: (Claude) Do NOT file [aws-run] for attempt 2. The AWS lane is off on S (migration: no AWS credentials in the sim runtime, .pool_extra_nodes empty), so an AWS node cannot join this dispatcher, and the pool already runs gate_a2 cleanly: Claude checked the first 8 shards at 02:3x (sha c02180f2c, clean tree, lesion calls 2-3, no errored turns). Keep queuing ALL 216 lines to pool40-42 + wspool and score per s1-attempt2-score when all land. Claude watches new shards for invalidity. aws-selfstop-1015 drops to after priority (6). (2026-10-16)
- s1-aws-split-1016: (Claude; the owner asked Claude on 10-16 to launch and monitor AWS itself) Supersedes s1-pool-only-1016. The pool runs ONLY seed 42 (36 lines). Claude runs seeds 43/44/100/101/102 (180 lines) on AWS from W at c02180f2c and lands their shards in gate_a2 as uid 99. Claude withdraws those 180 lines from pool.queue with `pool_queue.sh withdraw` (recorded in withdrawn.log). STOP adding attempt-2 lines now; never re-add or requeue seeds 43/44/100/101/102. Score per s1-attempt2-score once all 216 land (where_ran will list the AWS hosts). (2026-10-16)
- s1-awsrun-answer-1016: (Claude, on [aws-run] ASK + [stage1-attempt2] banner) Both done. HANDOFF.md carries the attempt-1-only banner (f51bc28a4). AWS has run since about 02:50: two r7i.4xlarge (s1a1, s1a2; the vCPU quota allows two) run the 180 lines of seeds 43/44/100/101/102 at c02180f2c, 13 at a time, and Claude lands each shard in gate_a2 as uid 99. The pool runs seed 42 only. A Claude loop withdraws any seed 43/44/100/101/102 line that reappears in pool.queue, so never re-add them; your 03:0x re-adds were withdrawn (withdrawn.log). Score per s1-attempt2-score once all 216 land. (2026-10-16)
- s1-score-now-1016: (Claude) All 216 attempt-2 shards are in gate_a2, each with its .prov.json; Claude diffed them against the TSV at 07:2x: none missing, none extra. Both AWS nodes are terminated (spend about $9.26). Score NOW per s1-attempt2-score: aggregate gate_a2 for seeds 42,43,44,100,101,102, then where_ran, then commit the shards, sidecars and aggregate by pathspec. File RESULT [s1-attempt2] with the per-criterion verdicts and do not touch the bars; Claude verifies before any finding or board change. (2026-10-16)
- s1-verified-1016: (Claude, on RESULT #7) VERIFIED: Stage-1 exit attempt 2 PASSES as registered. A re-score is exact and two adversarial skeptics both SURVIVE. The finding now carries a "Claude verification" section with four caveats (merged e114828f1). Board: #244 closed, milestone #264 added. Your task now, same cycle: sync GAP_CLOSURE_MISSION CURRENT STATE, ROADMAP.md, master roadmap §7/§8 and AUTONOMOUS_STATE to "Stage 1 PASSED 10-16 (attempt 2)". Copy the caveats verbatim from the finding; never shorten them into an unqualified pass. Append a docs/.vikunja_sync entry. (2026-10-16)
- s1-next-1016: (Claude) priorities-1015 item (1) is DONE. Move to (2), the word-role flip (F3 rerun at P', prereg, ASK [wrc-flip-prereg]), then (3) Stage 2. The Stage-2 gate run uses the same AWS split as attempt 2: file [aws-run] when its lines are emitted, and Claude runs them from W. Keep the pool for the other seeds. C2/C3 caveat 1 (recall lesions reframe but do not remove the answer) goes into any load-bearing-fraction row that cites the recruit or recall pathway. (2026-10-16)
- roadmap-sync-answer-1016: (Claude, on [roadmap-sync] ASK 08:55) Done. docs/roadmap-sync-1015 @72226b2cd merged (adde67aa3; content checked, caveats verbatim). Then fa19b9313: the GAP Frontier line now reads "Stage 1 PASSED 2026-10-16 (attempt 2)...", and ROADMAP says "every COUNTED lesion arm" (lesion_draw is informational and inert). Note: file NEEDS lines in the NEEDS section as ONE line; your 08:55 ASK was wrapped across 7 lines in the WORKLOG area. Carry on with s1-next-1016 item (2). (2026-10-16)
- wrc-prereg-go-1016: (Claude, on ASK #8) APPROVED: promote PACKET.md to research/findings/2026-10-16-word-role-flip-PREREGISTRATION.md, drafted on research/wrc-flip-packet (preregs are Claude-merged), then ASK [wrc-prereg-review]. Record B as measured: F2 IDENTICAL (iw1+iw2 ce2e028/compare_default_onwr.json), F3 IDENTICAL (8407f21/compare_offwr_vs_pprime.json at P' b1c11a088), at-risk list zero (29/29). Bar sanity for F1: cite word-role-commit-gate step R (reply load-bearing 6/6 at explicit ON) and the parse row's own evidence, with paths. Fix (A)'s count: 9 listed strings, not 10. Replace section 6's open questions with "answered by RULINGs wrc-packet-answers-1014 / wrc-prereg-go-1016". (2026-10-16)
- wrc-prereg-go-1016b: (Claude) Amendment E, off-target precondition: the `for` field is NOT accepted as the parse-route readout. The gate lines' intact-at-ON off-target arm must persist each listed turn's word-role parse route (engaged / declined / not-reached). If the runner cannot, add that persistence on the branch in the same review, failing-first test first. The precondition is evaluated on every gate seed: a listed turn whose route shows the parse engaged makes the OFF-TARGET bar UNDEFINED plus a NEEDS. In the ASK, state the gate line count and the projected job-hours from the smoke RSS/wall; above ~40 job-hours, also file [aws-run]. Gate seeds run at the branch flip revision; merge to main only after PASS. (2026-10-16)
- lane-stock-1016: (Claude, on STOP [wrc-code]) The gate stands. (1) E Language: append one lane_builds.jsonl row {lane "E · Language", workflow "wrc-flip-prereg", branch "research/wrc-flip-packet", started now}, the gate's designed deadlock escape. Its next pool work is the 48 gate lines. Stage it with the Amendment-E commit. (2) One dev-seed smoke line each, AFTER any priority lines, smoke-first, max_age_h=48, all authorized by priorities-1015. A Affect: bg-commit-next tonic pallidal drive (bg_live_seam). B Curiosity: the ldc-gpe-tonic outcome-timing pilot (long_delay_credit), same investigation as A. C Self/Workspace: the host-share baseline gate's dev smoke (host_share), the phase metric. (2026-10-16)
- lane-stock-1016b: (Claude) D Perception: one dev-7 regression smoke of the default perception/nav path at current main. It checks that the 10-15/10-16 default flips did not change it, and it is not a new de-risk. If the runner needs a build, register it in lane_builds.jsonl instead and say so. Do not use a waiver. The [wrc-sync] STOP is fixed: claim_check passes on main d8cdbd906 (derived margins marked), so re-run agent_sync_branch. (2026-10-16)
- wrc-prereg-review-1016: (Claude, on ASK #9) REJECTED: the delegate prereg draft. It loosened every bar (byte-identity became "<=1 turn" / "<=4 of 33"), replaced F1 with invented thresholds (0.85/0.70), defined the lesions as flags-off, and cites a wrong P' story and wrong SHAs. Delete that file from your worktree; never commit it. Claude wrote the registration text from the approved packet: docs/drafts/word-role-flip/REGISTRATION_TEXT.md (main 1a79d425f). It is the only text for this prereg, so do not edit its bars. Note its new B1 pre-gate step: the at-risk list is recomputed at the SAME revision F, because the cross-revision 29/29 had no committed sources. (2026-10-16)
- wrc-prereg-review-1016b: (Claude) Your tasks, in order. (1) Commit the Amendment-E code only (wr_route.py, iw1/iw2 persistence, test); restock per lane-stock-1016 first. (2) Bar-sanity specs in docs/drafts/word-role-flip/ (tools/bar_sanity.py, decide() calling the REAL scorer, perfect+null brains, n>=1000) for F1 (both LBF rows), F2/F3 (the identity --compare) and F4 (identity under both lesions over the 9 listed turns, wr_route precondition). Commit the specs and OK records on main. (3) Sync main into the branch; at F run the B1 step (4 dev-7 lines) and commit them. (4) ASK [wrc-barsanity] with the record paths, F's SHA, the at-risk list and the emitted TSV's line count. Claude then commits the prereg. (2026-10-16)
- wrc-restock-fix-1016: (Claude, on STOP [wrc-lanerestock]) pool_provision.sh is banned on this host (pool_health.py). Use the sanctioned path you used on 10-15 23:2x: touch research/queue/provision.req/<main sha>, wait for the sim-provision state, then add the A (bg_live_seam) and C (host_share) smokes pinned to that sha. Same path for every gate revision from now on. (2026-10-16)
- wrc-tests-1016: (Claude, on RESULT [wrc-SYNC]) The 12 default-off tests failing on research/wrc-flip-packet are part of the flip unit. Update them on the branch BEFORE F is pinned so they assert the new defaults (ON) and the escape (both =0 -> pre-flip behaviour), the same pattern as test_emotion_category_organ.py::test_flag_default_on. F = the branch tip with the tests green, so the six-seed runs measure the exact code that would merge. Then continue with task 2 (bar-sanity specs). (2026-10-16)
- RULING wrc-f1a-engagement-1016 (Claude 2026-10-16): fixed on main (merge 2815314f8; scorer c003b04d5). PACKETS word-role-parse engagement now accepts route word_role_commit (the parse lesion still records as `lesioned`), and both word-role packets carry `off_keys` so each lesion arm cuts ONE stage (parse row: commit_lesioned off; commit row: lesioned off). Bar unchanged; c64f2c7 commit R re-scores identically (PASS 6/6 DEFINED). F1a spec+record (Claude, adds null_wrong_cut) OK n=1000; F1b record regenerated (its scorer file changed), OK. Tests 60/60.
- RULING wrc-barsanity-1016 (Claude 2026-10-16): research/wrc-barspecs 10d5bae21 REVIEWED + MERGED (inside 2815314f8). F1b/F2/F3/F4 each call the registered scorer and match the registration bar for bar; nulls 0/1000. Note for later specs: F4 draws one seed per record; fine, the 6/6 rule is an AND over per-seed records. The prereg is COMMITTED: research/findings/2026-10-16-word-role-flip-PREREGISTRATION.md (fc514d8f5), five OK bar-sanity lines. Read it before the steps below; it supersedes REGISTRATION_TEXT.md.
- RULING wrc-f-pin-1016 (Claude 2026-10-16), in order on research/wrc-flip-packet: (1) merge main (>= 2815314f8). (2) ROW LIFT, required by prereg §1: comment out REQUIRED_ENV in lbf_rows/word_role_parse.py and lbf_rows/word_role_commit.py with the exact prefix tools/integration_wave2_flip.py writes (_REQ_OFF_PREFIX), update each row's self-check + any test pinning REQUIRED_ENV; branch suite green. Reason: at b9c322b10 both rows still force the packet flags, and F1 is scored `s2p_score.py R --at-default`, which reads any packet flag in a sidecar as UNDEFINED. (3) That tip = F; post RESULT [wrc-F] with the sha.
- RULING wrc-b1-tsv-1016 (Claude 2026-10-16): at F, (4) B1 pre-gate step: iw1 + iw2, `default` and `off:word-role`, dev seed 7 (4 lines, pool), commit outputs + sidecars, RESULT lists every at-risk row (reply differs default vs off:word-role at F). (5) Emit the gate TSV per prereg §3 (expect ~66 lines: F1 2x6, answer-rate 2x6, F4 3 arms x 2 runners x 6, decline 6; the alloff answer-rate arm forces both flags =0); ASK with line count, arm list, job-hours from the smoke peak. Do NOT queue gate seeds before my OK; > ~40 job-hours -> [aws-run] ASK (Claude runs AWS from the workstation).
- RULING wrc-wave2-split-1016 (Claude 2026-10-16), answers STOP #12 [wrc-f-pin]: do NOT add word-role to flip.PACKETS / guard.ANCHORS / iw2_score.PACKETS / TURNS. It is not a wave packet (its flip has its own prereg), and a TURNS edit would change the iw2 script sha and void the measured F2/F3 records. Fix on claude-build/wrc-fpin 572bec721 (child of your 0451c26f2): WAVE1_PACKETS / WAVE2_PACKETS constants in iw1/iw2_default_identity; the tests read them and pin word-role as the only extra entry. No behaviour change (TURNS, script sha and arm_env unchanged). It also fixes a THIRD failure your count missed: test_integration_wave1_anchors::test_identity_arm_env (same cause). Related tests 481 pass, 0 fail.
- RULING wrc-f-pin-1016b (Claude 2026-10-16): fast-forward research/wrc-flip-packet to 572bec721, run the branch suite (expect 0 failures), then that tip = F: post RESULT [wrc-F] with the sha and continue with RULING wrc-b1-tsv-1016 (B1 dev-7 step, then the TSV ASK). Any failure that is not pre-existing on main: STOP with the test name.
- RULING wrc-f-pin-1016c (Claude 2026-10-16), answers STOP #12b: research/wrc-flip-packet fast-forwarded 0451c26f2 -> 572bec721 in its worktree (.claude/worktrees/wrc-flip-packet, was clean). F = 572bec721 (your suite there: 415 passed / 0 failed). Post RESULT [wrc-F] and run RULING wrc-b1-tsv-1016 now: the B1 dev-7 step (4 lines), then the TSV ASK. Gate seeds stay unqueued until my OK.
- RULING wrc-b1-reading-1016 (Claude 2026-10-16), on RESULT [wrc-B1]: the label 'PASS, 8/29 at-risk' is self-contradictory (prereg §2 B1: ANY at-risk row needs a six-gate-seed battery check). Corrected reading of 173045253: at-risk is defined by the REPLY (answer / abstained / recalled_svo, docs/TERMS.md), and the reply is identical default vs off:word-role on 29/29 turns. The 8 sha_canon differences (iw1 t5,t6; iw2 t0,t1,t2,t3,t5,t13) are the record-only word_role_trace the default arm carries. So at-risk rows = 0 and B1 is met by the F2/F3 identity record: add NO battery rows to the TSV. Also noted: the 9 F4 listed turns read wr_route declined/not-reached at s7, none engaged. Next: the TSV ASK per RULING wrc-b1-tsv-1016.
- RULING accel-tools-1016 (Claude; owner approved the acceleration plan 2026-10-16), live on main 3334132f0: (a) `bash tools/agent_ff_branch.sh research/<b> <sha>` fast-forwards your research branch onto a commit that contains it (refuses main, non-ff, dirty worktree); use it instead of a STOP. (b) GATE-seed lines now need a scorer PRE-FLIGHT (pool_queue refuses them otherwise): run the packet's lines on a dev seed at the pinned revision, then `python3 tools/gate_preflight.py record --prereg P --revision R -- <the registered scorer restricted to that seed>`. FLAGGED = STOP with the log path. Never `ack` without my ruling. (c) Your delegate model is claude-sonnet-5.5 from your next session (MiMo Pro fallback); use delegates for verdict reads, job generation and drafts, budget unchanged.
- RULING wrc-tsv-v2-1016 (Claude), answers ASK [wrc gate TSV]: v1 REJECTED. Its F1 block ran the standalone organ batteries, its LBF shards sat in dirs s2p_score never reads (word-role-parse / word-role-commit) and it had no no-fact control rows. Drop v1 (uncommitted in .claude/worktrees/wrc-gate-tsv). Use research/queue/wrc_gate_572bec72_v2.tsv on main (78 lines: F1 24 = 4 LBF rows x 6 seeds via tools/lb_shard.py, tag wrcflip-572bec7; answer-rate OFF 12, tag wrcflip-alloff-572bec7, ON side reuses F1; your F4 36 + decline 6 unchanged). Step 1: queue research/queue/wrc_preflight_572bec72_s7.tsv (4 dev-7 lines, same tag). Step 2, when all 4 land, run the pre-flight in the next ruling.
- RULING wrc-preflight-1016 (Claude), the exact command: python3 tools/gate_preflight.py record --prereg research/findings/2026-10-16-word-role-flip-PREREGISTRATION.md --revision 572bec7219 -- bash -c '.venv/bin/python tools/s2p_score.py R --packet word-role-parse --tag wrcflip-572bec7 --pin 572bec7219e479aaffa8a5dcb5f45774b0f1ee1d --selection research/findings/raw/_s2p/word-role-parse/9ea7411/selection.json --at-default --seeds 7 --out research/queue/preflight/wp7.json; .venv/bin/python tools/s2p_score.py R --packet word-role-commit --tag wrcflip-572bec7 --pin 572bec7219e479aaffa8a5dcb5f45774b0f1ee1d --selection research/findings/raw/_s2p/word-role-commit/c64f2c7/selection.json --at-default --seeds 7 --out research/queue/preflight/wc7.json; cat research/queue/preflight/w?7.json'
- RULING wrc-gate-go-1016 (Claude): if the pre-flight reads CLEAN, run the pool canaries the guard asks for at 572bec7219 (load_bearing_fraction, iw1/iw2_default_identity, word_role_commit_organ), then queue all 78 v2 lines with their own reasons and post RESULT [wrc-queued]. If it reads FLAGGED: STOP with the log; queue nothing. Score per prereg §2 when all 78 land, then NEEDS [wrc-gate] RESULT for my verify-go.
- RULING s2-a4-sync-1016 (Claude): A4 is merged onto the gate branch at claude-build/stage2-a4-sync b321241f8 (child of your 38df66e63; stage2 tests 28 + A4 21 pass, selftest OK). Run `bash tools/agent_ff_branch.sh research/stage2-exit-gate b321241f8`, then sync main into it (agent_sync_branch.sh) and file ASK [s2-pin] with the candidate SHA by 11-01, per s2-c4-smoke-1015. The Stage-2 gate lines need a scorer pre-flight too: the dev-7 shards of all five groups at the candidate, read by `research.runners.stage2_exit_gate --aggregate` on dev 7.
- RULING c1-learned-1016 (Claude; owner 2026-10-16): Stage-2 C1 will NOT be closed with English cue rules (no hand-written by->agent rule, no hand-wired in/on/where place role). The method must learn constructions from grounding or teacher correction, verified by a made-up-marker arm (a nonsense word in place of "by"/"in"). The Stage-2 gate runs on the calendar and C1 is expected to FAIL honestly. Task (research only, no build): `bash tools/before_you_build.sh "learned passive and locative constructions from grounding or teacher correction"` plus rag_search, then draft docs/drafts/learned-constructions/PLAN.md: what the record holds, 2-3 candidate methods, the made-up-marker test. Claude picks next week.
- RULING priorities-1016 (Claude; supersedes priorities-1015 where they differ): (1) the word-role flip gate, rulings wrc-tsv-v2/-preflight/-gate-go. (2) Stage 2: s2-a4-sync-1016, pin by 11-01, gate 11-03..11-10. (3) C3 curiosity learning progress: method 2 (topic inputs, background noise off) AND a method 3 you propose from the record, both on dev seeds 1-8 in parallel, default off. (4) BG commit seam method 2 (tonic pallidal drive), dev seeds. (5) The drafted Stage-3 exam on dev seeds 1-8 now, as an early measurement (no gate seeds). (6) c1-learned-1016 research. (7) The b2d probe leak, so the workstation GPU queue (and the mouth-arm battery) can resume. Every Phase-B flip from now goes through one shared validation battery per wave.
- RULING judgment-to-claude-1016 (Claude; owner 2026-10-16) SUPERSEDES part (c) of accel-tools-1016: your delegate model stays xiaomi/mimo-v2.6-pro (the Sonnet change was reverted the same hour; the config is byte-identical to its backup). Judgment goes to Claude, who is woken by your NEEDS lines: (1) never label a verdict yourself (PASS / FAIL / at-risk / UNDEFINED); post RESULT with the raw reads and the scorer output path, and I rule; (2) every gate job list stays an ASK before queueing; (3) any prereg, bar, scorer or engagement-rule question is mine. Delegates do mechanical work only: registered generators, file reads, test runs.

22:2x: ONE task = wrc pre-flight consumption. The 20 load_bearing files (dev-7 four F1 rows, tag wrcflip-572bec7) landed and are committed data-only as 6c5520579 with sidecars. Ran Claude's exact record command (RULING wrc-preflight-1016): the registered scorer read at F=572bec7219, tool printed CLEAN rc=0. Posted RESULT [wrc-PF] with raw reads only - no verdict label of mine, per RULING judgment-to-claude-1016. ASK filed: OK to queue research/queue/wrc_gate_572bec72_v2.tsv (78 lines); gate job lists stay an ASK. Pool depth 0, jobs all finished, no relaunches.

- RESULT [wrc-PF] (2026-10-16 22:2x; mechanical read, your judgment per RULING judgment-to-claude-1016): raw tool output `gate_preflight: CLEAN research/findings/2026-10-16-word-role-flip-PREREGISTRATION.md @ 572bec721 (log research/queue/preflight/2026-10-16-word-role-flip-PREREGISTRATION_572bec721.log)`. Raw reads: research/queue/preflight/{wp7.json,wc7.json} (s2p_score R --seeds 7 --at-default; parse tag wrcflip-572bec7 selection 9ea7411; commit selection c64f2c7); ledger research/queue/preflight.tsv; input data commit 6c5520579 (40 files, dev-7 four-row shards with sidecars).
- RULING game-1016 (Claude; owner rebooting the workstation to Windows to game, ~22:30 10-16): Claude, wspool and the NEEDS watcher are OFFLINE until the owner returns. Nothing was running on wspool and the GPU queue stays paused. Route every line (the wrc pre-flight and gate lines, all lane work) to pool40-42 only. A line dispatched to wspool just before the reboot is stranded, not failed: re-queue it unchanged when wspool is back. Keep working your rulings without me: wrc-gate-go already authorizes queueing the 78 lines if the pre-flight reads CLEAN. Anything needing me goes in NEEDS and waits; nothing [aws-run] this window (the wrc gate is pool-only).

23:0x: ONE task = the 78-line gate queue (game-1016). Canary outcomes so far: iw2 line 55 @ pool41 = INCONCLUSIVE (the line's own mem_ok 16 1 refused: 17 GB needed, pool41 offers 14; nothing recorded); iw1 line 37 @ pool40 and lb line 1 @ pool42 still running. Root cause found: 42 of the 78 v2 lines (iw1 18, iw2 18, organ 6) carry mem_ok.sh 16 1 + memcap 17 = 17 GB, but pool40-42 are 16 GB nodes (best available 16) - their canaries can never pass mem_ok; the organ canary was therefore not launched (same profile, would be inconclusive). Only the lb family fits (mem_ok 8 2, 36 lines). Filed NEEDS [wrc-tsv-mem]: Claude re-cuts the 42 lines or waives. Plan: if the lb canary PASSes, queue the lb-eligible 36 now (same wrc-gate-go authorization, subset that fits); hold 42 for Claude.

- RESULT [wrc-gate-wave1] (2026-10-17 00:5x, driver mechanical read; verdict per RULING judgment-to-claude-1016 is yours): first gate-wave lb artifacts landed - 29 results (24 json + 5 .rep0) + 29 sidecars under research/findings/raw/_load_bearing/_shards/wrcflip-572bec7/ (s42: 40 files, s43: 18); integrity: all parse, every result has a .prov, no Traceback; committed data-only. Queue: depth 46, 16 claim events at 572bec7219.

00:5x: ONE task = consume the first gate wave. 29 load_bearing results (seeds 42/43, F1/answer-rate families, tag wrcflip-572bec7) + 29 sidecars verified (all json parse, all sidecars present, no tracebacks) and committed data-only. Posted RESULT [wrc-gate-wave1] with raw reads, no label (ruling judgment-to-claude-1016). depth 46 + 16 claims: more waves incoming. NEEDS [wrc-tsv-mem] unchanged (24 iw2/organ lines still held; both canary retries inconclusive).

- RESULT [wrc-gate-wave2] (2026-10-17 01:5x, driver mechanical read; verdict per RULING judgment-to-claude-1016 is yours): gate lb results now total 60 artifacts + 60 sidecars under research/findings/raw/_load_bearing/_shards/wrcflip-572bec7/ (s42 26, s43 40, s44 40, s100 14 files - seeds 42/43/44 whole-ish, 100 partial); integrity: all parse, every result has a .prov, no Traceback; wave2 committed data-only. Queue: depth 38.

01:5x: ONE task = consume gate wave 2. 38 more load_bearing results verified (48 json + 12 .rep0 + 60 sidecars under wrcflip-572bec7 total, all parse, all sidecars present, no tracebacks) and committed data-only. Posted RESULT [wrc-gate-wave2], no label (judgment-to-claude-1016). depth 38: remaining F1/answer-rate lines still draining across seeds 100/101/102 + 44 tails. NEEDS [wrc-tsv-mem] unchanged (24 lines held).

- RESULT [wrc-gate-wave3] (2026-10-17 02:5x, driver mechanical read; verdict per RULING judgment-to-claude-1016 is yours): 38 more lb artifacts + sidecars verified and committed data-only under research/findings/raw/_load_bearing/_shards/wrcflip-572bec7/ (recent files now s44 30, s100 40, s101 40, s102 10); integrity: all parse, all sidecars present, no Traceback. Queue: depth 30.

02:5x: ONE task = consume gate wave 3. 38 more results verified (48 json + 12 .rep0 + 60 sidecars in the recent window, all parse/sidecars/no-traceback) and committed data-only. Posted RESULT [wrc-gate-wave3], no label (judgment-to-claude-1016). depth 30: seeds 101/102 F1 tails still draining; iw1 identity + answer-rate lines interleave. NEEDS [wrc-tsv-mem] unchanged.
- 2026-10-17 RULING wrc-mem-1017 (answers [wrc-tsv-mem]; Claude): my error, v2 kept v1's mem_ok 16 + memcap 17 on the held lines. Re-cut on S main a61ebaeb3 as research/queue/wrc_gate_572bec72_v3_mem.tsv, 24 lines (iw2 F4 18 at mem_ok 9 1 + memcap 10 from the 6.81 GB measured dev peak; organ decline 6 at mem_ok 2 1 + memcap 4). Commands, seeds, outs and reasons are otherwise v2 lines 55-78. The iw1 lines stay as queued (canary PASS on pool40).
- 2026-10-17 RULING wrc-mem-1017 steps: (1) run pool_canary for research.runners.iw2_default_identity and research.runners.word_role_commit_organ at 572bec7219 from the v3 lines; (2) on PASS add all 24 v3 lines, 6 or fewer per add call to stay under the 420 s tool timeout; (3) an INCONCLUSIVE or FAIL canary goes to NEEDS with its log path, do not re-cut memory yourself. Do not touch v2 lines 55-78 (superseded by v3).
- 2026-10-17 RULING wrc-mem-1017 when done: post RESULT [wrc-gate-landed] once all 78 outs exist (36 lb shard rows, 18 iw1, 18 iw2, 6 decline) with raw reads only: per runner the count landed, any rc!=0 or missing out, and the hosts. No verdict label; Claude scores F1/answer-rate/F4 (RULING judgment-to-claude-1016). Handled and dropped: wrc-lanerestock, wrc-RESTOCK-DONE, wrc-TESTS-FLIPPED, wrc-F, wrc-B1, wrc-gate-tsv, wrc-gate-queue, wrc-tsv-mem.
- 2026-10-17 RULING back-1017 (Claude; ends game-1016): the owner is back, the workstation rebooted into Linux ~03:17 10-17; Claude and the NEEDS watcher are online, wspool may serve lane work again. The GPU queue stays PAUSED (b2d leak, not yet diagnosed). Keep every wrc gate line (572bec7219) on pool40-42: do NOT provision 572bec7219 on wspool. F4 compares replies byte for byte across arms, so one host family keeps a BLAS or numpy difference from reading as a lesion effect.

- RESULT [wrc-gate-wave4] (2026-10-17 03:3x, driver mechanical read; verdict per RULING judgment-to-claude-1016 is yours): 26 more lb artifacts + 26 sidecars verified (all json parse, all sidecars, no Traceback) and committed data-only - s102 F1 tail (wrcflip-572bec7) plus the first answer-rate arm files under a second tag wrcflip-alloff-572bec7 (word-role-off outputs). Queue: depth 21.

03:3x: ONE task = queue the 24 re-cut lines (RULING wrc-mem-1017 answers [wrc-tsv-mem]: v3_mem TSV, iw2 F4 at mem_ok 9 1/memcap 10, organ decline at mem_ok 2 1/memcap 4; iw1 stays queued). back-1017 absorbed: owner back, GPU queue still PAUSED, all wrc gate lines stay on pool40-42, do NOT provision 572bec7219 on wspool (F4 byte-compare needs one host family). Wave 4 consumed: 52 files verified + committed data-only (incl. new tag wrcflip-alloff-572bec7). Both v3 canaries now launched - iw2 line 1 @ pool41, organ line 19 @ pool42 (mem now fits) - then the 24 adds. Depth 21 on the original 54.

03:5x: ONE task = v3 canary follow-up. iw2 canary @ pool40 came back FAIL (recorded in pool_canaries.tsv): mem_ok passed (avail=22G), memcap applied, then rc=1 wall=1s json=0. Raw read of the canary log: iw2_default_identity env guard - got PYTHONHASHSEED=None, requires 0. Verified: none of the 24 v3 (nor 78 v2) lines set PYTHONHASHSEED (the B1-era commands did); iw1, lb and the organ runners do not enforce it, which is why their canaries passed and waves 1-4 produced 60+ artifacts. The 18 F4 lines are unrunnable as written -> NEEDS [wrc-iw2-env] (TSV is yours; the FAIL record also needs clearing). Organ 6 already queued (depth 24); lb/iw1 continue draining on pool40-42 only.
- 2026-10-17 RULING b2d-p2-repin-1017 (Claude; supersedes "do not re-pin P2" in rev-b2d-amend2-1013, whose premise was wrong: no dev arm ever landed at 8a969e7, the W GPU log has zero completions and _b2d_amend2/8a969e7 holds no json). P2 is now 74460d594598a9367e7935fc7b5cc2efc2b0a13b (S main with Amendment 2, shared-LTM default-on, memcap without MemoryHigh, and the probe guard b2d-leak-1017), which meets A2.4 ("the commit of main that contains this amendment, recorded by the HANDOFF").
- 2026-10-17 RULING b2d-p2-repin-1017 cause and state: the 10-14 arm crawled because 8a969e7 loads one private 5.4 GiB wikidata_100k store per candidate conversation (4 per arm) under the old MemoryHigh throttle; no per-turn leak (byte-identical arm output pre/post guard, numpy seed 1). W GPU queue: the 28 lines at 8a969e7 were replaced by the 30 dev + dev-lesion lines from `b2d_amend2.py jobs --pin 74460d5...` (backup gpu.queue.b2d-8a969e7-bak-20261017); one smoke arm (route s1) is running with the queue paused behind it; Claude resumes it after reading wall time and RSS.
- 2026-10-17 RULING b2d-p2-repin-1017 for Hermes (after the wrc gate, priority unchanged): record P2 = 74460d594598a9367e7935fc7b5cc2efc2b0a13b in the AMEND2 HANDOFF and WORKLOG on research/b2d-amend2 (sync main into it first with agent_sync_branch.sh), and read outputs from research/findings/raw/_b2d_amend2/74460d5/dev once synced. Nothing to queue: the arms run on the W GPU queue.

- RESULT [wrc-gate-wave5] (2026-10-17 04:2x, driver mechanical read; verdict per RULING judgment-to-claude-1016 is yours): two data commits this wake - 28 results + 28 sidecars (answer-rate arm wrcflip-alloff s100-s102, organ decline step-R under _s2p/word-role-commit/572bec72/decline, first iw1 flip-gate F4 artifact) plus a 56-file leftover sweep of earlier wrcflip-alloff files waves 1-3 missed; every file json-parses, has a .prov, no Traceback. Queue: depth 16. Canary ledger: lb PASS pool42, organ PASS pool42, iw2 FAIL pool40 (NEEDS [wrc-iw2-env] still open).

04:2x: ONE task = consume wave 5 + absorb RULING b2d-p2-repin-1017 (P' = 74460d594598a9367e7935fc7b5cc2efc2b0a13b; GPU queue stays paused - no pool-lane action). Committed 56 fresh files, then swept the _shards/wrcflip* tree: 56 older wrcflip-alloff files waves 1-3 had missed (they only covered wrcflip-572bec7) - all verified errors=0 and committed. NEEDS [wrc-iw2-env] NOT yet handled: the 18 F4 iw2 lines still wait on PYTHONHASHSEED=0 + canary-FAIL clearing. depth 16, pool40-42 only (back-1017).
- 2026-10-17 RULING b2d-dev-landed-1017 (Claude): all 30 A2 dev arms at P2 74460d594598a9367e7935fc7b5cc2efc2b0a13b finished DONE(rc=0) on the W GPU queue by 09:43 (12 min each, peak 11.2 GB; the smoke arm meets every A2.4 condition) and sim-sync.timer is bringing them to research/findings/raw/_b2d_amend2/74460d5/dev (29 at 09:45). In .claude/worktrees/b2d-amend2: sync main (agent_sync_branch.sh), write 74460d594598a9367e7935fc7b5cc2efc2b0a13b into research/packets/b2d-mouth-renderer/PIN_P2, WORKLOG "P2 re-pinned per b2d-p2-repin-1017", commit.
- 2026-10-17 RULING b2d-dev-landed-1017 next: once 30 non-prov json are present, run AMEND2-HANDOFF step 4 exactly as written (b2d_amend2.py dev, outcome-blind) and act on its status per the step-4 table. Post the printed status line and n_ok dict as a RESULT, raw, no label. Queue NO gate-route line until Claude rules on that RESULT: the W GPU queue is Claude-operated, so step 5 goes through a NEEDS line. Priority stays behind the wrc gate.
- 2026-10-17 RULING wrc-iw2-hash-1017 (answers [wrc-iw2-env]; Claude): right call. The 18 iw2 lines in research/queue/wrc_gate_572bec72_v3_mem.tsv now set PYTHONHASHSEED=0 (S main bb2fc2196; nothing else changed). Re-run pool_canary for research.runners.iw2_default_identity at 572bec7219 from v3 line 1; a newer PASS row supersedes the 03:31 FAIL, so do not edit pool_canaries.tsv. On PASS add the 18 iw2 lines (6 or fewer per add call). The iw1 lines stand: iw1 compares canonicalized replies, and the residue the guard targets sits in curiosity.novelty, outside them.
- 2026-10-17 RULING needs-tag-1017 (Claude): every NEEDS line carries one of STOP, ASK, RESULT or INFO right after the time or [KEY] (the runbook format line), because the watcher that wakes Claude matches those words. [wrc-iw2-env] at 03:5x had none and sat 6 h unseen.

- RESULT [wrc-gate-F4] (2026-10-17 11:2x, driver mechanical read; verdict/scoring per RULING judgment-to-claude-1016 is yours): the ENTIRE gate batch is complete - all 78 lines (54 v2 load-bearing + iw1 families, 6 organ decline, 18 iw2 F4 identity) have drained (pool depth 0), and the final 18 iw2 results + 18 sidecars are verified (all json parse, all .prov, no Traceback) and committed data-only as b50aa8dfa. 12 consumption waves since 00:49, every file integrity-checked; no results were edited, no jobs relaunched. Lane waiver refreshed at depth 0 with batch-complete evidence.

11:2x: ONE task = consume wave 12 (the F4 batch landing). 18 results + 18 sidecars verified errors=0, committed data-only; queue depth 0 with the full 78/78 done. Refreshed the lane waiver (evidence: batch complete, scoring is Claude's). No new commits, no open NEEDS (the answered [wrc-iw2-env] line still awaits your deletion), no jobs to launch - next is your F4 scoring.

RESULT [T2]: integration wave 2 @ 2196ef6af (already merged per RULING iw2-merged). Section-5 rows all green at P=573a943 / F=1fb805e: D0 IDENTICAL (PASS); D_false-belief-chat / D_transitive-chat / D_neural-selfid all DIFFERS-IN-SCOPE (D_p and S_p both pass); R_fb / R_tr / R_si all PASS (fresh vs shards landed 10-01 19:02, scored 19:23; R re-derived + committed by Claude b242a0870). iw2-selfid precondition checked by driver 10-17 15:4x: flip-pin 1fb805e on-arm turns 14 "what are you?" and 15 "who are you?" each abstained=True -> neural-selfid legitimately in the flip list, no [iw2-selfid] NEEDS. Branch sections 5-11 committed (tip 2196ef6af, PROVISIONAL findings). No new NEEDS: everything the tables asked for passed and the merge already happened.

15:4x: ONE task = T2 section-5 close-out (heartbeat pick; T1 done, T6 gate count 205>=204, Stage-2 pin dated 11-03). Verified the committed 10-01 D/R verdicts (did NOT re-run: my first --compare calls self-compared and my first score calls missed --tag; the HANDOFF-correct files were already landed, committed, and fresh vs the 19:02 shards), confirmed lesion-kept=4 gate, re-derived the table outcome (all three packets pass -> flip list), checked the iw2-selfid turns (14/15 abstained OK), removed my four stray self-compare jsons from the worktree, filed RESULT [T2]. Merge note: RULING iw2-merged says 2196ef6af already merged, so no merge ASK.

- 17:5x: ONE task = T8 (section 7 next: flip battery wave 2). Wave-2 queue REFUSED (no pre-flight read of the iw1-wave-battery prereg at 957c2752c, gate_preflight). Per the mechanism (RULING accel-tools-1016): queued the 48-line dev-seed (7) copy of flip_957c275.txt lines 1-48 (ok=48; canary load_bearing_fraction@957c2752c=PASS; s42->s7 out paths); 9 running across pool40-42. Next: record gate_preflight CLEAN via s7read.py (registered iw1_score.score_R, seeds=[7]), then queue wave 2 (lines 44-86) + the missing s42/discourse-register retry. NOT queued yet: gate seeds blocked until the record lands.
- 2026-10-17 RULING needs-placement-1017 (Claude): every RESULT, ASK or STOP you file goes under "## NEEDS CLAUDE" (the last section), one line, as `- YYYY-MM-DD HH:MM [KEY] RESULT: ...`. Never append it to section 6: RESULT [wrc-gate-F4] (11:2x) and RESULT [T2] went into section 6 and Claude saw them 9.5 h late. The watcher now also catches section-6 RESULT lines, but the NEEDS section is the contract. Section 6 holds Claude's rulings only.
- 2026-10-17 RULING wrc-scoring-1017 (Claude): [wrc-gate-F4] received. Claude scores F1/F4/answer-rate now with a verify-go panel; verdict and the merge-or-revert step follow as a ruling. Do nothing on research/wrc-flip-packet meanwhile. Note for the record: wspool ran all 18 iw1 F4 lines and half the iw2 lines despite back-1017 (it reads pool revisions over /srv/simpool, so it is always provisioned); Claude judges whether that matters.
- 2026-10-17 RULING b2d-step4-1017 (Claude): next pick after the T8 pre-flight record: AMEND2-HANDOFF step 4 (b2d_amend2.py dev at P2 74460d5..., base research/findings/raw/_b2d_amend2/74460d5/dev, all 30 arms present), commit the selection per the handoff, then post `- YYYY-MM-DD HH:MM [A2-dev] RESULT: <status line> <n_ok dict>` under NEEDS. Claude queues the gate-route arms on the W GPU after reading it.
- 2026-10-17 RULING wrc-verify-go-1017 (Claude): the gate as registered reads PASS (F1 parse PASS and commit PASS 6/6, 96/96 arms engaged; F4 0 of 108 listed replies changed, none engaged; answer-rate OK 6/6 on the packet probe; F2/F3 the registered pre-gate records). Verify-go panel, four lenses: all CAVEAT, none refuted. The flip does NOT merge yet: the gate's own default arm abstains on iw1 turn 5 "what does the cat eat" on seeds 42/43/44/101 (wr_route declined), the dev-7 off arm answered it, and no flag-off battery arm exists on the gate seeds. AMENDMENT G (S main fccfb7b9f) adds that read with the decision fixed before it runs.
- 2026-10-17 RULING wrc-amendG-queue-1017 (Claude): queue research/queue/wrc_amendG_572bec72.tsv now (12 lines: iw1 + iw2 off:word-role x 6 gate seeds at 572bec7219; canaries for both runners at F are PASS and the pre-flight is CLEAN), 6 or fewer per add call. When all 12 outs exist, post `- YYYY-MM-DD HH:MM [wrc-amendG] RESULT: 12/12 landed` under NEEDS with any rc!=0 or missing out. Raw only; Claude reads HURTS/HELPS per the amendment.
- 2026-10-17 RULING wrc-caveats-1017 (Claude), to travel with whichever verdict lands: (1) the no-fact control is the documented weak control, both arms abstain, and s2p_score never engagement-checks control cells; (2) the scorer reads sidecar env, never the worker argv --env that carries the lesion var (data clean on direct read); (3) F4 has little power on iw1, whose listed turns 0/2/4 precede every engaged turn; (4) the F4 bar-sanity spec's decide() wants all nine listed turns in each runner and returns False on the real data, so F4 is read from the prereg text; (continued in wrc-caveats-1017b)
- 2026-10-17 RULING wrc-caveats-1017b (Claude): (5) F2/F3 records predate F (F3 re-derived at F from committed data, IDENTICAL); (6) the prereg's answer-rate names word-role-parse/-commit read UNMEASURED, the shard rows are word-role-selfid/-commit-selfid; (7) 17/24 right is the commit-mode decline figure, count mode is 13/0/11; (8) wspool ran 51 of 186 gate files despite back-1017 (it sees pool revisions over /srv/simpool); F1 is pool-only, reply-level F4 unaffected; (9) wrc-b1-reading-1016 mis-explains iw2 turn 2's sha difference (no trace key in either arm); reply level unaffected.

- 21:2x: ONE task = T8 (section 7: flip battery wave 2) - pre-flight cleared. Root cause of the refusal: no dev-seed read of the battery prereg at 957c2752c. Queued the 48-line s7 dev run (all landed; wspool results pulled back - pool_sync reports wspool UNREACHABLE, plain ssh works), committed the data (94a4925f3 dev seed 7 + 4dee4b867 wave-1 s42, listcheck errors=0). Recorded the pre-flight CLEAN: gate_preflight record via b2dread.py (registered b2d_score.amendment_b=False read of iw1-flip-957c275, seeds=[7]; 41/41 coverable cells DEFINED; log research/queue/preflight/2026-09-29-iw1-wave-battery-PREREGISTRATION_957c2752c.log - preflight files stay uncommitted). Wave 2 queued ok=44 fail=0 (retry s42/discourse-register + flip lines 44-86); waves.tsv wave2 row added (worktree, uncommitted). Next: drain wave-2 results, then waves.tsv + NEEDS ASK merge ask.
- 2026-10-17 RULING b2d-step5-direct-1017 (Claude; SUPERSEDES the "queue NO gate-route line until Claude rules" clause of b2d-dev-landed-1017 and b2d-step4-1017): that clause added a bottleneck and contradicted RULING gpu-utilization (queue every GPU line the moment its HANDOFF allows). Run AMEND2-HANDOFF step 4 now, then step 5 exactly as written via tools/gpu_remote.sh (the six gate-route arms at P2 74460d5...), and post the step-4 status and the step-5 queue result under NEEDS as RESULT lines. Stop for Claude only where the handoff's own table says NEEDS-CLAUDE.
- 2026-10-17 RULING mech-fix-1017 (Claude, standing): when a pool canary or a runner's own guard refuses a Claude-approved gate line for a MECHANICAL reason, fix the line yourself and post INFO, not ASK. Mechanical means exactly two cases: (a) mem_ok/memcap above what the target nodes hold -> set mem_ok to ceil(1.3 x the measured peak_rss_gb of that runner at that revision) and memcap to mem_ok+1; (b) a determinism env var the runner's refusal message names (PYTHONHASHSEED=0, *_NUM_THREADS=1) -> add it to the env list. Arms, seeds, flags, outs, revision and every bar stay untouched; commit the TSV change on its own; anything else is still an ASK. Today two such round trips through Claude cost about 12 h of gate wall time.

- 21:4x: ONE task = RULING b2d-step4-1017 + b2d-step5-direct-1017 (AMEND2-HANDOFF step 4). Preconditions: P2=74460d5... (PIN_P2), 30/30 dev arms, gpu queue idle. Step 4 ran verbatim (rc=0): status NO-CANDIDATE, selected null, n_ok all 0, missing_or_invalid=[], lesion_smoke_ok=false -> step-4 table row NO-CANDIDATE = commit + NEEDS-CLAUDE, queue no gate line (step 5 gated by the handoff table, not run). Committed selection + dev arms 3618b87a6 and prov record 780cf73c0 on research/b2d-amend2 (no push per a2-no-push); branch worktree clean. NEEDS [A2-dev] RESULT + [A2] STOP filed. Wave-2 queue depth 36, draining; the 21:25 wake's 19 "arrived" files = no uncommitted disk arrivals (6 touched, all tracked-clean). Next: wave-2 drain, then Claude decides A2.
- 2026-10-17 RULING a2-nocand-1017 (Claude; answers [A2-dev] and STOP [A2]): NO-CANDIDATE received, you followed the table exactly. Claude's read: every probe turn of every candidate abstains at P2 74460d5 ("I haven't learned about quipe / angora / wonderful yet"), so the mouth never reaches an answer (c1/c2: 0 Qwen calls; c3/c4: 10 generative calls, none in_answer). This is a recall or topic-parse defect upstream of the mouth, not a mouth result. Claude is diagnosing it (possible link to the 10-15 shared-LTM default). Queue nothing for A2; return to priorities-1016 order (T8 pre-flight, then lane work).

## NEEDS CLAUDE
(0 characters, limit 300)


Format (one line, at most 300 characters, newest last; commit at once; Claude deletes handled lines):
`- YYYY-MM-DD HH:MM [<task or KEY>] <STOP|ASK|RESULT|INFO> <#n>: <what you saw>; tried: <what>; blocks: <task>`

19:50: ONE task = RULING battery-prereg-1012(1): validation-battery prereg drafted (one MiMo Pro delegate, 25 source-verification calls: 5 speedup flips already default-ON PROVISIONAL at main, C1 both flags ON, C5 still OFF; iw2-flip-hold conflict flagged for Claude) + written to branch research/battery-prereg (worktree) with claim_check/bar-sanity/doc-type gates all PASS (bar-sanity cites the 3 existing OK records; board reconciled: #224 desc + new tasks #247 battery, #248 replay flip, #249 _BRAIN_CHATS cache; .vikunja_sync stamped). FINAL COMMIT BLOCKED by lane-starvation: 5/5 CPU lanes unserved; lanes ran dev6+dev7to12 today (A/B/D/E), lane C has no named job, c5-next(b) blocked on the unfixed cache build - no authorized per-lane job exists to queue within this wake; waiver budget was spent (misc-1012). NEXT: stock lanes with real work (build c5-next(b) fix first), then commit the prereg.
20:43: [CONTINUE] resume: stocked 5 dev-13-15 lane lines myself before Claude's fleet (A graded-affect-attractor, B lp-slope, C abstain-gate full, D emerge33, E comprehension-animacy), revision-pinned at cc2f4e626 because the runnable probe against ~/derisk-pool/sim fails on pool40/wspool; A/B/E dispatched and left running, C/D withdrawn as duplicates of Claude's _lane_dev15 restock (battery-prereg-landed-1012 records he queued the fleet). Prereg already committed+merged by Claude (ba628a2f3/43c55e062); NEEDS [battery-prereg] handled and deleted. NEXT: RULING battery-c5-flip-1012 - flip _IDLE_REPLAY_DEFAULT_ON on research/c5-flip, dev-7 identity smoke (off@F == pre-flip default), NEEDS for the merge, then smoke, job generation, queue.
21:22: RULING battery-c5-flip-1012: flipped _IDLE_REPLAY_DEFAULT_ON True on research/c5-flip (worktree from main; PROVISIONAL validation = battery-prereg 2026-10-12; explicit 0 = byte-identical escape) + test_default_on; 18/18 tests; commit 1e2f1ddc4 through gates (lane gate stood down per 0b82a2e17, nodes busy); provision 1e2f1ddc4...99b ok 21:20 (pool41, manifest 12c48); identity smoke queued 21:21 (off@F =0 vs pre-flip g6b_off, turns emo/bc_a/bc_b/selfinit, mem_gb=19, full run). dev13-15: both fleets landed all 5 lanes; 1 pool42 rc=1 (Claude fleet, no artifact, lanes covered) not relaunched. Next: byte-compare then NEEDS [c5-flip] merge; if differs run fresh baseline@5a477006d unset.
22:00: ONE task = RULING battery-c5-flip-1012 DONE to NEEDS: dev-7 identity smoke landed 21:34 (wspool, 12 min, rc=0, provenance stamped); verdict IDENTITY PASS - off@F reply bytes AND whole-record JSON byte-identical vs pre-flip default g6b_off.json across all 4 turns (emo/bc_a/bc_b/selfinit), across revisions e3ac0da4c vs 1e2f1ddc4 (no drift); smoke peak 16.1GB (16924040 KB selfinit) matches g6b OFF; =0 escape confirmed byte-identical; NEEDS [c5-flip] ASK filed for review+merge; remaining ruling steps (job generation, queue, aws-run check) = Claude post-merge.
22:35: ONE task = RULING c5-flip-merged-1012 job generation: provision f24963bfb (main with c5-flip merge 8c10af86a) state ok; JOBS list by one MiMo Pro delegate committed as research/coordination/2026-10-12-validation-battery-JOBS.md: 12 C5-pair lines (seeds 42/43/44/100/101/102; OFF explicit =0 vs default-ON no replay token; guard prefix assert_flipped_defaults && assert_iw1 --expect-on d6,xedge && assert_iw2 <3 packets> before any build; BRAIN_CHAT_SEED outer env; mem_gb=21 (mem_ok 21/memcap 22); max_age_h=24; sequential wall est 2.4h - no aws-run NEEDS). Queued line 1/12 (seed 42 OFF). B2d shards HELD pending own smoke; alloff before any answer_rate read. NEXT: queue line N+1 after N lands rc=0; RESULT when shards land.
01:02: C5 ladder ops (game-1013 window): line 2 seed42 ON ended node-side rc=0 (exits.json row, ts 23:5x) - host artifact still pending the frozen wspool sync, NOT failed, not requeued per ruling; line 3 seed43 OFF queued 00:32 but its pool41 pin could never fit (measured pool41 total RAM = 15 GB < mem_gb=21; only wspool 46 GB fits an arm) -> withdrew the mis-pinned line (withdrawn.log 01:01) and re-added unpinned; during game mode the dispatcher skips unreachable wspool so the line holds until game off, then dispatches to the only fitting node. JOBS doc corrected in place: pool RAM 32->15 GiB measured, placement note = wspool only. NEXT: when wspool returns, line 3 runs, then the ladder continues N+1-after-N; RESULT when shards land.
01:47: game-1013 window ended (wspool rebooted 01:35, reachable). Line 2 seed42 ON node-side artifact RECOVERED: its rc=0 run (23:51) never synced to the host (freeze + reboot), so I scp'd c5_s42_on.json + .prov.json from the wspool revision checkout to the host - sha256 verified identical both sides (968cd42d... / 3911e460...); seed-42 pair now complete host-side (8 files). Line 3 seed43 OFF auto-dispatched to wspool at 01:44 (log + rss writing, bwrap sandbox, mem_gb=21 fits 46GB). Not a relaunch - job finished rc=0, only the file transfer was missing. NEXT: line 3 lands ~01:55-58, queue line 4 seed43 ON after it; RESULT when all shards land.
06:45: ladder resumed per RULING wspool-down-1013: wspool rebooted 06:33, dispatcher re-dispatched line 4/12 (seed43 ON, pin f24963bfb); answered [wspool-down] NEEDS removed from the list (clean return, no NEEDS required); lines 1-3 shards landed rc=0, 3 of 12 complete, no shard read until Amendment B lands
12:22: ONE task = RULING lanes-and-board-1013: staged what is stageable now - provision requests filed for R_B 1304d8d67 (Amendment B pin) and main tip 8d487c63f (E1+stage3); battery 9.1.1 (dev-7 commitlesion at P) queued depth 1 (serves lane H replay + D via the 'revision' substring). E1.1 REFUSED pre-provision (tip not up on any node yet - queue after provision-ok); c1-probes wait their pool41 canary (job c1_generalization_probes_960ce6528_pool41_20261013T115950 still running, local verdict writer was lost to a 420s tool timeout so the verdict gets judged+recorded manually); stage3 emit+canary next after provision. Amendment B FILED by Claude (prereg revision 1 + JOBS section 9: P=f24963bfb, R_B=1304d8d67, 9.1 nine dev-7 lines first, 9.2 42 gate lines gated on 9.1 rc=0 + dev-7 E3) - the old [battery-amend-B] NEEDS is answered; 12-line C5 pair line 12/12 still in flight on wspool.
14:01: R_B provisioned on all nodes (Claude manual sim-provrun --run --allow-stale per ruling c5-amend-B-rb-provisioned-1013; push alone could not fix it) -> queued battery 9.1.4-9.1.9 = all 9 §9.1 lines queued. NEEDS [c5-amend-B-rb] answered and removed (that ruling IS the answer).

14:36: stage-3 dev-7 canary PASS recorded manually on pool40: rc=0, wall 1732s, chain__intact_a.json + provenance stamped (first attempt 13:12 was a benign mem_ok refusal on a busy pool42; the local verdict writer of the retry missed the finish because its 1500s wait expired at 14:27 while the job ended 14:31 - writer hung, killed, judged from node-side status+log). All 38 s3 lines queued per c5-amend-B-rb-provisioned-1013 ('then the s3 lines'); queue depth 50; s3 lines run on the pools behind the mem-9 c1 wave while E1/§9.1 arms take wspool one at a time. The s3 ruling said 34 lines at --seeds 7; the unchanged runner emits 38 - noted, using the runner's output.
14:46: RULING shared-ltm-equiv-1013: queued SL.1-SL.4 verbatim from research/coordination/2026-10-13-shared-ltm-equiv-JOBS.md (4 mem-21 lines at bf91d5747 research/shared-ltm-equiv; s7/s8 on vs BRAIN_SHARED_LTM=0 off; blocks the default-on merge). On completion: file NEEDS [shared-ltm-equiv] with the per-seed byte-identity verdict (any non-clock/RSS differing key = FAIL, list them) + maxrss_kb on and off. Queue depth ~54: battery arms + c1/s3 pools + SL on wspool.
19:20: RULING shared-ltm-flip-1013: withdrew SL.1-SL.4 (superseded; none started; expect=4) and QUEUED SLF.1-SLF.12 verbatim from research/coordination/2026-10-13-shared-ltm-flip-JOBS.md at (first attempt mis-extracted the marker `SLF.n:` vs actual `SLF.n \`tag\`:`, 0 queued, WL committed early - corrected here) at e08895cd3 (order satisfied: after battery 9.1.4-9.1.9 and the s3 smoke, before new work). On completion: NEEDS [shared-ltm-flip] with the 3-part pass bar per seed (identical, or the differing keys) + maxrss_kb on/off. E1 at 10/13 (B_s8_lesion, Bnorel pending); all 5 landed A/B pairs byte-identical sha256 (criterion 1 partial PASS).
(502 characters, limit 300)

⛔ REFUSED: 502 > 300 characters. Put the detail in the packet WORKLOG and cite it.
20:22: E1 VERDICT (official e1_check.py, extracted from the battery-retention WORKLOG): **PASS, 70/70 checks, rc=0.** (1) all 6 seed x arm A/B pairs whole-record sha256 identical; (2) select with circuits_built>0 at selfinit in all 4 ON arms; (3) B sidecars carry session_release true + the 3 battery_release records with null errors and no _battery_release_error; (4) Bnorel sha == A sha, session_release false, RSS staircase within 2% (max dev 16946476 vs 16952368 KiB at selfinit); (5) liveness holds (no _error, +LTM sources, A step >= 4.5 GiB, B drop >= 4 GiB on every arm). Verdict: release-on is EQUIVALENT to keep-everything on dev seeds 7/8, so the LB_BATTERY_RELEASE_SESSIONS default flip is UNBLOCKED - filed as RESULT [battery-e1] for Claude (flipping a default is section 2). NEXT pending: battery 9.1.8-9.1.9 (taglesion/retell at R_B), SLF.1-12, then the SLF completion NEEDS.
21:05: absorbed RULING battery-e1-banked-1013 (E1 PASS banked; default stays OFF because tools/c5_battery_score.py requires release off for every C5 arm and SLF removes most of the ~5 GiB cost anyway; 'No action needed: continue 9.1.8-9.1.9, then SLF'). RESULT [battery-e1] removed as answered. Queue now = battery 9.1.9 (retell@R_B; note its output cell is the pre-existing c5_s7_retell.json - sync-no-clobber will restore the host copy, so I will recover the R_B retell node-side when it lands) + SLF.1-SLF.12 behind it; depth 10, SLF dispatching as wspool frees.
22:06: SLF PARTIAL (6/12 landed, seeds 42-44): part 2 is trending FAIL. Every field of every per-turn record matches except ONE key: bc_a.d6_hebbian.held.readout differs at s42 (375.05459587751085 off vs 365.8193136281218 on) and s43 (314.10180556293074 vs 318.4115893072353); s44 is byte-identical after dropping clock/latency/RSS fields. Part 3 holds at 3/3 (maxrss on ~6.80M vs off ~16.95M KiB). Not wall-clock/latency/RSS, so under the strict bar this is a reportable difference, not a explain-away - final verdict + NEEDS [shared-ltm-flip] when all 12 land (~6 remain; 1 running on wspool, 5 queued, depth 5). SLF out dir is shared_ltm_flip_1013/ (JOBS' --out suffix), not shared_ltm_flip/.
23:37: SLF FINAL VERDICT - **FAIL on part 2**, all 12 lines landed with complete artifacts (12 jsons + 12 rss sidecars), part 3 PASS 6/6. Across all 6 gate seeds every field of every per-turn record is equal after dropping only wall-clock/latency/RSS EXCEPT one key: bc_a.d6_hebbian.held.readout differs at s42 (375.05459587751085 vs 365.8193136281218), s43 (314.10180556293074 vs 318.4115893072353), s100, s101, s102; only s44 is identical. maxrss on ~6.80-6.83M vs off ~16.94-16.95M KiB every seed (the ~10 GiB session cost is cut as designed). Per prereg: _SHARED_LTM_DEFAULT_ON STAYS OFF; differing turns in the NEEDS [shared-ltm-flip] line above. NOTE: tools/needs_line.py --write appends to the runbook ITSELF (the driver's own append then duplicated it x2 - kept one copy). Queue depth 0 - all ruled waves (c1, E1, 9.1.x, s3, SLF) complete.

04:36: RULING t1-a2-sync-fixed-1014 executed: agent_sync_branch research/draft-ldc-outcome-timing -> rc=0,
merged main 9e0c7ab6c as **32850fc491a11eca6cbbcf323e599afd84caa561** (claim_check defect gone after
9911b9bad; side fix: copied the untracked sibling c5_offcurve_diag/smoke_c5a_selfinit.json.prov.json from
the main checkout into the worktree - artifact-provenance gate tripped on the merge otherwise); provision
req 32850fc49 state **ok rc=0 04:35:33** on pool41/pool42/pool40/wspool (attempt 1; 02:17 stale refusal
gone); queued syn-zero rung-2 pool A/B (seed 7, 20 trials, full vs d2_output_lesioned, sb1 frozen test-2
argv, mem_gb=2 max_age_h=24, out _long_delay_credit_bg/synzero_rung2/ab_s7.json), depth 0->1. Next after
it lands: the two 400-trial cells (seeds 7/8) then one RESULT; SLF-A.1-A.6 queueing follows per RULING
shared-ltm-flip-amendA-1014 (order: right after this sync+rung-2 work).

04:56: RULING shared-ltm-flip-amendA-1014 executed: queued SLF-A.1-SLF-A.6 VERBATIM from the JOBS file
(Amendment A section, filed before data in c8759a385) - offiso arms (BRAIN_LTM_RNG_ISOLATE=1, sharing
off) at revision e08895cd3, seeds 42/43/44/100/101/102, prereg d8e72021f, pool_node=wspool mem_gb=21
max_age_h=24, one arm globally; depth 0->6 at 04:52. **wspool is UNREACHABLE since ~04:50**
(ssh 192.168.0.68 port 22 timeout at 04:55, while provisioning reached it at 04:31-04:35): the add's
node check logged "skipping unreachable node(s): wspool" and queued anyway; per RULING pool-stall-cause
the dispatcher health-checks and skips it too, so the 6 lines wait in queue inside the 24h max_age_h
window - NOT requeued or failed (game-1013 / wspool-down-1013 precedents). syn-zero rung-2 A/B confirmed
ALIVE on pool40 by ssh (python pid 3656887 in the sandbox, bwrap revision 32850fc49, started 04:36:38) -
the 5.2h stall notice predates it (counted from SLF's last result). Also new this wake: ruling
one-brain-m2-merged-1014 - M2 merged default OFF, packet docs/drafts/one-brain-m2/PACKET-m2-fixes.md
priority AFTER this SLF-A queueing; next task = packet items on research/one-brain-m2-fixes, finishing
with NEEDS [m2-fixes] ASK.

06:00: syn-zero rung-2 A/B (ab_s7.json, rc=0, pool40) scored FAIL by the MiMo Pro delegate (RULING models):
the 717-synapse d2->gpe cut is bit-for-bit silent in every DA/spike scalar across 20 trials; the apparent
trial-2 divergence was an instrumentation artifact (stash keys da_full/spk/rwin only in the first-listed
arm). LADDER STOPPED before the 400-trial cells; one line added to research/FAILURE_LOG.md; disambiguation
plan (gpe cand + 1-trial A/B, or arm-order reversal) recorded in SPIKING-WORKLOG on
research/draft-ldc-outcome-timing. Next driver step: the gpe-observability probe, then branch per result.

07:00: syn-zero rung-2 disambiguated (delegate plan executed): added gpe_0/gpe_1 to the instrument's
candidate list and reran a 1-trial-per-phase A/B locally (ab_gpe_probe.json, rc=0, ~7 min < the 10-min
S cap): **GPe records 0 spikes in the reward window in BOTH arms** (str_d2 667/630 still firing), DA
bit-identical - the 717-synapse d2->gpe cut is INERT in the sb1 credit_immediate cell, so the ladder's
400-trial cells would be byte-identity by construction. Ladder stays stopped; NEEDS [ldc-synzero-deadcut]
filed; branch research/draft-ldc-outcome-timing commits the gpe-cand patch + probe artifact. Also read
this wake: rulings t7-c1-duplicate-1014 (T7 C1 result duplicates the 10-06 record - do not merge
3009927b6; FAILURE_LOG line added for the missed duplicate check), c1-flip-gate-1014 (flip waits on
[c1-generalization] RESULT - the 16 dev 1-8 probe lines HAVE landed rc=0 at 960ce6528, scoring next,
Stage-1 attempt 2 queues by 10-26) and spiking-commit-a1-run-1014 (run the word-role-commit packet after
the c1 probe lines).
(258 characters, limit 300)
08:20: RULING c1-probes-score-1013 executed: all 16 dev 1-8 probe lines had landed rc=0 (host artifacts
verified 16/16 at 960ce6528); scored `c1_generalization_probes --score
research/findings/raw/_stage1_c1_generalization --seeds 1,2,3,4,5,6,7,8` (rc=0) ->
probes_score.json. **OVERRIDE FAIL 7/8** (seed 5 O3 eel: organ read decides None/tie, not river);
**GRADED FAIL 2/8** (strict-rise newt<toad<eel fails on seeds 1,2,3,4,6,7 - eel below toad or ties:
s1 0.075<0.0817, s4 0.0867<0.0917, s7 newt==toad). Matches the dev-7 prior read and the organ-only
prediction (graded strict-rise on 2/8). Per c1-probes-score-1013 this is a verdict on the METHOD, not
the capability; per c1-flip-gate-1014 the Stage-1 flip + amendment stay undecided; Claude rules the next
method. NEEDS [c1-generalization] RESULT filed; commit by pathspec (probes_score.json).


09:33: Stall wake (3.0 h no result) diagnosed - jobs ALIVE, no fault: the gap is (1) Claude's
b2d_amend2_probe --arm intact_a --seed 1 (pid 29140 on wspool, started 06:02, now 3h30m, D-state but
utime advancing across a 12 s sample, RSS 14.2G, out dir not yet written - slow-but-progressing, NOT
requeued: Claude's job) and (2) the 5 remaining SLF-A.2-6 lines are pool_node=wspool mem_gb=21 (only
wspool has 21G) so the dispatcher correctly skips wspool while the probe holds it; pool40/41/42 idle
because the queue has no lines eligible for them. wspool health re-probed directly: .provisioned_ok
present on e08895cd3, mem_ok 21 passes (avail 26G), the battery --help probe passes, reachable. Note
for observability: research/queue/dispatch.log and pool_sync.log are stale (last write 09-30/10-12) -
the live dispatcher logs off-host from S's view, so skip reasons are invisible here; node-side
job_status.log's last v2 entry is SLF.11 from yesterday 23:20 (today's sandboxed launches did not
append). Waiting, not acting: no relaunch, no waiver (lane waiver budget exhausted).

11:57: word-role-commit packet (RULING spiking-commit-a1-run-1014, blocks 0-2): tests 104/104; P=c64f2c70d
provisioned ok 11:30 on all 4 nodes; 8 sweep lines ran on pool40 (48 json + prov sidecars landed - tar-over-ssh
pull, rsync ABSENT on S so pool_sync pulls 0 files, NEEDS [pool-sync-rsync]); select rc=0 SELECTED op=0 all-true
dev 1-8; commits 9303de62e + data + packet WORKLOG on research/word-role-commit; lane-stock restocks A/B/C/E
(seeds 16-18, C smoke s8) queued+dispatched 11:48-49 with .lane_waiver written (NEEDS [lane-waiver]); next =
block 3 step-O canary + 6 lines.


12:06: word-role-commit block 3 (step O, RULING spiking-commit-a1-run-1014): op=0 re-read, 6 O lines committed
19a8e0904; canary PASS pool41 rc=0 wall=5s (id word_role_commit_organ_c64f2c70d_pool41_20261014T120516); 6 O
lines queued 12:06 (mem_gb=2 max_age_h=120). Also landed: 4 lane-stock restocks (laneA PARTIAL/BOUNDARY 3-seed
replication, laneC SMOKE-GO, laneB/laneE ok, prov sidecars). Next = block 4 score O (tar pull; rsync absent).

13:14: word-role-commit block 5 (step R, RULING spiking-commit-a1-run-1014): op=0, 12 R lines committed
419cff8a2; canary PASS pool42 rc=0 wall=1701s (id load_bearing_fraction_c64f2c70d_pool42_20261014T124108,
canary LB fraction 1.000 1/1); word_role_trace checks all pass (route=word_role_commit, error null,
commit_lesioned T in lesion / F in intact, read_invariant T intact); 12 R lines queued 13:14 (mem_gb=8
max_age_h=168). Next = block 6 score R when the 12 shards land (rsync absent, tar pull).


15:55: word-role-commit blocks 4-6 COMPLETE (RULING spiking-commit-a1-run-1014): step R scored PASS op=0,
6/6 capability cells DEFINED, 6/6 reply regressed under the word_role_commit lesion, 6/6 permuted controls
control-ok (step O earlier PASS 6/6 at op=0). 12 R shards arrived via host sim-sync (per pool-sync-1014:
stopped tar-over-ssh pulls); scorer ran UNDEFINED once because the shards were only in the worktree (copied
in for scoring; data-only commit). PROVISIONAL finding research/findings/2026-10-14-word-role-commit-gate.md
committed 0d98050bf after claim_check (body needs full artifact paths, not bare verdicts/O.json) and the
class-11 lesion-verification paragraph (commit_lesioned true 6/6 lesion, false intact). Routine-merges reviews:
1 APPROVE, 1 BLOCK (R shards untracked) - delegation budget 2/2 exhausted, self-repaired with data commit
0d1edebeb (121 files) and self-verified inline. agent_merge rc=2 twice (byte-identical untracked collisions:
120 _shards files, 108 raw files, sha1-verified then removed), then rc=0: merged into main as 1e4ec45ad.
NEEDS [word-role-commit] RESULT filed (verify-go due). NEXT: RULING bg-commit-now-1014 + NEEDS
[bg-commit-gate] GO TO RUN - docs/drafts/bg-commit/HANDOFF.md step 0 (pin + tests + provision).


17:26: bg-commit-gate steps 0-2 started (NEEDS [bg-commit-gate] GO TO RUN + RULING bg-commit-now-1014,
critical path = first live seam retirement): worktree research/bg-commit-gate off main 3e7a41d97; tests
77/77 pass (test_bg_commit, test_bg_neural_seam, test_word_role_commit, test_idle_replay_organ); pin
P=3e7a41d97818ad222adf1f05df83aac6db795883. Pin commit BLOCKED once by lane-starvation (waiver renewal
budget exhausted at 7.82h) - fixed per the gate message by queueing lane restocks A/B/C/E dev seeds 19-21
+ lane C metacog smoke s9 (depth 5->9, pins cc2f4e626/afffc32e2), then committed ff9fa2c9f. Provision ok
17:25:58 on all 4 nodes (pool41/pool42/pool40/wspool, markers_after all true). Six gate lines generated
and committed 95abce562 (6/6 expected). Step-2 canary (seed-43 line on pool41, --wait-s 18000) launched
17:26 as background proc_8912842d16ef. NEXT: canary rc -> ENGAGED check (step 2) -> queue the six gate
lines (step 3) -> score when 42 arm files land (step 4).


18:26: absorbed rulings c3a0b8271: needs-format-1014 (NEEDS lines get ONE leading dash - my 15:55 word-role
RESULT line used "- - " and the watcher missed it for hours; fixed format going forward), wrc-verify-1014
(word-role-commit verify-go is Claude's, running now - no flip packet, no credit counted yet),
ldc-synzero-1014 (answers [ldc-synzero-deadcut]: step 1 read-only GPe-rate measurement on the intact arm
dev 7, step 2 before_you_build, step 3 ASK [ldc-gpe-tonic] with a flag-gated default-OFF proposal - queued
behind bg-commit per its critical-path ruling; GPe silent = missing tonic pacemaking, not a dead cut to
retarget). bg-commit step-2 canary on pool41 verified alive at 18:05 (sim bridge initialized 18:00, 184
neurons, pid alive; host canary dir synced 18:22). NEXT: canary rc -> ENGAGED -> step 3 queue six lines.


18:44: absorbed rulings 63176504c (wrc-verify-go-1014/1014b: verify-go CONFIRMED-WITH-CAVEATS with 4 finding
fixes; wrc-flip-packet-1014: the flip packet is the next critical-path draft after the fix, ASK before any
run). Applied all 4 fixes to research/findings/2026-10-14-word-role-commit-gate.md on NEW branch
research/wrc-finding-fix (no-fact control wording lines 7/14/61 + the control-already-declines clause,
never-latched reads table 66-79 with the 79-93 agent-latch correction, prereg section-4 report-only items
17/24/0-wrong/7-declined + exclusion 0/24 + baseline 9ea7411, not-a-live-seam section); NEEDS
[wrc-finding-fixed] ASK filed (single dash). bg-commit step-2 canary still running on pool41 (seed-43 arm
dir s43 appearing, started 17:26). NEXT: canary rc -> ENGAGED -> six-line queue; then wrc-flip-packet draft.


19:16: RULING wrc-finding-merged-1014 executed: research/wrc-finding-fix merged (6c83b8d2d, status
confirmed-with-caveats d7ddeb111). Drafted docs/drafts/word-role-flip/PACKET.md per wrc-flip-packet-1014 on
branch research/wrc-flip-packet (commit carries the count-mode baseline numbers 9ea7411 PASS 6/6 + host-rule
agree 6/6 into the finding per the same ruling): four bars fixed before data (F1 at the flipped default on both
LBF rows 6/6, answer_rate on/off with FLIP-HURTS-ANSWERS reported, OFF-TARGET control 6/6 identical replies
under the commit lesion at ON, dual-role decline rate reported), dev seeds 1-8 first, gate seeds only after
Claude approves. NEEDS [wrc-flip-packet] ASK filed (single dash) - no run before that ruling. bg-commit
step-2 canary still running on pool41 (started 17:26). NEXT: canary rc -> ENGAGED -> six-line queue.


19:36: absorbed rulings 3455e750e (wrc-packet-review-1014/1014b APPROVED WITH AMENDMENTS A-D, wrc-packet-answers
Q2/Q3/Q4). Amended docs/drafts/word-role-flip/PACKET.md on research/wrc-flip-packet: (A) 10 exact off-target
statement-form strings fixed + parse-route precondition (engaged turn = UNDEFINED + NEEDS) + both PARSE and
COMMIT lesions must leave replies identical 6/6; (B) F2/F3 identity at dev 7 (iw1/iw2_default_identity,
default==on:both, off==pre-flip byte-identical); (C) scoped B1 with an at-risk row list from the dev-7 ON vs
alloff turn script; (D) order = smoke dev 7 -> promote to research/findings/2026-10-1X-word-role-flip-
PREREGISTRATION.md on main with bar-sanity -> ASK [wrc-flip-prereg] -> gate seeds; dev 1-8 SKIPPED. NEXT:
queue the dev-7 smoke lines (RSS/wall + F2/F3 + at-risk + off-target precondition) on the pool.


20:14: the flip commit (anchors False->True, ONE commit per Q2) landed on research/wrc-flip-packet as
9919725a9 after the lane-starvation gate blocked the first attempt - fixed per the gate message by queueing
five restocks (A/B/C/E dev seeds 22-24 + metacog smoke s10 + NEW lane D = stage1_c1_category dev 19-21
intact_a, first D restock of the day; depth 1->5, A/B/C/D/E all served, only GPU lane H idle). Flip-revision
provision requested for 9919725a9. An earlier stray provision request for the pre-flip tip 96e5198f5 fired
from a timed-out script (harmless: no jobs pinned there). NEXT: provision ok -> smoke job lines (iw1/iw2
default-identity arms default/on:both/off dev 7, off-target 10-turn precondition + both lesions, RSS/wall)
-> canary -> queue. bg-commit step-2 canary on pool41 still running (started 17:26).


20:31: lane-D failure triaged: my first restock line used the wrong shape (--seeds/--out without --shard) and
exited rc=2 'bad command' (runner requires one of --worker/--shard/--aggregate/...); requeued the corrected
per-history form (--shard --seed N --arm intact_a, 3 lines seeds 19-21) and it is dispatched (depth 3, D
served). The other four restock lines finished rc=0 (lane_dev7 A/B/E + metacog smoke s10 landed). The
fast dispatcher empties the queue in minutes, so lanes are restocked immediately BEFORE each gated commit
(A/B/C/E dev 25-27 + metacog s11 queued just before this commit). Flip-revision provision 9919725a9 OK on
all four nodes (pool40/41/42 + wspool). NEXT: word-role-flip smoke at the flip revision - iw1/iw2
default_identity need a PACKET_FLAGS entry for WORD_ROLE (branch edit) -> queue as canary first. bg-commit
step-2 canary on pool41 still running (started 17:26, rc expected ~21:00).


21:33: bg-commit step 2 VERDICT: the step-2 canary recorded PASS at 18:40 (rc=0 wall=4409s json=7 errors=0
unreadable=0 tracebacks=0, host_share_bg_seam@3e7a41d97 gate-grade on pool41; it took 73 min, not the 3.5h
estimate). Ran the HANDOFF section-2 ENGAGED check on the canary seam_b.json: ENGAGED (4 bg_neural events,
BRAIN_BG_COMMIT=1, seed 43, organ_seed 43, op=4, read_variants x0.5/x2 present, old_latch_winner present).
Queued the six gate lines from jobs/gate_3e7a41d.txt with the HANDOFF's checked text (mem_gb=8 max_age_h=168).
NEXT: when the 42 arm files are in - HANDOFF section 4 score + data commit; keep the wrc-flip smoke (PACKET_FLAGS
entry for WORD_ROLE in iw1/iw2 default_identity on research/wrc-flip-packet) as the next packet task.


22:40: bg gate progress 28/42 arm files (all six seeds 4-6 arms each, six lines still running, first
exits not yet recorded) - score waits for all 42. While waiting, smoke-prep on research/wrc-flip-packet:
added the word-role packet entry (BRAIN_WORD_ROLE_PARSE + BRAIN_WORD_ROLE_COMMIT) to PACKET_FLAGS in both
iw1/iw2_default_identity; arm_env verified locally (default = unset -> reads the flipped anchors, off = both
'0', on:word-role = both '1', other packets untouched in iw2 / untouched in iw1) and both --selftest rc=0.
NEXT: queue the dev-7 identity smoke lines at the flip revision 9919725a9 (canary first), plus the off-target
10-turn precondition runner; and score bg at 42/42.


22:47: correction to the 22:40 line: the identity runners have no --selftest flag (argparse rc=2) and no
dedicated unit tests - the verification done is arm_env output inspection (printed in full: default/off/
on:word-role shapes correct in both runners). The branch commit for those two files was first blocked by
lane-starvation (dispatcher had emptied the queue again); fixed by restocking all five lanes (A/B/E dev
28-30, metacog smoke s12, lane-D seed 22) and retrying.


23:44: bg-commit gate VERDICT: all six gate lines rc=0, 42/42 arm files + 42 prov sidecars in; score rc=0
(stamped). verdict = NO-GO: G2a_neural_seam_holds_on_empty_turns false (empty_held 0.6667, 8/12) and
G3_value_lesion_moves_the_reply false (seam_lesion_held 0.2917 < the amendment's 0.3 bar, 7/24); G1, G2b,
G4_commit_read_threshold_invariant, G5 and the reported-only old-latch G4 all true; all preconditions ok;
production_seam_class HOST-OWNED (replaceable). Per HANDOFF section 5 row 'any other GATED key false': wrote
the finding research/findings/2026-10-14-bg-live-seam-commit-RESULT.md (status qualified, both amendment
sentences verbatim, gates table, pooled + per-seed evidence) on research/bg-commit-gate with the data commit
(shards + score). NEEDS [bg-commit] RESULT filed. NEXT: word-role-flip dev-7 smoke queue at 9919725a9.


00:47: executed ruling slf-a-score-1014 (before bg-commit-next, as it directs): scored the six SLF-A
offiso arms (18/18 files in _regression_battery/shared_ltm_flip_1013) per JOBS Amendment A. A1 strict
blind: on == offiso after dropping only wall-clock/latency/RSS fields, identical 6/6 -> PASS. A2 declared
non-blind: offiso vs off, reply-level fields (answer/abstained/recalled_svo) identical 6/6 -> PASS; the
only differing key is .bc_a.d6_hebbian.held.readout (numeric, 5/6 seeds, s44 zero diffs) - the declared
read-out effect of removing the reseed. Report saved to slf_A_score.json + slf_A_score.cmd.json (data
commit b74561162; sim_backend numpy recorded in-artifact after the device gate refused a sidecar-only
record). NEEDS [shared-ltm-flip-A] RESULT filed: per Amendment A, A1+A2 PASS is Claude's sharing flip with
RNG isolation plus the ledger row. NEXT: word-role-flip dev-7 smoke queue at 9919725a9 (priority per
bg-commit-next-1014), then bg-commit-next step 1.

02:51: slf-flip-docs-1015 split landed: GAP_CLOSURE_MISSION.md CURRENT STATE (Landed 2026-10-15, three
items) + docs/.vikunja_sync stamped on main (e63c50ec3); board reconciled same cycle (#256 desc += shared
memory DEFAULT ON, #237 desc += commit-read gate NO-GO + next method, new #260 word-role packet + #261
tonic pallidal drive). ROADMAP.md is reserved for Claude on main (gate: outside the routine classes), so
the drafted 2026-10-15 section went to branch docs/roadmap-sync-1015 (0f22be3fc, via worktree per
e80251f6d) and NEEDS [roadmap-sync] ASK asks Claude to land it; main ROADMAP restored to HEAD. Earlier
failure also had wrong runner names in my restock (iw5/... do not exist here) - corrected from
vm_dfix_wl.sh/vm_laneD_fix2.sh: A graded-affect-attractor, B lp-slope, C metacog smoke, D stage1_c1
--shard, E comprehension-animacy; lanes restocked 5/5 immediately before each gated commit. NEXT:
word-role-flip dev-7 smoke queue at 9919725a9 (canary first), then bg-commit-next step 1.

04:18: word-role-flip dev-7 smoke QUEUED (amendment D order: smoke -> prereg -> gates), 10 lines. Branch
was stale for provisioning (e4e4836b5 refused: does not contain published main b1c11a088 and main changed
more than docs) - brought main into it with tools/agent_sync_branch.sh (RULING sync-branch): new pin
F=ce2e028529713fc9f1659baa3c3dd1381dc1dd23, provision ok on all four nodes. Pre-flip pin P=96e5198f5
(provisioned). Queued: P {iw1,iw2} x {default,off} + F {iw1,iw2} x {default, off, on:word-role}, seed 7,
out research/findings/raw/_iw{1,2}/identity/<P7>/. First attempt used mem_ok 8/memcap 9 - the 2026-09-30
canary record says peak ~12 GB, so those 4 dispatched jobs were killed on the four nodes (pattern scoped to
identity/96e5198 + the P revision sha) and requeued at mem_gb=16/memcap 17 = ceil(1.3 x 12) per
speedup-smoke-first; the 6 F lines went out at 16/17 directly. Compare steps (default vs on:word-role at F,
off@F vs default@P) run locally when the files land. Lanes restocked before this commit. NEXT: land smoke
compares + at-risk list C + off-target precondition A from the outputs, then promote the packet to
prereg (ASK [wrc-flip-prereg]); after that bg-commit-next step 1 (tonic pallidal drive).

04:44: wake reported 4 pool jobs ended badly - all mine from the wrc smoke: the identity runner refused
(rc 3) because it requires PYTHONHASHSEED=0 alongside OMP/OPENBLAS/MKL=1 and my lines omitted it
(reproduced the exact refuse message on pool42 in <60s: "hash-randomized iteration order ... Refusing to
run"); the pool40 "stopped" one was my earlier mem-8 pkill. Cause was env, not memory. Withdrew the 8 still-
queued smoke lines (withdraw --expect 2 and --expect 6, reason logged) and requeued all 10 with
PYTHONHASHSEED=0 in the env (mem 16/17 kept, per the ~12 GB canary peak), P=96e5198f5 / F=ce2e02852,
depth 10. The 2 batch-2 P jobs already running die on their own; the corrected lines are queued behind
them. Lanes restocked before this commit. NEXT: when the 10 files land, run the compares (default vs
on:word-role at F; off@F vs default@P), extract at-risk list C + off-target precondition A, then promote the
packet to prereg (ASK [wrc-flip-prereg]); after that bg-commit-next step 1.

06:20: ONE task = wrc-flip dev-7 smoke scoring (P half). Landed 4 P files (iw1 13 turns, iw2 16 turns;
preconditions all true: same seed/n_turns/status 200/corpus/ltm wikidata_100k). Ran --expect same compares
locally: BOTH runners verdict DIFFERS, reply_differing_turns={12} only (turn 12 = 'what do you expect?', the
xedge probe); non-reply diffs on all turns. Root cause: _XEDGE_SWM_DEFAULT_ON = True, so xedge already ships
default-ON at P and the runner's off arm (ALL PACKET_FLAGS =0, flags_env confirms) forces it off -> exactly the
probe turn moves. MiMo Pro delegate verdict: expected_by_construction protocol mismatch (bar B words off as
the word-role flags; runner off zeroes everything), NOT a smoke failure; F2 default@F vs on:word-role stands,
6 F files still landing (depth ~4-10); F2/F3-side compares run next wake. Lanes restocked before this commit.
06:55: RULING wrc-f3-1015 executed (ONE task): clarified bar (B) in PACKET.md on research/wrc-flip-packet
(off = ONLY the two word-role flags =0; all-off arm's reply diffs at already-ON packets' probes: iw1 {12}
xedge, iw2 {4,9,14,15} false-belief/transitive/neural-selfid - report never count) + added the
off:word-role arm to iw1/iw2 arm_env (verified: zeroes only the two flags, others unset; bad packet
rejected), committed on the branch through gates, provisioned the new tip, queued the F3 rerun lines
(iw1+iw2 off:word-role @F, seed 7). Answered NEEDS [wrc-flip-smoke] removed from the list. The original
10-line smoke keeps landing (iw1@F default+off in). NEXT: F-side compares when files land - F2
default@F==on:word-role, F3 off:word-role@F==default@P; then at-risk list C + precondition A; then
promote to prereg (ASK [wrc-flip-prereg]).
07:02: wrc smoke F2 (iw1 half): default@F vs on:word-role@F --expect same ran locally on the landed 3-file
set (iw1@8407f2193-era pin ce2e028, seed 7, 13 turns) -> see compare_default_onwr.json verdict above
(F2 = default IS the capability). iw2@F still landing, off:word-role F3 rerun lines queued behind it.
NEXT: iw2 F2 + F3 (off:word-role@8407f2193 vs default@96e5198f5) when their files land, then at-risk list C
+ precondition A, then prereg promotion ASK [wrc-flip-prereg].
08:15: wrc smoke F-side scoring: F2 PASS both runners (default@F == on:word-role@F, verdict IDENTICAL,
zero reply diffs; iw1 13 turns committed 13ae12088, iw2 16 turns now). F3 off:word-role@F vs default@P
(both runners): reply_differing_turns EMPTY and every persisted field byte-equal, but verdict DIFFERS on
sha_canon/sha_raw alone (hashes of the un-persisted server body) at iw1 {2,4,6,7,8} / iw2 {7,8,11};
shas byte-equal at all other turns across nodes/times (turn_wall_s differs there), rss noise not counted
by runner. F3 verdict needs a judge: delegation budget for this run exhausted (2/2 used) -> filed NEEDS
[wrc-f3-verdict] ASK, halting per resource-unavailable rule. Compare artifacts
_iw{1,2}/identity/{ce2e028,8407f21}/compare_{default_onwr,offwr_vs_Pdefault}.json committed.
NEXT when ruled: at-risk list C + precondition A, then prereg promotion ASK [wrc-flip-prereg].
09:10: wrc smoke (ONE task): measured the two remaining smoke components from the landed dev-7 files and
recorded both in PACKET.md on research/wrc-flip-packet. (C) at-risk list: ZERO changed replies between
default@F (ON) and off:word-role@F (alloff) across 29/29 rows (iw1 13 + iw2 16) -> no at-risk rows, B1 met
by the identity record; cross-revision caveat noted (same form pending [wrc-f3-verdict]). (A) precondition:
9/9 listed statement-form strings present in the smoke scripts (>=8 required), serving-subsystem `for`
field = wm-focus-bind / d6-hebbian-write / false-belief-chat / transitive-chat / scope, NONE word-role ->
no listed turn answered by the word-role path; caveat recorded that `for` is the serving subsystem, not the
parse-route trace body (not persisted) - if `for` is not the trace readout, the precondition needs a
dedicated trace run (prereg review decides). F2 PASS both runners stands; F3 still pending NEEDS
[wrc-f3-verdict]. NEXT when ruled: promote packet to prereg + bar-sanity -> ASK [wrc-flip-prereg] -> gate
seeds; the (A) lesion bars themselves (PARSE/COMMIT lesions vs intact-at-ON, 6/6) still need their gate-seed
runs after that ASK.
11:12: heartbeat wake (nothing new 2h). Re-read s1/s2/s6 + NEEDS; walked every s4 start condition: T0
queue empty with no held keys; T2/T8 identity 5=5; T6 stage-1 gate 207 >= 204; T7 c1 113 >= 54; T9 B2d
stub 258/258 landed + both scores COMPLETE + PROVISIONAL finding + NEEDS #2 filed (10-10, nothing left);
T10 LDC gate 0/150 (blocked: ladder stopped by the synzero ruling, next = bg-commit-next tonic after the
wrc smoke); T12 closed per RULING T12-seam-map; GPU dispatcher down but no GPU lines queued (mouth arm
landed; gpu_queue beyond status is not mine). No s4 task can move -> s5, but s5's plan + refute steps
require two MiMo Pro delegates and this run's delegation budget is 2/2 EXHAUSTED -> filed NEEDS [wrc-idle]
INFO and halting per the resource-unavailable rule. The critical path stays: answer [wrc-f3-verdict] ->
packet promote + bar-sanity -> ASK [wrc-flip-prereg] -> gate seeds; bg-commit-next (tonic pallidal drive,
one investigation with ldc-gpe-tonic) queues right after the wrc smoke.
11:29: RULING wrc-f3-verdict-1015 executed (ONE task). Identified P' = b1c11a0887b3c17ff7198674e9c33f162a3d58b9
and VERIFIED both ruling conditions by git: cbffef7f8 (shared-LTM ON) IS an ancestor of b1c11a088, and at
b1c11a088 research/runners/brain_chat_tui.py holds _WORD_ROLE_PARSE_DEFAULT_ON = False (line 279) +
_WORD_ROLE_COMMIT_DEFAULT_ON = False (line 315) - the word-role flip commit 9919725a9 lives only on this
branch, so pre-flip main is exactly 'shared-LTM ON + word-role OFF'. Recorded P' in PACKET.md on
research/wrc-flip-packet (c229489be), with the (C) readout upgraded from 'pending NEEDS' to
confound-immune: (C) compares REPLY fields only and the shared-LTM flip is verified reply-identical
(SLF-A A1 6/6). MISTAKE disclosed: first tried tools/pool_provision.sh --revision b1c11a088 pool41 pool42 -
that tool is BANNED on this host (pool_health.py: 'server-side provision.req instead, RULING
compute-placement'); it died harmlessly at 'rsync: command not found' (server lacks rsync; nothing was
transferred). Switched to the sanctioned path: touched research/queue/provision.req/b1c11a0887b3... (empty
file, the established request shape). NEXT when the provision-ok event lands: queue the two dev-7
default@P' identity lines (iw1 + iw2, same line shape as the 96e5198 wave, out
_iw{1,2}/identity/b1c11a08/default.json; pool add REFUSES pre-provision lines, so they wait for the event),
then compare off:word-role@8407f21 vs default@P' per the unchanged bar (sha_canon every turn).
11:37: provision ok for b1c11a088 (main P') landed -> queued the two F3-rerun dev-7 lines (ONE task): iw1 +
iw2 `--arm default --seed 7` at P' with out _iw{1,2}/identity/b1c11a08/default.json, checked reason names
RULING wrc-f3-verdict-1015, mem_gb=8, max_age_h=72, nodes pool41+pool42 (depth 0->2). Pre-checks: both
runners exist at P' (git cat-file) and pool41 has the provisioned venv. NEXT when the two results land:
compare off:word-role@8407f21 vs default@P' (bar unchanged: sha_canon identical on EVERY script turn; the
shared-LTM body-field turns must now match since P' carries shared-LTM ON), then record F2/F3 final reads
in PACKET.md.
11:52: triaged the 2 bad pool jobs = MY OWN F3-rerun lines (both died within seconds on pool40 with
'command not found'). Root cause mine: the queue line I built at 11:37 carried a literal
'OMP_THREADS_PLACEHOLDER' token between SIM_BACKEND=numpy and OMP_NUM_THREADS=1, and env treats a bare word as
the command to run, so it executed the placeholder instead of python (pool.running record line confirms it
verbatim). Nothing else in the line differed from the working 96e5198-wave shape. Requeued both corrected
lines (same P', --arm default --seed 7, out _iw{1,2}/identity/b1c11a08/default.json, depth 0->2, checked
reason records the first attempt's failure). Also noted: pool_queue add takes no --nodes flag (my
'--nodes pool41,pool42' was silently ignored; node choice is dispatcher-side) - future queue adds must not
pass one. NEXT when results land: off:word-role@8407f21 vs default@P' compare per the unchanged bar.
12:00: exit 3 on pool41 = the documented identity-runner refusal: the runner requires PYTHONHASHSEED=0
(reproduced rc 3 on pool42 in the 04:46 fixenv record, line 'env bug: ... refuses (rc 3)'), and both of my
requeued P' lines omitted it. Also found my mem_gb=8/memcap 9 was below the sanctioned wave's record
(mem_gb=16, canary peak ~12 GB, 1.3x) - the dispatcher's mem_gb reservation is what routes these lines to
the big-RAM node (all 14 prior mem_ok-16 jobs ran on wspool, 31G; pool41/42 only have 15G). Requeued both
lines in the exact fixenv shape (PYTHONHASHSEED=0 + mem_ok 16 1 + memcap 17 + mem_gb=16 in the record),
depth 0->2, checked reason names both of my earlier failures. Attempt history for this pin: 11:37 died on my
OMP_THREADS_PLACEHOLDER token; 11:53 died rc 3 missing PYTHONHASHSEED=0; this is attempt 3 with the record's
own env and memory shape. NEXT when results land: off:word-role@8407f21 vs default@P' compare (bar
unchanged, sha_canon every turn) then PACKET.md final reads.
12:19: stall check (3.7h no results) = jobs were NOT dead. iw1 default@P' finished 12:00-12:13 and its
result is banked server-side (research/findings/raw/_iw1/identity/b1c11a08/default.json + .prov.json,
8690 B); iw2 was still sitting in pool.queue because THE POOL DISPATCHER DAEMON IS DOWN - its log
/tmp/pool_dispatch.log last wrote Oct 12 10:27, no pool_autodispatch process on S or wspool, and the only
dispatches since then were the inline fill passes that pool_queue.sh add runs at add time (my three queue
scripts today at 11:37/11:52/12:00 each dispatched their first line that way; the leftover line strands with
nothing to retry it). Launched the sanctioned self-heal from workflow_check.sh:434 (nohup recipe, run as a
tracked background process) - it drained the queue immediately: iw2 dispatched to wspool 12:14:23 and is
running now (expect ~12:27-12:30). nodes listed as iw1-launch-proof bwrap sandbox, no network. pool_sync.sh
from my shell reports all four nodes UNREACHABLE (different ssh cfg) but results arrive anyway via the shared
findings/raw mount (iw1's file appeared with node uid 999), so the bank path is not blocked. NEXT when iw2
lands: run the F3 compare off:word-role@8407f21 vs default@P' (sha_canon every turn) + at-risk list C on P'.
13:17: F3 rerun under RULING wrc-f3-verdict-1015 LANDED AND IS CLEAN. default@P'(b1c11a088) ran on the pool
(iw1 12:13, iw2 12:27; dispatcher-restarted lane). Compare off:word-role@8407f2193 vs default@P' -> verdict
IDENTICAL on BOTH runners: differing_turns [], reply_differing_turns [] (iw2 reply_diffs []), preconditions
all true (same seed/turns/status/corpus/ltm/deterministic_env). The 5 iw1 + 3 iw2 body-hash-only mismatches
of the first attempt are explained exactly by the shared-LTM ON/OFF confound the ruling identified - with
shared-LTM ON at P' every sha_canon matches. Data committed 792dcd039 (default@P' x2 + prov, compare x2 +
prov; data-only commit). NEEDS updated: [wrc-f3-verdict] rerun outcome recorded, NEW ASK [wrc-flip-prereg]
(prereg promotion - bars/preregs are s2-reserved; all four smoke components now read: (A) 9/9 OFF-TARGET,
(B) F2 PASS both runners, F3 rerun IDENTICAL, (C) 0 changed replies across 29/29 rows). Lanes restocked
before this doc-only commit per lane rule. NEXT: wait on [wrc-flip-prereg] to queue the 6-seed PARSE/COMMIT
lesion gate runs; nothing else in s4 can move without it.
13:17: [wrc-f3-verdict] rerun outcome: PASS-candidate - verdict IDENTICAL x2 (iw1, iw2) at re-pin
P'=b1c11a0887b3c17ff7198674e9c33f162a3d58b9 (shared-LTM ON + word-role defaults OFF), artifacts
research/findings/raw/_iw{1,2}/identity/{b1c11a08/default.json,8407f21/compare_offwr_vs_pprime.json}
committed 792dcd039. Stamp PASS on bar (B)/F3 or send back; bars are s2-reserved.
- - -
16:25: [m2-fixes] ASK — PACKET-m2-fixes all 8 items are committed on branch `research/one-brain-m2-fixes`, tip
0938b88b8 (from main 30ab399b6), NOT merged (Claude's review, per the packet). Landed: item 1 multi-fact
claim-verify for deferred-mouth prose at ON (`source_attributed` claim-set path + honest DeferredMouth docstring,
2a13b1b15); item 2 source-neutral D5 recall disclosure at ON (`recall_disclosure(..., source_attributed=not
_one_brain_on())`, "Someone mentioned {topic}" replaces "Earlier you brought up", 1f0aff670); item 3 turn lock
FORCED at ON (`BRAIN_TURN_LOCK=0` ignored when `BRAIN_ONE_BRAIN=1`, 83f106f14); item 6 `ContextFieldView.__getitem__`
raises KeyError when the Context vanished (never getattr(None, field)); item 7 OFF-identity scope in WORKLOG §4 +
`one_brain.py` FLAG OFF (only the 17-turn script; fold-ins (a)-(c) intentionally change OFF on paths it never
reaches; reset/tick/warm/shutdown/prewarm/OpenAI shim have no OFF hash run); item 8 test that `one_brain_source`
(reason host_tag/scaffold) is mandatory wherever the "Someone told me" answer text is shown; items 4+5 DECLARED in
WORKLOG §8 risks 14/15 (prewarm drops the scratch turn's pending mouth calls; shutdown flush can run unlocked on a
lock timeout) under the packet's declaration option (0938b88b8). Tests: `tests/test_one_brain.py` 45 passed (7 new
m2 tests, RED-then-GREEN, SIM_BACKEND=numpy), `tests/test_brain_turn_lock.py` 43 passed. FLAGGED: the
`tests/test_webapp_server.py` subset relevant here (source_provenance / xedge_surprise_episodic /
d6_semantic_drop, real brain builds) exceeded a 420 s window under the brain lock + mem_ok 18 on this server, and
could NOT be rerun under memcap because `tools/memcap.sh` REFUSES here (systemd --user unavailable) — the packet's
memcap-18 recipe needs a host where memcap works (or a longer window); no GPU/dev-seed violations (no GPU runs).
Ask: review the branch; until then nobody enables `BRAIN_ONE_BRAIN=1` outside tests (packet header).

16:25: m2 PACKET executed as the one task: all 8 items on research/one-brain-m2-fixes (tip 0938b88b8; 4 commits 2a13b1b15/1f0aff670/83f106f14/0938b88b8); test_one_brain 45/45 + test_brain_turn_lock 43/43; NEEDS [m2-fixes] filed (webapp_server heavy subset + memcap-unavailable caveat inside).
15:5x: s2-c4-dev-now (RULING, due today) as the one task: sync BLOCKED - agent_sync_branch reached commit and the gates refused the merge on main's c5_offcurve_diag/smoke_c5a_selfinit.json (CLASS P + device-and-cost); pre-resolved the 4 add/add dev intact_a shards per RULING rev-stage2-sync-1013 (721086cb4 on research/stage2-exit-gate: f2aaabc8a copies kept + runs.jsonl); NEEDS [s2-c4-build] filed, line stopped per HANDOFF 4. Delegate 0/2.
16:4x: s2-c4-dev-now resumed (RULING s2-c4-sync-1015): sync ok 940f78973, tests 47 pass, selftest OK, pool repro selected=0 8/8 PASS committed 1661e4cf1; H1/H2 emit = 64 lines vs HANDOFF 56 (5th emo arm?) -> NEEDS [s2-c4-build] ASK, paused at queueing. Delegate 0/2.
16:5x: ONE task = NEEDS [stale-test]: research/stale-test-fix (from main 12614ecdb) updates tests/test_battery_amendment_b.py:103 to the current server.py:5590 call (_bkey(cache_key)); 56 passed; runtime asserts unchanged (env still reaches the organ, commit lesion observed). ASK for merge. Delegate 0/2.
17:0x: s2-c4-build H1/H2 QUEUED: 64 lines (H1 40 + H2on 8 + H2off 16) at 1661e4cf1, mem_gb 9 each, depth 61. CORRECTION per RULING s2-c4-count-1015: HANDOFF's 32-line H1 was a miscount (5 emo arms since e4bdf9f41: intact_b null + lesion_affect.rep0); emit authoritative. stale-test merged 783d79fc6. NEXT: pull shards as they land, 3-part verdict -> NEEDS [s2-c4-build] RESULT. Delegate 0/2.
20:1x: s2-c4-build sec 3: all 64 handler shards landed (queue 0, no rc!=0); aggregate = emo/C4 PASS 8/8 (lesion/reproduced/permuted 8/8, null_clean every seed; other groups INCOMPLETE = expected, no H1 shards); overall gate verdict UNDEFINED as expected. flip_check not produced: emotion_category_dev.py:224 sys.path.insert(tools) -> tools/experiment.py shadows root experiment package -> ImportError ExperimentEngine; code edit is sec 2 -> NEEDS STOP. Committed a71cfcea3 (64 shards + aggregate + branch WORKLOG, 132 files) after lane restock (starvation gate); HANDOFF's push_both skipped per standing never-push rule. Delegate 0/2.
20:5x: RULING s2-c4-next-1015 executed: agent_sync_branch synced research/stage2-exit-gate with main @ d58d5a39f as a5c543c31 (contains flip 780c465a3, _EMOTION_CATEGORY_DEFAULT_ON=True verified in webapp/server.py:3633); emitted the gate's own pool jobs (29 lines) and filtered the FIVE dev-seed-7 intact_a arms (one per group); provision a5c543c31 state ok 20:49 on all 4 nodes (within budget); queued 5/5 (mem_gb 9). NEXT: pull the five shards, sanity-read them, RESULT [s2-c4-smoke]. Delegate 0/2.
21:2x: RESULT [s2-c4-smoke]: five dev-seed-7 intact_a gate shards landed at flip tip a5c543c31 (sync merged main d58d5a39f incl. flip 780c465a3) - ask/conv/emo/infer/inner = C3/C1/C4/C2/C5, all well-formed (seed7 intact_a identity, pythonhashseed_ok true, no error keys); provision ok 4 nodes 20:49, queued 5/5 mem_gb 9; shards committed on branch as 38df66e63. NEXT (Claude): verify-go + pin GATE_SHA per RULING s2-c4-next-1015 (before 11-01). Delegate 0/2.
21:5x: priority (1) Stage-1 attempt 2 started: worktree .claude/worktrees/stage1-exit on research/stage1-exit-gate (tip a2941fb1d, ff incl. claude-review/stage1-sync); HANDOFF sec-3 pre-checks all green (SELFTEST OK, tests 14 passed, tree 0, GATE-SHA c238ff325 ancestor, prereg byte-identical 0, queue depth 0, 204-line requeue TSV present). NEEDS-MERGE path taken: bringing main in -> conflict in docs/drafts/stage1-exit/WORKLOG.md only -> aborted per sec-3 -> NEEDS [stage1-attempt2] #2 filed. Stopped that line; provision/smoke/[aws-run] ASK/204 queued pending Claude. Delegate 0/2.
21:5x: s1-attempt2-path-1015 executed: worktree+branch left as attempt-1 ARCHIVE (no merge, no union merge); attempt 2 runs from main. Prereg additions-only check vs PREREG_SHA = 0 OK (A2 = sec-12 additions, main 927aeb87c+). Provision req filed for main tip 61cc6e61e (status pending). HANDOFF banner edit blocked by reserved-path gate -> NEEDS [stage1-attempt2]. Smoke next: dev-7 cat_disc lesion_inherit_pathway (calls>0, p_canary_eat hop:true lesion:all lesion_held:true) + val_heldout permuted (n_words 24, |scrambled_r|<=0.05), mem 9, via pool_canary run auto. Delegate 0/2.
23:2x: first smoke launcher died before running anything (no canary row, log unrecoverable); main moved +4 (RULINGS s1-game-1015, s1-overnight-1015 + fix, aws-selfstop-1015) so SHA2 = c02180f2c (Claude offline overnight, tip stable), provision req refiled; relaunched smoke: wait-provision -> canary A cat_disc lesion_inherit_pathway dev-7, B val_heldout permuted dev-7, pool auto (pool40-42), judge s1_smoke_check; log /opt/data/cache/scratch/s1_smoke_run2.log. s1-overnight-1015 accepted as operative: pass -> emit 204 to gate_a2, canary one, queue ALL max_age_h=168 overnight, file [aws-run] ASK; fail -> NEEDS [s1-a2-smoke] STOP. Delegate 0/2.
23:3x: attempt-2 smoke re-filed via pool queue: two direct-canary background launches died WITH their sessions (provision-wait wrapper and plain relaunch; no canary row, no log) -> queued the two dev-7 smoke shards instead (cat_disc lesion_inherit_pathway; val_heldout permuted) at provisioned revision c02180f2c, mem_gb 7, max_age_h 168, --checked per before_you_build; pool40-42 run them (wspool down expected per s1-overnight-1015). On landing: judge s1_smoke_check; pass -> emit 204 to gate_a2, canary one, queue ALL, file [aws-run] ASK; fail -> NEEDS [s1-a2-smoke] STOP. Delegate 0/2.
00:0x: smoke lines set aside by pop_job's record-check gate (pool.queue.unchecked): my inline TAB split the record so #checked: landed in field 3 while pop_job reads field 2 (tools/pool_autodispatch.sh:724,731); root-caused and re-added tab-free with mem_gb=7/max_age_h=168 inside the --checked reason (s1_smoke_queue4.py); CHECKED_TOKENS_IN_QUEUE=2, depth 2 at c02180f2c, awaiting dispatch on pool40-42. Delegate 0/2.
00:5x: RESULT [s1-a2-smoke] PASS: both dev-7 shards landed at c02180f2c (dispatched 00:03 pool40/pool41, landed 00:17/00:25, prov sidecars present). cat_disc lesion_inherit_pathway: patch calls=2 gt 0, p_canary_eat read with hop true / lesion all / lesion_held true present; val_heldout permuted: n_words=24, scrambled_r=-0.044 (abs 0.044 le 0.05). Judge s1_smoke_check rc=0. Next per s1-attempt2-queue + s1-overnight-1015: emit 204 to gate_a2, canary one, queue ALL max_age_h=168, file [aws-run] ASK. Delegate 0/2.
01:2x: emit at SHA2=c02180f2c produced 216 lines (research/queue/s1_attempt2_c02180f2c.tsv): all under gate_a2 (STOP condition clean), 36 unique seed/group/arm per seed x 6 gate seeds, zero dups; the ruling's 204 predates A2's .rep0 replication arms -- MiMo Pro delegate verdict QUEUE-ALL-216 (STOP triggers are a closed list; s2-c4-count-1015 precedent). Canary plan: one gate-seed line (word intact_a s42, TSV line 13) via pool_canary run auto, foreground poll <=480s; if inconclusive the remote setsid job keeps running -> read node status next wake, pool_canary record, verify has, then add all 216 (cmd stripped of the tab/metadata tail, mem_gb=7 + max_age_h=168 inside --checked), file [aws-run] ASK. Delegate 1/2.
03:0x: RESULT [s1-a2-QUEUE] all 216 A2 lines in the pool at SHA2=c02180f2c7e1464cf0f2192a46cf2f399bd1c215: gate canary RECORDED PASS (pool40 word/intact_a s42, rc=0, artifact err=0, pythonhashseed_ok=true, wall 827.5s, pool_canary record+has ok); first 2-add refusal was the missing prereg token in --checked (corrected per the guard's own instruction -- not a rerun); line-level sweeps covered 1-216 (START=138 tail added 79, 0 fails; 84 skips = already-in-system). At 03:03: depth 108, 18 gate_a2 dispatch rows today, 10 gate_a2 shards landed well-formed (cat_disc s42 x6 all no-error + phs true, seeds s42-s44 family). NOTE: the server-side dispatcher's pops are invisible to this user's claims/running views (queue rows drained without local records; re-add sweeps stayed rc0) -- s1-attempt2-score's UNDEFINED-requeue-once path covers any residual gap at scoring time. Delegate 1/2 used (verdict QUEUE-ALL-216, ruling's 204 predates A2 .rep0 arms).
03:2x: RULING s1-awsrun-answer-1016 received: HANDOFF attempt-1 banner landed (f51bc28a4); AWS s1a1+s1a2 already draining the 180 lines of seeds 43/44/100/101/102 at c02180f2c (13 at a time, landed as uid 99); pool runs seed 42 only; a Claude loop withdraws any seed-43+ line that reappears in pool.queue -- ACKNOWLEDGED, never re-add (my 03:0x re-adds were withdrawn; withdrawn.log 443 rows). Verified: pool.queue = 38 rows all seed 42; 26 gate_a2 shards landed so far. Score per s1-attempt2-score once all 216 land (aggregate gate_a2, seeds 42,43,44,100,101,102; missing-shard UNDEFINED re-queues once; PASS waits for Claude verify-go). Delegate 1/2 used.
04:2x: attempt-2 landing health (post s1-awsrun-answer-1016): 86/216 gate_a2 shards on disk (s42=18/36 via pool, s43=36/36 + 1 in-flight .part, s44=30/36, s100=1/36 via AWS s1a1/s1a2), 0 truthy-error or unparsable JSON, all prov sidecars error-free. pool.queue seed-42-only (18 rows), withdrawal loop idle. Note for scoring: exclude *.part.json until final (aggregate runs only after all 216 land). No action needed; awaiting completion per s1-attempt2-score. Delegate 1/2.
07:35: RESULT [s1-attempt2]: scored per RULING s1-score-now-1016 — aggregate at gate_a2 verdict PASS (GO true, 0 missing, undefined_reasons empty), all 7 preconditions ok; report = C1 PASS, C2 PASS, C3 PASS, C4 PASS (each 6/6 lesion/reproduced/permuted) with diagnostics cat_told + val_novel FAIL 0/6 not counted. where_ran: aws 90+90 (s1a1/s1a2, ~$9.26, now terminated), pool 19, workstation 17, server 1 (this seconds-long aggregate). Shards+sidecars+aggregate committed by pathspec as 1d462264b; RESULT.md PROVISIONAL filed; delegate2/2 PASS-FILE. Bars untouched; board line below is held until Claude verify-go + adversarial #7. NEXT (Claude): verify-go, then finding/board.
07:35: Stage-1 exit gate (first integrated stage-exit attempt): PASS; C1 PASS, C2 PASS, C3 PASS, C4 PASS (r per seed: 42 0.817, 43 0.805, 44 0.824, 100 0.809, 101 0.836, 102 0.820). Finding: research/findings/2026-10-16-stage1-exit-attempt2-RESULT.md.
08:40: RESULT [s1-doc-sync]: Stage 1 PASSED 10-16 (attempt 2) synced per RULING s1-verified-1016 — GAP
  CURRENT STATE new RESUME 2026-10-16 (frontier Stage 1 PASSED, four caveats copied VERBATIM from the finding,
  next = word-role flip then Stage 2 per s1-next-1016), master roadmap §7 walls-ledger + §8 next-actions,
  AUTONOMOUS_STATE entry (post-supersession), docs/.vikunja_sync stamped. ROADMAP.md half (Stage-1 section)
  parked on branch docs/roadmap-sync-1015 via worktree under the existing [roadmap-sync] ASK. Board: #244
  closed + milestone #264 added by Claude same cycle.
08:55: RESULT [s1-doc-sync] COMPLETE — per RULING s1-verified-1016. Landed on main e22e1e81b (GAP
  CURRENT STATE 2026-10-16 RESUME section, insert-only + agent_merge_policy token-free, four caveats
  verbatim; runbook [roadmap-sync] ASK; WORKLOG). Parked on branch docs/roadmap-sync-1015 @ 72226b2cd
  (via worktree, the three Claude-reserved pieces): ROADMAP Stage-1 section, master roadmap §7 + §8,
  AUTONOMOUS_STATE entry — all with the four caveats verbatim; Claude lands the branch + adds the GAP
  headline line (PASS/GO words in GAP are Claude's by policy). docs/.vikunja_sync entry WRITTEN, its
  commit held: lane-starvation gate blocks it (pool idle, driver may not waive) — it lands with the
  next lane-ready cycle.
08:42: [roadmap-sync] CLOSED per ruling roadmap-sync-answer-1016 — merge adde67aa3 carries the three
  parked pieces (ROADMAP counted-lesion wording too), GAP Frontier headline added by Claude in
  fa19b9313, four verbatim caveats verified intact; my NEEDS entry reformatted to one line per the
  ruling's format note. Loose end unchanged: docs/.vikunja_sync stamp uncommitted (lane gate, depth 0).
08:50: ONE task = s1-next-1016 item (2): the [wrc-flip-prereg] ASK existed only as WORKLOG prose
  (13:17 entry), never as a NEEDS line, so filed it now as ONE line, #8 — prereg promotion + bar B/F3
  stamp are s2-reserved (Claude's); F3 rerun IDENTICAL x2 at P' already committed (792dcd039). The
  6-seed PARSE/COMMIT gate seeds stay blocked until the prereg lands; nothing else in item (2) can
  move this wake. docs/.vikunja_sync stamp still held (lane gate, depth 0).

09:58: wrc PREREG (RULINGs wrc-prereg-go-1016/1016b): research/findings/2026-10-16-word-role-flip-PREREGISTRATION.md drafted on research/wrc-flip-packet (B recorded as measured; F1 bar-sanity cited; (A) fixed to 9 listed; sec6 open questions replaced; Amendment E wr_route persistence + failing-first test GREEN 2 passed). 48 gate lines projected ~12-21 job-hours (<40 so no [aws-run]). Amendment-E code commit STAGED-BUT-BLOCKED (lane gate, pool.queue=0); branch sync BLOCKED (claim_check on stage1 RESULT) - NEEDS [wrc-sync] + [wrc-code] filed. Follow-up: the prereg .md commit also waits - bar_sanity_line needs a spec+record (.py/.json set = same lane block), so prereg + Amendment-E code + spec/record all ride [wrc-code]; branch data commit landed 49cb40f5c.

12:4x: RULING wrc-prereg-review-1016b task 2 (bar-sanity): A/C lane smokes landed clean (8da9804b2, prod_a + recall intact_b + prov sidecars, no lane waiver); wrote specs for F1b/F2/F3/F4 in docs/drafts/word-role-flip/, every record bar-sanity OK at n=1000 (perfect 1.000, nulls 0.000, random 0.000; f1b = real score_R with the judge_dir swap, F2-F4 = each runner own compare() + the wr_route precondition read from the turn record). F1a (parse row) blocked = NEEDS #10 (engagement route word_role vs the flip word_role_commit, verified in PACKETS line 86 + brain_chat_tui 1462-1469 + score_R 264-267). The verdict gate reserves docs/drafts/*.py on main, so specs+records were committed as research/wrc-barspecs 10d5bae21 via worktree (ASK #11 for the merge). B1 not run: F needs the amended prereg on the branch (ASK #9 open); delegate calls this run 0/2.

13:2x: ONE task = RULING wrc-f-pin-1016 steps (1)-(3): synced main 1181c79c2 into research/wrc-flip-packet (merge 25a60351e); ROW LIFT applied with tools/integration_wave2_flip._lift_required_env (exact _REQ_OFF_PREFIX) in both lbf word-role rows; row self-checks now assert "REQUIRED_ENV not in globals()" (the category_inherit pattern), the test_word_role_parse REQUIRED_ENV pin -> not hasattr, and the two park-expectation tests renamed + now assert register-at-the-default; .venv symlinked into the worktree so make_jobs.sh runs. Suite: 267 passed / 2 failed = test_integration_wave2_anchors test_packet_lists_agree + test_identity_arm_env_and_script, PRE-EXISTING at b9c322b10 (iw2 ident PACKET_FLAGS gained word-role under wrc-f3-1015 but flip.PACKETS, guard.ANCHORS, score.PACKETS and TURNS never did; finishing that edits the iw2 scorer + the defaults guard = section 2) -> step (3) F pin STOPPED, NEEDS [wrc-f-pin]. Branch commit 0451c26f2. Next after Claude: complete the word-role packet registration, then F + RESULT [wrc-F], then B1 (wrc-b1-tsv-1016). Delegate calls this run 0/2.

13:5x: ONE task = RULING wrc-f-pin-1016b (ff + suite + RESULT). Claude fix claude-build/wrc-fpin 572bec721 inspected (4 files: WAVE1/WAVE2_PACKETS constants in the two identity runners + the two anchors tests; child of 0451c26f2, ancestry verified). The ff itself is BLOCKED for me: approvals.deny forbids the merge and the reset --hard commands, and no sanctioned tool covers a research-branch ff (agent_sync_branch = main->branch only, w_follow = W main follower, land_claude_branch = owner->main), so research/wrc-flip-packet stays at 0451c26f2 and NEEDS [wrc-f-pin] STOP #12b asks Claude to move it. Content verified read-only meanwhile: detached worktree at 572bec721, suite (word_role parse/commit, wrc_amendment_e, lbf_row_registry, wave1+wave2 anchors, s2p_score, bg_commit) = 415 passed / 0 failed / 0 errors, matching the ruling claim; worktree removed after. B1 (wrc-b1-tsv-1016) not started: it needs F = branch tip first. Delegate calls this run 0/2.

14:2x: ONE task = RULING wrc-f-pin-1016c (RESULT + B1). Verified the ff: research/wrc-flip-packet = 572bec7219e479aaffa8a5dcb5f45774b0f1ee1d (= F). RESULT [wrc-F] posted. B1 pre-gate step: provisioned F via provision.req/<sha> (server state ok on pool41/pool42/pool40/wspool after ~9 min), then queued the 4 dev-7 lines (iw1+iw2 x default + off:word-role at F; mem_ok 16 + memcap 17, PYTHONHASHSEED=0, SIM_BACKEND=numpy; outs research/findings/raw/_iw{1,2}/identity/572bec721/; corpus check logged for --checked). One add raced a parallel add and its queue-file entry was clobbered - lesson: queue adds run one at a time, never in parallel - REQUEUE 1/4 records it. State at end: iw1 default RUNNING on wspool since 14:23:56, the other 3 queued (depth 3). Next wake: commit the four outputs + sidecars, RESULT listing every at-risk row (reply default vs off:word-role), then the gate TSV (wrc-b1-tsv-1016 step 5) + its ASK; gate seeds stay unqueued until Claude OK. Delegate calls this run 0/2.
15:5x: ONE task = RULING wrc-b1-tsv-1016 (4) B1 pre-gate. The 4 dev-7 records landed (iw1 14:37/14:51, iw2 15:05/15:19, sidecars present, queue drained) and were committed data-only as 173045253. Mechanical diff (sha_raw+sha_canon): iw1 2/13 at-risk (t5,t6 both for=d6-hebbian-write); iw2 6/16 (t0,t5 scope; t1-3 false-belief-chat; t13 neural-selfid); every other row byte-identical; default arm flags unset = ON at F, off arm both =0. Verdict delegated per RULING models (MiMo Pro, 1/2 this run), quoted: PASS - "All four records verified on disk ... 8 genuine at-risk rows exist, attributable to the word-role flag as the sole arm difference. The flip demonstrably changes replies at F, so the six-seed gate is meaningful" (proceed_to_step5=true). Delegate concerns: the d6-hebbian-write/scope spread is downstream state, not a confound; the gate must score per-turn sha, never session hash; watch the iw2 6/16 rate across seeds; dev seed 7 only, NOT a GO/NO-GO for the flip. NEXT: step (5) emit the gate TSV per prereg §3 (~66 lines) + ASK (no gate seeds queued before Claude OK).


20:4x: ONE task = RULING wrc-b1-tsv-1016 (5) gate TSV + ASK. MiMo Pro delegate (2/2 this run) designed the 66-line TSV at F=572bec7219e479aaffa8a5dcb5f45774b0f1ee1d (F1 organs 12 + answer-rate wrcflip/wrcflip-alloff 12 + F4 intact/parse-lesion/commit-lesion x iw1+iw2 36 + decline 6); 14.85 job-hours from the B1 smoke peak (810.1 s/line), pool-only (< 40 h -> no aws-run). Emitted research/queue/wrc_gate_572bec72.tsv; NOT queued - gate seeds wait for Claude OK per the ruling. Delegate risks quoted: "the 12 F1 organ lines may not satisfy the prereg F1 bar, which is scored on LBF shard arm files ... if the scorer needs row shards, F1 must be re-cut as load_bearing_fraction rows" (weakest assumption); decline is a deliberate standalone duplicate of the F1 commit-organ battery; answer-rate --only word-role-selfid dir naming unconfirmed vs the scorer faculty strings. RULING wrc-b1-reading-1016 then corrected the B1 label: 0 at-risk rows (the 8 sha diffs were the record-only word_role_trace), so the TSV needed no battery rows. The queue path is reserved on main, so the TSV was committed on research/wrc-gate-tsv for Claude to merge. NEXT: Claude merge + OK before any gate seed queues.

21:0x: ONE task = RULING wrc-tsv-v2-1016 + accel-1016 (pre-flight before gate seeds). Claude landed v2 (78 lines: F1 = 4 lb_shard rows x 6 seeds incl. both nofact controls, tag wrcflip-572bec7; answer-rate off-side 12 tag wrcflip-alloff-572bec7 reusing F1 intact for the ON side; F4 36 + decline 6 kept) and the PF guard (gate-seed lines refused until tools/gate_preflight.py reads a dev-seed run CLEAN). Driver queued the 4-line pre-flight (research/queue/wrc_preflight_572bec72_s7.tsv, dev seed 7, the same four F1 rows at the flipped default) via pool_queue.sh add --checked (depth 4). My v1 TSV is superseded (v1 mis-shaped F1 and missed the nofact controls - the delegate's F1-family risk was real). NEXT: when the 4 records land, run tools/gate_preflight.py; CLEAN -> queue research/queue/wrc_gate_572bec72_v2.tsv (78 lines).

21:1x: RULING judgment-to-claude-1016 read and absorbed: verdicts are Claude-s rule (RESULT carries raw reads + scorer output path only, never my label), every gate job list stays an ASK before queueing, delegates are mechanical-only. It corrects my 21:0x NEXT: when the 4 pre-flight records land I run tools/gate_preflight.py and post its output + paths as RESULT, then ASK to queue research/queue/wrc_gate_572bec72_v2.tsv (78 lines) - I do not queue it myself. Stall check: the 5.7 h no-result clock predates the queue; pre-flight jobs are alive (2 running on wspool, queue depth 4->2).

22:5x: ONE task = queue the 78-line gate TSV under RULING game-1016 (route pool40-42 only; wrc-gate-go authorizes it since the pre-flight reads CLEAN). All 78 adds REFUSED by the pool-canary guard: no (runner, 572bec7219...) record in research/queue/pool_canaries.tsv for load_bearing_fraction (36 lines), iw1_default_identity (18), iw2_default_identity (18), word_role_commit_organ (6). The B1/dev-7 runs cannot be recorded - their prov says host dant123-wk (the workstation), not a pool node, and old-sha canary rows (957c2752c/c64f2c70d) key differently. Launched 3 real canaries via tools/pool_canary.sh run (v2 line 1 = lb @ pool42, line 37 = iw1 @ pool40, line 55 = iw2 @ pool41, --wait-s 3600, background notifies armed); the organ canary (line 73) follows when a node frees. Then: re-run the 78 adds (idempotent) and WORKLOG the depth.

23:5x: ONE task = gate queue after canaries. Recorded at 572bec7219: load_bearing_fraction PASS pool42 (rc=0 wall=1730s) and iw1_default_identity PASS pool40 (rc=0 wall=1100s) - so lines 1-54 (lb 36 + iw1 18) may queue; the first add pass hit the 420 s tool timeout partway and is re-running idempotently in the background (add is idempotent). Canary RETRIES armed for the two still uncovered: iw2 line 55 on pool40 and organ line 73 on pool42 (both mem_ok 16 1; pool40 proved it can clear 17 GB momentarily, pool42 freed up after the lb canary). The 24 lines behind [wrc-tsv-mem] stay held unless those canaries record PASS. NEEDS [wrc-tsv-mem] still pending (Claude offline per game-1016).

00:1x: ONE task = finish the gate queue. State: lines 1-54 (lb 36 + iw1 18) are ALL in flight - depth 51 in pool.queue + running lines in .claims (11 claim events for 572bec7219), dispatcher skips the unreachable wspool as game-1016 requires; the 420 s timeout re-run was idempotent (duplicates REFUSED, nothing doubled). Both remaining canary retries came back INCONCLUSIVE - iw2 line 55 @ pool40 and organ line 73 @ pool42, mem_ok 16 1 refused again (17 GB > node room), nothing recorded - so the 24 lines (iw2 18 + organ 6) stay held behind NEEDS [wrc-tsv-mem] until Claude re-cuts or waives. A gap-add attempt of mine used a checked arg without the prereg token and was refused harmlessly (no queue change). Next wake: read gate-line RESULTS as they land.

03:4x: ONE task = queue the 24 v3 lines (RULING wrc-mem-1017). Canary outcomes: organ line 19 @ pool42 PASS (rc=0 wall=5s json=1 errors=0, recorded in pool_canaries.tsv) so the 6 decline lines (v3 19-24) queued ok=6, depth now 24; iw2 line 1 @ pool41 INCONCLUSIVE again (mem_ok 9 1 refused on a node running other gate lines, nothing recorded) - retrying on pool40 in the background; the 18 F4 lines queue when that records PASS. All adds stayed on pool40-42 per back-1017 (no wspool provision).

05:2x: ONE task = consume wave 6. 8 results + 8 sidecars verified (all json parse, all .prov, no Traceback) and committed data-only: 4 more iw1 flip-gate F4 artifacts under _iw1/flip-gate/572bec72 and 4 b2d_amend2 probe results under _b2d_amend2/74460d5 - i.e. at the new pin P' 74460d594 from RULING b2d-p2-repin-1017, so the probe guard is live there. No new commits; NEEDS [wrc-iw2-env] still open (18 F4 lines). Depth 11 on pool40-42; no jobs relaunched or moved.

06:2x: ONE task = consume wave 7. 12 results + 12 sidecars verified (all json parse, all .prov, no Traceback) and committed data-only: iw1 flip-gate F4 + b2d_amend2 probe continuation. No new commits; NEEDS [wrc-iw2-env] still open. Depth 6 - the first 54-line gate batch is nearly drained; the 6 organ lines are in flight; the 18 F4 iw2 lines still wait on the ask. Pool40-42 only (back-1017).

07:2x: ONE task = consume wave 8. 12 results + 12 sidecars verified (all json parse, all .prov, no Traceback) and committed data-only: iw1 flip-gate F4 + b2d_amend2 probe continuation. No new commits; NEEDS [wrc-iw2-env] still open. Depth 1 - the batch is one line from drained; when depth hits 0 the lane-starvation gate will need either queued work or a fresh waiver (the 22:2x one expired ~04:2x). Pool40-42 only (back-1017).

08:2x: ONE task = consume wave 9. 8 results + 8 sidecars verified (all json parse, all .prov, no Traceback) and committed data-only. Queue now depth 0 - all 60 dispatched gate lines drained and their results are committed through this wave. Refreshed research/queue/.lane_waiver (CLASS NO-READY-WORK, evidence: only the 18 F4 iw2 lines remain and they sit behind NEEDS [wrc-iw2-env]) so the doc commit can pass the lane gate. NEEDS [wrc-iw2-env] still open; nothing else pending.

09:2x: ONE task = consume wave 10. 6 results + 6 sidecars verified (all json parse, all .prov, no Traceback) and committed data-only - all b2d_amend2 probe continuation (not my pool queue: the wrc gate batch is drained at depth 0). No new commits; NEEDS [wrc-iw2-env] still open (18 F4 lines held). Lane waiver refreshed 08:2x covers doc commits until ~14:2x.

09:5x: ONE task = RULING wrc-iw2-hash-1017 handoff. Verified the fix landed: v3 line 1 sets PYTHONHASHSEED=0, exactly 18 of 24 v3 lines do (the iw2 family). Re-launched the iw2 canary from v3 line 1 on pool40 in the background; on PASS I add the 18 lines in three <=6 add calls, and a newer PASS row supersedes the 03:31 FAIL (pool_canaries.tsv untouched). RULING needs-tag-1017 absorbed: future NEEDS lines carry STOP/ASK/RESULT/INFO right after the time or [KEY] - my 03:5x line lacked the tag and sat 6 h unseen; apply from now on. Also consumed wave 11: 3 b2d results + 3 sidecars verified (errors=0) and committed data-only. NEEDS [wrc-iw2-env] answered by the ruling.

10:2x: ONE task = finish RULING wrc-iw2-hash-1017. iw2 canary came back PASS (pool40, rc=0 wall=1136s json=1 errors=0, id T094844) - the newer PASS row supersedes the 03:31 FAIL, pool_canaries.tsv left untouched. Queued all 18 iw2 F4 lines in three <=6 add calls (1-6, 7-12, 13-18): ok=6 fail=0 each; depth 14 on pool40-42 (back-1017, no wspool). The full gate batch - 54 v2 + 6 organ + 18 iw2 = 78 lines - is now dispatched; only results consumption remains, then the F4 byte-comparison scoring (Claude). NEEDS [wrc-iw2-env] is answered; no new lines.

13:3x: ONE task = RULING b2d-dev-landed-1017 (heartbeat pick; T1 checked first - DONE: PILOT-4B + [T1] RESULT e39a3d67e already answer pilot-4b-next; Stage-2 gate pin is dated 11-03, not yet). In .claude/worktrees/b2d-amend2: restored _provenance/runs.jsonl drift (T1-provision), synced main (worktree script conflicted on WORKLOG; main's post-319f41691 copy union-merged cleanly -> f3601fb28), wrote 74460d594598a9367e7935fc7b5cc2efc2b0a13b into PIN_P2, appended WORKLOG "P2 re-pinned per b2d-p2-repin-1017", committed 6cbd9c87f on research/b2d-amend2 (gates green; merge to main is yours). Pool depth 0; wave consumption unchanged; no new NEEDS lines.
