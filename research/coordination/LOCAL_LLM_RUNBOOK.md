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

**T4. Host-share baselines.** `$WT/host-share-baselines/docs/drafts/host-share-baselines/HANDOFF.md` sections 7-11
(GATE `f7f2bd169`); WORKLOG `docs/drafts/host-share-baselines/WORKLOG.md`. Start: section 7's count = 108; stalled
12 h below it: re-queue ONCE (section 3 block, run for `F=research/queue/requeue_host_share_f7f2bd169.tsv` and for
`F=research/queue/requeue_host_share_f7f2bd169_dispatched_failed.tsv`; never `/tmp` or `59edf3255` lines). First: section 8. Write: finding + section-9
sentences, RESULT; BASELINE-VALID also #8.

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

## NEEDS CLAUDE

Format (one line, at most 300 characters, newest last; commit at once; Claude deletes handled lines):
`- YYYY-MM-DD HH:MM [<task or KEY>] <STOP|ASK|RESULT|INFO> <#n>: <what you saw>; tried: <what>; blocks: <task>`

- 2026-09-30 16:50 [open-items] INFO (seeded): other judgment items stay in each HANDOFF's NEEDS-CLAUDE list
- 2026-10-01 19:54 [LDC-grid] INFO (corrected): grid ALIVE and progressing - 33 runner procs each on pool41 + pool42, outputs landing (4 + 8 in the nodes' checkouts); the earlier 0/0 read S-side ps (pool jobs never run on S) plus the S output dir (copy-back pending).
- 2026-10-01 23:12 [T13] STOP #1: cupy re-run DONE(rc=1) 19:39 - 3 passed/1 failed; failing tests/test_bounded_conductance_integration.py::test_byte_identical_when_off; tried: re-run on W gpu after test fix 9e8edbec1; blocks: T13 merge
- 2026-10-02 04:56 [T23] 18 mouth cells already present: b2d0929-mouth/s{42,43,44,100,101,102} x3 fac, mtimes 10-01 10:57-19:39, ran at P=5818dc6 from /home/dant123/derisk-pool/revisions export, uncommitted, no s7; tried: n/a (packet table STOP); blocks: T23 preflight+18-line queue
- 2026-10-02 07:28 [transitive-spiking-reach] canary R INCONCLUSIVE x2: mem_ok refused 8G on pool42 07:26 and pool41 07:27 (same block as [fb-spiking-witness] STOP #1); 12 R lines held; O canary PASS + 6 O queued; blocks: T19 sections 4-5
- 2026-10-02 08:47 [fb-spiking-witness] O PASS + R PASS at research/fb-spiking-witness @ 40c43d520 (op 4 readout; verify-go 2 MiMo Pro refuters both SURVIVES; finding research/findings/2026-10-02-fb-spiking-witness-gate.md PROVISIONAL); confirm + merge; unblocks P4 (s2p-flip)
- 2026-10-02 09:31 [word-role-parse] canary R INCONCLUSIVE x2: mem_ok refused 8G on pool42 + pool41 09:29 (same 8G block as the [fb-spiking-witness] and [transitive-spiking-reach] lines); 12 R lines held; O canary PASS + 6 O queued; blocks: word-role-parse sections 4-5
- 2026-10-02 16:55 [host-share-baselines] BASELINE-VALID (e7475748c): k/v/bg HOST-OWNED, organ-c HOST, metacog host-replaceable, ignition MIXED. HANDOFF #8: sync summary docs + board and hand the bg-select and ignition numbers to the step-3 builder as its "before" values.
- 2026-10-02 17:45 [selection-seam-need-reversal] BG live-seam retirement gate NO-GO: only G4 fails (latch read not threshold-invariant, latch_thr_change 0.25) - finding research/findings/2026-10-02-bg-live-seam-retirement-RESULT.md (0466e8c24). No fix started; NEEDS the next method.
- 2026-10-02 18:45 [draft-seam-map] T12 amend committed locally (a6aff478c; amendment verbatim, >-stripped, Markdown-only) but research/draft-seam-map is unpushed (amend + main merge ahead of gitea) - push is reserved for Claude.
- 2026-10-03 04:06 [T1] RESULT #1: Step 4a pilot @50f52c23e; 41 cells + PILOT.md @4e6b19ec2: DA signs 1/24, sanity>=.8 1/12, credit>=.8 0/12 -> A0 branch = eligibility-tag action-specificity (marginal 1 seed, 0.0013); tried: below-chance decl; blocks: A0+Step 4b
- 2026-10-04 00:35 [draft-ca3-symmetric-ti] RESULT #1: dev 4/4 row-2 WRITE-ABSENT shape, but row 11 fires 4/4 (rev residual 34-47 pA) so the table reads UNDEFINED(dev):instrument; verify-go caveat applied; ASK: row-11 scope rev-scoped or per-seed? blocks: WRITE-ABSENT word, sec-13
- 2026-10-04 00:45 [draft-ca3-symmetric-ti] STOP #2: agent commit refused - verify-go found a mischaracterization in research/findings/2026-10-03-ca3-symmetric-ti.md (row 11 void 4/4 -> headline must read UNDEFINED(dev):instrument; rewrite ready in session); editing an existing finding is reserved; tried: commit-through-gates; blocks: finding caveat landing, routine-merges
- 2026-10-04 01:20 [stage1-exit] #7: first stage-exit attempt scored 204/204 via the registered scorer -> stage verdict FAIL (expected, prereg s9) but C2 word PASS (correct 6/6, lesion_recruit 6/6+reproduced, permuted 6/6); verify-go adversarial skeptics required before any board/ledger word for the PASS (HANDOFF s10 #7); tried: aggregate AGG_RC=0; blocks: board line, finding promotion
- 2026-10-04 01:20 [stage1-exit] INFO: result commit local on research/stage1-exit-gate in $WT/stage1-exit (finding research/findings/2026-10-04-stage1-exit-gate-RESULT.md + aggregate); push_both skipped - push reserved for Claude
- 2026-10-04 02:25 [stage1-host-share] RESULT MEASUREMENT-VALID at research/stage1-host-share (§6 rc=0, gate 132/132, V1-V7 all pass); C2 PARTIAL: all 5 seams HOST-OWNED, panel host 1.000 / neural 0.000 (n 48); C3 PARTIAL: all 5 HOST-OWNED, host 0.967 / neural 0.033 (n 30); confirm + fold the credit lines into the Stage-1 exit report (Claude's edit); finding research/findings/2026-10-04-stage1-host-share-RESULT.md (PROVISIONAL)
- 2026-10-04 04:00 [T24] STOP/RESULT: cupy lived FAIL (5/6); finding+aggregate merged on main 85ab5881e; 42 cell JSONs WITHHELD: corpus-check-required refuses 2 shards (1.1h+1.4h, no corpus check before the 10-02 queueing; after-the-fact evidence impossible); tried: before_you_build + FAILURE_LOG row; blocks: T24 data land + close
- 2026-10-04 04:05 [T24] INFO: the corpus-check lapse FAILURE_LOG row is written on S's main worktree but UNCOMMITTED (the agent backstop reserves research/FAILURE_LOG.md for Claude on branch and main alike); lands with your data ruling
- 2026-10-04 05:25 [T5] merge research/selection-seam-need-reversal @ ca5ac43ae: policy-proven agent-merge refusal (branch diff vs main touches tools/lab.py, webapp/server.py, GAP_CLOSURE_MISSION.md = outside the routine classes; no merge attempted, per agent_merge's own policy text) - finding 2026-10-02-bg-live-seam-retirement-RESULT (NO-GO, G4 only) + its six-seed score still local on the branch; unblocks landing the NO-GO record. Step-6 re-run today reproduced the verdict exactly (score.json's 8-line argv/timestamp delta left uncommitted).
- 2026-10-04 06:12 [T12] resource unavailable: research/findings/2026-09-27-tone-bg-nchannel-tonic-firing-wall.md does not exist on main (nor in docs/RETRACTED.md) - T12's Start grep cannot run, so the seam-map amendment block has no target; where did the file go?
- 2026-10-04 06:45 B2d mouth arm: yield refused (refusing to yield: GAME_MODE is set (owner gaming/away) -- see: bash tools/game.sh status) - the 18 registered mouth lines ARE queued from S per RULING b2d-mouth-gpu (gpu_remote.sh add 18/18, remote depth 18, dispatcher up, state PAUSED); they start when the owner's card is free and yield succeeds; re-attempt `bash tools/gpu_remote.sh yield --max-hours 12` next session
- 2026-10-04 08:15 [T23] B2d mouth arm failed: DONE(rc=143) 08:04:51 ~9s into first dispatch, b2d0929-mouth/s42/affect-coloring; GAME_MODE pause-term suspected (guess); tried: log read + yield retry still refused; 17 lines queued PAUSED; blocks: T23 mouth
- 2026-10-04 13:35 [§5 xedge-prov-surprise-reciprocal] TWO items. (1) SELFTEST STOP (2nd failure, §2 one-fix-round): the read-isolation guard on the new draft runner cannot fail-in-its-failing-direction — attempt 1 (contradict-drive induction + full-trial read) fix-off reads bitwise IDENTICAL (strong trial drive is contractive); attempt 2 (template-2-style weak read) degenerates to surprise=0.0 (surprise only fires on the mismatch trial). Runner honestly returns SELFTEST FAIL on branch research/draft-xedge-prov-surprise-reciprocal (WORKLOG 5d60e5e5a). Needs Claude: port template-2's amb_read margin probe verbatim (requires the ambiguous-pattern machinery this port dropped) or rule the guard optional for this draft. (2) POOL ADD REFUSED (x2, ADD_RC=2): no reachable node for revision 62a83bbbe086a023c82425932d83920f11890682 — wspool unreachable (GAME_MODE/workstation paused), pool40 BROKEN (code missing), pool41/42 healthy but unprovisioned for this revision. Corpus check ran 13:2x, smoke line shape-checked RC=0, provision.req touched 13:24 — if /opt/host-status/sim-provision/62a83bbb*.json does not land, node restore is Claude's; the ready-to-add line retries next wake either way.
- 2026-10-04 13:56 [§5 xedge smoke STOP] SECOND failure = STOP (§2 one-fix-round). Root cause fully identified from synced stderr (research/findings/raw/_xedge_prov_surprise_reciprocal/diag_smoke_seed7.txt): run_gate expects pool.bridge but the ported ProvSurprisePool names it pool.b (template-1 attr) -> AttributeError at onebrain_crossedge_gate.py:218 on both attempts (exits ts 1791135275 and 1791135757, pool40, no smoke_seed7.json written). One-line fix on branch research/draft-xedge-prov-surprise-reciprocal: expose `@property bridge: return self.b` (or rename b->bridge throughout); NOT memory (memcap 8G enforced fine on pool). Please land the one-liner or hand it back to the driver; then the smoke line re-adds verbatim (shape-checked pattern in scratch xedge_smoke_requeue.sh minus the 2> diag redirect). Companion open item: selftest STOP (13:35 NEEDS above).
- handled: [T23] B2d rc=143, B2d mouth yield refused (GAME_MODE) -> RULING game-mode-pauses (2026-10-04)
- 2026-10-04 17:14 [s1-c4-void] STOP: re-provision at c238ff325 -> state=already; ruling's confirm ls: _affect_learned_valence_map.json MISSING on pool41+pool42 (S has it); tried: provision.req touch (iw2-provision); blocks: 72 val_ re-queue under _stage1_exit/gate_c4rerun
