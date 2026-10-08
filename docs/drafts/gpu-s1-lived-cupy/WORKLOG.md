# WORKLOG: gpu-s1-lived-cupy (T24)

Branch `research/s1-lived-cupy` from main at `c96b7e445`.
GATE `c238ff325a38e40a68b1f9751822630ff729377b` reachable (guard-at-pin verified).

- 2026-10-02 19:43 (local model) — section 0: branch `research/s1-lived-cupy` created from main `c96b7e445`;
  GATE `c238ff325` reachable (stage1-exit-gate branches listed, guard-at-pin OK). section 1: 2 dev smoke lines
  (seed 7, intact_a + intact_b) queued via `gpu_remote.sh add --rev c238ff325…`, depths 1-2, no ADD-FAILED;
  yielded the GPU up to 2 h. devcheck next session when both `dev/s7` JSONs arrive (sim-sync).

- 2026-10-02 20:45 (local model) — section 1 devcheck: **GO 653.2 s** (`devcheck.py dev/s7`; intact_a 19:55 +
  intact_b 20:06 both landed, DONE(rc=0), null held). 653.2 <= 2400 -> no INFO line. section 2: all 42 gate-seed
  lines queued at `c238ff325` (depths 1-42, no ADD-FAILED), GPU yielded up to 12 h. sections 3-4 (aggregate +
  informational write-up) wait for T6's gate to be scored (RULING `s1-lived-cupy`).

- 2026-10-03 12:58 (local model) — **section 2's ONE retry executed** for the two missing shards. The 10:28
  yield drain said "42 job(s) ran" but only 40/42 shards landed: s43 `lesion_recall` + `.rep0` absent (the other
  5 s43 arms landed 01:05-05:20). Evidence recovered by grepping the workstation's full
  `research/queue/gpu_queue.log` (10.9 GB / 112M lines — `gpu_remote.sh log N` only ever tails a minutes-long
  window): `DONE(rc=143)` 01:41:01 (`--arm lesion_recall`) and `DONE(rc=143)` 04:38:30
  (`--arm lesion_recall.rep0`) — rc=143 = SIGTERM (external kill under pause/yield semantics), not a code fault.
  Section 2's precondition ("ended DONE(rc!=0) (log)") is therefore met literally; this is line 1 of the single
  allowed retry — a second failure stays missing and §3 reads UNDEFINED for seed 43. Judgment routed per RULING
  models: verdict + independent refutation delegates both answered
  "RE-QUEUE ONCE each — the rule's precondition is verified, not missing" and
  "DOES NOT REFUTE — the record's own registered handling; rc=0-without-shard is impossible in the runner, so an
  absent --out after a drained run is proof the line failed." Pre-add checks: both `--out` files re-verified
  absent; queue empty (no re-add of queued lines); `make_jobs.sh` unchanged since 9320b428f so the regenerated
  gate lines are byte-identical to the originally queued 42 (delegates also diffed them against the recovered log
  commands); commands still carry the unchanged `--prereg …-PREREGISTRATION.md --allow-gate-seed` at code pin
  `c238ff325`; no numbers read (blind-until-T6-scored intact). Actions: 2 × `gpu_remote.sh add --rev c238ff325
  '<line>'` (queued depths 1-2, rev c238ff325a38e40a68b1f9751822630ff729377b), then
  `gpu_remote.sh yield --max-hours 12` (local-llm stopping gracefully, deadline epoch 1791089919).
  Counting note for §3: two stale `.part.json` debris files sit beside finished s42/s100
  `lesion_episodic` shards — exclude them from shard counts. sections 3-4 still wait for T6's gate score.

- 2026-10-04 03:2x (local model) — **sections 3 + 4 DONE.** Preconditions: T6's gate scored (stage FAIL, C3
  FAIL, 01:03 wake); 42 lived shards in `$P/$O` (44 glob = 42 + 2 stale `.part.json`, excluded per the note
  above); no mouth/lived lines left in either queue. `rsync` absent on S (exit 127, logged skip) →
  `cp -a $P/$O/. $WT/s1-lived-cupy/$O/` (same end state, dev included). Runner byte-identical first:
  `git -C $WT/stage1-exit diff --quiet c238ff325 HEAD -- research/runners/stage1_exit_gate.py` → **same**
  (the module is not on main, so the aggregate runs cwd `$WT/stage1-exit`). Aggregate rc=0; report: stage
  verdict UNDEFINED (INCOMPLETE) by design (other criteria not run), `lived` C3 **FAIL — correct 5/6,
  lesion_recall 0/6, lesion_episodic 5/6, permuted 5/6, reproduced 6/6 + 6/6; seed 102 fails
  (correct=False; lesion_episodic did not change p_wolf_discuss)**. PROVENANCE fix (root-caused, not
  bypassed): the first aggregate wrote no sidecar — `_stamp_outputs` stamps only paths under the running
  package's root and my `--out` was an absolute path into the OTHER worktree → silently filtered (n=0, no
  warning). Re-ran byte-identical with a RELATIVE `--out` under `$WT/stage1-exit` → stamped
  `aggregate.json.prov.json` (`sim_backend: numpy`), moved the three files over; `cmp` confirms the
  re-run output identical. Stage-out tmp files in `$WT/stage1-exit/.../gpu_lived_cupy/` remain untracked
  scratch (delete bursts are scanner-blocked; never staged). Finding
  `research/findings/2026-10-04-stage1-lived-cupy-replication-INFORMATIONAL.md` (PROVISIONAL line, the
  HANDOFF's fixed sentences, `lived` block pasted unedited).
- Board line: 2026-10-04 stage1-lived-cupy: informational cupy C3 lived FAIL (5/6 seeds, seed 102 absent), numpy gate C3 FAIL at the same seeds; not a gate arm. Finding: research/findings/2026-10-04-stage1-lived-cupy-replication-INFORMATIONAL.md.
- Next: commit branch (finding + `$O` + this WORKLOG), merge per RULING `routine-merges` (informational,
  RULING `results-without-pass`), RESULT `[T24] RESULT: cupy lived FAIL (5/6); numpy gate C3 FAIL;
  research/s1-lived-cupy @ <sha>`. Never push (sim-sync publishes main).

- 2026-10-07 20:30 (Hermes) — **T24 data landing, per RULING `T24-data` (2026-10-04).** Staged `$O` + this
  WORKLOG on the branch; `corpus-check-required` refused exactly the recorded paths — cited here by path,
  these stay UNCOMMITTED on `research/s1-lived-cupy`, never re-stamped or bypassed:
  `research/findings/raw/_stage1_exit/gpu_lived_cupy/s42/lived__lesion_episodic.json` (1.1 h),
  `research/findings/raw/_stage1_exit/gpu_lived_cupy/s42/lived__lesion_episodic.json.part.json` (stale debris),
  `research/findings/raw/_stage1_exit/gpu_lived_cupy/s42/lived__lesion_recall.rep0.json` (1.4 h) — the ruling's
  "2 refused shards" plus the `.part` debris. The other 40 cells + dev + aggregate + sidecars commit now
  (42 - 2 = 40; the two cells the gate refuses are s42 `lesion_episodic` + s42 `lesion_recall.rep0`).
  Next: two MiMo Pro reviewers (RULING `routine-merges`), `tools/agent_merge.sh research/s1-lived-cupy`,
  then RESULT `[T24] RESULT: cupy lived FAIL (5/6); numpy gate C3 FAIL; research/s1-lived-cupy @ <sha>`.

- 2026-10-07 20:55 (Hermes) — **T24 data landing, RULING `T24-data` executed.** Branch `research/s1-lived-cupy`
  @ `15d012909` = 40 gate-accepted cells + dev + aggregate + sidecars (86 files) + this WORKLOG; the 2
  corpus-check-refused s42 shards + `.part` debris stay UNCOMMITTED on this branch (paths cited in the entry
  above), never re-stamped or bypassed; main's pre-merge untracked copies of the 86 incoming files verified
  `cmp`-identical and removed so the merge can write the tracked versions (content unchanged). Two MiMo Pro
  reviewers (RULING `routine-merges`): adversarial verify-go **SURVIVES-WITH-CAVEAT**; merge approval
  **APPROVE-MERGE** (`agent_merge_policy` pre-screen "OK, 87 routine change(s)"; aggregate/finding/WORKLOG
  numbers verified identical). Caveats carried into the RESULT:
  (1) `lesion_recall` is **UNDEFINED at pin `c238ff325`** (the env-flag never engages on the production bus
  path — `2026-10-04-stage1-exit-gate-RESULT.md` Claude correction 2), so 0/6 and 6/6-reproduced must NOT read
  as "recall not load-bearing";
  (2) the landed finding's sentence "The numpy gate C3 FAILs at the same seeds" is **mischaracterized**: that
  as-scored FAIL rested solely on the void lesion_recall clause, was corrected to UNDEFINED (instrument) and
  re-measured PASS 6/6 at pin `7851dd220` (Amendment A1) — not a cross-backend FAIL corroboration; the
  finding edit is §2-reserved, so a NEEDS line asks Claude;
  (3) 2 of the 42 cells behind the counts are the uncommitted refused shards (s42 `lesion_episodic`,
  s42 `lesion_recall.rep0`);
  (4) seed 102 is a real miss, record complete (all 7 shards present, undefined_reasons=[], null_clean=true);
  (5) backend provenance: shards record `backend='cupy'` from the request; only 2/42 `.prov.json` carry a
  resolved-cupy child stamp, the rest show the parent-process numpy stamp.
