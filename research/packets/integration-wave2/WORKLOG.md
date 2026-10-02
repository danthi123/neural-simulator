# T2 Integration wave 2 WORKLOG

## 2026-10-01 16:55 (Hermes, local model): section 5 scored; commit blocked by stale branch gate

- Worktree created from LOCAL `research/integration-wave2` @ 758519b3f (RULING iw2-R9: never from origin).
  R precheck done: `grep -c lesion-kept tools/iw2_score.py` = 4 >= 1, so R scoring was valid.
- Packet status: identity 5/5, iw2-R shards 36/36 -> Start holds. Section 1.2 T0 check: no pool-hold.
- Section 5 scoring run (S paths; env.sh/copy_results.sh workstation paths overridden per runbook, no code edits):
  - D0: ok=true, verdict IDENTICAL (expect same) -> D0 PASS.
  - D_false-belief-chat / D_transitive-chat / D_neural-selfid: all DIFFERS-IN-SCOPE (D_p + S_p both pass).
  - R_fb / R_tr / R_si: all PASS, n_defined=6, n_reply_load_bearing=6 each.
  - Section-5 table row 1 -> all three packets flip-eligible; iw2-selfid extra condition MET (on-arm turns
    14 "what are you?" and 15 "who are you?": abstained=true, recalled_svo=null) -> FLIP =
    false-belief-chat,transitive-chat,neural-selfid.
- copy_results done (identity + 3 shard tags, P corrected to the S checkout); verdicts + landed results
  STAGED in the worktree, uncommitted.
- BLOCKED: section-5 commit stopped by the biology-binding gate (101 x "SOURCE PATH DOES NOT RESOLVE"
  ~/Projects/sim-catalog/...). Root cause: the branch's `tools/biology_check.py` predates main's 2026-09-30
  container-HOME fix (`_projects_dir`/`_expand` absent); HOME=/opt/data/home in this container. The fix exists
  only on main; merging is reserved -> NEEDS-CLAUDE "[iw2-gate]" written to the runbook. Per HANDOFF stop
  conditions: STOP (never --no-verify, never a waiver, never editing a gate file).
- NOT done (waiting on the merge): section-5 commit, section-6 flip apply/tests/guard/commit, PIN_F append,
  provision F (compute-placement: provision key is workstation-interactive), section 7 job files + queue waves.
- Note for the finding's Deviations (HANDOFF NEEDS-CLAUDE #3): neural-selfid's "what/who are you" turns now
  abstain (verified in on_neural-selfid.json @ 573a943).
## 2026-10-01 17:52 (Hermes, local model): merge landed; section-5 commit now blocked by verdict-preconditions (scorer-side fix reserved)

- Claude merged main into the branch (dacf6cb00; my 377 staged files kept staged, biology gate now passes).
- Re-ran section 5: copy + add OK; commit BLOCKED by `verdict-preconditions`: R_fb/R_si/R_tr assert PASS with
  no `preconditions` block (3 problems; D0/D_* pass — `iw2_default_identity.py` emits one; dev_s7 has no R
  precedent). Root cause: branch `tools/iw2_score.py::score_R` (lines 251-282) never calls `tools/verdict.Verdict`
  — the gate requires runtime-emitted preconditions in the artifact. Fixing it = editing a scorer = section-2
  reserved -> NEEDS-CLAUDE "[iw2-R-pre]" written, STOP per HANDOFF stop conditions (never --no-verify).
- Flip NOT run (section 6 follows the section-5 commit). Provision F not attempted. This WORKLOG line and the
  verdicts stay staged until the scorer emits preconditions and the R verdicts are regenerated.
## 2026-10-01 19:29 EDT  section 6: flip commit F (PROVISIONAL, branch only)
- Synced main into the branch (agent_sync_branch, 994e5dd74): brought the fixed
  score_R + the PASS R verdicts WITH preconditions + R shards; my stale local R
  copies were deleted (superseded by main's committed ones).
- Section 5 verdicts (from main): D0 PASS, D_p DIFFERS-IN-SCOPE, R_fb/R_tr/R_si
  all PASS with preconditions -> FLIP list = all three packets.
- Created $W/.venv + $W/data symlinks per HANDOFF line 28 (missing -> anchors
  test failed on a relative .venv path).
- anchors pytest: 123 passed, 3 skipped; guard-ok; flip status all three ON.
- Board synced BEFORE the commit (board_sync gate): append-desc on #232
  (false-belief), #141 (transitive reasoning), #209 (self-identity rank-13) -
  provisional note only, NOT marked done by the flip (validation pending).
- F commit through gates, no push (standing rule: sim-sync publishes main),
  then PINS row F + pool_provision isolated pool41 pool42.

## 2026-10-01 20:43 EDT - section 7 step B/F start (provision ok, jobs committed, wave 1 held for canary)
- RULING iw2-provision: provision request for F=1fb805e9e REFUSED (stale vs published main
  8df14a06d) -> agent_sync_branch main into the branch @ 94e0760a7 (contains main + the flip);
  PINS re-pinned F = 94e0760a7 (append-only row, env.sh takes the last F); provision requested
  for 94e0760a7 -> state ok, "provisioned on pool41, pool42" (6998 files, 111 MiB + 278 MiB data).
- make_jobs flip @ F = 276 lines (240 LB + 36 guard), identity = 3 lines; all 276 carry the
  required assert_iw2_node + assert_iw2_defaults <FLIP> prefix (0 violations); committed
  3f9d77449 (jobs + PINS row + shard PIN.txt; no push - standing rule).
- wave 1 queue REFUSED by pool_queue: no pool canary record for (load_bearing_fraction,
  94e0760a7). Canary launched on pool41 (line 1 of flip_94e0760, background, wait<=14400s).
  NEXT WAKE: on PASS -> queue wave 1 (sed 1,43) + step F identity (3 lines) + waves.tsv, then
  waves 2-7 while fewer than 10 iw2-flip lines remain.

## 2026-10-01 20:59 EDT - CORRECTION: F back to 1fb805e9e (RULING iw2-provision UPDATE 00:40Z)
- Ruling: "flip F 1fb805e9e IS provisioned (owner-approved one-off). Do NOT sync or re-pin flip F:
  run steps B and F at 1fb805e9e as the HANDOFF says." The 94e0760a7 re-pin (my response to the
  stale-refusal) is superseded; its PINS row stays as audit trail - append-only, env.sh takes the
  LAST F row = 1fb805e9e.
- make_jobs needs HEAD==PIN when the branch head differs beyond the packet dir (94e0760a7 has
  runbook + session2-packets code) -> built the battery in a detached worktree at 1fb805e9e
  (git worktree add; .venv + data symlinks per HANDOFF line 28): flip_1fb805e.txt 276 lines
  (240 LB + 36 guard, 0 prefix violations), identity_1fb805e.txt 3 lines; copied into the branch;
  superseded 94e0760 job files removed (they were never queued - pool_queue refused, 0 landed).
- Canary (load_bearing_fraction, 1fb805e9e) NOT yet run: both pool nodes still refuse mem_ok 8 2
  (LDC grid's memcap scopes hold 5-6G committed-unused). NEXT: canary when mem_ok passes -> wave 1
  (sed 1,43) + step F identity + waves 2-7 while fewer than 10 iw2-flip lines remain.

## 2026-10-01 22:12 EDT - canary PASS; wave 1 (43 lines) + step F identity (3 lines) QUEUED
- Canary (load_bearing_fraction, 1fb805e9e, gate seed) PASS on pool41 at 22:09 (rc=0, json=1,
  errors=0; recorded in pool_canaries.tsv) after the LDC grid's last cells freed the memory window
  (mem_ok 8 2 passed: avail 10G - 4G committed). The canary's own outcome is n/a 0/0 by design -
  the redirected _pool_canary dir carries no interaction data; the rule reads only rc/json/errors.
- Queued through pool_queue: wave 1 = 43 lines (sed 1,43 of flip_1fb805e.txt, reason "iw2 step B:
  prereg ...wave-battery-PREREGISTRATION.md ; mem_gb=8 ; max_age_h=216") + step F = 3 identity
  lines (mem_gb=11). iw2-flip lines in pool.queue: 0 -> 43. waves.tsv rows written (epoch
  1790907126 = 1-43, epoch 1790907133 = F-identity-3).
- Pacing: waves 2-7 only when fewer than 10 iw2-flip lines remain (HANDOFF section 7). After every
  line has landed: score B, F2, F3 (section 7 decision table).

## 2026-10-02 03:55 EDT - wave 2 (T11)
- Queued flip lines 44-86 (43 lines, failures=0) at F=1fb805e; start condition iw2-flip<10 held (q=4). Prefix rule verified (cd revisions/1fb805e9e + assert_iw2_node + assert_iw2_defaults). waves.tsv row appended. Next: wave 3 (87-129) when q<10 again.
