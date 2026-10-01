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
