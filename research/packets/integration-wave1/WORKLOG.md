
## 2026-10-04 17:00 — §5 step D scored (S-adapted paths); R waits for wmf shards

Woken by lb arrivals (16). Re-read §1/§2/§6 + NEEDS (2 new rulings: game-mode-pauses, commit-attribution).
T0 no-hold clean; T9 no wave (43 b2d stubs queued). T8 state check: PINS.tsv has only the P row (F absent →
§6/§7 untouched); §5 never ran; the 5 P-pin identity files landed on S main (RULING iw1-stepB: stand, no
re-run) → step D Start satisfied. R step shards on S: d6 6/6 seeds x4, swm 6/6 x4 COMPLETE; wmf still in
flight (s42-s100 33/33, s101 10, s102 6 files at 16:55) → R scorers NOT run yet (status.sh "lines done"
gate not met; scoring partial R would misread the table).
Ran step D (4 identity compares) with workstation env.sh adapted to S paths: $I = S main identity/049d4bb,
$ID = S worktree copy, $V = S worktree verdicts/049d4bb (env.sh points $P/$W at /home/dant123/Projects/sim,
which does not exist on S). Results, all matching expectations: D0 IDENTICAL (PASS; off escape inert);
D_d6-hebbian-write DIFFERS PASS (internal turns 0-11, reply text unchanged — dev note said response-body
only); D_wm-focus-bind DIFFERS PASS (reply turn 3); D_xedge-surprise-worldmodel DIFFERS PASS (reply turn 12).
Table so far: D0 PASS + D_p PASS x3 → R decides each packet's flip membership. Next (event wake on wmf
landing): the 3 iw1_score R runs + copy_results + the combined data(iw1) R commit + §6 if all PASS.
No push (wake rule: sim-sync publishes main; branch push left to the HANDOFF's next commit or Claude).

## 2026-10-04 23:10 -- section 5 scored: D0 PASS; R PASS x3; flip list = all three

- R verdicts at P=049d4bb (scorer exit codes, shared table row "D0 PASS, D_p PASS, R_p PASS"):
  d6-hebbian-write PASS (6/6 DEFINED, 6/6 reply-load-bearing),
  xedge-surprise-worldmodel PASS (6/6, 6/6),
  wm-focus-bind PASS (6/6, 6/6; on=regressed off=pass, direction ok).
- All three packets go into the flip list (HANDOFF section 6 order:
  d6-hebbian-write,wm-focus-bind,xedge-surprise-worldmodel).
- Data commit below carries verdicts D0/D_* + R_* and the landed R shards
  (provisional). Not pushed (wake rule: sim-sync publishes main).
- copy_results.sh line 7: added the RESULTS_P override this commit carries,
  matching the iw2 sibling script (this host's primary checkout is S, not the
  workstation path).
- Next wake: section 6 flip commit (runbook scripted exception, iw1/iw2 flip on
  the packet branch) then provision F; section 7 only after RULING iw1-stepB.
# integration-wave1 WORKLOG

## 2026-10-06 11:3x EDT - T8 section 5 completed (driver: Hermes)

- Start check (§4): identity `049d4bb` = 5 json non-prov, iw1-R lb = 18 -> exactly holds.
- Step D was already committed `0f79b39ad` (D0 IDENTICAL, D_d6/D_wmf/D_swm DIFFERS, preconditions all true).
- Step R verdicts (re-run 2026-10-04 23:09): d6 PASS, swm PASS, wmf PASS; each n_defined 6,
  n_reply_load_bearing 6/6. The earlier commit was refused by the verdict-preconditions gate
  (NEEDS [iw1-R-pre], filed 08618051d); Claude's fix was marked handled 2026-10-05, so this retry is
  the ported fix taking effect.
- Section 5 table row 1 fires for all three packets (D0 PASS, D PASS, R PASS) -> d6, swm, wmf all go
  into the flip list. Next: section 6 (flip commit F, provision F).
- Not done as written: `tools/push_both.sh` skipped (driver contract: never push; sim-sync publishes).
- 516 lb shards + 13 _iw1 files staged by copy_results.sh earlier; verified no forbidden paths
  (runs.jsonl / requeue / research/queue / external_search) among the staged files.

## 2026-10-06 11:5x EDT - T8 section 5 BLOCKED at commit (driver: Hermes)

- Section-5 data commit refused by verdict-preconditions: R_d6/R_swm/R_wmf carry no `preconditions`
  block (score_R returns without one; mtime shows the branch scorer untouched since 10-04 16:56).
- RULING iw1-R-pre (runbook 868) gives the recipe: AFTER main is merged into this branch,
  `git am research/packets/integration-wave1/patches/0001-iw1-score-earn.patch` (the patch lives only on
  main; arrived there 10-05 02:39), re-run the three R scorers, recommit, file verify-go NEEDS.
- `tools/agent_sync_branch.sh research/integration-wave1` (RULING sync-branch) REFUSED by claim_check on
  main-side finding `2026-07-25-gap5-learned-band-emergence-STDP-directed-traversal-6seed-GO.md`
  (line 22/23/32 ratios 1.662.../0.601.../0.991... "not in any cited artifact"). Editing an existing
  finding = §2-reserved. A raw `git merge --no-commit main` (diagnostic, aborted rc=0) also shows real
  conflicts in `research/findings/raw/_provenance/runs.jsonl` (committing runs.jsonl = §2-forbidden) and
  `webapp/server.py`. Same class as open NEEDS 8d30b0ec6.
- State left: R data (516 shards + 13 _iw1 files) on disk in this worktree, UNCOMMITTED (reset from the
  index so the diagnostic merge could run); step-D commit 0f79b39ad intact; branch tip unchanged
  0f79b39ad; abort restored the worktree. tools/verdict.py already present in the branch (not the
  dependency); the patch file itself is the dependency.
- STOP per contract: NEEDS [iw1-sync] filed on main; merge + finding edit are Claude's.

## 2026-10-06 - NEEDS [iw1-sync] resolved by Claude review (branch claude-review/iw1)

- Merge: main @ 2f256bb0a merged into this branch's tip c02658287. Only research/FAILURE_LOG.md conflicted (both
  sides' rows kept). runs.jsonl and webapp/server.py merged cleanly in a clean worktree (every row of both sides
  kept; disjoint server.py hunks). The sync refusal's real claim_check failure in a worktree is main's
  2026-10-03 ca3-symmetric-ti finding, whose cited `parts/seed*_merged.json` are git-ignored and so absent from
  worktrees; copying them in (not committed) clears it. Main's 2026-07-25 gap#5 STDP finding was also marked
  as quoting run logs that no longer exist (claim kept, no number changed).
- RULING iw1-R-pre applied (patch 0001, through the commit gates). The three R scorers were re-run on the landed
  shards. R d6 PASS, R swm PASS, R wmf PASS: 6/6 DEFINED and 6/6 reply-load-bearing each, preconditions block
  ok. The decisions match the 2026-10-04 scoring field for field.
- verify-go (three independent skeptics: gate-cheat/instrument, control-integrity, reproducibility/seeding).
  Nothing was refuted at the registered criterion. Checks that hold for all three:
  - no organ error in any arm;
  - the null is clean and the lesion reproduces;
  - all 258 sidecars are at P, git_archive, numpy;
  - six distinct brains (BRAIN_CHAT_SEED per seed, 154-447 fields differ across seeds).
  - **d6-hebbian-write: PASS, survives.** Intact recalls "the wolf hunts the deer". With in-conversation writes
    frozen, the reply abstains ("haven't learned about wolf yet") on 6/6 seeds, while the host KB record still
    exists. The synaptic write owns whether the fact is held. The host owns the held/not-held threshold and the
    block-to-words map (declared shortcuts in d6_hebbian_store.py).
  - **xedge-surprise-worldmodel: PASS, NARROWED.** The reply change is value-only: the stated pool-rate margin
    drops 70-201 Hz with the online update (intact +220..+363 Hz vs lesion +393..+466 Hz). The predicted sign and
    the sentence are unchanged. It meets docs/TERMS.md load-bearing by the letter (the answer text differs), as
    the prereg predicted ("PASS on the margin text"). It is not a content change.
  - **wm-focus-bind: PASS, NARROWED.** The hold is load-bearing for which referent the reply follows. But the
    referent that wins is a seed-fixed register preference (the prereg's declared residual: intrinsic
    excitability) and ignores the question:
    - on 6/6 seeds, with the flag on, 2 of the 4 order-swapped sessions abstain ("I don't know about that.");
    - the flag-off positional route and the lesion answer all 4 with the stored fact ("The dog chases cat.").
    On this probe, the flip lowers answerability.
  - Instrument note: `iw1_score._allowing` strips the packet's flags from the sidecar env check, so the scorer
    cannot see a missing or stray lesion flag. The skeptic verified the lesion flags in each arm's argv `--env`
    instead: they are correct.
  - D0 note: D0 is IDENTICAL by exact per-turn hash at seed 7, but the identity sidecars do not carry
    PYTHONHASHSEED=0.
- Section-5 table: row 1 fires for all three, so the flip list is `d6-hebbian-write,wm-focus-bind,xedge-surprise-worldmodel`
  as registered. Each flip finding must carry its narrowing above, word for word.
- 2026-10-06 16:09: RULING iw1-sync pre-FF check executed: 526/529 uncommitted files
  byte-identical to claude-review/iw1, copy_results.sh too, but R_{d6,swm,wmf}.json differ
  (local pre-rescore vs branch re-scored; PASS both, branch adds preconditions) => any
  difference is a STOP: NEEDS [iw1-sync] STOP filed (main ef6402174). Fast-forward waits
  on Claude; worktree left as-is, nothing cleared.
- 2026-10-07 04:52: RULING iw1-ff-go pre-checked: copy_results.sh re-verified byte-identical
  to claude-review/iw1 this arc; untracked still 22 entries (18 shard seed-dirs + identity/ +
  the 3 superseded R_*.json DIFF copies the ruling moves to aside), file-level identity as
  recorded 16:09 above (526/529). FF PRE-CHECK FAILED before any merge ran:
  `merge-base --is-ancestor HEAD claude-review/iw1` = false; branch-only = this packet's own
  STOP marker 0d029d553, target 3c06a1c8f branched from the pre-marker parent c02658287
  (1 vs 289 commits). iw1-ff-go: anything but a fast-forward is a STOP for NEEDS => NEEDS
  [iw1-ff] filed on main. Cleanup deliberately NOT executed (deleting the 526 copies without
  the FF would destroy data for nothing); worktree untouched, 22 untracked + 1 modified.
- 2026-10-07 07:05: RULING iw1-ff + iw1-ff-go executed. FF to fef5bf63a (ancestor pre-checked
  by Claude; HEAD verified, tree clean). Cleanup: 3 stale pre-rescore R_*.json moved to
  /opt/data/cache/aside/iw1-pre-rescore/ (the tracked re-scored copies now stand); 526
  untracked copies verified byte-identical vs fef5bf63a blobs (526 match / 0 mismatch) and
  deleted; copy_results.sh restored. answer_rate (paired, on=iw1-R-{d6,swm}-049d4bb,
  off=iw1-alloff-049d4bb): BOTH UNMEASURED — off cells did not exist; on side answered 100%
  on every measured seed. Reports: verdicts/049d4bb/answer_rate_{d6,swm}.json. Per RULING
  flip-hurits-answers (UNMEASURED => queue flag-off cells first), the section-6 flip commit
  is DEFERRED until they land and answer_rate re-runs. DEFECT: make_jobs.sh alloff exits
  rc=1 silently at line 76 ($( [ bisect ] && printf ) fails the assignment under
  set -euo pipefail; bisect unaffected) — job files were generated with an output-identical
  one-line if/else patch in a DETACHED TEMP worktree at the R pin 049d4bb0b (never
  committed anywhere); NEEDS [iw1-make-jobs] filed on main for the branch fix (section 7
  decision-table row 2 calls alloff at F). Provision 049d4bb = state "already" (pool41/42);
  queued 12 flag-off lines (6 d6 + 6 xedge, ok=12 fail=0). Workaround-vs-NEEDS judgment
  delegated to MiMo Pro (verdict B). Flip list unchanged: d6, xedge (wm-focus-bind held).

## 2026-10-07 10:45 EDT - make_jobs.sh:76 fixed per RULING iw1-make-jobs; off-cell files confirmed byte-identical
- One line, exactly as ruled: `OUT="$OUTD/${STEP}_${P7}_${FAC}$( [ "$STEP" = bisect ] && printf '_%s' "$PKTS" || true ).txt"`
  (under set -e the false test failed the assignment; output is identical).
- Regenerated NOTHING in the packet dir. The confirmation diff the ruling asks for: the fixed
  script run at the R pin 049d4bb0b (detached temp checkout, IW1_JOBS_OUTD to a scratch dir)
  emitted 2 files / 12 lines that are BYTE-IDENTICAL to the committed jobs/alloff_049d4bb_*.txt
  (`ALL BYTE-IDENTICAL`, diff -q on both).
- Note: the fixed script cannot run from this worktree HEAD (2c609face) - the pin-descendant
  guard fires ("HEAD differs from pin outside the packet dir and raw results"; claude-review
  work sits between them), so generation stays at the pin, as section 7 already requires.

## 2026-10-07 15:48 EDT - answer_rate paired: off-side UNMEASURED by design (NEEDS answer-rate-off)
- 12/12 iw1-alloff cells landed (6 seeds x d6-hebbian-write + xedge-surprise-worldmodel). Paired mode:
  on 2/2 answered on all 6 seeds; off 0/6 -> `off - UNMEASURED` (the tool never reads as no-harm).
- Off shards hold ONLY lb.json (verdict `unmapped-in-battery`, load_bearing null); no intact arms to
  glob. The ON and OFF job lines differ ONLY in the flag values (1 vs 0) and --out.
- Cause: at the R pin the lbf_rows modules gate registration on REQUIRED_ENV (d6: all three
  BRAIN_D6_* flags =1; xedge: BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL=1). alloff zeroes them, so the
  rows never register, _faculty_row returns None, and the runner early-returns before spawning arms.
  The module docstring states it is by design: "OPT-IN until the flip ... the flip script deletes the
  REQUIRED_ENV line in the same commit that flips the anchor."
- 049d4bb is NOT an ancestor of main (integration-wave1 lineage). The off-side answer-rate awaits the
  T5 flip or a new off-probe design: filed NEEDS [answer-rate-off] ASK (runbook 1e652853b on main).

## 2026-10-08 02:50 -- answer-rate off-side MEASURED (off = lesion arm): d6 OK, xedge OK

Woken by 8 stage1_c1_flipdev results; re-read §1/§2/§6 + NEEDS (no new commits since 13d5ca525). RULING
answer-rate-off executed. answer_rate.py was absent on this branch and a full branch sync is refused by gate
lane-starvation (5/5 CPU lanes unserved; waiver budget 6.00h/6h charged, frees ~18:48), so the three infra
files came from main one by one -- 097989c89 (answer_rate.py + test), 1c5c76ef9 (iw2_score.py, whose
answer-rate wiring the imported test expects; the branch copy predated it) -- all tools/ + tests/ paths, i.e.
infra-exempt per gate -- then --off-arm landed as 4214cb8e9. `paired --off-arm <stem>` reads the LESION arm
beside the intact arms as the off side (off tag cells abstain by design via REQUIRED_ENV); exactly one of
--off-tag/--off-arm is required (both/neither -> exit 2); the label carries `off = lesion arm`.
tests/test_answer_rate.py 10/10 pass, incl. the new fixture test (lesion arm read, lb.json excluded,
render carries the label).

The read (6 seeds, base = research/findings/raw/_load_bearing/_shards), label `off = lesion arm`:

- d6-hebbian-write -- on iw1-R-d6-049d4bb vs off-arm lesion_d6_hebbian_write: OK, measured 6/6. Every seed:
  on 2/2 answered (abstained 0, empty 0); off 0/1, abstained 1 (the lesion abstains -- the d6 "haven't
  learned about X yet" phrase). No seed has on < off, so no FLIP-HURTS-ANSWERS.
- xedge-surprise-worldmodel -- on iw1-R-swm-049d4bb vs off-arm lesion_xedge_surprise_worldmodel: OK,
  measured 6/6. Every seed: on 2/2 answered; off 1/1 answered. No FLIP-HURTS-ANSWERS.

Verdict: neither flip hurts the answer rate with the default off -- criterion CLEAR for d6 and xedge. The d6
lesion abstaining 6/6 while intact answers 2/2 is the load-bearing shape (off answers LESS than on: the
opposite of the harm the check guards). The flips themselves are §2 (flipping a default): NEEDS
[answer-rate-off] RESULT filed on main with these numbers.

## 2026-10-08 23:59 EDT - iw1 flip RE-APPLIED after the waiver budget freed (driver: Hermes)

- Re-applied per ack-1008 (`tools/integration_wave1_flip.py apply d6-hebbian-write
  xedge-surprise-worldmodel --date 2026-10-08`; wm-focus-bind HELD OFF, `iw1-wmf-hold`). Guard
  `tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel"` exit 0;
  `tests/test_integration_wave1_anchors.py` 146 passed, 3 skipped (1.90s). Status prints
  d6 on / wm off / xedge on.
- Why it waited: the 2026-10-08 03:42 commit attempt was lane-starvation BLOCKED (5/5 CPU lanes
  UNSERVED; at the time I mis-read the `.lane_waiver` route as §2-denied — corrected in the landing
  note below; `_shared_queue_root()` evaluates main's queue), budget freed ~18:47.
  Validation pending shared prereg steps B, F (PROVISIONAL, branch only).
- CORRECTION 2026-10-09: writing `.lane_waiver` is NOT §2-denied — RULING `lane-builds` (2026-10-07)
  grants the driver standing permission (CLASS NO-READY-WORK + lane_check evidence as the reason +
  a same-wake `[lane-waiver] INFO` NEEDS; never for a lane with a ready line). Used exactly that at
  00:57 in main's queue root (lane_check: B/C/E queued=0; gate idle B/C/D/E, 0 keyed lines;
  D's rows = stale register + CLASS L keepalives the gate discounts); `[lane-waiver] INFO` filed
  (main 12c14c11e).
- LANDED 2026-10-09 01:05 EDT: commit `cfce97fe1` — 8 files (5 flip files + this WORKLOG +
  `board_sync.json` + `.vikunja_sync`) through the full pre-commit suite, never --no-verify.
  `board-sync-on-status-change` cleared by the real Vikunja receipt: next-rung task #245 created +
  Learning-labelled (wire + run the shared-prereg validations before any merge); no task closed by the
  provisional flip (#228, #202 stay open). Branch clean afterwards; never pushed; RESULT filed on main
  under `[flip-commit-blocked]`.
