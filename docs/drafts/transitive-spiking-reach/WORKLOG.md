# WORKLOG transitive-spiking-reach (T19)

## 2026-10-02 07:05 EDT - sections 0-1
- Section 0: worktree research/transitive-spiking-reach created from main at PIN=9ea7411eb;
  tests 41 passed in 1.52s (test_transitive_spiking_reach + test_s2p_score); pin committed;
  provision requested -> state ok at 07:05 (attempt 12).
- Section 1: make_jobs sweep -> 8 lines committed; queued 8/8, failures=0 (pool depth 389).
- NEXT: section 2 when all 56 sweep files land in research/findings/raw/_s2p/transitive-spiking-reach/9ea7411/sweep/.
## 2026-10-02 07:23 EDT - section 2 SELECTED op=0 (table row below), section 3 lines made
- 56/56 sweep files in; select rc=0 -> SELECTED op=0, table rows all true:
  {"0": {1..8: true}} (dev seeds 1-8, grid_len 7, missing []). Commit 5da6c475c
  (sweep + selection + gate sidecar selection.json.prov.json - the gate pairs *.prov.json,
  same pattern as T18).
- Section 3: O=6 + R=12 lines committed at op=0.
- NEXT: section 4 canaries (O pool41, R pool42) then queue O+R.
## 2026-10-02 07:28 EDT - section 4: canary-O PASS + 6 O queued; canary-R INCONCLUSIVE x2 -> NEEDS
- canary-O pool41 07:25: PASS rc=0 wall=2s json=1 errors=0 (id ..._T072509). O lines queued 6/6,
  failures=0 (pool depth 385).
- canary-R: rc=3 on pool42 07:26 AND pool41 07:27 - "the line's own mem_ok refused" (rc 1, 8G
  need; nothing launched, nothing recorded). Both nodes per the table's other-node retry.
  12 R lines HELD (pool refuses gate lines without a canary PASS).
- NEEDS-CLAUDE added: [transitive-spiking-reach] canary R INCONCLUSIVE x2 - same systemic 8G
  block as [fb-spiking-witness] STOP #1 (still open).

## 2026-10-05 02:28 -- section 4: R canary PASS recorded; R lines queued

- Fresh attempts per RULING canary-node/canary-auto: 8 launches across wspool+pool40 hit two infra failures (rssh launch false-negative; the line's own mem_ok refused while twin canaries held RAM). The transitive R canary nevertheless ran to completion on wspool: status rc=0 after 1596 s -> recorded PASS (module+revision covers both packets: same runner, same sha).
- Queue adds this wake: transitive ok=12 fail=0, word-role ok=12 fail=0 (12 expected each) with the HANDOFF --checked note.
- fb-spiking-witness canary not re-run: its step R already scored PASS at 40c43d520.
- Trigger: results wake 2026-10-05 01:42 (22 files).

## 2026-10-05 02:53 -- queue hold detail for the NEEDS ASK

- RULING iw2-merged(b) landed on main at 02:06 EDT; the 12 s2p-R-tr-9ea7411 lines were added at 02:25 EDT (this branch's section-4 run), so they must not dispatch.
- Verified: none started (pool.queue rows ~129-140, pool.queue.running empty, no load_bearing_fraction procs). Pinned rev 9ea7411 predates the required chase trace (n_ignited per hop + error field), so a rerun at this pin can never record it.
- pool_queue.sh has add|list|depth only -- no hold/withdraw -- and moving/editing lines is forbidden to the driver. NEEDS ASK filed 02:52 for Claude to hold/withdraw and re-queue at a post-fix revision.
- If they slip out before Claude acts: quarantine _load_bearing/_shards/s2p-R-tr-9ea7411/ (never gate on it) and rerun at the fixed revision.
- The 12 word-role lines (s2p-R-wr) are NOT covered by the ruling and stay queued.

## 2026-10-05 08:14 -- RULING transitive-engagement steps 2-3 executed (re-pin + corpus + dev shard)

- Step O pin check: `git diff --quiet 9ea7411eb..83efc766c -- research/runners/transitive_spiking_reach_organ.py`
  rc=0, organ UNCHANGED -> step O stands (no re-run).
- Corpus step: `tools/before_you_build.sh` rc=0, recorded to corpus_checks_shared.jsonl.
- Re-generation at the new pin `83efc766c` (main tip carrying the ruling; old pin 9ea7411eb): the HANDOFF's
  own generator ran from the MAIN worktree because `agent_sync_branch research/transitive-spiking-reach`
  REFUSED -- gates flag an uncited measurement in main's
  `research/findings/2026-10-05-iw2-transitive-chat-default-on-flip-validated-PROVISIONAL.md` (a finding edit,
  section-2 reserved; branch unchanged, no --no-verify). Generated `docs/drafts/transitive-spiking-reach/jobs/R_83efc76.txt`
  = 12 lines, tag `s2p-R-tr-83efc76`, OP=0 (carried from the old selection), plus `Rdev7_83efc76.txt` = 2 dev-seed
  lines (seed 7 capability + ctl).
- Queue order per the ruling: corpus first (done), then ONE dev-seed R shard (the s7 capability line) queued
  BEFORE the 12; on its result check the lesion arm shows `transitive_trace.lesioned` true with no error
  (packet checklist item 4), then queue the 12.
- word-role-parse lines: untouched, still queued (ruling).
- PINS.tsv gains a `P2` row at `83efc766c` (the original `P` row stays for the old pin).

## 2026-10-05 09:35 EDT - dev7 item-4 check PASS; gate-seed canary launched before the 12

- Dev-seed shard (s2p-R-tr-dev7-83efc76, seed 7 capability row) ran on the pool from pin
  83efc766c. Packet checklist item 4 PASSED: lesion arm records transitive_trace.lesioned
  true with route spiking_reach and no error; intact_a/intact_b record lesioned false,
  same route, no errors (the run's lb.json row summary carries no per-turn trace by
  design; .rep0 matches). Script: /opt/data/cache/scratch/t1_dev7_check4.py.
- The 12 re-pinned R lines (jobs/R_83efc76.txt) passed queue_job_shape_check but
  pool_queue.sh add REFUSED all 12: gate seed 42 needs a pool canary record for
  (research.runners.load_bearing_fraction, 83efc766c) in pool_canaries.tsv. Per RULING
  canary-auto, launched `pool_canary.sh run auto '<line 1>' --wait-s 1800` (pool40
  confirmed to hold the revision); background proc_1171c64f5048.
- Next: on the canary's PASS record, queue the 12 (record text prepared in
  /opt/data/cache/scratch/t1_r12_queue.sh), then WORKLOG.
- word-role lanes untouched (present in pool.running/queue.claims); the eight queued
  6b9cb47f lines left alone per RULING canary-node.

## 2026-10-05 10:12 EDT - gate-seed canary PASS at 83efc766c; the 12 R lines queued

- pool_queue.sh add refused the 12 (gate seed 42 needs a pool canary record for
  (research.runners.load_bearing_fraction, 83efc766c)). Canary launched per RULING
  canary-auto: run auto picked wspool; the launch ssh call hung past POOL_CANARY_SSH_TIMEOUT
  (60s) and reported rc=3, but the detached job ran to completion (verified: log,
  ps, status file). Id load_bearing_fraction_83efc766c_wspool_20261005T093506,
  rc=0 wall=1136s, 0 Tracebacks, 0 mem refusals; local scan of lb.json: n_files=1
  n_bad=0 n_unreadable=0. Recorded PASS (gate) via pool_canary.sh record - evidence
  in research/queue/pool_canaries.tsv. NEEDS [pool-canary] ASK filed for the
  launch-hang defect (it also fired on pool40 twice: two duplicate canary runs there,
  each writing its own _pool_canary id - left to finish, unrecorded).
- The 12 R lines (jobs/R_83efc76.txt) ADDED x12 FAIL=0; queue depth 20. Packet step 4
  (queue gate seeds 42/43/44/100/101/102) complete for transitive; word-role lines
  untouched (in pool.running/queue.claims).

## 2026-10-05 12:25 EDT - steps O and R scored (scorer: main s2p_score, absolute paths)
- Assembled 83efc76/{sweep,O} from 9ea7411 (organ ODIFF 0 per RULING transitive-engagement);
  select -> SELECTED op=0; 12 R shards (36 arms + 12 lb rows) copied from the S-side results.
- O at its own pin 9ea7411: PASS rc=0, 6/6 all_pass (P2=83efc766c per PINS P2 row).
- R at 83efc766c: UNDEFINED rc=3 - capability and permuted-control cells DEFINED 0/6.
  Root cause: tools/b2d_score.py:281-283 collects any field carrying an 'error' KEY even when
  the value is None (str(None) -> 'None'); rule7 (286-287) filters turn/organ only, never a
  real error -> every capability + control row reads C-UNCONFIRMED.
- Data verified clean: transitive_trace.error typed None in all arms checked; route
  spiking_reach; lesion flags match; engagement_state OK on all 6 seeds (non_ok=0); row
  verdict regressed, load_bearing=true, control clean, attributable_fraction=1.0.
- Scorer fix is s2-reserved: NEEDS [transitive-spiking-reach] (RESULT + ASK) filed on main;
  R re-scores after the fix.

## 2026-10-07 09:49 EDT - R rescored per RULING transitive-R-rescore (PASS op=0, NARROWED)
- Synced main into the branch (agent_sync_branch.sh): merged main @ 2405f9134 as 246db7fbd.
- Re-scored with the same recorded data (NO new runs): `$PY tools/s2p_score.py R --packet
  transitive-spiking-reach --tag s2p-R-tr-83efc76 --pin 83efc766c39b1385372c794e9ba87ac1cade442e
  --selection research/findings/raw/_s2p/transitive-spiking-reach/83efc76/selection.json --out
  research/findings/raw/_s2p/transitive-spiking-reach/83efc76/verdicts/R.json` -> rc=0,
  "s2p R transitive-spiking-reach: PASS op=0". The b2d rule7 fix (a15010cba) counts only
  non-null error; the data was always clean (rule7 entry above).
- verify-go (RULING): TWO independent adversarial refuters (reproducibility+instrument-trust;
  gate-cheat+control-integrity) -> nothing refuted, verdict NARROWED, commit-with-caveat:
  (1) op=0 is the selected operating-point GRID row (DETECT_MIN=10 host readout) - NOT "zero
  host steps"; (2) replies are byte-identical across the six seeds (only detect_spikes 40/41
  varies) - 6/6 shows determinism, not seed robustness; say so wherever 6/6 is written;
  (3) answer-rate lens UNMEASURED - no flag-off cells exist for these probes (on-tag answers
  2/2 per seed); (4) engagement SURVIVES - all 48 arm files carry transitive_trace (route
  spiking_reach, error null, lesion flags matching, intact reached=True detect 40/41 vs
  lesion reached=False detect 0); where-ran: pool=24, workstation=24, verdict server, all at
  pin 83efc766c; scorer re-run byte-identical; tests test_s2p_score + test_b2d_score 49 pass.
- verdicts/R.json.prov.json refreshed to describe this rescore (s2p_score writes no sidecar
  itself; the 10-05 sibling still described the UNDEFINED run).
