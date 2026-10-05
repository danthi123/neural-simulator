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
