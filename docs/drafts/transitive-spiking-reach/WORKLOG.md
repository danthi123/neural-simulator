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
