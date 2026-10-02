# WORKLOG: fb-spiking-witness (T18)

## 2026-10-02 00:31 EDT - sections 0 + 1 done (session start)
- Section 0: worktree `research/fb-spiking-witness` created from main @
  PIN=40c43d520cbee67210b85ad767e88bdd176f88c1 (symlinks .venv + data); PINS.tsv written.
  Tests `SIM_NO_PROVENANCE=1 nice -n 10 pytest -q tests/test_fb_spiking_witness.py
  tests/test_s2p_score.py` -> **16 passed in 2.83s** (rc=0). Pin committed
  "ops(fb-spiking-witness): pin P=40c43d520". Provision via provision.req handshake
  (RULING compute-placement + iw2-provision) -> state ok (attempt 1/3, 04:26-04:31Z).
- Section 1: `make_jobs.sh fb-spiking-witness sweep 40c43d520...` -> sweep_40c43d5.txt
  **8 lines** (expected 8), committed. Queued all 8 through `pool_queue.sh add --checked
  "... dev seeds 1-8 ... mem_gb=2 ; max_age_h=72"`: 0 failures, depth 405 -> 413.
  (Dev seeds: no verdict, per packet rule.)
- NEXT (section 2): when 64 sweep files land in $P/research/findings/raw/_s2p/fb-spiking-witness/
  <P7>/sweep/ -> cp + `s2p_score.py select`; rc=0 step 3 (OP), rc=1 PROVISIONAL finding +
  NEEDS-CLAUDE, rc=3 requeue missing seeds once. Queue is 413 deep (FIFO), so sweep results
  arrive late; sections 2+ continue on later wakes.
