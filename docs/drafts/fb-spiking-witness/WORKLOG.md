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

## 2026-10-02 00:57 EDT - sections 2-4 (partial): select op=4, data committed, O queued, R blocked
- Section 2: 64/64 sweep jsons landed (queue ran them fast); `tools/s2p_score.py select` ->
  **SELECTED op=4** (rc=0). First commit blocked by device-and-cost + artifact-provenance
  (the scorer writes no sidecar). Fixed per the gate's OWN fix text with a truthful sibling
  `selection.json.prov.json` (exact argv, host, backend python-stdlib/cpu, seeds 1-8);
  data commit **d868f5ee5** (132 files). Lesson: do NOT set SIM_NO_PROVENANCE for select -
  the HANDOFF's line has no such flag.
- Section 3: `op=4` -> O_40c43d5.txt **6 lines** + R_40c43d5.txt **12 lines** (exact counts);
  landed in d868f5ee5 (staged alongside the data).
- Section 4: canary O on pool41 **PASS** (rc=0, wall=11s, 00:56). canary R **rc=3
  INCONCLUSIVE twice** (pool42, then pool41 per the packet rule; mem_ok refused the 8G+2 line
  both times) -> NEEDS-CLAUDE "[fb-spiking-witness] STOP #1" added (00:57). O lines queued
  (6/6, 0 failures). R lines NOT queued - blocked on the canary.
- tool-health note: `tool_health --emit` shows **auto-tune ROTTED on S** - `dearpygui` is
  declared in requirements.txt but absent from .venv, and pip install is blocked by the
  sandbox; non-blocking for commits (warning lane) but the smoke cannot go green on S until
  the venv is repaired. Repair item emitted to research/coordination/tool_health.json.
- NEXT: when nodes have room, canary R once more (or Claude's ruling), queue 12 R lines
  (mem_gb=8, max_age_h=168); then section 5 (score O + R when 6 O files + 12 LB shards land).

## 2026-10-02 03:14 EDT - section 4 DONE: canary R recorded PASS, 12 R queued; NEEDS #1 resolved
- Canary R: launch attempts at 01:57 and 02:54 both printed "launch over ssh failed" while the
  setsid job ACTUALLY LAUNCHED both times (T015721 completed rc=0 wall=932s, json=4, 0
  tracebacks, wrote _pool_canary/.../lb.json; T025427 still running at 03:14 - left alone,
  never relaunched). Root cause: pool_canary's rssh wraps ssh in `timeout 60`; on a loaded
  node the client is killed AFTER dispatch - client reports failure, job runs to completion.
  The 01:58:45 attempt was a true mem refusal (rc=1, avail 4G) - correctly inconclusive.
- Recorded the completed T015721 run through the tool's own `record` subcommand (evidence
  "rc=0 wall=932s json=4 errors=0 unreadable=0 tracebacks=0; id=...T015721 ...") ->
  `check-line` rc=0 (gate open).
- Queued 12 R lines (mem_gb=8, max_age_h=168), failures=0 (depth 395).
- NEEDS "[fb-spiking-witness] STOP #1" removed from the runbook (resolved by the record).
- NEXT: section 5 - O half ready (all 6 O jsons landed + parse); score O + R once the 12 LB
  shards (s2p-R-fb tag) land from the R cells now queued.
