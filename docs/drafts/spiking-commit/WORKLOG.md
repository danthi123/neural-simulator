# word-role-commit — packet log

Pin P = c64f2c70d8232336992aab074f199c36a6560d45 (branch `research/word-role-commit` from main `c64f2c70d`).
Host, numpy CPU, `SIM_NO_PROVENANCE=1` for the S-side tests/select.

## Step 1: dev-seed operating-point sweep + select (2026-10-14)

- Block 0 tests: 104 passed (word-role-commit + word-role-parse + s2p_score).
- Provision of P: `ok` 11:30:29, on pool41, pool42, pool40, wspool.
- 8 sweep lines queued 11:31, dispatched pool40 11:30:54–11:31:35, all landed: 48 `op<I>_s<S>.json` +
  48 prov sidecars. `rsync` is absent on S so `tools/pool_sync.sh` pulls 0 files — retrieved with a
  tar-over-ssh pull (NEEDS `[pool-sync-rsync]`).
- select rc=0: verdict SELECTED, op=0, grid all-true for dev seeds 1–8, missing []. `selection.json`
  committed with a truthful `.prov.json` sidecar (device-and-cost + artifact-provenance).

| rc | do | result |
|---|---|---|
| 0 SELECTED, op = 0 | step 3 | DONE — WORKLOGged; step O lines at op=0 (block 3) next. |
