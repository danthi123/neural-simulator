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

## Step O: canary + queue (2026-10-14 12:05-12:06)

- op re-read = 0; 6 O lines committed (`19a8e0904`).
- Canary PASS on pool41: rc=0, wall=5s, json=1, errors=0,
  id `word_role_commit_organ_c64f2c70d_pool41_20261014T120516`
  (log `~/derisk-pool/canary/word_role_commit_organ_c64f2c70d_pool41_20261014T120516.log`).
- All 6 O lines queued 12:06 (mem_gb=2, max_age_h=120). Step R waits for block 4 (score O).

## Step R: canary + trace checks + queue (2026-10-14 13:09-13:14)

- op re-read = 0; 12 R lines committed (`419cff8a2`).
- Canary PASS on pool42: rc=0, wall=1701s, json=1, errors=0,
  id `load_bearing_fraction_c64f2c70d_pool42_20261014T124108`
  (registry line 2026-10-14T13:09:57; canary LB fraction 1.000, 1/1 env-lesion faculties).
- Trace checks on the canary shard (s42) ALL PASS: `word_role_trace.route = word_role_commit` and
  `error = null` in both arms; `commit_lesioned = true` in `lesion_*.json`, `false` in
  `intact_a_*.json`; `commit.read_invariant = true` in intact.
- All 12 R lines queued 13:14 (mem_gb=8, max_age_h=168). Step: score R when the 12 shards land.

