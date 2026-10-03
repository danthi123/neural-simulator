# WORKLOG: gpu-s1-lived-cupy (T24)

Branch `research/s1-lived-cupy` from main at `c96b7e445`.
GATE `c238ff325a38e40a68b1f9751822630ff729377b` reachable (guard-at-pin verified).

- 2026-10-02 19:43 (local model) — section 0: branch `research/s1-lived-cupy` created from main `c96b7e445`;
  GATE `c238ff325` reachable (stage1-exit-gate branches listed, guard-at-pin OK). section 1: 2 dev smoke lines
  (seed 7, intact_a + intact_b) queued via `gpu_remote.sh add --rev c238ff325…`, depths 1-2, no ADD-FAILED;
  yielded the GPU up to 2 h. devcheck next session when both `dev/s7` JSONs arrive (sim-sync).

- 2026-10-02 20:45 (local model) — section 1 devcheck: **GO 653.2 s** (`devcheck.py dev/s7`; intact_a 19:55 +
  intact_b 20:06 both landed, DONE(rc=0), null held). 653.2 <= 2400 -> no INFO line. section 2: all 42 gate-seed
  lines queued at `c238ff325` (depths 1-42, no ADD-FAILED), GPU yielded up to 12 h. sections 3-4 (aggregate +
  informational write-up) wait for T6's gate to be scored (RULING `s1-lived-cupy`).
