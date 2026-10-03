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

- 2026-10-03 12:58 (local model) — **section 2's ONE retry executed** for the two missing shards. The 10:28
  yield drain said "42 job(s) ran" but only 40/42 shards landed: s43 `lesion_recall` + `.rep0` absent (the other
  5 s43 arms landed 01:05-05:20). Evidence recovered by grepping the workstation's full
  `research/queue/gpu_queue.log` (10.9 GB / 112M lines — `gpu_remote.sh log N` only ever tails a minutes-long
  window): `DONE(rc=143)` 01:41:01 (`--arm lesion_recall`) and `DONE(rc=143)` 04:38:30
  (`--arm lesion_recall.rep0`) — rc=143 = SIGTERM (external kill under pause/yield semantics), not a code fault.
  Section 2's precondition ("ended DONE(rc!=0) (log)") is therefore met literally; this is line 1 of the single
  allowed retry — a second failure stays missing and §3 reads UNDEFINED for seed 43. Judgment routed per RULING
  models: verdict + independent refutation delegates both answered
  "RE-QUEUE ONCE each — the rule's precondition is verified, not missing" and
  "DOES NOT REFUTE — the record's own registered handling; rc=0-without-shard is impossible in the runner, so an
  absent --out after a drained run is proof the line failed." Pre-add checks: both `--out` files re-verified
  absent; queue empty (no re-add of queued lines); `make_jobs.sh` unchanged since 9320b428f so the regenerated
  gate lines are byte-identical to the originally queued 42 (delegates also diffed them against the recovered log
  commands); commands still carry the unchanged `--prereg …-PREREGISTRATION.md --allow-gate-seed` at code pin
  `c238ff325`; no numbers read (blind-until-T6-scored intact). Actions: 2 × `gpu_remote.sh add --rev c238ff325
  '<line>'` (queued depths 1-2, rev c238ff325a38e40a68b1f9751822630ff729377b), then
  `gpu_remote.sh yield --max-hours 12` (local-llm stopping gracefully, deadline epoch 1791089919).
  Counting note for §3: two stale `.part.json` debris files sit beside finished s42/s100
  `lesion_episodic` shards — exclude them from shard counts. sections 3-4 still wait for T6's gate score.
