# WORKLOG — host-share baselines (plan step 2), branch `research/host-share-baselines`

Worktree: `.claude/worktrees/host-share-baselines` (from main @ f52c6d246). Lane C (Self/Workspace), registration
copied from the primary checkout's uncommitted `research/coordination/lane_builds.jsonl`.

- 2026-09-29 ~12:00  Setup. Research the record: `before_you_build.sh` (hits: 2026-08-11 production-chat-pipeline-is-
  largely-host inventory; 2026-08-13 gnw-bus-default-flip), RAG on host-share / intervention / value choice / metacog
  margin. Read the plan sections 1, 2, 6, 7.
- ~12:10  Reviewed the local model's seam map + probe (branch `research/draft-seam-map`). 24 rows verified against the
  code; classification corrections (host shadow fact list, K-SELECT first-match row missing). Owner question answered:
  the degraded panel IS confounded (8/16 straddles are misrecalls, off-distribution). Review:
  `docs/drafts/seam-map/CLAUDE-REVIEW-2026-09-29.md`. Evidence runner: `research/runners/_host_share_draft_confound_check.py`.
- ~12:20  The worktree had no `data/` (untracked corpus): symlinked `data -> /home/dant123/Projects/sim/data` (the
  convention the other worktrees use; gitignored). Without it the tiny-demo build hits a FileNotFoundError on
  `data/corpus/tinystories.txt` in the learned-animacy cue.
- ~12:35  Built `research/runners/host_share.py` (4 seams x decoding/substitution/intervention, sham arm, live chat
  handler via the battery worker, in-process wrappers only, no sim/ or webapp/ edit) + `tests/test_host_share.py`
  (6 pass, no brain build). Selftest OK.
- ~12:40  COMMIT BLOCKED (expected): `gates/compute_idle_persistent` blocks non-exempt commits until the pool/GPU are
  busy or ~18:30. Work is STAGED in the worktree (not committed): lane_builds.jsonl, the review, host_share.py, the
  confound-check runner, the test. Retry only after `tools/parallel_audit.py` stops reading UNDER-PARALLELIZED.
- ~12:40  Dev timing (seed 7, battery worker, 2 sessions): ~19 min wall under load 17-22 on the 20-core box, ~6.2 GB
  RSS per ChatBrain build. Added a cupy DEV job to the GPU queue (seed 8, `--backend cupy`, dev-only; the gate stays
  numpy). IT DID NOT RUN: it started at 12:33:32 and exited rc=1 in under a second because `mem_ok.sh 12` refused
  (local RAM held by the numpy dev arms); its log was never created.
- ~13:00  DEV FINDING about the harness: a multi-session arm keeps every session's ChatBrain in the server's
  per-session cache (`_BRAIN_CHATS`), so RSS grew to 11 GB on the 2nd session (12 GB cap, MemoryHigh throttling), and
  the process-shared organs (metacog / value-choice / comprehension singletons, workspace bridges) would carry state
  from one session to the next. FIX: each session now runs in its OWN fresh subprocess; `run_shard` merges the
  per-session parts into one shard file (resumable). Killed the two multi-session dev workers; restarted per session.
- ~13:47  The first cupy GPU dev job ran at 12:33 and exited rc=1 in <1 s: `mem_ok.sh 12` refused (local RAM held by the numpy dev arms). Re-queued with mem_ok 10.
- ~14:00  Mid-flight audit received (`.claude/worktrees/_audit_2026-09-29/host-share.md`, 7 confirmed issues) plus
  coordinator rulings. Applied, one fix round each: (1) classification by INTERVENTION outcome in a new importable
  schema `research/runners/host_share_schema.py` (F=1,S=0 = host_causal_replaceable, never neural; TRANSDUCED
  dropped; selftest case added); (2) new seams `bg-select` (host salience pair + pre-filter) and `ignition`
  (norgan_hop argmax + THR); (3) schema importable + spec in the prereg; (4) v-choice neural alternative = each
  candidate block's spiking decode margin (not the lesion), D UNDEFINED when the organ declines, per-seed organs via
  `install_seed_threading` (value choice, BG selector, shared salience; FAILURE_LOG row); V5 held-fixed check; LTM pinned
  (`BRAIN_LTM_SHIP_DEFAULT=1`) + V6 same-source bar + `ltm_fallback` outcome; learned-content fraction = substrate-
  selected facts vs Python-selected (host shadow list / value-choice candidate list), lexical share kept as
  `fact_word_share`; call-graph seams (organ-c, metacog, ignition) flagged `downstream_neural_path=False` and kept
  out of the pooled panel; metacog S restricted to downstream fields with an upstream side-effect check. LOWs: the
  confound checker pinned to the reviewed draft SHA 34017865436d; organ-c reported by known/unknown sub-kind.
  REFUTED audit item (k-select wrapper on a non-production path): already fixed before the audit landed (the
  fact-shard wrapper, live on dev seed 7).
- ~14:20  The re-queued cupy dev job had started with the pre-audit code; stopped it (mixed revision) and re-queued it
  on the final code. Owner mode change: execution packet for the local model (`HANDOFF.md`), then stop.
- ~14:25  Dev verification launched on the final code (dev seed 7, numpy): recall intact_b, recall
  bg-select:intervene, compete v-choice:substitute -> `research/findings/raw/_host_share/dev/s7/`.
- ~14:45  Dev verification on the final code PASSED (prereg section 10): all recall seams live in the sham, bg-select
  and v-choice manipulations applied, source `tiny-demo +LTM`, seeds threaded. Removed stray `.part0.prov.json` /
  `.log.prov.json` sidecars (the runner now deletes part sidecars itself). Everything staged; the prereg stays
  untracked for its own commit. STOP here per the owner's mode change: the local model executes `HANDOFF.md`.
- 2026-09-30 ~14:40  PRE-DATA FIX (pool canary; no gate data existed, the prereg is unchanged). The four s42 recall
  lines dispatched to pool41/pool42 on 2026-09-29 (intact_a, intact_b, k-select:intervene, k-select:substitute) and
  a sandboxed canary of `recall bg-select:intervene` on pool42 (2026-09-30 13:57) all exited rc=1 in the worker's
  argparse: `run_shard` spawned `--worker --seed 42` without the parent's `--allow-gate-seed`, and the worker re-runs
  the gate-seed check (`seed 42 is a GATE seed: pass --allow-gate-seed`). The remaining 104 lines were moved to
  `research/queue/pool.queue.held` before they ran. Fix: `run_shard(..., allow_gate_seed=)` passes the parent's flag to
  every worker (`--shard` and `--seed-all`); `--backend` and `--labels` were already passed, and no other flag is
  gate-only. Failing-first tests in `tests/test_host_share.py` run the real CLI in a real child process with only
  `run_worker` stubbed: at 59edf3255 they fail with the pool's exact error; after the fix 13 passed, SELFTEST OK,
  SCHEMA SELFTEST OK. The gate revision moves from 59edf3255 to the commit that merges origin/main over this fix (main
  changed no sim/, webapp/ or research/runners/ file since 59edf3255, so relative to 59edf3255 the only brain-path change
  is this fix); it is written to `GATE_SHA`, provisioned `--isolated` on pool41 + pool42, canaried sandboxed, and its
  104 lines are regenerated with `--emit-pool-jobs` (+ `max_age_h=168`) into
  `/home/dant123/Projects/sim/research/queue/requeue_host_share_<sha9>.tsv` for the coordinator to queue.
- 2026-09-30 ~15:25  Gate revision is now f7f2bd1690783e60200d443c7442f63d3e5994a5 (fix cf7db278a + merge of origin/main
  b604085a6), provisioned `--isolated` on pool41 and pool42 (venv check ok, remote brain build ok). SANDBOXED CANARY
  PASS on pool41 (`tools/pool_sandbox_selftest.sh` from the pool-job-sandbox worktree): the real line `--shard --seed 42
  --group recall --arm bg-select:intervene --allow-gate-seed` at f7f2bd169 exited rc=0 after 940 s; the shard has 4
  turns, no errored turn, source `tiny-demo +LTM`, all six seams recorded, bg-select applied, PYTHONHASHSEED ok, seeds
  threaded (42), provenance git_sha f7f2bd169 clean. That shard is a valid gate shard at the gate revision; its line
  stays in the requeue set. The 104 held lines are regenerated at f7f2bd169 in
  `research/queue/requeue_host_share_f7f2bd169.tsv` (identical to the held lines except the sha, plus `max_age_h=168`);
  the four s42 recall lines that already died on 2026-09-29 (intact_a, intact_b, k-select:intervene,
  k-select:substitute) are in `requeue_host_share_f7f2bd169_dispatched_failed.tsv`. Queueing is the coordinator's call.

## 2026-10-02 16:44 EDT - section 8 scored (T4)

- 108/108 gate cells (gate f7f2bd169) scored by the registered scorer -> **BASELINE-VALID** (V1 sham-clean 6/6, V3 applied, V4 registrable, V5 drift 0, V6 tiny-demo +LTM, PYTHONHASHSEED ok, seed threading ok, missing shards 0). Finding: research/findings/2026-10-02-host-share-baselines-RESULT.md.
- Board line (Claude moves to the board): 2026-10-02 host-share baseline (plan step 2): BASELINE-VALID; measured panel host_decision_share 0.958, neural_decision_fraction 0.042; classes k-select HOST-OWNED, v-choice HOST-OWNED, bg-select HOST-OWNED; content learned 0.667, python-selected 0.333. Finding: research/findings/2026-10-02-host-share-baselines-RESULT.md.
- NEEDS-CLAUDE #8 recorded in the runbook (sync summary docs + board; hand the bg-select and ignition numbers to the step-3 builder as its "before" values).

## 2026-10-11 05:14 EDT - re-queue cells re-scored (T4 line re-tripped)

- Start condition (section 7 count = 108) re-met after the t4-fire-and-dev re-queue + clobber restore; re-aggregated
  with the registered scorer at the same gate f7f2bd169 -> **BASELINE-VALID again, identical numbers** (host_decision_share
  0.9583333333333334, neural_decision_fraction 0.041666666666666664, learned 0.6666666666666666, python-selected
  0.3333333333333333, missing_shards 0, sham noise 0.0, V1-V6 all pass) = reproduces the 2026-10-02 verdict (merge
  c0679d474). Per RULING host-share-clobber the re-run cells are already-committed duplicates: duplicate RESULT file
  removed, provenance re-stamps reverted, tree clean, nothing committed from this pass. Runbook T4 entry marked DONE
  so later wakes do not re-score.
