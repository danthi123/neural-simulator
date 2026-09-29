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
