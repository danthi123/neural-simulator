# HANDOFF (overwrite each session; <= 40 lines)

- **TASK:** 12-14 milestone, multi-week life with nightly sleep (board #286). Design
  `brain/research/2026-11-19-multiweek-life-design.md` (§5 bars, §13 registration). Older state: `brain/experiments/results/`.
- **TYPE:** build · **DRIVER:** claude · **TIMEBOX:** Stage-2 methods by 12-08, 2 days each; read/land 12-10 to 12-14.
- **STATE (11-23):** Stage 1 read 11-21 (no capability passes). Stage-2 DEV screens read:
  - K-P (`keep_bank='prior'`): KEEP at DEV (11-22). Same-device sensitivity vs B-fp16 (results/2026-11-23-kp-vs-bfp16-same-device-1.5b.md):
    language win device-robust (KL n15 0.32 vs 2.47); self-name wrong 0.77 vs 0.33 and refusal 38/582 vs 0 HOLD; collapse
    guard device-sensitive, unresolved (reader's "KILLED" refuted 2/2).
  - D2-PR2 PR-a (`replay_priority='loss_a'`): PASSES 3.1-3.3 by the prereg letter (results/2026-11-23-d2pr2-pra-dev-read-1.5b.md:
    leak 48/460 vs 135/460, blind week-3 read no new class). Refuters 2/2 said HELD on s321 collapse nights 1-2 (would trip the
    §10 per-seed kill; B 0/3). Ruling: enters TEST with that risk registered in the test2 prereg (≥2/6 PR-a seeds killed -> NARROWED at best).
  - **TEST split v2 FROZEN** (a8f7c4429 on main): bank sha 5e775778, cues 794da243; DEV row check s322 passed.
- **K-P IDENTITY GATE READ (11-23, `results/2026-11-23-kp-identity-gate-read.md`, 2 refuters not refuted):** K-P-U UNDEFINED
  (A-position KL 1.195x K-P, bar 1.25x), K-P-0 FAIL (bar 1 by 0.006 nats). No DEV lives. Test2 arms fixed (prereg Amendment T2):
  B, K-P unchanged, D2-PR2. Lever 1/1b on K-P's identity cost spent; lever 2 only via before-build + prerequisite judgment.
- **RECALL-VOICE DEV READ (11-24, results/2026-11-24-recall-voice-dev-read-1.5b.md): M1 lived_key and M2 speaker contrast
  both KILLED** (M1: lang KL s321 1.264x; M2: told 105/223 < K-P 123, collapse). Prereg §5 stop rule: D2 lever count 5, no
  further perspective lever before a ROUTING method (wrong-episode fetch = 28 of K-P's 41 fetch-hurt rows). Self/other-tag
  source-hold DRAFT (brain/research/2026-11-24-self-other-tag-prereg-DRAFT.md) BLOCKED by 2 Fable reviews.
- **FLIP-NIGHT DIAGNOSIS (11-24, results/2026-11-24-flip-night-diagnosis.md, host-only + pool40 cascade test):** B drifts into
  the user's voice in the weights (27/42 nights, absorbing); K-P cuts it (5/42, recovers). Remaining flips are both kinds: M1
  s319 in the weights, s320 intact an in-conversation cascade (gone from a fresh state). Battery rows within a night are not
  independent; is_first_person/is_echo count refusal loops (caveat for every pooled count, incl. the TEST v2 read).
  Nothing live on AWS or pools. pool40 holds a 2 GB copy at ~/brain-lane/flip_repo (delete when done).
- **DONE 11-23:** test2 freeze gates built + landed (6998bc538): `life_costs.py`, `life_compare.py`, G9 mode validated (read 2),
  provenance + test2 dirty-tree guard in life.py, Amendment T1 (no margin widening; G11 a setup check). Residuals: `--resume`
  of a test2 life skips the dirty-tree guard; AWS test2 lives must set BRAIN_CODE_COMMIT + BRAIN_CODE_CLEAN=1 (launcher not
  yet updated); two leftover tmp dirs under brain/experiments/results/2026-11-18-inquiry-gate-dev-1.5b/ block a local test2 start.
- **ROUTING/CASCADE (11-24, brain/research/2026-11-24-routing-cascade-research.md + §5 amendment):** wrong fetches are
  strength winners (110/141 weighted-only), so first method = raw-similarity rank + weighted fire gate, then a strength cap.
  Screens need per-night state: **KR = K-P rerun with `life_kw --night-snapshots`**, RUNNING on AWS g5 (owner 11-24
  "rerun today, increase the cap as needed": cap $80 for 11-24 via aws-guard drop-in; `_scratch/cap_restore_1125.sh`
  moves it out at 11-25 00:05 UTC, back to $50 - check it ran). s319 i-0b9a3ef1c1efd384e (17:59, commit 2e7a4f6a6),
  s320 i-05c8a39f252a86db6 (22:04), s321 i-0c3ccd978db1f0df7 (22:44; both commit 4864e3274, no life-path change).
  `_scratch/kr_launch_then_pull.sh` pulls each when stopped (~02:30-07:30 UTC 11-25). Do NOT commit while a launch is
  rotating zones (HEAD must equal BRAIN_CODE_COMMIT). Gate: each KR life must equal lifeKP row for row.
  Screen prereg: brain/research/2026-11-24-routing-screen-prereg.md (landed 6701dda0b).
- **NEXT STEP:** (1) KR pulled -> `routing_replay.py validate` vs lifeKP, `replay` on pool40 CPU, `score`, refuters, land.
  (2) Launch 12-04 AWS g5 fp16: 6 seeds x (B, K-P, D2-PR2). (3) Label-shuffle null for
  `life_bars.py`. (4) `tools/where_ran.py` ignores the life JSON host field. (5) sim-rag-autoupdate.service fails to start.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OWNER (11-22):** "Finished rebooting, can you confirm everything looks good? I have another session using local compute
  so you may proceed autonomously, just don't use workstation compute until I say to." -> W GPU/CPU still held; use pools/AWS.
- **INFRA:** gpu-unlock works (kernel 7.3.5-2). AWS: all 11-22 batches pulled, instances terminated; $50/day cap restored
  (drop-in moved to `_scratch/cap-override.conf.expired-2026-11-22`). When W is released: GPU fp32 parity check
  (lang_guard nll_off == CPU 1.930773), resume or stop `_scratch/kpcpu/run.sh`. Worktree cleanup done 11-22: 557 removed,
  extras archived in `~/Projects/sim-worktree-archive/` (17 GB, MANIFEST.tsv each), 41 `archive/wt/*` branches; 60 skipped
  (changed <2h), rerun `_scratch/wt_audit/archive_and_remove.py` later. Snapshots untouched (owner: wait).
- **OPEN:** s102 g6 i-008ed35fe857dce55 STOPPED (state `research/queue/.aws_brain.s102-stopped-20261117`): pull then terminate. RAG index stale.
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
