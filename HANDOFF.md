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
- **ROUTING SCREEN READ (11-25, brain/experiments/results/2026-11-25-routing-screen-read.md): D+G and SC both KILLED.**
  KR lives = lifeKP (0 mismatches, 3 seeds; AWS instances terminated; cap back to $50 at 00:05 UTC). Offline replay on
  pool40-42 CPU, 1 void night (s321 n5). D+G: right 44 -> 54 (need 55), killed by K3 (right->wrong 1/5/4) on any reading;
  SC: right 46, K1/K2/K4. Ranking changes trade wrong fetches for no fetch. Arm D (unbarred reference): right 59 wrong 34
  but told no-fetch 37 -> 124 - no result; a D test needs its own prereg on fresh seeds with a reply/answer-rate bar.
  Refuters x3 SURVIVES_WITH_CAVEATS, applied. Lever count on routing (D1) now >15 + 2: research gate stays fired.
- **ROUTING CEILING (11-25, brain/experiments/results/2026-11-25-routing-ceiling-read.md): the KEY is a big lever.**
  Rank of the right episode on the 217 KR told rows, raw order only: S0 right@1 90; S_all_live 168 (non-repeated rows
  89/89 vs S0 56); S12 160 but item-dependent; last layer 21. DG(S0) dead (sign-unstable over map seeds); S0q dead as
  registered (double-centred: not a test of cue offsets); S0 drift vacuous by construction. A's weighted+gated
  competition turns S0's 90 raw-first rows into 44 right. Refuters x3 SURVIVES_WITH_CAVEATS, applied.
- **ROUTING SCREEN 2 (11-25, brain/experiments/results/2026-11-25-routing-screen2-read.md): S_all + slept bar KILLED**
  (K1 49 < 55, K2 s321 falls, K5 told no-fetch +26/+33/+32). Wrong 136 -> 40, false fires 239 -> 67. Of 168 raw-first
  right rows only 49 fetched: the right episode's weight tanh(eff) median 0.36 (Volvo 0.25, rotating 0.35). Candidates:
  bar fit on young (2-day) episodes vs 5-15-night-old ones; store strengths shaped by S0 fetches. Refuters x3 applied.
- **ARM S READ (11-25): KILLED** (`brain/experiments/results/2026-11-25-sall-live-dev-read-1.5b.md`). Live S_all + slept bar:
  intact told 125/223 (K-P 123), h-off 133, hurt 25 > helped 17; frozen right fetches 18/223 (K-P 45), none 127 (K-P 37);
  collapse + G8a + h-off echo guards fail. The wrong fetches are mostly the PERMANENT self-name episode ("My name is
  Nova") on 65/223 told rows (K-P 0). Skeptic: KILLED stands. Instances terminated. Next routing method (own prereg):
  keep permanent/self episodes out of the S_all competition, or gate on rank margin, not absolute strength.
- **CASCADE SCREEN RUNNING (11-25):** prereg brain/research/2026-11-25-cascade-screen-prereg.md, build 268ea9041
  (`Brain(own_decay)`, default OFF), pool40/41/42 s319/320/321 restarted 19:11 UTC (first start stopped in error, kept as
  casc_s*.aborted1.json), ~7.5 h; out ~/brain-lane/route_repo/_scratch/route/kr/casc_s<seed>.json. Night 2 s319 already
  shows D25 replies going empty (told ok 6 -> 1): fading own replies may erode the turn format. Then K5 (own_gain_probe
  --gains d0.5 d0.25 on pool40 E / pool41 B), `cascade_screen k5` + `table`, read.
- **AFTER A W REBOOT:** rerun `bash _scratch/s_launch_then_pull.sh` pull part only (or pull research/queue/.aws_brain.rvS<seed>
  per stopped instance) and re-arm tools/attention_watch.py --loop.
- **TIMEBOX (11-25, routing arc open since 11-23, past 2 days): KEEP, new limit 11-27.** One last method: S_all competition
  without permanent/self episodes, offline screen on KR snapshots (pool, $0). If it fails: STOP routing, bank as
  "missing prerequisite (near-duplicate separation) at this scale", and run 12-04 test2 with the current memory. No 6th lever.
- **NEXT STEP:** (1) Cascade screen: when done, K5 then `cascade_screen k5` + `table`, read, refuters, land. (2) Routing: before-build +
  prereg for an S_all competition that excludes permanent/self episodes (or a rank-margin gate), offline on KR snapshots first. (3) Launch 12-04 AWS g5 fp16: 6 seeds x (B,
  K-P, D2-PR2). (4) Label-shuffle null for `life_bars.py`. (5) `tools/where_ran.py` ignores the life JSON host field (and
  replay/ceiling/screen JSONs carry none). (6) sim-rag-autoupdate.service fails. Pool40-42 ~/brain-lane/route_repo holds
  the KR snapshots + trunks (keep until the routing arc closes).
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
