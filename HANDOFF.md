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
- **CASCADE SCREEN READ (11-26): KILLED** (`results/2026-11-25-cascade-screen-read.md`): own_decay 0.5 / 0.25 make the
  brain stop answering (empty told replies 367 / 528 of 726; told ok 170 -> 66 / 7); only 2/19 flip nights are a pure
  cascade (B fresh-state V 165 vs A 141), so the user-voice form is mostly in the weights. own_decay stays OFF.
- **ROUTING ARC STOPPED (11-26, timebox):** screen 3 part X (skip permanent) and part J (J-space key) both KILLED
  (`results/2026-11-26-routing-screen3-read.md`; J: Jt/Jv 30-47 vs hidden state 121-134; skeptics: KILLED stands,
  artifact unlikely). Banked in `brain/PREREQUISITES.md` (Memory selection): rank is not the limit, the weighted gate
  is; missing near-duplicate separation, self/other tag, reachable strength. Pointer only: re-keyed S_all 184/217.
  Pool route_repo copies can go once test2 is read.
- **WEEKLY (11-26): DONE** `brain/eval/transcripts/2026-11-26-1.5b-cuda-s42.md` (AWS A10G; pool40 CPU run OOM-killed at the
  11 GB cap during the 1.5B sleep, rc 137). 6/6 intact and 6/6 hippocampus-off, vs 3/6 and 2/6 on 11-19 (CPU), but the
  device differs and no default changed: NOT a progress claim. From now on run the weekly on AWS cuda
  (`brain/research/jlens_launch/jobs_weekly.txt` pattern, ~$0.5) so the series is same-device.
- **VOICE RESEARCH ROUND (11-26, workflow, 5 agents; external: Yonelinas 2019, Berres 2024, Qiu 2026):** biggest K-P
  defect by whole-transcript read = told facts answered in the user's voice (+ self-name answered with a user/told name).
  Cascade-screen data split it: fresh state per question still 44/273 intact, 48/279 h-off user voice (n2-5 10 -> n6-15 38),
  so whole-night flips = the cascade clustering a weight base rate. Candidate C1 (source-contrast) = the BLOCKED source-hold
  DRAFT renamed (skeptic). **VOICE SPLIT READ (11-26): UNDEFINED** by the registered rule
  (`brain/experiments/results/2026-11-26-voice-split-read.md`; 3 seeds valid, 560/560 reproduction). D = h-off B minus
  both-off+telling-in-context P(uv|knows) = +0.10, CI [-0.01, 0.20]. Report only: the effect reverses across items (cat,
  sister +; car -), the ctx control shifts replies to "You ...", and on name items the RAW cortex (no sleep, no fetch,
  knows 0) already answers "My cat is called Fluffy" (36/84 vs h-off 32/84). Sleep supplies the content (knows 196 vs 0),
  not demonstrably the frame. Source-hold DRAFT stays BLOCKED. Two sonnet skeptics: UNDEFINED holds, wording fixed.
- **VOICE CEILING RUNNING (11-26):** prereg `brain/research/2026-11-26-voice-ceiling-prereg.md` + Amendment R1
  (pre-data sonnet review, 7 BLOCKING fixed), harness `brain/experiments/voice_ceil.py` (d7132cd0f). Trains the slept
  weights (KL, reply positions) toward the pretrained cortex's in-context answer on ~20 rotating told items, tests the
  never-trained name items (pixel, marguerite); arms A0 / T / sham S, nights 6/10/14, hippocampus off, fresh state.
  pool40/41/42 s319/320/321 from 11:46 UTC, ~2.1 h; `~/brain-lane/route_repo/_scratch/route/kr/vc_full_s*.json`,
  `vc_done_full_s*.txt`. Smoke (s319 n6, T only): engaged (guard KL 0.55 -> 0.10, agreement 0.90), 17.7 GB peak,
  ~20 min per training arm; its log printed T n6 uv|knows 0/4 (a leak, disclosed; smoke rows not used).
  Read: pull to results/2026-11-26-voice-ceiling/, `voice_ceil table vc_full_s*.json --vs
  results/2026-11-26-voice-split/vs_s*.json --lives results/2026-11-26-voice-split/lives/*.json --out verdict.json`.
  TRANSFER -> before-build a voice night; NO-TRANSFER / ABSTAINS -> close the class, bank in PREREQUISITES.
- **TEST2 LAUNCHER CHECKED (11-26):** `launch_plan.sh plan` worst case $47 / $45 / $39 per UTC day (<= $50); env
  BRAIN_CODE_COMMIT/CLEAN set in `go`; no tmp dirs left. Ready for `go 2026-12-04` from a clean tree.
- **Pi:** lane was empty since ~11-13; two bounded items added 11-26 (where_ran host field; scorer-consistency audit).
- **AFTER A W REBOOT:** rerun `bash _scratch/s_launch_then_pull.sh` pull part only (or pull research/queue/.aws_brain.rvS<seed>
  per stopped instance) and re-arm tools/attention_watch.py --loop.
- **TIMEBOX: routing arc CLOSED 11-26 (STOP as pre-committed).** One last method: S_all competition
  without permanent/self episodes, offline screen on KR snapshots (pool, $0). If it fails: STOP routing, bank as
  "missing prerequisite (near-duplicate separation) at this scale", and run 12-04 test2 with the current memory. No 6th lever.
- **NEXT STEP:** (1) Read the voice ceiling (rule fixed; skeptic before landing; timebox 11-28). (2) (was 1-2: voice split read, done.) (3) Launch 12-04 AWS g5 fp16: 6 seeds x (B,
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
