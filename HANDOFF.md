# HANDOFF (overwrite each session; <= 40 lines)

- **TASK:** 12-14 milestone, multi-week life with nightly sleep (board #286). Design
  `brain/research/2026-11-19-multiweek-life-design.md` (§5 bars, §13 registration). Older state: `brain/experiments/results/`.
- **TYPE:** build · **DRIVER:** claude · **TIMEBOX:** Stage-2 methods by 12-08, 2 days each; read/land 12-10 to 12-14.
- **STATE (11-21): STAGE 1 READ** (`results/2026-11-21-stage1-registered-test-1.5b.md`, six TEST seeds, every flag,
  5 skeptics: every verdict survives). P0 NOT VALID (never-told false 0.40-0.76, old bins thin) -> savings UNDEFINED
  (G_null_sub +0.67, p 0.053); cued recall, transfer, self-report measured negatives; language guard FAIL (held-out NLL
  1.93 -> 3.8-4.25, KL 13-17x night 1, keep term holds only its 3 texts); §7 FAIL (readers prefer the untrained cortex
  6/6, week 1 over week 3 6/6). Measurement milestone met on the §5 letter; no capability passes. Programme kill
  ruled NOT met (only s44 meets the per-seed kill, nights 14-15; the single-night reading would meet it: owner may override).
  Aggregator `brain/eval/life_bars.py` (6e1f4932e, built on DEV only). Defects in order: language drift in sleep, first-person
  echo (150/311 wrong H answers), stock self-reply leak (24%), told keys on never-told questions.
- **D2-PR READ DONE (11-22, `results/2026-11-22-d2pr-dev-read-1.5b.md`, 5 refuters): PR-l and PR-x both UNDEFINED**
  (engagement: loss weight concentrates replay, top-6 above the strength-only reference; PR-l 5/6, PR-x 0/6). PR-l leak
  -44% robust over nights (p<5e-5), guards pass; PR-x crosses fp16/fp32 so no KILL. Neither goes to TEST. KEEP the
  replay-priority lever: next method PR-l + per-episode share cap (own prereg; name the day-1-2 window; fp32 every arm).
  Report-only echo read `brain/eval/life_echo.py`: echo equal hippocampus-off (in the cortex); PR-l lowers it.
- **LIVE: D2-PR2 DEV screen** (prereg `brain/research/2026-11-22-replay-priority-2-prereg.md`, built c91c0c200, `loss_a`
  = L_e**0.6): PR-a s319 pool41, s320 pool42 (same nodes as B, CPU fp32, started ~17:15 UTC 11-22, `outLIFE/lifePRa_s<S>`);
  s321 pool40 node-mxl1462drl (started by hand ~13:20 UTC 11-22; the watcher's pgrep matched its own ssh line). Read with `d2pr_bars.py` (results/2026-11-22-d2pr-dev-read-1.5b/) + §7 blind read. Lever 3 of 3.
- **K-P (language-keep method, prereg `brain/research/2026-11-21-sleep-language-keep-prereg.md` §5-6):** built (d777f71c9,
  default OFF byte-identical). Gate run 2 LAUNCH (gate-overridden): one night from s316 n15 took held-out KL 2.40 -> 0.86
  (B 2.80), NLL 4.18 -> 2.81 (B 4.59), replay still writes. DEV screen s319-321 on AWS g5 (`_scratch/stage2/launch_kp.sh`,
  state `.aws_brain.kp<S>`, out `_scratch/brainv2/aws/<batch>/lifeKP_s<S>`), read vs `lifeS1F_s<S>` with §3 bars.
  Gate replications (report only, results/2026-11-22-kp-gate-replications-1.5b.md): s319 and s321 repeat s316
  (held-out KL 0.76/0.81 vs B 2.79/2.76). s320 running on W (`_scratch/kpgate_w/run_w320.sh`); add its row when done.
- **NEXT STEP:** (1) K-P DEV read when AWS lives land (§3 bars, gate-overridden label). (2) D2-PR2 read when PR-a lands. (3) Fresh TEST split (§9; more old-age items per seed, P0). (4) Label-shuffle null for `life_bars.py` (§5).
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OWNER WORDS (2026-11-20, verbatim):** "Looks like the GPU fell off the bus earlier? I won't be home to reboot until
  Monday so continue with CPU, pool, and AWS until then. The MTALLM Claude session also may need AWS for GPU access, I
  am preapproving that access as needed so don't ask for permission."
- **OPEN:** W GPU off the bus until Mon 11-23 (CPU, pools, AWS only; AWS shared with MTALLM, keep cap headroom). s102 g6
  i-008ed35fe857dce55 STOPPED (state `research/queue/.aws_brain.s102-stopped-20261117`), retry pull then terminate. RAG
  index stale. MTALLM stopped 11-21; final hand-off `~/Projects/mtallm/experiments/reports/sim_handoff.md` (84ecd10) read: info only; its §6 says a 1.5B in-context reader invents untold attributes when many people are in view (check any in-context judge at that density).
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
