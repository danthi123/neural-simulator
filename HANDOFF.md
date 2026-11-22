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
- **LIVE RUNS:** D2-PR screen (prereg `brain/research/2026-11-21-replay-priority-prereg.md`, DEV s319-321 vs B):
  B lives `lifeS1F_s<S>` on pool41/42/40 (nights 14/14/11 at 15:45 UTC), then PR-l starts on each pool by watcher
  (`_scratch/stage2/pool_pr_watch.sh` -> `outLIFE/lifePRl_s<S>`). PR-x s319 on the W CPU (pinned copy
  `_scratch/stage2/wcode_e5f15f6`, out `_scratch/stage2/wout/`, ~27 min/night, started 13:30 UTC). PR-x s320/s321 on AWS
  g5 after 00:05 UTC 11-22 (`_scratch/stage2/launch_prx2.sh`, state `.aws_brain.prx<S>`; ships the worktree brain/ at launch:
  keep brain/ committed until then). Read with `life_bars.py`, `life_leak.py` and the prereg bars; add the language-guard NLL.
- **NEXT STEP:** (1) language-guard method: before-build on the sleep keep set (broader ordinary-text interleave vs a
  norm bound); separating reads first (`life_langguard.measure()` ORDINARY vs HELDOUT on saved homes; keep-only night).
  (2) D2-PR read (timebox 11-23). (3) First-person echo record check. (4) Label-shuffle null for `life_bars.py` (§5).
  Any re-test needs a FRESH TEST split (§9).
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
