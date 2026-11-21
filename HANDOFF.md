# HANDOFF (overwrite each session; <= 40 lines)

- **TASK:** Sleep consolidation: told facts recalled from the cortex's own synapses after sleep, with the hippocampus
  off. Board #279 (and #280 high-use facts, #281 a lesion visibly changes replies).
- **TYPE:** build
- **DRIVER:** claude
- **STARTED:** 2026-11-02 · **TIMEBOX:** 2 days per problem (first decision point 2026-11-04: KEEP or KILL the text-replay method)
- **DONE MEANS:** six seeds, hippocampus off after sleep: told facts recalled, little drift, false recall not above shuffled.
- **MILESTONE:** day 14 (2026-11-16), brain/ARCHITECTURE.md §4. Day 7 (2026-11-09): first [weekly] run by Hermes.
- **STATE (11-15):** lesion item MET; `self_completion` ON by default (seeds 300-305 GO); after the night "What's your
  name?" still recalls the wrong telling (S0 loses 6/6; S_all clears its bar 2/6); self-key and teacher-night NOT A
  LEAD; gain 0.5 on for 1.5B; key default KEEP S0. Details: `brain/experiments/results/2026-11-1[45]-*.md`.
- **LIVE RUNS (11-20 ~16:40 EST):** Stage-1 15-night run check on DEV, defaults (`--days 15 --weekend
  --full-probe-nights 1,5,10,15 --frozen-probe --span --transfer`): pool40 s315, pool41 s316, pool42 s317
  (`~/brain-lane/life_arm_s1dev.sh <seed>` -> `outLIFE/lifeS1D_s<S>.*`, about 9-10 h). Checks night time, the 15-night
  harness and the instrument before the registered Stage 1 (six seeds, TEST). Three TEST 7-night B lives (s42-44)
  started by mistake were stopped within 10 min, before any probe read (homes moved to `*.stopped`); TEST not read.
  **Stage-1 DRESS REHEARSAL (11-20 eve):** s318 DEV, every flag (`--pruned-a --cue-ladder --interview` + the above),
  AWS g5 (`_scratch/stage1/jobs_s318_full.txt`, MAX_MIN 720); pull with `tools/aws_brain_batch.sh sync|pull`.
  **M-SCALE-E READ (11-20, s315-318): KILLED, the M-scale family ends.** Recall guards pass (H int 53 vs 42/96, H off
  55 vs 37, day-1-2 off 64 vs 40/113, intrusions 10 vs 37, ‖A‖ 0.73-0.78x), but transcript FAILS 4/4 (4 refuters):
  identity loss ("My name is Dan.", "Qwen", "Bramble"), a ✿ marker leak (s316), told names overwritten ("Luna"),
  job fact lost (`results/2026-11-20-life-mscale-e-dev-1.5b.md`). Stage 1 runs with the defaults.
  **M-SCALE-3 READ (11-20, SAB s315-318): M-scale KILLED as registered.** Recall holds (H int 65 vs B 42/96, day-1-2 off
  74 vs 40/113, ‖A‖ fixed 0.72-0.91x) but norm escapes 2.2x night-1 on 3/4 seeds (cut before replay: waking cortex =
  budget + one night) and transcript fails 4/4 (s315 identity loss "My name is Dan baker."; stuck phrases, stutter loops;
  3 readers refuted) (`results/2026-11-20-life-mscale-sab-1.5b.md`). Next = M-scale-E (cut after replay, SHY; 3beb11654).
  **M-SCALE DEV READ (11-20, s315-317):** S (B factors) does NOT pass prereg 7: H int 51 vs 30/72, H off 53 vs 27
  (3/3 seeds), but norm > 2.2x night-1 and ‖A‖ 1.59-1.64x on 3/3; transcript: night 7 clearly better (no "My cat is
  called Pixel", keeps "Nova"), S-only defects on 2/3 (s316 one ✿text2✿ marker leak + flat day-7 replies, s317 stuck
  appended phrases) -> M-scale-3: SAB on s318 (AWS) + extra dev SAB s315/316/317 on pool40/41/42
  (`results/2026-11-20-life-mscale-dev-1.5b.md`). **R-raw KILLED** (K1 6 < 16, K2 s312 0.75)
  but bar-limited (R fired 20/100) and K2 low-n; S0 raw ranking (arm D) dissolves the hub (top-1 <= 0.22 on 4/4) and
  right fetches 8 -> 14, fails K1 by 2 and K3 (`results/2026-11-20-rank-raw-screen-1.5b.md`). Hub levers = 2 ->
  research before a third. **SCREENS (s310-312):** M-use KILLED; M-scale open speech recovers 3/3 seeds
  (`results/2026-11-19-life-screens-1.5b.md`). Stage-0: H falls to 0-2/6 by night 7 on all 5 seeds
  (`results/2026-11-19-life-stage0-1.5b.md`).
- **WEEKLY READ (11-19, seeds 300-305, curiosity ON):** intact 31/36 vs hippocampus-off 35/36
  after the night: every intact error is a wrong reinstated episode ("What's your name?" -> Marguerite telling 6/6);
  slept facts intrude in both arms (fox named "Pixel" 12/12). Curiosity false fires harmless; no collapse
  (`results/2026-11-19-weekly-defaults-300-305-1.5b.md`).
- **RESULT (11-19):** curiosity ON by default at 1.5B (`Brain(inquiry="auto")`, flipped 2bf228e2f). Six-seed test PASS,
  NARROWED: coined asks 0.08 -> 0.33, asks-or-admits +0.38, invents -0.12, common untouched; 12 fixed items (gain on ~7),
  fires on some real names ("What's pho?"; in the weekly 7 false fires, replies ordinary). Weekly guard: first run
  leaked "Zorbit" after sleep -> lesson kept outside the episode list, never replayed; final guard W-a 70 vs 71.
  Milestone (c) MET at scaffold level (`results/2026-11-19-inquiry-{test,weekly}-1.5b.md`).
- **TIMEBOX (11-17), in-context collapse: KILL further levers at 1.5B for now.** Six methods, none a lead (habituation,
  reply monitor, own-reply gain flat + copies-only, carry penalty, fake-think prefix: 53/72, worse) and 2.9B loops more
  (`results/2026-11-16-onset-*.md`). Banked: the own-reply write is the amplifier. On the GPU decision host the
  cortex alone loops on open turns in 6/72. Reopen with a state-level whose-turn marker or a new cortex family.
- **INVENTED FACTS (11-17, 1.5B), 4 levers failed:** M1 FALSIFIED; in-context ceiling (pretrained cortex with the
  tellings in context invents 40%, the slept brain 99%); teacher night NARROWED at best, no flip: refusals distil as
  an unconditional reply (2/6 open-conv seeds refuse nearly everything). `results/2026-11-17-{confab-m1,incontext-abstain,teacher-night}-1.5b*.md`.
- **TIMEBOX (11-18, a day early; s102 cannot change it): KILL self-teacher levers for invented facts at 1.5B.** The
  1.5B cortex in context binds a wrong told fact as often as the right one (9 vs 7/64); distilled refusals turn
  unconditional. Verdict on the method; reopen with a cleaner in-context cortex (2.9B: a third bad targets).
- **RESULT (11-18 eve):** S_all + slept bar, seeds 1132-1137: reworded right memory 15/18 vs S0 4/18 (6/6) = milestone
  selection MET as read-out; replies flat (27 vs 24; slept cortex alone 27); S4 3/6 FAIL: NO flip. 3 skeptics PARTLY
  REFUTED (selection only). Unslept bar first failed (3 vs 3); slept fit (0.165, 0.02) in
  `KEY_BARS_SLEPT_BY_SIZE`. `results/2026-11-18-sall-{open-conv,slept-fit,slept-open-conv}-1.5b.md`.
- **GATE (11-18):** registered seeds: reworded right memory 14/18 vs 1/18 (6/6) replicates; S3 intrusions 21 vs 17 FAIL
  (collapse-driven, equal collapse rates); no flip. Weekly: own name fetched 5/6 vs 0/6, named right 3 vs 2 (W2 FAIL),
  no cost. `results/2026-11-18-sall-slept-gate-1.5b.md`.
- **NEXT STEP:** 11-30 status: (a) selection MET as read-out, (c) curiosity MET at scaffold level (ON), (b) mood NOT MET:
  iteration 7 Stage 0 FAIL, STOP per prereg (`results/2026-11-19-mood7-stage0-1.5b.md`: appraisal sign 100%, NEG vs
  surprise-matched neutral AUROC 0.98, but fire 0.65 on s42 and ordinary recall takes 56% of follow-up turns). Next
  mood method (after 11-30) must act alongside recall. NOW: 12-14 milestone (multi-week life), design `brain/research/2026-11-19-multiweek-life-design.md`.
  Baseline done (Stage 0, 5 seeds). B vs S read done (S fails norm/‖A‖ + S-only defects); SAB killed (norm 3/4, transcript 4/4). M-scale-E KILLED
  (transcript 4/4). NOW: Stage 1 with the defaults. Stage-1 pieces BUILT (b83a6c036: `--pruned-a`, `--cue-ladder`,
  `--interview`, 223 tests) and REGISTERED (design §13: hashes, command, DEV-decided changes). Gate: the s318 all-flags
  dress rehearsal finishes; then launch six seeds on TEST by 12-01. TEST split WRITTEN + hashed (222 items, blind sonnet,
  `brain/eval/life_bank_test.py` + `.sha256`, d396963d9) and wired (`--split test`, hash-guarded, 7e6733373). Transfer
  read-out built (`--transfer`, dT = T - pretrained T0 per life: raw T has a domain-fluency offset, 0.58 at 0.1B; the
  design bar "both-off T about 0" must read dT; dab53c019). Span read built (`--span`: span-NLL + log-odds vs 3 foils
  on told items, for P0 forgetting / savings; 58ae5968a). Stage-1 TODO before 12-01: savings P0, cue ladder, 15-night run check. Hub defect: R-raw KILLED; fetch rarely moves the reply at
  night 7, so it ranks below M-scale; research round (before-build 7) on D + open-turn fire gate vs decay-clock rule.
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OPEN:** s318 batch 20261120-0412 pulled + terminated. s102 g6 instance i-008ed35fe857dce55 STOPPED (no capacity to restart; state file set aside as
  `research/queue/.aws_brain.s102-stopped-20261117` in the worktree), retry its pull later, then terminate; RAG refresh fails its quality check (index stale).
