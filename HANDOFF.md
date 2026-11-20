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
- **LIVE RUNS (11-19 23:59):** dev lives, 7 nights, `~/brain-lane/life_arm.sh <seed> <arm>` -> `outLIFE/life_s<S>_<A>.*`:
  pool40 s315 B, pool41 s315 S, pool42 s316 B; queue s316 S, s317 B, s317 S (prereg
  `brain/research/2026-11-19-life-muse-mscale-prereg.md`). **SCREENS (s310-312):** M-use KILLED (right fetches denied
  3/4, rule >40%); M-scale open speech recovers 3/3 seeds (intrusions -> 0), H mixed 2->4, 2->2, 4->2
  (`results/2026-11-19-life-screens-1.5b.md`). Stage-0: H falls to 0-2/6 by night 7 on all 5 seeds; hub + norm growth
  (`results/2026-11-19-life-stage0-1.5b.md`). Hub defect needs a new method (not a raw-margin gate). **WEEKLY READ (11-19, seeds 300-305, curiosity ON):** intact 31/36 vs hippocampus-off 35/36
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
  Baseline done (Stage 0, 5 seeds). Read the 6 dev lives (B vs S, nights 4-7 pooled, >= 2/3 seeds + whole
  transcript); S passes -> six-seed registered life (Stage 1). Hub defect, next method (before-build first): scale the
  use boost by content match (raw sim above familiarity) instead of a flat 0.5, so a weak match earns little.
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OPEN:** retry s102 AWS pull, then terminate; RAG refresh fails its quality check (index stale).
