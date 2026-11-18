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
- **FOUND (11-16):** day 2 collapses into one repeated line on 4/6 seeds (`results/2026-11-15-open-conv-baseline-1.5b.md`,
  control `results/2026-11-16-open-conv-day2-control-1.5b/`). Rehearse NO-GO, habituation fails R4 (board #290).
- **LIVE RUNS (11-18 pm):** pool40/41/42 `brain/experiments/jobs/2026-11-18-pool-sall-slept-open-conv.sh` (prereg
  amendment 1, seeds 1132-1137, d then s=S_all+slept bar; out `~/brain-lane/sslept/`, `done.txt`).
  AWS i-008ed35fe857dce55 stopped holding nmtc_s102 (pull: capacity errors); old i-0129d9e410417218b stopped.
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
- **RESULT (11-18):** S_all on open conversation, S1 FAILS: reworded right memory fetched 3/18 vs S0 3/18; S_all fires
  less (6 vs 15), skeptic PARTLY REFUTED "bar is the cause" (undetermined). S0's wrong fetches cost reworded replies
  (hippocampus off 10 vs S0 7). `results/2026-11-18-sall-open-conv-1.5b.md`.
- **SLEPT FIT (11-18):** after a night S_all still ranks the right telling first on rewordings (top-1 0.83/0.69):
  bar cause. Frozen `KEY_BARS_SLEPT_BY_SIZE` (0.165, 0.02), `key_bar="slept"` OFF. `results/2026-11-18-sall-slept-fit-1.5b.md`.
- **NEXT STEP:** when sslept DONE: score S1-S3 (as `results/2026-11-18-sall-open-conv-1.5b/scores.txt`), S4 blind reader
  if S1-S3 hold, skeptic, decide per prereg amendment 1. S1 fails again: NO-GO S_all; next a self-key on slept cortex.
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OPEN:** retry s102 AWS pull, then terminate; RAG refresh fails its quality check (index stale).
