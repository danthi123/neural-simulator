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
- **REORIENTED (owner 11-15):** stop passing our own scripted tests; primary measure is `brain/eval/open_conv.py`
  (held-out bank, new conversation per seed; read the transcripts).
- **FOUND (11-16):** day 2 collapses into one repeated line on 4/6 seeds (`results/2026-11-15-open-conv-baseline-1.5b.md`,
  control `results/2026-11-16-open-conv-day2-control-1.5b/`). Switches, default OFF: sleep_rehearse, habituation.
- **RESULT (11-16):** `results/2026-11-16-rehearsal-night-1.5b.md`: rehearse NO-GO, habituation fails R4 (hides loops as
  paraphrase). Board #290.
- **LIVE RUNS:** none. Teacher-night AWS i-008ed35fe857dce55 stopped holding nmtc_s102 (unverified finish;
  pull = InsufficientInstanceCapacity, retry `bash tools/aws_brain_batch.sh pull`). Old i-0129d9e410417218b stopped too.
- **POOL:** pool40/41/42 run brain/ jobs from `~/brain-lane` (venv `~/brain-lane/venv` on 40/41, `~/derisk-pool/sim/.venv`
  on 42; pattern `brain/experiments/jobs/2026-11-14-pool-*.sh`). pool40 host key re-trusted 11-14 (reinstall).
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
- **NEXT STEP (11-18):** the 11-30 milestone, not more 1.5B-ceiling defects. Night write has 3 failed levers (rehearse,
  habituation, reply-only R4 4/6), so its gate has fired. Take #282 reworded recall (due 11-23): S_all+contrast mu 3
  GO-narrowed unslept (`results/2026-11-14-sall-contrast*.md`), aunt lure 6/6, untested after a night; judge on open-conv.
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OPEN:** retry s102 AWS pull, then terminate. Chores: board timer -> main-following checkout; lane JSONs no host/sha.
