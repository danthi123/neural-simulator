# HANDOFF (overwrite each session; <= 40 lines)

- **TASK:** Sleep consolidation: told facts recalled from the cortex's own synapses after sleep, with the hippocampus
  off. Board #279 (and #280 high-use facts, #281 a lesion visibly changes replies).
- **TYPE:** build
- **DRIVER:** claude
- **STARTED:** 2026-11-02
- **TIMEBOX:** 2 days per problem (first decision point 2026-11-04: KEEP or KILL the text-replay method)
- **DONE MEANS:** six seeds, hippocampus off after sleep: true-replay recall above the no-sleep and shuffled arms on
  every seed, ordinary-text drift small, false recall not above shuffled; then sleep wired into the live brain
  (restart = sleep, adapters persisted) so the weekly transcript shows it.
- **MILESTONE:** day 14 (2026-11-16), brain/ARCHITECTURE.md §4. Day 7 (2026-11-09): first [weekly] run by Hermes.
- **STATE (11-03 pm):** 1.5B live brain six seeds (`results/2026-11-03-live-brain-1.5b.md`): hippocampus off after
  sleep 9.33/12 vs no-sleep 2.83 (6/6), shuffled 0; KEEP. Told-gate FAIL (`results/2026-11-03-told-gate-1.5b.md`):
  the lesson overrides a recalled fact when the gate fires; gate misfires 2-6/12 on told questions
  (`results/2026-11-03-gate-misfire-1.5b.md`). M2 a LEAD for never-told questions only. Weekly 1.5B preview
  (`results/2026-11-03-weekly-1.5b-preview.md`): told facts loop in free text after sleep; self-name only in the
  hippocampus. M3 FAIL at 0.4B. Brain defaults unchanged (new switches OFF; default cortex still 0.1B).
- **STATE (11-05):** hippo-route fix at 1.5B, six seeds on AWS (`results/2026-11-05-hippo-routing-1.5b-live.md`):
  NOT A LEAD (F3 fails). S_all + 1.5B bar never fetches a wrong memory (g 9 right/0 wrong x6) and the hippocampus now
  helps (f 7.67 vs a 4.50, g 4.83 vs b 2.83, 6/6), but after sleep it still costs recall vs off (f 7.67 < c 9.33):
  reinstating the RIGHT memory into a slept cortex loses 10 of 33 answers (gain 4 fit unslept). Pi's
  `research/pi-sleep-varied-replay` is superseded (not merged); its lane was empty because its workspace sat on it.
- **LIVE RUNS:** none. AWS instance terminated. W paused for the owner's other project (`tools/game.sh on`;
  resume: `bash tools/game.sh off && systemctl --user start sim-rag-autoupdate.timer`).
- **READY WORK (one next job per resource):** W GPU (when back): measure the slept cortex's own answer confidence as a
  "do I know this" signal (serves both the reinstatement gate and the lesson arbitration), then intrusion/looping of
  told facts after sleep. Pi: `PI_LANE.md` (seeded-key scorer first, before the 11-09 weekly; 12-facts rescore; mood
  re-analysis), CPU only. Mini-PCs: none queued.
- **NEXT STEP:** pre-register reinstatement gated by cortex confidence (or gain falling with consolidation) at 1.5B;
  then gate accuracy + arbitration for the lesson. Finish the 1.5B adoption plan.
- **DELEGATED:** Pi: `research/coordination/PI_LANE.md` open items on `research/pi-lane-1105` (from main, steered 11-05). Hermes:
  [weekly] every Monday from 11-09; WORKLOG at most once a day (RULING overnight-1102).
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **SPEED LEVERS (built):** AWS lane `tools/aws_brain_batch.sh`; Pi's Claude gated by live plan usage (`usage.ts`).
- **OPEN:** none blocking. Chores: repoint the board timer to a main-following checkout; pool40 host key (owner's call).
