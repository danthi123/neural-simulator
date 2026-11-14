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
- **STATE (11-14):** the 1.5B brain now answers facts told today, before sleep. Live six-seed mixed day
  (`brain/experiments/results/2026-11-13-live-gain-1.5b.md`): gain 0.5 GO (L0-L4), new facts 3.50/6 vs 1.17 at the old
  gain 4 and 0.17 hippocampus-off; told 8.00/12 vs 4.83; no slept-fact cost vs c; skeptics NARROWED (false recall
  uninformative on this probe). **Flipped:** `Brain(gain="auto")` default = 0.5 at 1.5B, 4.0 elsewhere (0.1B unchanged;
  0.1B refit NOT A LEAD, `2026-11-13-gain-refit-0.1b.md`). Oracle refit: `2026-11-13-gain-refit-1.5b.md`. Scorer fix
  re-scored 0.1B live-brain and told-gate (awake hippocampus ~1.5/12, errata in those docs).
- **LIVE RUNS:** none (AWS pulled + terminated; ~$2.5 today). Pool: all three nodes run brain/ jobs from
  `~/brain-lane` (brain/, rwkv, 0.1B + 1.5B models; venv `~/brain-lane/venv` on pool40/41 (torch 2.11 CPU, installed
  11-14 with the owner's OK), `~/derisk-pool/sim/.venv` on pool42; job pattern in
  `brain/experiments/jobs/2026-11-14-pool-key-default.sh`). pool40's host key re-trusted 11-14 (reinstalled 10-01;
  fingerprints checked on the host). W is dedicated to the owner's other work (RULING w-dedicated-1113); AWS cap ~$50/day.
- **KEY DEFAULT (11-14):** KEEP S0 (`brain/experiments/results/2026-11-14-key-default-1.5b.md`; K2 failed, mixed day
  S_all +0.83 < 1). S_all leads by about a fact everywhere with no cost; at gain 0.5 S0 already answers 4.83/12 unslept.
- **ECHO (11-14):** three levers, none passed its bar (oracle lived NOT A LEAD; live lived NO-GO, routing 36 -> 4/72;
  live lived_key NO-GO +0.33 but echoes 13 -> 4 and answered-given-right-episode 37 -> 43 of 54;
  `results/2026-11-14-lived-*.md`). Switch `Brain(episode=)` stays "tell". Routing (S0 right on 36/72 told) is now the
  limit of the unslept 1.5B brain.
- **READY WORK:** pool (CPU): S_all + lived_key (store a key read from the telling for S_all; then a six-seed
  unslept test vs S0 + tell, prereg first). AWS: confidence gate for false recall on the slept cortex (c says a told fact on
  6.83/12 never-told questions). Pi: `research/coordination/PI_LANE.md` (on `research/pi-lane-1105`, unmerged).
- **NEXT STEP:** a research round on perspective before a 4th echo lever; else (2) the false-recall gate; record-check both first
  (`2026-11-03-confabulation-next-methods.md`, confab M1-M3, abstain v2). The 1.5B cortex default (milestone 11-16)
  is what puts all of this in the weekly transcript: consider flipping the default cortex to 1.5B (needs a prereg).
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **SPEED LEVERS (built):** AWS lane `tools/aws_brain_batch.sh`; Pi's Claude gated by live plan usage (`usage.ts`).
- **OPEN:** none blocking. Chores: repoint the board timer to a main-following checkout.
