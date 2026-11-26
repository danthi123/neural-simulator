# Voice split prereg (2026-11-26): is the user's voice in told answers learned by sleep?

DEV diagnostic. No new Brain switch, no method. Offline on K-P's per-night homes (the KR night snapshots on pool40-42, which reproduce lifeKP with 0 mismatches), CPU fp32, $0. It runs on the same seed-to-node assignment as the cascade screen, so that screen's fresh-state rows act as a reproduction check.

## 0. Record brief
- **Biggest defect.** The research round (11-26) read the whole transcripts of lifeKP s319-321.
  - User voice in told answers ("My cat is called Pixel") is the biggest conversational defect.
  - A related defect: "What's your name?" is answered with the user's or a told name.
  - The skeptic marked the exact ranking UNDEFINED (its counts differ by detector), but the family is not disputed.
- **What the cascade screen already shows.** Pooled from `results/2026-11-25-cascade-screen/casc_s*.json`, told rows, nights 2-15, using `own2`:
  - Fresh state per question (B): intact 44/273, hippocampus-off 48/279. In the running battery (A): 62 and 54.
  - So the whole-night flips are mostly the cascade clustering a base rate that a fresh state still shows.
  - That base rate grows with age: hippocampus-off B is 10 on nights 2-5 and 38 on nights 6-15.
- **What is not known.** Is the fresh-state rate caused by sleep (the slept adapters), or is it the pretrained cortex's own habit once it has the fact in context?
  - The cascade screen had no pretrained-synapse arm. The source-hold DRAFT
    (`2026-11-24-self-other-tag-prereg-DRAFT.md`) stays BLOCKED until this is known.
  - One block reason was that the flips are whole-night, not item-level. The fresh-state rate is item-level, so that reason is now open again.
  - Its Stage-0 kill (both-off 2 vs hippocampus-off 0 of 72, night-matched) used the running battery on 4 nights only.
- **External.** Hippocampal item-context binding (Yonelinas 2019, Nat Rev Neurosci); sleep strengthens source memory (Berres 2024, Mem Cogn; Qiu 2026, Cognition). In brain terms: if the voice is sleep-learned, the replay is consolidating the item without its source.
- **Prerequisites.**
  - Sleep consolidation: PARTIAL.
  - Self/other tag: MISSING.
  - A diagnostic needs neither.
- **Expected outcome:** none registered. Either verdict decides what comes next (section 4).

## 1. Arms and conditions (Amendment R1, before data: pre-data review, sonnet, 3 BLOCKING fixed)
Same homes, nights, seeds and nodes. Reseed is 500+night, as in the cascade screen.

**Condition B** (each question from a fresh state):

| arm | synapses | fetch | note |
|---|---|---|---|
| intact | slept | yes | |
| hippocampus-off | slept | no | |
| consolidated-off | raw | yes | not "pretrained only", see below |
| both-off | raw | no | |

The consolidated-off fetch reinstates a stored matrix delta, not text. Deltas and keys were stored through the adapters active at the time, so this arm still carries sleep's influence through them. It is a conservative control.

**Condition ctx** on both-off (fresh state; the pretrained habit with the fact in context):
- The item's ledger telling is heard first, and the brain answers it.
- Then the question is asked.
- Told rows only. Other rows run as B.

**Condition A** on consolidated-off and both-off: report only.

**Code.** `brain/experiments/voice_split.py`, which keeps the rows' fetch fields (`recalled`, `right_episode`, `wrong_fetch`).

## 2. Measures (told rows: `cascade_screen.is_told`)
- **knows.** The reply names the told fact: any ledger key of the item, as a whole word.
- **uv.** The reply is in the user's voice (`cascade_screen.own2`, verbatim).
- **Rates.** P(uv | knows) is primary. P(uv) on all told rows is reported beside it.
- **Report only.**
  - The hedge rate (the forms own2 excludes).
  - Right-episode fetches per arm.
  - Nights 2-5 vs 6-15.
  - The self-name replies.
- **Banked n.** From the cascade screen B rows, knows is about 54-63 per seed and arm, concentrated in 3 items asked about 14 times each. So rows are not independent, and the statistics resample items.

## 3. Rule (fixed before data)
- **VOID seed.** Its intact-B and hippocampus-off-B replies match the cascade screen's on fewer than 90% of rows, or its `brain_kw` differs (the run errors).
- **UNDEFINED.** Fewer than 2 live seeds, or hippocampus-off B or ctx has knows < 20 on any live seed.
- **The statistic.** D = pooled P(uv|knows), hippocampus-off B minus ctx. Its 95% CI comes from an item-clustered bootstrap: resample (seed, item) cells, 2000 draws, rng seed 20261126.
- **SLEEP-LEARNED.** Both of:
  - CI lower bound > 0;
  - hippocampus-off > ctx on ≥ 2 live seeds.

  Sleep adds user voice beyond the cortex's own in-context habit.
- **NOT-SLEEP.** The CI lies inside ±0.10.
- **FETCH-PRIMED.** Neither of the above, and the CI of consolidated-off minus hippocampus-off has its lower bound > 0.
- **UNDEFINED.** Anything else.
- **Report only, each with a CI.**
  - hippocampus-off − consolidated-off;
  - intact − consolidated-off;
  - hippocampus-off − both-off;
  - every comparison above, also on P(uv) over all told rows.

## 4. Consequences
- **SLEEP-LEARNED.**
  - Unblock the source-hold DRAFT through an amendment that answers each remaining block item: Stage 1 on pool or AWS (W is held), a sham arm, the cited near-miss-guard neighbour, the stop rule (the routing arc is closed), and the record check.
  - Its first read is fresh-state P(uv|knows), not whole-night flips.
- **NOT-SLEEP.**
  - The voice is the pretrained cortex's habit once it has the fact, so no night method targets it.
  - Bank it in PREREQUISITES (Self and identity).
  - The next target is the reinstatement side (what the fetch puts in context) or the cascade (state), under a new before-build.
- **FETCH-PRIMED.** The voice rides on what the fetch reinstates. The next target is the reinstatement side, under a new before-build.
- **UNDEFINED.** Report it; no lever follows from it.

## 5. Cost and timebox
- **Runtime.** About 100 s per condition and night, so 7 × 14 × 100 s ≈ 2.7 h per seed, with the three seeds in parallel.
- **Timebox.** Read by 11-27. Nothing on AWS. This does not touch test2 (12-04..06).
