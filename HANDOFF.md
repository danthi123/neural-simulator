# HANDOFF (overwrite each session; <= 40 lines)

- **TASK:** Sleep consolidation: told facts recalled from the cortex's own synapses after sleep, with the hippocampus
  off. Board #279 (and #280 high-use facts, #281 a lesion visibly changes replies).
- **TYPE:** build
- **DRIVER:** claude
- **STARTED:** 2026-11-02 · **TIMEBOX:** 2 days per problem (first decision point 2026-11-04: KEEP or KILL the text-replay method)
- **DONE MEANS:** six seeds, hippocampus off after sleep: told facts recalled, little drift, false recall not above shuffled.
- **MILESTONE:** day 14 (2026-11-16), brain/ARCHITECTURE.md §4. Day 7 (2026-11-09): first [weekly] run by Hermes.
- **STATE (11-15):** lesion item MET (`2026-11-15-sleep-lesion-weekly-1.5b.md`). **`self_completion` now ON by default**
  (`results/2026-11-15-self-completion-replication-1.5b.md`: seeds 300-305 GO; day-1 own name 24/24 vs 13/24 pooled;
  caveat: "I have one too" self-claims 6 vs 16 of 192, p~0.29; weekly `--no-self-completion` = flag-off arm).
  After the night "What's your name?" still recalls the sister telling: S0 loses on 6/6 slept homes (S0 key is the
  layer-0 attention delta, the night cannot reach it); S_all ranks self first 6/6 but clears its 0.22 bar 2/6
  (`2026-11-15-self-recall-after-night-1.5b.md`). Self episodes keyed by their question: steal every question,
  K2 0/12, NOT A LEAD (`2026-11-15-self-key-question-1.5b.md`; S0 is form-dominated). Teacher-night gate: own cues
  0/12 (cortex writes them to the brain, one greedy draw), never-told clean 1/87: no build
  (`2026-11-15-teacher-night-gate-1.5b.md`). **(11-14):** gain 0.5 on for 1.5B; key default KEEP S0 (S_all +1 fact,
  short of the bar, `2026-11-14-key-default-1.5b.md`); echo levers banked NO-GO; near-miss guard NOT A LEAD.
- **REORIENTED (owner 11-15):** stop passing our own scripted tests; primary measure is `brain/eval/open_conv.py`
  (held-out bank, new conversation per seed; read the transcripts).
- **FOUND (11-16):** open-conv baseline: after one night day 2 collapses into one repeated line on 4/6 seeds
  (`results/2026-11-15-open-conv-baseline-1.5b.md`); control: the night causes the stored-line loop, and the
  pretrained cortex also loops (mirror, then echo) on 3/6 (`results/2026-11-16-open-conv-day2-control-1.5b/`).
  Fixes, default OFF: `Brain(sleep_rehearse)` and `Brain(habituation=N)`; prereg `research/2026-11-16-rehearsal-night-1.5b.md`.
- **LIVE RUNS:** AWS batch 1 (default vs rehearse, seeds 1106-1111) from worktree `_scratch/wt-rehearse`; AWS batch 2
  (H, HR) from `_scratch/wt-habit` -> in EACH: `bash tools/aws_brain_batch.sh pull`; transcripts are the .json
  outputs. Pool report-only default vs HR seeds 1112-1114 (`~/brain-lane/hab/`). Old S_all fit batch: instance
  i-0129d9e410417218b stopped, pull failed (no g6 capacity) from `wt-mixday`; low priority, pull when capacity returns.
- **POOL:** pool40/41/42 run brain/ jobs from `~/brain-lane` (venv `~/brain-lane/venv` on 40/41, `~/derisk-pool/sim/.venv`
  on 42; pattern `brain/experiments/jobs/2026-11-14-pool-*.sh`). pool40 host key re-trusted 11-14 (reinstall).
- **NEXT STEP:** score R1-R4 (collapse = a day-2 reply repeated >3 of 16; R4 blind sonnet reading of all arms); land;
  flip what holds. Then when-to-recall (needless recalls 2-4 per conversation) judged on open_conv.
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OPEN:** none blocking. Chores: board timer -> main-following checkout; brain/ lane JSONs stamp no host/sha.
