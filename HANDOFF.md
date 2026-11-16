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
- **FOUND (11-16):** open-conv baseline: after one night day 2 collapses into one repeated line on 4/6 seeds
  (`results/2026-11-15-open-conv-baseline-1.5b.md`); control: the night causes the stored-line loop, and the
  pretrained cortex also loops (mirror, then echo) on 3/6 (`results/2026-11-16-open-conv-day2-control-1.5b/`).
  Fixes, default OFF: `Brain(sleep_rehearse)` and `Brain(habituation=N)`; prereg `research/2026-11-16-rehearsal-night-1.5b.md`.
- **RESULT (11-16):** `results/2026-11-16-rehearsal-night-1.5b.md`: no switch flips. rehearse (whole-text guard) NO-GO
  (recall 25->10/36: the guard KL also holds the told sentence, so it fights replay); habituation passes R1-R3, fails
  R4 (blind reading tied 4/6), and hides the night's loop as paraphrase (pool 1112). Collapse is seed-dependent (1/6
  fresh). Biggest visible defect when no collapse: invented / misplaced facts (readers: 27 right, 19 wrong / 6 seeds).
  Board #290.
- **LIVE RUNS:** pool40/41/42 report-only `sleep_rehearse="reply"` (KL on reply tokens only, prereg addendum 3) on
  seeds 1112-1114 (`~/brain-lane/hab/ocy_s*`, `done3.txt`); compare with the pool default via
  `rehearsal_night_score.py` + `stored_line_reuse.py`. Old S_all fit batch: i-0129d9e410417218b stopped in `wt-mixday`
  state; pull when g6 capacity returns (pull now waits for status-ok; poll with `sync` while running).
- **POOL:** pool40/41/42 run brain/ jobs from `~/brain-lane` (venv `~/brain-lane/venv` on 40/41, `~/derisk-pool/sim/.venv`
  on 42; pattern `brain/experiments/jobs/2026-11-14-pool-*.sh`). pool40 host key re-trusted 11-14 (reinstall).
- **NEXT STEP:** if reply-only keeps pool recall near default, run it vs default on fresh AWS seeds with R1 counting
  paraphrased loops too (`stored_line_reuse.py`; write addendum 4 first). Then the invented-facts defect (never-told
  questions answered with a made-up fact) and when-to-recall, both judged on open_conv by reading transcripts.
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OPEN:** none blocking. Chores: board timer -> main-following checkout; brain/ lane JSONs stamp no host/sha.
