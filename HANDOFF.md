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
- **RESULT (11-16):** `results/2026-11-16-rehearsal-night-1.5b.md`: no switch flips. rehearse (whole-text guard) NO-GO
  (recall 25->10/36: the guard KL also holds the told sentence, so it fights replay); habituation passes R1-R3, fails
  R4 (blind reading tied 4/6), and hides the night's loop as paraphrase (pool 1112). Collapse is seed-dependent (1/6
  fresh). Biggest visible defect when no collapse: invented / misplaced facts (readers: 27 right, 19 wrong / 6 seeds).
  Board #290.
- **RESULT (11-16, later):** `results/2026-11-16-reply-rehearse-1.5b.md`: reply-only rehearsal NO-GO (AWS 1115-1120:
  R1-R3 hold, R4 blind 4/6 < 5/6; pool 1121-1125 recall 7/30 vs 16/30, abstains on every question on 1124/1125).
  Biggest defect now: in-context collapse (echo / one line / one template) in either arm, 3/12 AWS conversations.
  Echo-onset probe: the pretrained cortex alone echoes in 1/12 sampled conversations, then on 15/16 turns.
- **LIVE RUNS:** AWS g5.xlarge us-east-1b from `_scratch/wt-rehearse` (detached at 3a2a92ed6): default vs
  `--reply-monitor 4`, seeds 1126-1131, prereg `research/2026-11-16-reply-monitor-1.5b.md` (M1-M4, answer rate,
  engagement). Poll `bash tools/aws_brain_batch.sh sync`, then `pull` (terminates). Pool report-only monitor arm:
  pool40 1121, pool41 1124, pool42 1125 (`~/brain-lane/hab/done6.txt`; defaults already in the pool-replication dir).
  Old S_all batch i-0129d9e410417218b still stopped (wt-mixday state).
- **POOL:** pool40/41/42 run brain/ jobs from `~/brain-lane` (venv `~/brain-lane/venv` on 40/41, `~/derisk-pool/sim/.venv`
  on 42; pattern `brain/experiments/jobs/2026-11-14-pool-*.sh`). pool40 host key re-trusted 11-14 (reinstall).
- **NEXT STEP:** pull + score the reply-monitor batch (`rehearsal_night_score.py`, arm ocm; `withheld` in rows =
  engagement), paired blind R4, skeptic, flip only on all bars. If NO-GO: research gate fires (2 levers on in-context
  collapse: habituation, monitor). Then invented facts and when-to-recall, judged on open_conv transcripts.
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OPEN:** none blocking. Chores: board timer -> main-following checkout; brain/ lane JSONs stamp no host/sha.
