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
- **RESULT (11-16, later):** `results/2026-11-16-reply-rehearse-1.5b.md`: reply-only rehearsal NO-GO (AWS 1115-1120:
  R1-R3 hold, R4 blind 4/6 < 5/6; pool 1121-1125 recall 7/30 vs 16/30, abstains on every question on 1124/1125).
  Biggest defect now: in-context collapse (echo / one line / one template) in either arm, 3/12 AWS conversations.
  Echo-onset probe: the pretrained cortex alone echoes in 1/12 sampled conversations, then on 15/16 turns.
- **RESULT (11-16, night):** reply monitor (`results/2026-11-16-reply-monitor-1.5b.md`): no-worse bars pass on ties,
  no benefit (the one collapse identical in both arms): NOT flipped. Own-reply gain probe (`results/2026-11-16-own-gain-probe.md`):
  the write of the brain's own reply into the state IS the amplifier (forced copy repeats 20/20 at g=1), but g<=0.25
  loses turn-taking (asks questions back) and E fails at every g: no lead. Copy-delta: no copy direction to suppress.
- **LIVE RUNS:** pool baseline: pretrained cortex alone, spontaneous collapse onsets (echo + questions back) over day-2
  scripts 1112-1124 x 6 samples (`~/brain-lane/hab/onset_base_pool4N.json`). No AWS instance live. Old S_all batch
  i-0129d9e410417218b still stopped (wt-mixday state).
- **POOL:** pool40/41/42 run brain/ jobs from `~/brain-lane` (venv `~/brain-lane/venv` on 40/41, `~/derisk-pool/sim/.venv`
  on 42; pattern `brain/experiments/jobs/2026-11-14-pool-*.sh`). pool40 host key re-trusted 11-14 (reinstall).
- **NEXT STEP:** in-context collapse (timebox 11-18). Copies-only own-gain: counts pass, reading fails (2/6 E samples
  speak as the user from turn 2): keeping copies out of the state is necessary, not sufficient; the state needs a
  positive marker of whose turn it is. Read the onset baseline (how often echo / role confusion start), then design
  that marker (research method 2 or a self/other tag). Then invented facts on open_conv.
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OPEN:** none blocking. Chores: board timer -> main-following checkout; brain/ lane JSONs stamp no host/sha.
