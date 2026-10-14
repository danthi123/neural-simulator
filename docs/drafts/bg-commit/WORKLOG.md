# BG live seam, spiking commit stage: WORKLOG

*Claude build agent, 2026-10-14, branch `claude-build/bg-commit` from S main cc9b4bf3e. Task: retire the BG live seam
with the spiking commit stage (`docs/drafts/spiking-commit/PLAN.md` section 3), the next method after the 10-02 NO-GO
on G4. Dev seeds 1-8 only; gate seeds 42/43/44/100/101/102 are Hermes's, after the amendment is committed.*

## 2026-10-14: STEP 1 design check. Verdict: PROCEED. The channels compete on one bridge, driven by the turn.

Read first: the 09-29 prereg (all), the 10-02 RESULT, `research/runners/host_share_bg_seam.py`, the BG organ
(`research/runners/bg_action_selection_production_organ.py`, region `neural-seam`, 315-589) and the selector it builds
(`research/runners/_vocal_action_selector_gate.py` 116-381), PLAN.md (all), the A1 commit code
(`research/runners/word_role_parser_organ.py` 175-200, 251-372, 508-626) with `tests/test_word_role_commit.py`, and
`docs/drafts/kselect-commit/WORKLOG.md` (k-select STOPPED because its candidates are host-evaluated; not repeated here).
`tools/before_you_build.sh "BG live seam latch read not threshold-invariant (G4)"` and the RAG surfaced only the
prereg, the RESULT and older Gate-B threshold findings; no prior commit-stage attempt on this seam.

### One bridge, spiking competition

`NeuralSeamSelector.ensure_built` (organ 540-551) builds ONE `SimulationBridge` via `build_selector_bridge(...,
extra_regions=..., extra_pathways=...)`: the Gate-A v2 selector plus the neural-seam regions. On that bridge, per
channel k (0 SPEAK, 1 STAY-SILENT): `proposal_k -> str_d1_k -> gpi_k -| thal_k -> commit_k -> motor_k`, with the
indirect path through `str_d2_k -> gpe_k -> selector_stn -> gpi_*`, cross-channel inhibition `commit_fs_k -| commit_j`
(selector 316-323), and the seam's own latch `motor_k -> latch_k`, `latch_fs_k -| latch_j` (organ 419-430). The two
channels compete through these synapses; nothing between the stimulus and the motor pools is computed in Python.

### Driven by the turn's own activity, not relayed host numbers

The only currents the host writes during a trial (`_run_neural_seam_trial`, organ 442-466): the selector's equal
tonic currents on GPi and thalamus (identical for both channels), shared arousal on `practice_arousal`
(candidate-agnostic), the default-mode `hold` tonic 350 pA (identical on every turn), and the rendered stimulus: each
word of the message drives a fixed crc32-addressed 12-neuron group of `heard` at 500 pA (`render_utterance`, 387-394;
the declared body/senses transduction). Which channel gets more drive is decided by synapses: `heard -> str_d1_SPEAK`,
`hold -> str_d1_STAY_SILENT`, `heard -> hold_inh -| hold` (organ 407-416). No count, score, comparison or rate is
relayed in. With `BRAIN_BG_NEURAL_SEAM=1` the host admission and `salience(msg)` are not called (pinned by
`tests/test_bg_neural_seam.py`). So the alternatives are spiking populations on one bridge; this is the PLAN's case,
not k-select's. No STOP.

### The current latch read (the thing G4 failed on)

- `research/runners/bg_action_selection_production_organ.py:497-500`, `latch_read(rate_thr)`: rate of each
  `latch_k` over the last `READ_WINDOW` = 100 ms (line 368) of the 500 ms trial; a channel is active iff its rate
  `>= rate_thr`; exactly one active -> that channel, else None.
- Threshold: `LATCH_READ_RATE = 0.02` spikes/neuron/ms (line 369), applied at line 517; the G4 variants at 519-520.
- Why it failed (gate shards `research/findings/raw/_bg_live_seam/gate/s*/seam_b.json`): every G4 change is
  `hsb_q` ("what does the dog chase"), SPEAK latch rate 0.039 on all six seeds, so the x2 read (0.04) drops it to
  None: 6/24 = 0.25. `hsb_hi` sits at 0.002-0.004 on three seeds (None at every read level). The old latch holds a
  graded rate after the drive ends, so its read is a rate threshold, not a persistence read.

### Found while reading (reported; not changed on the flag-off path)

- `webapp/server.py:5685` calls `decide_action_neural(msg)` with the default organ seed 42 on every chat seed. In the
  10-02 gate the first two turns read identical latch rates on all six seeds (0.0993 / 0.039): the six "seeds" share
  one organ build and differ only through later RNG state. Under the new flag the organ seed follows
  `BRAIN_CHAT_SEED` (declared in the amendment). With the flag off nothing changes. This also matters for dev work:
  a live dev run with the default seed would build the organ at gate seed 42.
- Lever count against this defect: the 09-29 latch (lever 1, NO-GO on G4) and this commit stage (lever 2). The commit
  stage is the PLAN section 2 research output (navigation commit burst + GNW rung-1 latch + the G4 lesson), not a
  fresh guess; if it fails G4 on dev seeds the research gate fires before a third lever.

### Design (steps 2-3)

- `research/runners/spiking_commit.py`: the one copy of the stage. One topology spec (per key k: source -> G_k
  `W_RG` [commit gate], G_k -> G_k `W_GG`, G_k -> GI_k `W_GGI`, GI_k -| G_j for j != k `W_GIG` [exclusion gate],
  G_k -> S_k `W_GS`, S_k -> S_k `W_SS` [latch gate]) with two emitters: dense explicit-wiring entries (A1's
  `inject_explicit_wiring` build) and BrainRegion/RegionPathway parts (region-built bridges such as the BG selector),
  plus the pure latch read (`read`) and its x0.5 / x2 re-read. A1 imports it; a byte-identity test pins A1's battery.
- BG: sources are `motor_k` (the selector's output pools). The stage is appended to the neural-seam build on the same
  bridge; the between-trial `selector_reset` also reaches G and S (candidate-agnostic). The trial schedule is
  unchanged (drive 300 ms, zero-stimulus hold 200 ms); the commit read is the last `READ_STEPS` of the hold. The old
  latch stays built and its read is recorded from the same trial (reported, never used).
- Flag `BRAIN_BG_COMMIT` (default OFF, read only under `BRAIN_BG_NEURAL_SEAM=1`), row `BRAIN_BG_COMMIT_OP`, lesions
  `BRAIN_BG_COMMIT_LESION` in {commit, latch, excl}, each a transmission gate on the organ's bridge.

## 2026-10-14: STEP 2. One copy of the stage, A1 moved onto it

`research/runners/spiking_commit.py` already existed (idle-replay organ, C5, 2026-10-06); extended, not replaced:
one topology (`projections`), two emitters (`commit_plan`, unchanged output for K = 1..4 against the old body;
`region_parts` for region-built bridges, same edge set, pinned by a test), and `sequence_read` (A1's incremental read).
A1 (`word_role_parser_organ.py`) now builds and reads through it: `commit_battery(1, 0)` hashes to
`6e0bd58e...` before and after (`tests/test_bg_commit.py::test_a1_commit_battery_byte_identical`);
`tests/test_word_role_commit.py` and `tests/test_idle_replay_organ.py` pass. Declared remaining copy: the
emotion-category organ still wires the stage inline. Side effect: the A1 and idle-replay bar-sanity records name
scorer files whose sha256 moved (behaviour unchanged); see the end of this worklog.

## 2026-10-14: STEP 3. The stage on the BG channels (dev seeds 1-8 only)

Region `bg-commit` of the organ: sources `motor_k`; stage appended after every neural-seam region and pathway; the
selector's reset reaches G and S; after the build's warmup the same reset + washout runs once (dev seed 1, row 0: the
warmup's thalamic transient had ignited BOTH latches before the first trial, carry 19 / 16 spikes). Flag-off: the
neural-seam trial records on dev seed 1 hash to `4b54f5c4...` before and after (pinned).

**What the first probe showed (row 0 = A1's row 0, latches coexist).** The read is clean (latched 65-104 spikes in
the 30 ms window, unlatched 0-1), but the BG race can switch channel inside the 300 ms drive (an early burst of one
motor pool, then the other), so both latches turn on: 96 co-commits in 320 trials, empty-turn hold 77/112. A1's
latches coexist by design (several roles at once); a BG trial is one decision. Rows 1-4 add a single-slot exclusion
to the shared stage (default absent, so A1 and idle-replay are unchanged): `W_SGI` (a latch keeps its own GI pool on:
the first commitment keeps the decision) or `W_GIS` (a bursting gate's GI silences the other latch: a later
commitment replaces it). Checked against the hypothesis that the post-drive schedule decides empty turns: the held
STAY-SILENT gate fired during the drive on 96 of 98 held empty trials (row 4), so the withdrawal time is not the
decider.

**Sweep** (`research/runners/_bg_commit_derisk.py`, raw `research/findings/raw/_bg_commit/dev_organ/`; per row and
seed: the live four-turn sequence + 12 trials per class on one warm organ, and 12 trials per lesion organ):

| row | change | read invariant | latched min / unlatched max | empty held | content held | co-commit | agree old read | seeds all_pass |
|---|---|---|---|---|---|---|---|---|
| 0 | A1 row 0 | 320/320 | 65 / 1 | 77/112 | 0/104 | 96 | 0.738 | 3/8 |
| 1 | W_SGI 60 | 320/320 | 65 / 1 | 94/112 | 8/104 | 7 | 0.778 | 2/8 |
| 2 | W_SGI 60, W_XG 30 | 320/320 | 67 / 1 | 95/112 | 8/104 | 6 | 0.781 | 2/8 |
| 3 | W_GIS 60 | 320/320 | 65 / 1 | 98/112 | 7/104 | 0 | 0.956 | 1/8 |
| 4 | W_GIS 60, W_XG 30 | 320/320 | 67 / 1 | 98/112 | 7/104 | 0 | 0.963 | 1/8 |

The old latch read on the same trials: empty 99/112, content 6/104. Commit lesion and latch lesion: 0 commits in
every row (the commit lesion silences every gate while the motor pools still fire); exclusion lesion: 64/96 trials
commit. The read is threshold-invariant on every trial of every row: the G4 lesson holds at organ level.

**Selection.** The rule written before the sweep (first row with all_pass on all eight seeds) selected NONE. Its
capability checks (C2 empty held >= 0.75, C3 no content trial held, C6 the live sequence) test the race, not the read:
the old latch read on the SAME trials fails them on 5 of 8 seeds (content held on 1-3 trials of 13 on seeds 1, 6, 7,
8). A bar that the replaced read also fails cannot choose between reads (the bar-sanity lesson, here on a dev rule
that was never bar-sanity checked). Declared post-data rule (dev data only, no gate seed run, recorded in
`selection.json` and the amendment): among rows whose stage checks (C1 invariance, C4 lesions, C5 membrane and no
carry-over) hold on all eight seeds, the one whose committed action agrees most often with the old latch read on the
same trials: it retires the read without changing what the seam decides. All five rows pass the stage checks; row 4
(0.963; row 3 0.956) is selected. Rows 3 and 4 differ from the old read by one trial in each capability count. On the
live sequence the question turn is SPEAK on 8/8 dev seeds in rows 1-4.

**Value lesion with the stage on (row 4, reported, G3 preview):** commits continue (0.5-1.0 per class) but the held
rate no longer follows the class (empty 0.25-0.75, content 0.0-0.75): the class information reaches the commit only
through the value synapses.

**Live dev canary** (seed 7, all seven arms, `--commit`): see the next entry.
