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
