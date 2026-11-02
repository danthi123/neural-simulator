# Stage-2 exit gate, Amendment A7: the retake re-pin

status: SIGNED (Claude, 2026-11-01). Drafted by a build agent on branch research/stage2-retake-amendments. `GATE_SHA` = M is
written in a commit after the merge, once the A7.1 checks hold.
Ruling (Claude, 2026-11-01): criterion-only variant.
host-decision: none a re-pin moves the gate revision only; it adds no host code and changes no bar, arm, turn or scorer rule
bar-sanity: N/A a re-pin changes no bar, arm, turn, seed or scorer; the scorer's own records are AMENDMENT-A6.md's

Amends `research/findings/2026-09-30-stage2-exit-gate-PREREGISTRATION.md` (byte-identical; `PREREG_SHA` stays
`24dac499e`) for the second attempt (window 2026-11-03..11-10 per HANDOFF; due by 11-21, RULING stage2-next-1026).

## A7.1 The pin

- **What the first attempt actually ran at.** Every 10-26 gate shard's sidecar reads `git_sha` `9bdb92ee8`. The
  repo's `docs/drafts/stage2-exit/GATE_SHA` file still reads `43763e637`, which Amendment A4 declared void. That pin
  never ran a gate line. This amendment sets the record straight: the first attempt's pin is `9bdb92ee8`.
- **Content baseline for the retake:** main `41d76bdbe` (2026-11-01, merge of research/c3-flip-default-on). It is the
  first main revision with both Stage-2 fixes ON by default: the emotion organ's bounded conductance (merge `0b79336ec`)
  and the C3 topic stop + topic-locked phasic DA (flip commit `7f024a1c1`). It is already provisioned on the pool,
  where the 11-01 dev preview ran.
- **`GATE_SHA` = M**, the main commit that merges this branch (A6 + A7 + the runner + its tests), chosen by Claude.
  Checks, all of which must hold before any gate line is emitted:

      git merge-base --is-ancestor 41d76bdbe $M
      git cat-file -e $M:docs/drafts/stage2-exit/AMENDMENT-A6.md && git cat-file -e $M:docs/drafts/stage2-exit/AMENDMENT-A7.md
      git diff --stat 41d76bdbe $M -- sim webapp research/runners ':!research/runners/stage2_exit_gate.py'

  The last diff must list only changes this amendment names below (default OFF, or flipped with their own gate). At
  S main `493d173ae` it lists one file, `research/runners/_phaseB_onebrain_sequencerK_derisk.py` (`3ec75e204`,
  `BRAIN_C2_GATE_RESET`, default OFF). Any new default flip merged before M needs a line here, or a pin at an earlier
  main. Merged since then, all default OFF and named here (Claude, 2026-11-01): `research/c2-drain-wta` (merge `1adc40b3c`;
  `BRAIN_C2_QUIET_DRAIN`, `BRAIN_C2_ROLE_WTA`, `BRAIN_C2_ROLE_RELAY`, `BRAIN_C2_ROW_SHUFFLE` and their lesion knobs, all unset =
  the pre-change path, byte-identity pinned by `tests/test_c2_drain_wta.py`; it touches `one_brain_composer.py`, the sequencer
  runners, `_c2_patient_inhib.py`, `c2_hops_step0.py` and `stage2_exit_gate.py`'s C2 read only through those flags).
- Pool lines: unchanged in number and shape (210 with informational arms, 180 with `--no-informational`;
  `tests/test_stage2_exit_gate_a6.py` pins the bytes). Emit with `--emit-pool-jobs $M`. A provisioning request for M is needed (41d76bdbe's is not M's).

## A7.2 Every default that changed between the first attempt's pin (9bdb92ee8) and 41d76bdbe

Method: the env switches the production handler or its organs read, grep'd from `git diff 9bdb92ee8 41d76bdbe`
(webapp/, sim/, research/runners/ organs; 497 first-parent commits), checked against the merge notes. As a cross-check,
the c3-flip gate's flag-OFF arms (`research/findings/raw/_c3_flip/gate_off/`, pin `15db27a73` = after the emotion flip,
C3 flags off) were compared reply by reply with the 10-26 gate's intact_a on the six gate seeds (conv, infer, emo, inner).
The only primary-field change in 24 cells is s101 `feel_joy`, which now names joy; every scorer `all` is unchanged
except s101 emo, FAIL to PASS. The same comparison for the flag-ON arms (= the 41d76bdbe defaults) adds only the four
follow-up-question turns the c3-flip RESULT lists (infer s44 `c_prey_chase`, emo s100 `s_sadness_2`, s102 `s_joy_2`,
`s_fear_2`), with every scorer `all` unchanged.

| switch | change | where | expected effect on C1-C5 |
|---|---|---|---|
| `BRAIN_EMOTION_CATEGORY_BOUNDED` | OFF -> **ON** (`=0` escapes) | merge `0b79336ec` (RULING emo-flip-1028) | **C4:** the s101 joy miss was the 1 ms Euler overshoot in one 53.6 pF neuron; with bounded conductance the flip gate read C4 correct 6/6, lesion_affect changes the named category 6/6 and reproduces, permuted tracks 6/6 (emo-flip gate B1-B3). Expected C4 PASS. No other group: the flip gate's B6 (240 non-feel whole-record compares, 0 diffs) and the cross-check above. Every emo reply carries five engagement keys (expected leaf drift, never a reply change). |
| `BRAIN_CURIOSITY_TOPIC_STOP` (L) | OFF -> **ON** | `7f024a1c1`, merge `41d76bdbe` (RULING c3-flip-landed-1101) | **C3:** the per-topic stop. At 34a4e5f75 with L+P set, the C3 topic-stop gate passed every bar G1-G10 on 6/6 gate seeds (`research/findings/2026-10-29-c3-topic-stop-gate-RESULT.md`). The 11-01 dev preview at 41d76bdbe reads C3 PASS 3/3 (both orders, both lesions, permuted). Expected C3 PASS. `m_goat` stays UNDEFINED (`all_taught_moats_acquired` 0: the D4 gate refuses `the goat eats the hay`), which A4 (a) handles. C1/C2/C4/C5: only the follow-up question on abstained turns changes (c3-flip gate B1-B3 24/24); no scored clause reads it. |
| `BRAIN_CURIOSITY_PHASIC_DA` (P) | OFF -> **ON** | same | **C3:** P's bus dopamine replaces the host crave gain whenever L applies, so the handler writes no `curiosity_da` trace. The A4 (c) informational `da_lesion` arm therefore reads "NOT HELD, attempted lesion (persistence unverified): no curiosity_da trace" (dev preview, every seed) and its reads are withheld. Informational only; the verdict is unaffected. The arm is RETIRED at A7 (section A7.3). |
| `SIM_STABLE_INTEGRATION` | new, **OFF** | merge `f64b564bf` | none at defaults. RULING substrate-1027 (3): never set in a production or gate run (plain ON trips FLIP-HURTS-ANSWERS). The C5 question is the memo's (`C5-INTEGRATOR-DECISION-MEMO.md`). |
| `SIM_STABLE_RECAL` | new, **OFF** (refused without the above) | merge `34a4e5f75` | none at defaults; see the memo. |
| `SIM_PRESET_CENTRED_HET`, `SIM_TRAIT_CENTRED_HET` | new, **OFF** | merge `500d8986d` (and the 10-20 RS-centred fix) | none at defaults. |
| `CoreSimConfig.izh_membrane_substeps` | new, default **1** (byte-identical) | izh-substeps build (BG lane) | none at defaults. |
| `BRAIN_C2_COMPOSER_PATIENT_INHIB` (+ `_W_I2E`) | new, **OFF** | C2 composer lane (`_c2_patient_inhib.py`) | none at defaults (C2 step-0 entries only, never gate lines). |
| `BRAIN_C2_GATE_RESET` | new, **OFF** | `3ec75e204` (after 41d76bdbe, on S main) | none at defaults. |
| `BRAIN_CURIOSITY_TOPIC_STOP_LESION`, `BRAIN_CURIOSITY_PHASIC_DA_LESION`, `BRAIN_CURIOSITY_PHASIC_DA_CONST_SALIENCE`, `BRAIN_EMOTION_CATEGORY_COMMIT_LESION` | new lesion / control switches, **OFF** | the C3 and emotion lanes | none at defaults; no Stage-2 arm sets them. |

The emotion organ's always-on census moved on-device (`66ccdd924`, instrument refactor, no default change).

## A7.3 The A4 (c) `da_lesion` informational arm is RETIRED

Ruling (Claude, 2026-11-01). The A4 (c) arm set `BRAIN_CURIOSITY_DA_LESION=1`: the handler's DA/engagement crave-gain
coupling pinned to gain 1.0 (`webapp/da_curiosity_drives_chat.crave_decision`). Since `41d76bdbe`
`BRAIN_CURIOSITY_PHASIC_DA` is ON by default, and with L applied the handler skips `crave_decision`: the gain is taken
from the topic-stop organ's spiking SNc read (`research/runners/curiosity_topic_stop_organ.TopicStopOrgan.phasic_gain`,
through `da_crave_gain`), which never reads `BRAIN_CURIOSITY_DA_LESION`. So the flag reaches nothing on that path, the
arm writes no `curiosity_da` trace, and `da_lesion_held` reads NOT HELD on every preview seed. The arm measured a lesion
of a mechanism whose default changed; it was never counted.

No simple re-point keeps its original meaning. `BRAIN_CURIOSITY_PHASIC_DA_LESION=1` pins only the PHASIC afferent of the
SNc read (the tonic afferent intact, `webapp/curiosity_topic_stop_chat.py`), so the gain still varies with the session DA
level; it is not "the coupling pinned to gain 1.0". Pairing `BRAIN_CURIOSITY_DA_LESION=1` with
`BRAIN_CURIOSITY_PHASIC_DA=0` would restore the old path by switching off a default-ON mechanism, which lesions the
brain the gate does not run. Its persistence check (`da_lesion_held`, the `curiosity_da` trace) would also need a new
read of the organ's `phasic_da` trace. Either is a new arm with a new meaning, not a re-point.

What retirement means at A7: the arm's read is not reported as a lesion read and is not cited in the RESULT. The runner
still defines and emits the arm (its line is part of the 210 informational lines whose bytes
`tests/test_stage2_exit_gate_a6.py` pins); its shard is harmless and its `lesion_held` False keeps every read withheld.
Dropping it from `GROUPS["ask"]["informational"]` is a runner change that moves the pool-line pins and is not made here.
A C3 lesion of the coupling that P now owns, if one is wanted, is the C3 topic-stop gate's `lesion_P` arm
(`research/runners/c3_topic_stop_score.py`), already measured there.

## Expected verdict at M (a prediction, not a bar)

- **C1:** FAIL, as on 10-26 (no C1 fix has landed; the preview reads 0/3 with the same named gaps).
- **C2:** at A6, UNDEFINED/INCOMPLETE while the D4 gate refuses `heron`/`gnat`; at A5, FAIL "not exposure-driven" as on
  10-26. Correctness, lesion and reproduction are 6/6 on 10-26 and 3/3 on the preview.
- **C3:** PASS (gate-seed topic-stop gate + preview 3/3).
- **C4:** PASS (emo-flip gate 6/6 + preview 3/3).
- **C5:** PASS at the default integrator (10-26 6/6 + preview 3/3), cited only with the substrate-1027 caveat; see the memo.
- **Stage:** at A6, FAIL on C1 (criterion-only ruling 2026-11-01: an undefined C2 never hides a defined C1 failure;
  the C2 precondition is recorded unmet); at A5, FAIL (C1, C2).

## Consequences for the packet

- `PREREG_SHA` unchanged. `GATE_SHA` file: Claude writes M after the checks above, replacing `43763e637`.
- HANDOFF section 8's aggregate line gains `--amendment-level A6` if A6 is signed.
- The C3 `da_lesion` informational arm is retired (A7.3); its read is not cited.
