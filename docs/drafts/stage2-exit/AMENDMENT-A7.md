# Stage-2 exit gate, Amendment A7 (DRAFT, for Claude to sign): the retake re-pin

status: DRAFT (build agent, 2026-11-01; branch research/stage2-retake-amendments). Claude writes `GATE_SHA` on signing.
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
  main.
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
| `BRAIN_CURIOSITY_PHASIC_DA` (P) | OFF -> **ON** | same | **C3:** P's bus dopamine replaces the host crave gain whenever L applies, so the handler writes no `curiosity_da` trace. The A4 (c) informational `da_lesion` arm therefore reads "NOT HELD, attempted lesion (persistence unverified): no curiosity_da trace" (dev preview, every seed) and its reads are withheld. Informational only; the verdict is unaffected. Claude may retire or re-point that arm (a `BRAIN_CURIOSITY_PHASIC_DA_LESION=1` arm would be the analogue). |
| `SIM_STABLE_INTEGRATION` | new, **OFF** | merge `f64b564bf` | none at defaults. RULING substrate-1027 (3): never set in a production or gate run (plain ON trips FLIP-HURTS-ANSWERS). The C5 question is the memo's (`C5-INTEGRATOR-DECISION-MEMO.md`). |
| `SIM_STABLE_RECAL` | new, **OFF** (refused without the above) | merge `34a4e5f75` | none at defaults; see the memo. |
| `SIM_PRESET_CENTRED_HET`, `SIM_TRAIT_CENTRED_HET` | new, **OFF** | merge `500d8986d` (and the 10-20 RS-centred fix) | none at defaults. |
| `CoreSimConfig.izh_membrane_substeps` | new, default **1** (byte-identical) | izh-substeps build (BG lane) | none at defaults. |
| `BRAIN_C2_COMPOSER_PATIENT_INHIB` (+ `_W_I2E`) | new, **OFF** | C2 composer lane (`_c2_patient_inhib.py`) | none at defaults (C2 step-0 entries only, never gate lines). |
| `BRAIN_C2_GATE_RESET` | new, **OFF** | `3ec75e204` (after 41d76bdbe, on S main) | none at defaults. |
| `BRAIN_CURIOSITY_TOPIC_STOP_LESION`, `BRAIN_CURIOSITY_PHASIC_DA_LESION`, `BRAIN_CURIOSITY_PHASIC_DA_CONST_SALIENCE`, `BRAIN_EMOTION_CATEGORY_COMMIT_LESION` | new lesion / control switches, **OFF** | the C3 and emotion lanes | none at defaults; no Stage-2 arm sets them. |

The emotion organ's always-on census moved on-device (`66ccdd924`, instrument refactor, no default change).

## Expected verdict at M (a prediction, not a bar)

- **C1:** FAIL, as on 10-26 (no C1 fix has landed; the preview reads 0/3 with the same named gaps).
- **C2:** at A6, UNDEFINED/INCOMPLETE while the D4 gate refuses `heron`/`gnat`; at A5, FAIL "not exposure-driven" as on
  10-26. Correctness, lesion and reproduction are 6/6 on 10-26 and 3/3 on the preview.
- **C3:** PASS (gate-seed topic-stop gate + preview 3/3).
- **C4:** PASS (emo-flip gate 6/6 + preview 3/3).
- **C5:** PASS at the default integrator (10-26 6/6 + preview 3/3), cited only with the substrate-1027 caveat; see the memo.
- **Stage:** at A6, UNDEFINED (the C2 precondition); at A5, FAIL (C1, C2).

## Consequences for the packet

- `PREREG_SHA` unchanged. `GATE_SHA` file: Claude writes M after the checks above, replacing `43763e637`.
- HANDOFF section 8's aggregate line gains `--amendment-level A6` if A6 is signed.
- The C3 `da_lesion` informational arm needs a ruling (retire, or re-point to `BRAIN_CURIOSITY_PHASIC_DA_LESION=1`). Doing
  nothing is safe: it is reported NOT HELD and never counted.
