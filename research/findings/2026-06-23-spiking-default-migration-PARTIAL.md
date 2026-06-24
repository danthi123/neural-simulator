# Fully-spiking-default conversational migration — PARTIAL: word-order ✓ + read-out-norm already-neural; moat/cleanup spiking-default REVERTED (the agent-level `integrated_loop` OVER-ABSTAINS at the small-vocab test config — moat INTACT, recall breaks) (2026-06-23)

**Owner chose "do the full migration" (the 3 conversational defaults → spiking). Honest outcome: 2 of 3 done, the
3rd reverted pending threshold tuning. The no-confab moat was INTACT throughout — the failures are OVER-abstention
(the SAFE direction), NOT a single false-accept breach.**

## The 3 flips
1. **WORD-ORDER (`enable_neural_render`): DONE + VERIFIED, committed.** All agent constructors default True
   (`multi_turn_agent_v2.py:60` flipped; `brain_conversational_agent.py:153` + `multi_turn_agent.py:48` already True).
   Verified clean by the numpy gate (`test_multi_turn_agent` + `test_multicue_competition_agent`, **11/11 passed**).
2. **READ-OUT NORMALIZATION: already the production default** — not a runtime flag (a cortex code-gen choice;
   `consolidated_320_conversation_demo.py:215` `--readout` defaults `neural`). Nothing to flip.
3. **MOAT/CLEANUP (`integrated_loop` + `enable_spiking_cleanup`): REVERTED.** The per-substrate sentinel
   (onebrain→spiking-default) made **6 agent-level onebrain tests FAIL** (`test_one_brain_composer_agent.py`): the
   spiking K-way sequencer (`integrated_loop`, `match_thresh=0.06` — calibrated for the production 320-vocab/D=128)
   OVER-ABSTAINS at the SMALL test vocab → `what_does` returns `None`/`'unknown'` when it should recall the stored
   fact (`assert None == 'north'`, `assert 'unknown' == 'yes'`, reason_chain `None`, clause `None`, multiturn abstain).
   Cupy gate: **6 failed, 42 passed (2h10m)** — all 6 are agent-level `integrated_loop` ABSTENTIONS.

## The MOAT is INTACT (the load-bearing check)
The failures are OVER-abstention (the agent says None/unknown when it should answer) — the SAFE direction. NOT a
single false-accept (a confabulation where it should abstain). The no-confab moat is preserved; the regression is
RECALL, not the moat. The parity guards (`test_onebrain_spiking_cleanup` + `test_onebrain_integrated_loop_fold`)
PASSED — the spiking ops are answer-identical IN THEIR validated config.

## Why reverted + the follow-on
The `integrated_loop` spiking-default is correct for PRODUCTION (the demo works) but the `match_thresh` isn't
calibrated for the small-vocab agent TESTS → over-abstain. Reverting the agent sentinel restores the host-oracle
default (known-good; the agent-level tests pass); the production demo stays spiking (its explicit flags).
**FOLLOW-ON: config-aware `sequencer_match_thresh` calibration** (per vocab-size / D) so the onebrain agent can
default-spiking without over-abstaining at small vocab → then re-run the agent gate. The cleanup-only variant
(`integrated_loop=False`, `enable_spiking_cleanup=True`) is a possible partial (untested in isolation; the failures
were ALL `integrated_loop` abstentions).

## Kept (verified/correct), committed
- Word-order flip (`multi_turn_agent_v2.py:60`) — numpy gate 11/11.
- The latent-test fix (`test_onebrain_spiking_cleanup.py`: pin explicit `enable_spiking_cleanup=False` — the
  composer's OWN default is `True`, so the off-path guard must pin off explicitly; passed in the 42).

## Net
The migration is honestly PARTIAL: 2 of 3 done (word-order shipped, read-out-norm already-neural at production), the
3rd (`integrated_loop` agent-default) reverted because it breaks small-vocab recall — a bounded threshold-calibration
follow-on, NOT a moat failure (the moat held: over-abstention, zero false-accepts).

## UPDATE (2026-06-24, BURNDOWN 1A re-dispatch — the agent-level flips banked; C-2 root-caused deeper than "threshold")
The "3rd" above was treated as a `match_thresh` calibration follow-on. The burndown 1A de-risk
(`research/findings/raw/_burndown_1A_c2_smallvocab_derisk.json`, 3 seeds) shows it is NOT a `match_thresh` re-cal: at
the SMALL test vocab (V=15, K=4, fresh random codes) the divnorm-WTA **agent-role** cleanup decode produces ZERO firing
on ≥half the blocks (the action-role decode is clean) — `present_ok` stays 0/4 (seed 42) or 2/4 (seeds 43/44) at EVERY
threshold {0.06,0.02,0.005,0.001} because the winner match-pool rate is structurally 0.000 (nothing to detect). A
divnorm gain/sigma sweep (gain {0.11→0.001} × sigma {1,100}) found NO operating point that cleanly isolates the agent
winner (gain≥0.11 → empty; gain≤0.03 → winner lights WITH runner-ups, `clean_exact` 0/4 everywhere). Root cause = the
LOW cleanup MARGIN of fresh random codes at small V (agent winner ~2.4× runner-up), vs the high-margin stream-learned
320 codes where `integrated_loop` is GO 4/4. The moat is INTACT throughout (worst_off_rate 0.000, `host_absent_all_none`
True). ⇒ **C-2 stays default-OFF at the library agent default (the host-`_scan` oracle for the small-vocab path) and ON
at the production demo** (a code-MARGIN boundary, not a fixable threshold).

What the burndown DID flip (runner-level, NO `sim/` edit, `research/runners/brain_conversational_agent.py`): the
agent-level `enable_spiking_cleanup` (C-3) + `enable_learned_assoc` (C-5) now default to a **None sentinel that
auto-resolves ON for the onebrain production path, OFF for the rf/rate test-oracle + numpy-CPU path** (byte-identical).
So a plain `BrainConversationalAgent(composer_kind="onebrain")` is fully-spiking-cleanup + learned-assoc by default
(== the production demo). C-4 (`local_reciprocal_unbind`) was already the OneBrainComposer default; H-7 (on-bridge
complex-synapse store) is on-by-design for the OneBrainComposer. Validated on GPU: `test_onebrain_agent_matrix_and_moat`
+ `test_onebrain_negation_yes_no` + `test_onebrain_describe_and_reason` + `test_onebrain_multiturn_correction` +
`test_onebrain_multiturn_anaphora` (who/what · describe · yes-no/negation · multi-turn) all PASS with the flips on,
**moat 0 false-accepts**, and the rf-path `test_comprehend_store_and_qa` + `test_negation_yes_no` still pass
(byte-identical, no regression). Finding: `research/findings/raw/_burndown_1A_conv_spiking_flips.json`.

Notably, the 2 multi-turn onebrain tests above were among the 6 this PARTIAL saw FAIL when `integrated_loop` was on —
they PASS now with `integrated_loop` OFF + cleanup ON, confirming the over-abstention was an `integrated_loop` (C-2)
issue, NOT a cleanup (C-3) issue.
