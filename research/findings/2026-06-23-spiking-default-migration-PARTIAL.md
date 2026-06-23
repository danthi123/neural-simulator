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
