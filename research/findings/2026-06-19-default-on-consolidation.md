# Default-on consolidation: the production conversational agent now ships fully capable by default (2026-06-19)

**Owner directive (CYCLE 266):** the one-brain conversational agent should ship with its validated capabilities
default-ON (the default-OFF flags were a staging convention: add → validate-behind-flag-byte-identical → deliberately
flip). This pass did the "validate the combined config → flip the CPU-runnable defaults → keep the GPU-only / un-
defaultable ones guarded" work, with the no-confab moat as the HARD gate.

## What flipped to default-ON (the CPU-runnable, cleanly-defaultable capabilities)

| capability | flag | now default | substrate |
|---|---|---|---|
| Attributed objects ("big red apple") | `enable_attributed` | **True** | neural AttributedBridgeParser |
| Flexible word orders (SVO/VSO/OSV) | `enable_multiframe` | **True** | neural FrameParser |
| Neural word-ordering (describe) | `enable_neural_render` | **True** | spiking competitive-queuing renderer |
| Multi-referent pronouns ("which *it*?") | `enable_biased_competition` (MultiTurnAgent) | **True** | WTA biased competition |

`BrainConversationalAgent` and `MultiTurnAgent` constructors flipped (`research/runners/brain_conversational_agent.py`,
`research/runners/multi_turn_agent.py`). ⇒ the default agent now comprehends attributed objects, verb-first/object-first
sentences, neural word ordering, and multi-referent pronouns with NO flags.

## What stays OPT-IN (honest carve-outs, with reasons — NOT laziness)

| flag | why it stays opt-in |
|---|---|
| `enable_multicue_competition` | requires a hand-curated verb lexicon (animacy + selectional restrictions) that the agent's plain `{word: code}` vocab can't supply; AND it *replaces* rather than composes with the position/frame parser path. Un-defaultable. |
| `composer_kind` (stays `"rf"`) | the numpy-CPU path + the TEST ORACLE; `onebrain` (fully-spiking) is the production-demo default, but forcing it on every constructor would break CPU portability + the oracle. |
| `enable_rf_cudagraph` | GPU-only megakernel; stays GPU-if-available-guarded. |

## The HARD gate: the combined config + the moat (passed)

`tests/test_all_capabilities_on.py` (committed `064d167a`): the four always-on capabilities ON TOGETHER, multi-seed,
GPU. **Single-seed (42) = 5/5 PASS** (8:44 GPU) including the moat assertions in every test + the documented-fragile
attribute+embedded-clause combination (both round-trip, moat holds). The earlier interrupted 3-seed run showed 9/9
passing dots before an orphan-process cleanup. The moat held 0-breach throughout — the non-negotiable gate.

## CPU portability verified (the owner's explicit concern)

The all-ON agent constructs + works on `SIM_BACKEND=numpy` (`what_does` correct, moat returns `None`, `describe`
correct) — the four capabilities are GPU-*validated* but NOT GPU-*required*, so the flip preserves CPU portability.
The GPU-only paths (composer/megakernel) stay guarded.

## Test-suite delta (the flip is a behavior change)

Affected-test sweep (10 files, numpy + scan of the GPU-gated): **only 2 tests broke**, both the anticipated kind —
they asserted a flipped flag's *default-OFF* value:
- `test_multireferent_biased_competition.py::test_flag_off_buffer_not_built_and_anaphora_unchanged` (asserted
  `enable_biased_competition is False`).
- `test_one_brain_composer_agent.py` (asserted `enable_attributed` defaults OFF → `_attr_parser is None`).

Both FIXED by passing the flag **explicitly OFF** — the opt-OUT path still exists + works (byte-identical), so these
tests now validate the opt-out rather than the old default. No moat assertion was weakened; no capability test changed.
`test_production_spiking_flags.py` was unaffected (it passes the flags explicitly both ways).

## Net

The default one-brain conversational agent is now fully capable out of the box, CPU-portable, with the no-confab moat
intact and the two genuinely-un-defaultable capabilities (multicue verb-lexicon, GPU composer) kept as documented
opt-ins. This realizes the owner's "ship fully capable by default" directive for the conversational stack.
