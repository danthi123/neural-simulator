# Burndown #1 CLOSED — the production OneBrainComposer's cleanup SELECTION is now a fully-on-substrate spiking WTA (2026-06-20)

**Verdict: GO (CPU-validated; GPU CI-guard confirm in flight). The host `np.argmax`-over-membrane word-selection in the
SHIPPED `OneBrainComposer` is RETIRED — replaced by the validated spiking Izhikevich WTA, ANSWER-IDENTICAL to the host
argmax, moat 0 false-accepts, default-OFF byte-identical, NO `sim/` edit.**

## The shortcut (burndown #1, the single most-surprising one)

`research/runners/one_brain_composer.py` picked every recalled word with a host `np.argmax` over the rectified
matched-filter membrane (`self.words[int(np.argmax(scores[ri]))]`), at three read sites:
- `_read_block` (the per-block oracle read) — old lines 308/310
- `_decode_batched_mem` (the default batched read) — old lines 427/429
- `_decode_clause` (the recursive embedded-clause decode) — old line 489

The matched FILTER was already on-substrate (the complex-synapse `clean` matvec into the co-resident bridge's RF
membrane), but the WINNER-PICK — a COGNITIVE selection (which concept = the recalled word) — was host numpy. The
validated spiking-WTA cleanup (`RFPhasorComposer._spiking_cleanup`) was opt-in only on the legacy `rf` path and was
NEVER wired into the production `onebrain` default. CLAUDE.md's "cleanup CLEARED" clearance lived on the `rf` path only.
(`research/findings/2026-06-20-shortcut-burndown-inventory.md` #1.)

## The conversion (reuse-by-import; NO sim/ edit)

Added `OneBrainComposer(enable_spiking_cleanup=False)` (default OFF) + two methods:
- `_spiking_select(scores, words)` — the validated **NEF spiking WTA Stage-2** (Stewart-Tang-Eliasmith): input-normalize
  the rectified scores -> drive a cached Izhikevich concept bank (REUSED from the inner `RFPhasorComposer._izh_bank`,
  keyed by candidate count, same seed) -> integrate firing over the cleanup window -> **winner = argmax-over-FIRING** (a
  readout of the spiking competition = the body-read of which neuron won, NOT a host argmax over the membrane).
  Off-target concepts get ZERO normalized drive (rectified) so they stay silent -> a clean WTA. Degenerate fallbacks
  (zero peak / zero firing) read the argmax of the same non-negative scores = the same value the host path returns, so a
  silent competition never confabulates.
- `_select(scores, words)` — the single dispatch the three read sites share: spiking when `enable_spiking_cleanup`, else
  the byte-identical host argmax.

All three host-argmax sites now call `_select`. The matched FILTER (already on the bridge) is unchanged; only the
SELECTION moved from host argmax to the spiking WTA. The inner `RFPhasorComposer` is reused by import (its `_izh_bank` +
`_cleanup_drive_pA` + `_cleanup_window`) — nothing duplicated, NO `sim/` edit.

## Why the no-confab moat is preserved BY CONSTRUCTION (0 false-accepts)

- The cue-match abstention (`query_*`/`ask_yes_no`/`render_fact` `for/if` over decoded words) and the `confidence_gate`
  margin both read the SAME rectified `scores`. The WTA changes only WHICH winner is read from those scores.
- Piece A is validated == numpy argmax multi-seed (`2026-06-05-composer-cleanup-NEF-GO.md`), so the spiking WTA picks the
  same winner the host argmax did -> every abstention decision is unchanged -> the moat holds.
- Confirmed empirically: the parity test asserts `query_patient/query_agent/ask_yes_no` abstain on unstored cues/facts
  on the spiking path (0 false-accepts).

## Validation (CPU-first, SIM_BACKEND=numpy — the spiking WTA runs on both backends)

New CI guard `tests/test_onebrain_spiking_cleanup.py` — **4/4 PASS on CPU** (re-exercised on GPU):
1. `..._parity_and_moat` — spiking-cleanup OneBrain == host-argmax OneBrain == truth on the who/what/yes-no matrix; the
   no-confab moat abstains on unstored cue/fact (0 false-accepts) on the spiking path.
2. `..._per_block_and_batched` — the spiking WTA selection holds on BOTH read paths (batched default + per-block oracle).
3. `..._clause_parity` — the recursive embedded-clause decode (`_decode_clause`, site :489) selects inner words with the
   spiking WTA, == host == "cat look south" / "dog go cat look south".
4. `..._default_off_byte_identical` — default `enable_spiking_cleanup=False` keeps the host-argmax path.

End-to-end through the agent (CPU): `BrainConversationalAgent(composer_kind="onebrain", enable_spiking_cleanup=True)` —
`what_does`/`who_does`/`is_it_true` correct, unstored cue abstains. The cleanup selection is in spikes.

The standalone `rf` composer `_spiking_cleanup` == numpy parity was re-confirmed on CPU too (the mechanism this reuses).

**GPU CI-guard confirm (in flight / for the controller):** the existing `tests/test_one_brain_composer_agent.py`
(11 tests, GPU-only) must stay green — these construct the agent with the DEFAULT (host-argmax) path, so they validate
the default-OFF byte-identity end-to-end on GPU. (First run reported exit 0 before the summary flushed; a clean re-run is
in progress — see the command below.)

## Latency cost (the reason the library default stays OFF)

CPU, D=64, 3 facts: host-argmax 58 ms/query vs spiking-WTA 95 ms/query = **1.63x** (the 120-step Izhikevich WTA window
per role per block). Bounded but not free -> the library `OneBrainComposer`/`BrainConversationalAgent` constructors keep
`enable_spiking_cleanup=False` for the numpy-CPU + test-oracle path, exactly mirroring the rf->onebrain default pattern.

## "Make it the default" — the production demo flips ON, the library keeps the escape

Per the established project pattern (the flagship demo flips the default; the library constructors keep the conservative
default for numpy-CPU portability + the test oracle):
- `consolidated_320_conversation_demo.py` (the flagship 320-scale production conversation): **default `spiking_cleanup=True`**
  for the `onebrain` path (a `--no-spiking-cleanup` escape), so the WHOLE conversational turn (parse -> bind -> store ->
  unbind -> **select** -> abstain) is brain-based on one bridge. The `rf` composer (the oracle / numpy-CPU path) stays its
  host-argmax default.
- `BrainConversationalAgent` threads its existing `enable_spiking_cleanup` flag to BOTH composer substrates now (it
  already did for `rf`; now also for `onebrain`). Default OFF = byte-identical.

## Honest scope

- The SELECTION is now in spikes; the matched FILTER was already on-substrate. `argmax-over-firing` is a body-read of the
  spiking WTA result (legit), NOT a host computation of the selection — confirmed: the competition is the Izhikevich WTA.
- This closes burndown #1 (#2 is the same op on the `rf` composer — already opt-in there; the `rf` path is the explicit
  numpy oracle). The composer's exact-inverse FHRR bind algebra (#12) remains the separate deep frontier.

## Files

- `research/runners/one_brain_composer.py` — `enable_spiking_cleanup` param + `_spiking_select`/`_select`; 3 sites -> `_select`.
- `research/runners/brain_conversational_agent.py` — thread `enable_spiking_cleanup` to the onebrain composer.
- `research/runners/consolidated_320_conversation_demo.py` — `--no-spiking-cleanup` (default ON for onebrain).
- `tests/test_onebrain_spiking_cleanup.py` — the new 4-test CI guard.

## Reproduce

```bash
# CPU (numpy): the new guard (parity + moat + per-block/batched + clause + default-off)
SIM_BACKEND=numpy python -m pytest tests/test_onebrain_spiking_cleanup.py -q     # 4/4

# GPU: the existing OneBrain CI guard (default-OFF byte-identity, 11 tests)
SIM_BACKEND=cupy python -m pytest tests/test_one_brain_composer_agent.py -q      # must stay 11/11

# GPU: the flagship 320-scale production conversation, now spiking-cleanup default-on
SIM_BACKEND=cupy python -m research.runners.consolidated_320_conversation_demo --seeds 42 43 44 --composer onebrain
```
