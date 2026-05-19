# Remote-regime gate run was INSTRUMENT-INVALID: the falsify-first probe did not honor `--remote-regime` (it ran the default distinct-pathways readouts). No science conclusion drawn; instrument fixed and re-run.

## Status

Honest propagation of an instrument-validity failure (the discipline
requires propagating every outcome, including instrument failures, not
only science results). No fifth/terminal scientific conclusion is drawn
from this run — drawing one from an instrument that did not probe the
intended question would be exactly the overclaim the project's
anti-cheat discipline forbids (instrument-validity-FIRST). The frozen
verdict modules, the corrected module, and the no-confabulation moat
are byte-unchanged; no partition edit; no configuration-cranking.

## What happened (GPU, CuPy / RTX 3090, durably captured)

The cheap gating falsify-first was run with `--remote-regime
--falsify-first --only-modes full no_cls_replay --only-load 2 --seeds
42 43 44`. The durable log shows the `--falsify-first` gate-probe code
path printed:

- "FALSIFY-FIRST (default) ... full ep (episodic ORDER via
  order-preserving ONLINE trisynaptic CA3->CA1 completion, taken
  PRE-consolidation) = 1.0000"
- "full wm (... order-invariant offline-consolidated schema ...) =
  0.5000"
- "distinct-pathways does NOT jointly satisfy wm+ep ..."

This is the DISTINCT-PATHWAYS readout configuration (episodic order
read from the online trisynaptic path BEFORE consolidation), not the
remote-regime configuration the design and plan pre-registered. The
remote-regime test's defining requirement (Design B) is that episodic
order is read from the CONSOLIDATED store AFTER the hippocampus is
strict-silenced — precisely to test whether systems consolidation
retains serial order. The `--falsify-first` gate-probe branch did not
route through the remote-regime post-consolidation
hippocampus-silenced readouts (the spine has the strict-silence
mechanism and the post-consolidation episodic readout per its own
design comments, but the falsify-first probe path used the default
distinct-pathways readouts and explicitly labelled itself
"(default)").

## Why this is an instrument-validity failure, not a finding

The reported `full ep = 1.0` is simply the already-known
distinct-pathways online pre-consolidation result; it says nothing
about the remote-regime question (does the consolidated, hippocampus-
silenced store retain serial order?). The strong complementary-
learning-systems prediction for the remote regime (consolidation
builds an order-invariant schema, so consolidated `ep` should NOT
clear the bar) was NOT tested by this run. Per the instrument-
validity-first principle that has governed this entire arc, an
instrument that does not soundly measure the intended question yields
no science conclusion. This run is therefore discarded as a science
result; only the instrument defect is the finding.

## Precise defect and the fix

Defect: the `--falsify-first` gate-probe code path is hard-wired to the
default/distinct-pathways readouts and does not compose with
`--remote-regime` (it printed "FALSIFY-FIRST (default)" and the
distinct-pathways readout description even though `_REMOTE_REGIME` was
true).

Fix (instrument construction, not a result-driven change; frozen
verdict/bars/partition/moat untouched): make the `--falsify-first`
gate-probe path, when `--remote-regime` is set, take BOTH the consolidated
working-memory readout AND the consolidated episodic-order readout
from the post-consolidation, hippocampus-strict-silenced consolidated
store (the existing `_hippo_strict_silence` mechanism + the
post-consolidation `_episodic_order_readout`, exactly as the design
comments specify), instead of the default distinct-pathways readouts.
Then re-run the cheap gate validly (full + no_cls_replay, N=2, 3 seeds,
durable output capture).

## What this does and does not change

- It does NOT change the durable, scale-confident scientific results:
  the thrice-convergent falsification-and-correction of the original
  necessity prediction (independently goalpost-move-cleared), the
  structural dissolution of the encode-order contradiction, and the
  fourth (recent-memory) structural characterization all stand.
- It does NOT yield a fifth/terminal finding — that remains to be
  settled by a VALID remote-regime gate run (the strong CLS prediction
  is still the most probable outcome, but it must be measured by a
  sound instrument, not assumed and not inferred from an invalid run).
- Honesty ceiling remains binding; conversational capability is not
  achieved and is not claimed; all previously-validated assets are
  intact and byte-unchanged.

## Files / evidence

- Durable invalid-run log: `research/findings/raw/integrated_loop_remote_gate.log`.
- Remote-regime design + plan (the pre-registered correct readout): `docs/plans/2026-05-19-remote-memory-regime-necessity-test-architecture-design.md` (`aa90dac`), `docs/plans/2026-05-19-remote-memory-regime-necessity-test-implementation.md` (`07ae035`).
- Frozen verdict (`2048750`), corrected module (`36a7975`), no-confab moat: byte-unchanged.
