# R1 close — fold the K-way sequencer + divnorm-score pools onto ONE fabric bridge, device-resident cleanup→score handoff (2026-06-30)

**Type:** build (research/runners/ only; opt-in; default byte-identical). Closes R1 from
`research/findings/2026-06-30-tier2-integrated-spiking-loop-scoping.md` (Option 1).

## The residual (R1, precisely)

In the deployed `OneBrainComposer(integrated_loop=True)` query path, the cue-match CONTROL is on-substrate
(the spiking K-way sequencer + the on-bridge `input_divisive_norm` per-query normalization). BUT the cleanup
membrane (the OP RESULT, an RF region on `OneBrainComposer.b`) is read TO HOST and the host array is re-driven
onto a SEPARATE divnorm-score `SimulationBridge`:

- `block_cleanup_scores` (`_phaseB_onebrain_sequencer_derisk.py:86`): `mem = to_host(b.cp_membrane_potential_v)`
  → returns `(agent_scores, action_scores)` host vectors.
- `onbridge_divnorm_drive` (`_phaseC_S5_divnorm_derisk.py:166-175`): builds a host `cur` from those scores,
  `score_sb.cp_external_input_current[:] = from_host(cur)` → drives the SEPARATE score bridge.

That `to_host` of the cleanup score + the cross-bridge re-drive is **the last host DATA seam in the integrated
who/what query path**. It is NOT a host COMPUTATION (the normalization + match + select are all spiking) — it is a
host DATA TRANSFER between co-resident-in-principle Izhikevich fabric slices that are currently separate bridge
objects, with the cleanup score crossing host to get there.

## The close (the feasible, reuse-by-import mechanism)

A literal "everything (parser + RF + score-pool + sequencer) on `OneBrainComposer.b` with one `_run_one_simulation_step`
loop" is **infeasible without major `sim/` surgery** and is NOT what R1 requires: `b` is an `inject_explicit_wiring`
RF bridge (`region_manager=None`, so `couple_gate_to_pool` raises; v/u hold complex phasors that one Izhikevich step
destroys — the documented 5b KILL). The RF ops (`rf_resonate_steps`) deliberately bypass `_run_one_simulation_step`.

R1 is specifically **the cleanup-score VECTOR marshalled host-side**. The close, in two parts:

1. **Fold** the divnorm-score pool + the K-way sequencer into ONE Izhikevich "fabric" bridge `b_seq` (today two
   separate `SimulationBridge` objects). Reuse the existing validated builders' region/pathway specs, merged onto one
   `enable_brain_region_framework` bridge (disjoint region-name namespaces: `score_*` for the score pool, the
   sequencer's `cueA_*/cueX_*/d{b}{role}_*/mw.../m{b}/ans{b}/abstain/inh{b}` for the cascade). The score-pool→sequencer
   hand-off is then intra-bridge.

2. **Device-resident cleanup→score handoff** (the load-bearing R1 close): the cleanup membrane lives on
   `OneBrainComposer.b` (a cupy array). Instead of `to_host` → host vector → `from_host` onto a separate bridge, the
   cleanup score is **gathered + scattered DEVICE-SIDE** into `b_seq`'s score-pool `cp_external_input_current` (cupy →
   cupy, same GPU, NO `to_host` of the cleanup score). The divnorm divide then runs on `b_seq` exactly as before; a
   placed rheobase threshold reads which words fire (the per-block decoded-line drive); that drives the co-located
   sequencer. The READBACKS that remain — `to_host(b_seq.cp_firing_states)` for the score-pool firing and the
   sequencer match pools — are the legitimate **body read** ("which neuron fired"), the S7 boundary (like the nav
   cascade reading the winning motor pool), NOT the cleanup-score data seam. They are NOT R1.

### Per-role discipline preserved

The divnorm divisor is the mean over the flagged set. The score pool processes ONE role at a time (agent, then
action) through the shared V-pool so the divisor is that role's own per-query total — preserved verbatim in the
fused path (the device scatter drives one role's V words, settles, reads firing, then the next role), so the
on-bridge normalization is byte-faithful to the validated S5 op-point.

## API surface (opt-in, default byte-identical)

- New constructor arg `integrated_loop` accepts `False` (host `_scan` oracle — unchanged), `True` (the existing
  separate-bridge spiking path — unchanged, the revertible escape), and the NEW value `"fused"` (the folded,
  device-resident path). `bool(integrated_loop)` still selects the spiking branch in `_seq_block`; a small helper
  `self._fused = (integrated_loop == "fused")` selects the fused fabric inside `_ensure_sequencer`/`_seq_block`.
- A new runner module `research/runners/_seq_fused_fabric.py` holds: `build_fused_fabric_bridge(seed, V, K, ...)`
  (the merged score-pool + K-way sequencer on one bridge), the device-resident `fused_block_drives(...)` (the
  cleanup→score handoff with NO `to_host` of the cleanup score), and `run_fused_sequencer(...)` (drive the
  co-located sequencer from the device-resident drives). It reuses the existing builders' region/pathway helpers by
  import where possible; the genuinely new code is the merge + the device handoff.
- `_seq_imports()` is extended to also return the fused fabric fns (lazy, so the OFF path never imports them).
- `_ensure_sequencer`/`_seq_block` get a `self._fused` branch that builds + runs the fused fabric; the
  `integrated_loop=True` (separate-bridge) and `False` (host) branches are byte-unchanged.

## The contingent `sim/` edit (FLAGGED for byte-review; preferred NONE)

The fused fabric is an Izhikevich bridge stepped by `_run_one_simulation_step` (its own step path) — it does NOT use
the RF megakernel or the masked `rf_kick`, so the design §3.2 contingent tracker-mask edit is **not anticipated**.
The cleanup membrane read is the SAME `block_cleanup_scores` RF op (already masked to `c.rf_mask`), just keeping the
result on-device. IF a multi-op register break surfaces at K=8 (the design's central risk — a re-kick clobbers a
holding RF group), the contingent fix is the design §3.2 edit 1 (~6 lines, mask the `rf_kick` spike-tracker re-init),
default `None` = byte-identical, isolated commit FLAGGED for the controller's byte-review. Prefer NO `sim/` edit.

## GO bar / anti-cheats (the de-risk, in `_seq_fused_fabric.py` main)

- **==HOST:** the fused path's per-query decision (`query_patient`) == the host `_scan` oracle (`integrated_loop=False`),
  K ∈ {2,4,8}, 6 seeds. (CPU smoke uses a toy K subset; the controller runs the GPU 6-seed.)
- **MOAT 0-FA (HARD):** an unstored/absent/cross cue still abstains, every seed. A single false-accept = FAIL.
- **ZERO `to_host` of the cleanup score** in the cleanup→sequencer HAND-OFF (R1, precisely). Asserted by a
  monkeypatch counter on `sim.backend.to_host` scoped to `_seq_block` (the S4 cleanup → S5 score → S6 select path R1
  lives in): the composer RF bridge's `cp_membrane_potential_v` (the cleanup-score carrier) is NEVER read to host
  during the hand-off (it stays device-resident from the RF cleanup to the score-pool drive). The remaining `to_host`
  inside the hand-off (firing-state body reads on the fabric bridge — "which score-pool word fired") is the placed
  rheobase body-read, not the cleanup score.
  NOTE (R5, NOT R1): `query_patient`'s SEPARATE downstream `got = self._read_blocks()[idx]` re-decodes the SELECTED
  block to EMIT its patient word — that read happens AFTER the sequencer chose the block (S7, the answer body-read), is
  the SAME read every composer path does, and is the documented legitimate "which concept-neuron won" boundary
  (scoping R5, "effectively closed" under `enable_spiking_cleanup`). It is excluded from the R1 assert (the assert
  scopes to the cleanup→sequencer hand-off, not the whole query).
- **lesion fails safe** (sever the cleanup→score drive → decoded lines silent → abstain, never confabulate).
- **OFF == byte-identical:** `integrated_loop=True` (separate bridge) and `False` (host) produce identical answers
  to the pre-change code (the fused path is purely additive).

## Deliverable

1. The build (`research/runners/_seq_fused_fabric.py` + the `one_brain_composer.py` `integrated_loop="fused"` wiring).
2. A CPU smoke (`_seq_fused_fabric.py` main, `SIM_BACKEND=numpy`, toy K) proving: fused == host (==host), moat 0-FA,
   the `to_host`-of-cleanup-score-eliminated assert, lesion-fails-safe, OFF==byte-identical. (On numpy "device-resident"
   is a passthrough, but the assert that `block_cleanup_scores`'s host read is GONE from the fused path holds on both
   backends — it is a code-path property, not a backend property.)
3. Report: file:line of changes, the CPU-smoke result, the cleanup-score `to_host`-gone assert, the EXACT GPU 6-seed
   command (K{2,4,8}, seeds 42 43 44 100 101 102), the commit SHA, any `sim/` edit flagged.

## Hard rules

research/runners/ only unless the contingent register-mask sim/ edit is strictly needed (then ~6 lines, FLAGGED).
Strict failing-test → minimal-impl → CPU-smoke. Commit on `main`, STRICT git add (pre-existing modified findings
JSONs NOT staged). NEVER weaken the no-confab moat. Do NOT run the full GPU 6-seed.
