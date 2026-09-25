---
type: finding
status: live
claim_check: measured
date: 2026-09-25
lane: consumer-hardware-reference (SlotBinder teach latency) + scaffold-retirement (VSA composer -> learned)
mechanism: SimulationBridge._run_one_simulation_step, gated by cfg.sparse_activity_step (default False). When on
  and the dispatch guard `_sparse_activity_step_can_dispatch` accepts the current config, the E/I transpose
  matvec, the slow-NMDA recurrent matvec and the causal Hebbian pre/post coincidence read only the synapses of
  the neurons that fired last step; the gain-weighted decay and the gain-masked clip touch only the
  cp_plasticity_rate_gain != 0 / > 0 synapses (cached, version-bumped by every in-place gain writer). Every other
  step section, and the entire step when the flag is off or the guard refuses, is byte-for-byte the pre-existing
  code path.
seeds: [7] (dev seed only -- this finding is an EQUIVALENCE + speed measurement, not a recall-accuracy claim, so
  it does not need the project's 6-seed generalization bar)
instrument: bit-for-bit comparison (sha256 of the raw array bytes, plus per-step lockstep for state) between the
  flag off and flag on paths, on numpy, at two real corpus sizes; a sabotage control that removes the gain-index
  cache and confirms the comparison CAN fail
prereg: none -- this is a performance/equivalence result behind an additive, default-off flag, not a capability
  verdict subject to the production-gate prereg discipline
artifacts:
  - research/findings/raw/_slotbinder_sparse_step/equivalence_seed7_n8.json (full protocol: teach all 8, query
    all 8, ablate all 8, real day_33 facts)
  - research/findings/raw/_slotbinder_sparse_step/equivalence_seed7_n32_partial.json (N=32 topology built from
    32 real facts; teach/query/ablate the first 8 only -- a timing point at 4x the nnz, not a full-corpus claim)
  - tests/test_slotbinder_sparse_step_equivalence.py (8 tests: flag-off-is-default, dispatch-guard refusals,
    per-step lockstep bit-identity through a teach+read window, the composer contract on synthetic facts
    (dense AND fanout wiring), a sabotage control, and the two artifacts above reproduced as pytest cases)
verdict: the event-driven step is bit-identical to the dense step on numpy at both measured sizes (sha256 match
  on stored weights after teach/queries/ablation, on final neuron state, and on every answer, agent+patient+
  yes/no+attribute+moat+mismatch+ablated) -- MEASURED, not the "expected unchanged" fallback docs/TERMS.md
  requires for that phrase. Speedup: teach 7.7x at N=8 (566,400 synapses), 17.1x at N=32-topology (2,265,600
  synapses); overall per-step wall time (teach+query+ablation) 9.6x and 16.9x respectively. Default OFF; no
  behavior change for any existing caller. Scope: numpy/CPU only, and only for configs the dispatch guard
  accepts (see "What this does not cover" below) -- it does not by itself make the N=404 production-gate run
  practical, because the gate's dominant cost at N=404 is query-time reads (O(N) per query x N queries), not
  teach, and this flag speeds up the SAME per-step cost in both phases rather than changing the read protocol.
---

# The SlotBinder's event-driven step is bit-identical to the dense step on numpy, and 7.7-17x fewer seconds/fact at small N

## Why

`research/findings/2026-09-24-slotbinder-production-composer-gate-PREREG.md` AMENDMENT 1 recorded the N=404
dev-seed production-gate run sitting silent for 6h08m with no artifact. `research/FAILURE_LOG.md`'s 2026-09-25
row (added in the same commit as the code this finding documents) already corrected the READING of that stall:
a step count of the arm's own protocol puts the teach at well under 1% of the arm's total simulation steps at
N=404 -- the run was most likely deep in the query/ablation loops, not stuck in teach. That correction does not
by itself make N=404 fast; it says WHERE the cost actually is. This finding is the other half: whatever the
step costs, on the *dense* path every simulation step touches every synapse (~20 passes over `nnz`: the E/I and
slow-NMDA transpose matvecs, the NMDA/AMPA data split, the Hebbian pre/post gather, the gain-weighted decay, the
gain-masked clip), even though a synapse whose presynaptic neuron did not fire contributes an exact `+0.0`, and
a synapse with plasticity gain 0 is multiplied by exactly `1.0` by the decay. That is recomputed zero-work on
every step -- the wall reframe's own question ("what does the real system run alongside this that we replaced
with a constant?") does not apply here; this is the more basic case of doing arithmetic whose answer is already
known to be a no-op.

## What was built

`cfg.sparse_activity_step` (`sim/config.py`, default `False`). When true and
`SimulationBridge._sparse_activity_step_can_dispatch(cfg)` accepts the current configuration (`sim/bridge.py`),
`_run_one_simulation_step` takes an alternate route for three sections:

1. **Propagation** (`_sparse_ampa_transpose_matvec`, `_sparse_nmda_transpose_matvec`): builds a CSR sub-matrix of
   only the outgoing synapses of the neurons that fired last step (`cp_prev_firing_states`), in ascending
   presynaptic-neuron / CSR-row order -- the same order the dense path's `effective_connections_matrix.T @ x`
   sums in on numpy -- and does the transpose matvec on that sub-matrix instead of the full `nnz`-sized one.
2. **Causal Hebbian coincidence**: instead of `cp.where(pre_fired & post_fired)` over every synapse, it starts
   from the same fired-rows synapse set and filters by which of THEIR targets fired this step.
3. **Gain-weighted decay / gain-masked clip**: `_sparse_gain_index_sets()` caches the ascending indices where
   `cp_plasticity_rate_gain != 0` (decay) and `> 0` (clip), invalidated by an explicit version counter that
   every in-place gain writer in the class bumps (`set_plasticity_gate`, `set_plasticity_gate_by_indices`,
   `set_global_plasticity_gain`, the structural-pruning gain zeroing) -- so a stale cache cannot silently apply
   the wrong index set after a gate flips.

`_sparse_activity_step_can_dispatch` refuses (falls back to the unchanged dense step) whenever a feature the
event-driven path has not been verified against is active: short-term or structural plasticity, the
neuromodulator subsystem, inhibitory STDP, an experiment-engine run, transmission gating, graded dendritic
plateaus, GABA_B, deterministic-transpose-matvec mode, coincidence detection, or any Hebbian variant other than
the default causal pre(t-1)&post(t) rule with a named gain array (branchless plasticity, rate-window, symmetric,
or an enforced plastic mask). `SlotBinderComposer(sparse_step=...)` (default reads
`BRAIN_SLOTBINDER_SPARSE_STEP`, default unset = off) and `build_binder_bridge(sparse_step=...)` wire the flag
through for this composer specifically, and additionally set `enable_reward_modulation = False` on that private
bridge when sparse_step is on -- inert there (nothing ever moves `current_reward_signal` off
`reward_baseline`, and no eligibility trace is ever written), so this does not change behavior, only avoids an
all-`nnz` decay-of-zeros the event-driven path has not special-cased.

## Measurement

Instrument: `research/runners/_slotbinder_sparse_step_equivalence.py`, output
`research/findings/raw/_slotbinder_sparse_step/equivalence_seed7_n8.json` (N=8, full protocol) and
`research/findings/raw/_slotbinder_sparse_step/equivalence_seed7_n32_partial.json` (N=32 topology, first-8
protocol). It builds the binder twice (flag off, flag on) from the SAME real day_33 facts
(`_slotbinder_l2_sparse_derisk._sample_facts`, seed 7) and runs the
production gate's own slotbinder-arm protocol on each -- teach every fact via `store()`, `query_patient` every
fact, the moat probe (a never-stored pair, the gate's own RNG draw), the mismatch probe, then the zeroed-synapse
ablation re-query -- comparing sha256 of the raw bytes for: neuron firing thresholds right after build (seed
control), synapse weights after teach, after queries, and after the ablation re-queries, every answer, and the
final neuron state (v, u, firing, every conductance channel). No tolerance is used or needed: on numpy this is
exact.

| N | synapses (nnz) | protocol | teach s/fact OFF | teach s/fact ON | speedup (teach) | ms/step OFF | ms/step ON | speedup (overall) | bit-identical |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 566,400 | full (teach/query/ablate all 8) | 1.00626 | 0.130558 | **7.7x** | 4.96657 | 0.517288 | **9.6x** | yes (all 9 checks) |
| 32-topology | 2,265,600 | partial (teach/query/ablate first 8 of 32) | 4.68278 | 0.27434 | **17.1x** | 19.5767 | 1.15849 | **16.9x** | yes (all 9 checks) |

Both rows: `sparse_activity_step_dispatches: true` for the "on" path and `false` for "off" (the guard is doing
what it says), and every one of the 9 bit-comparisons (`thresholds`, `weights_after_teach`, `intact_answers`,
`intact_reads`, `probe_answers`, `weights_after_queries`, `final_state`, `ablation_answers`,
`weights_after_ablation`) reads `true`. The speedup GROWS with N (7.7x -> 17.1x teach) because the dense path's
cost is `O(nnz)` per step regardless of how few neurons fired, while the event-driven path's cost tracks the
(roughly N-independent, ~20-neuron) firing slot -- consistent with the mechanism, not just a coincidence of one
run.

`tests/test_slotbinder_sparse_step_equivalence.py::test_the_comparison_can_fail` sabotages the event-driven
decay (monkeypatches `_sparse_gain_index_sets` to return empty index sets, i.e. "decay nothing") and confirms
the stored weights then DIFFER from the off-path -- the bit-identity checks used above are demonstrated able to
fail in their failing direction, not just able to pass.

## What this does not cover

- **cupy is untested for bit-identity.** The transpose matvec is cuSPARSE's atomic scatter in BOTH paths on
  GPU, so summation order was already run-to-run nondeterministic there before this change; the dispatch guard
  does not depend on backend, but no cupy equivalence run backs this finding. GPU was unavailable this session
  (a local-model bake-off held the only GPU) -- a cupy equivalence + timing pass is the natural next step before
  this flag is turned on for a GPU production-gate run.
- **N=404 is not measured with this path.** The speedups above are measured at N=8 and N=32-equivalent nnz; they
  are not extrapolated to N=404 here. `research/FAILURE_LOG.md` 2026-09-25 already established that N=404's cost
  is dominated by query/ablation reads (`O(N)` scan per query x `N` queries in `SlotBinderComposer._match`, plus
  every ablated query scanning all N facts), not by teach -- this flag makes each of those reads' per-step cost
  cheaper too (the ON column above is teach+query+ablation combined), but does not change the O(N^2) read-count
  itself. Closing the N=404 wall fully needs a read-side fix (e.g. an indexed match) in addition to this.
- **Configs outside the dispatch guard's allow-list get zero speedup** (silent fallback to the dense step) --
  by design, since those combinations are unverified, not because they were tried and found unequal.
