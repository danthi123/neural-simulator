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
  cp_plasticity_rate_gain != 0 / > 0 synapses, cached by `_sparse_gain_index_sets` and invalidated in O(1) (an
  identity + nnz + mutation-version check, no content scan) because `cp_plasticity_rate_gain` is always a
  `_TrackedGainArray` (sim/bridge.py) whose own `__setitem__` bumps that version on every write, catching the
  ~40 research/runners sites that write it in place without going through a setter. Every other step section,
  and the entire step when the flag is off or the guard refuses, is byte-for-byte the pre-existing code path.
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
  - tests/test_slotbinder_sparse_step_equivalence.py (9 tests: flag-off-is-default, dispatch-guard refusals
    (config-level and bridge-state-level), per-step lockstep bit-identity through a teach+read window, the
    composer contract on synthetic facts (dense AND fanout wiring), a sabotage control, and the two artifacts
    above reproduced as pytest cases)
  - tests/test_sparse_gain_index_sets_self_heals.py (3 tests: the gain-index cache picks up an in-place write
    with no setter call, correctly, not just when a setter bumps a version)
  - tests/test_sparse_gain_index_sets_o1_staleness_check.py (4 tests: repeated no-op calls do not re-scan the
    array's contents, the mutation-version counter -- not content-diffing -- is the staleness signal, a
    whole-array reassignment is auto-wrapped and detected by identity, and the default flag-off dense step is
    byte-identical regardless of whether the gain array is wrapped)
verdict: the event-driven step is bit-identical to the dense step on numpy at both measured sizes (sha256 match
  on stored weights after teach/queries/ablation, on final neuron state, and on every answer, agent+patient+
  yes/no+attribute+moat+mismatch+ablated) -- MEASURED, not the "expected unchanged" fallback docs/TERMS.md
  requires for that phrase. Speedup (RE-MEASURED 2026-09-25 on the exact code merged in the slotbinder-fast-teach
  review's FINAL fix round, after the gain-index cache's staleness check was changed from an O(nnz) content
  comparison to an O(1) mutation-version check -- see "mechanism" above; the numbers below supersede an earlier
  measurement taken against a pre-review commit, which is no longer what this repo runs): teach 11.0x at N=8
  (566,400 synapses), 12.9x at N=32-topology (2,265,600 synapses); overall per-step wall time (teach+query+
  ablation) 12.1x and 15.0x respectively. Default OFF; no behavior change for any existing caller. Scope:
  numpy/CPU only, and only for configs the dispatch guard accepts (see "What this does not cover" below) -- it
  does not by itself make the N=404 production-gate run practical, because the gate's dominant cost at N=404 is
  query-time reads (O(N) per query x N queries), not teach, and this flag speeds up the SAME per-step cost in
  both phases rather than changing the read protocol.
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
   `cp_plasticity_rate_gain != 0` (decay) and `> 0` (clip). `cp_plasticity_rate_gain` is always stored as a
   `_TrackedGainArray` (a `cp.ndarray` subclass whose own `__setitem__` bumps a per-instance mutation-version
   counter on every write); the cache key is (object identity, nnz, that version), so ANY write through
   indexing -- `g[:] = 0.0`, `g[idx] = 1.0`, a named setter, or one of the ~40 research/runners sites that write
   the array in place directly -- invalidates the cache in O(1), with no per-call scan of the array's contents
   (an earlier version of this mechanism compared contents on every call, which was measured to cost as much as
   the work it was meant to save; see the Measurement section's note on the 2026-09-25 re-measurement).

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

**RE-MEASURED 2026-09-25** (slotbinder-fast-teach review FINAL fix round) against the exact code being merged --
the prior table in this finding was measured against a pre-review commit (010f6c117) whose gain-index cache no
longer exists in this repo (superseded first by an O(nnz) content-diffing self-heal, then by the O(1)
mutation-version check described in "What was built" item 3 above); neither intermediate version is what ships,
so the numbers below are the only ones that describe the merged code. Same instrument, same command, same real
facts, same seed -- only the code under test changed.

| N | synapses (nnz) | protocol | teach s/fact OFF | teach s/fact ON | speedup (teach) | ms/step OFF | ms/step ON | speedup (overall) | bit-identical |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 566,400 | full (teach/query/ablate all 8) | 1.00457 | 0.09157 | **11.0x** | 4.34870 | 0.35979 | **12.1x** | yes (all 9 checks) |
| 32-topology | 2,265,600 | partial (teach/query/ablate first 8 of 32) | 4.42653 | 0.34252 | **12.9x** | 20.11269 | 1.33858 | **15.0x** | yes (all 9 checks) |

Both rows: `sparse_activity_step_dispatches: true` for the "on" path and `false` for "off" (the guard is doing
what it says), and every one of the 9 bit-comparisons (`thresholds`, `weights_after_teach`, `intact_answers`,
`intact_reads`, `probe_answers`, `weights_after_queries`, `final_state`, `ablation_answers`,
`weights_after_ablation`) reads `true`. The speedup GROWS with N (11.0x -> 12.9x teach; 12.1x -> 15.0x overall)
-- that direction is safe: the dense path's cost is `O(nnz)` per step, so a 4x increase in nnz (N=8 -> 32
-topology) costs the dense path ~4.62x (4.34870 -> 20.11269 ms/step), consistent with `O(nnz)`. **The
event-driven path's cost is NOT N-independent, only sub-linear in nnz**: its own overall ms/step rises 0.35979
-> 1.33858 (~3.72x for that same 4x increase in nnz). Unlike the pre-review measurement of this same protocol,
per-fact teach time within either run does NOT climb monotonically here (N=8 ON: 0.09441, 0.08601, 0.08858,
0.09248, 0.08845, 0.09063, 0.09290, 0.09911 s; N=32 ON: 0.37464, 0.35935, 0.33658, 0.35879, 0.31402, 0.32386,
0.34630, 0.32662 s -- both flat within noise, no trend) -- the earlier "~2.5x climb within N=8" observation does not reproduce on this code and
is retracted as a claim about the mechanism; it was most likely either host load on that one earlier run or an
artifact of the since-superseded O(nnz) content-diffing cache (which this finding never isolated at the time).
Two methodological limits on the headline itself (which the measured numbers do support): each size is ONE
sequential off-then-on run on a shared host, not repeated trials, so run-to-run variance at either N is
uncharacterized; and `SlotBinderComposer(sparse_step=True)` bundles THREE changes on this private bridge --
the event-driven step measured here, `enable_reward_modulation=False` (which also drops an all-`nnz`
eligibility-trace decay every step, inert on this bridge's actual dynamics but still removed work), and the
event-driven helpers' own read-side accumulation -- and this measurement does not decompose the speedup between
them.

`tests/test_slotbinder_sparse_step_equivalence.py::test_the_comparison_can_fail` sabotages the event-driven
decay (monkeypatches `_sparse_gain_index_sets` to return empty index sets, i.e. "decay nothing") and confirms
the stored weights then DIFFER from the off-path -- the bit-identity checks used above are demonstrated able to
fail in their failing direction, not just able to pass.

## What this does not cover

- **cupy is untested for bit-identity OR speed.** The transpose matvec is cuSPARSE's atomic scatter in BOTH
  paths on GPU, so summation order was already run-to-run nondeterministic there before this change; the
  dispatch guard does not depend on backend, but no cupy equivalence or timing run backs this finding. GPU was
  unavailable this session (a local-model bake-off held the only GPU) -- a cupy equivalence + timing pass is the
  natural next step before this flag is turned on for a GPU production-gate run (prepared, not yet run:
  `research/findings/2026-09-24-slotbinder-production-composer-gate-PREREG.md` AMENDMENT 3). The event-driven
  helpers also add per-step HOST syncs the dense path does not have (`_sparse_csr_rows`'s `int(bounds[-1])`,
  `_sparse_rows_state`'s `cp.flatnonzero`) -- free on numpy (already host-side), but a device->host sync on
  cupy, so the numpy speedups measured above should NOT be assumed to transfer to GPU; they may be smaller
  there, or in principle even negative at small firing counts, until actually measured.
- **N=404 is not measured with this path.** The speedups above are measured at N=8 and N=32-equivalent nnz; they
  are not extrapolated to N=404 here. `research/FAILURE_LOG.md` 2026-09-25 already established that N=404's cost
  is dominated by query/ablation reads (`O(N)` scan per query x `N` queries in `SlotBinderComposer._match`, plus
  every ablated query scanning all N facts), not by teach -- this flag makes each of those reads' per-step cost
  cheaper too (the ON column above is teach+query+ablation combined), but does not change the O(N^2) read-count
  itself. Closing the N=404 wall fully needs a read-side fix (e.g. an indexed match) in addition to this.
- **Configs outside the dispatch guard's allow-list get zero speedup** (silent fallback to the dense step) --
  by design, since those combinations are unverified, not because they were tried and found unequal.
