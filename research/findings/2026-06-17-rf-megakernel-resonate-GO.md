# RF resonate megakernel — the orchestration latency fix, validated (~17×, answer-identical, default-off)

**Date:** 2026-06-17 (owner cleared higher-risk revertable edits → executed the deep core-engine piece)
**Status:** **GO.** One custom CUDA kernel does the whole resonate step (complex sparse matvec + dynamics, one
thread per neuron), collapsing the launch-bound ~15-kernel/step loop into 1/step. Bit-identical to the loop
(bridge-level) + answer-identical end-to-end + **~17× faster** (contention-skewed by concurrent gaming; a clean
quiet-GPU number is the follow-up). Additive, gated `cfg.enable_rf_cudagraph` (**default OFF → byte-identical**),
revertable by gate-flip or `git revert`. GPU-only (numpy backend + masked co-resident bridges fall back to the loop).
**Edit:** `sim/bridge.py` (`_RF_MEGASTEP_SRC` RawKernel + `_rf_resonate_steps_megakernel` + the gated dispatch in
`rf_resonate_steps`), `sim/config.py` (`enable_rf_cudagraph`), `research/runners/rf_phasor_composer.py`
(`enable_rf_cudagraph` pass-through). **Tests:** `tests/test_rf_megakernel.py`.

## The problem (recap)

The conversational composer's per-op latency was ~160 ms, 97.7% the 208-step resonate loop — each step issued
~15-20 sequential CuPy kernels (4 cuSPARSE matvecs + ~10 element-wise), looped 208× in Python → ~3-4k tiny kernel
launches/op, GPU ~99% idle (launch-bound). The period can't shrink (clauses need the full window,
`2026-06-17-resonate-period-free-speedup.md`), so the fix is cheaper steps.

## The fix — the fused megakernel (the GeNN "merged kernel" pattern, per the literature review)

A custom CUDA `RawKernel` (`rf_megastep`) where **one thread per neuron** does the ENTIRE step:
- the complex **sparse CSR matvec** for its row (the FHRR bind-through-synapses input) — accumulated in double,
  matching the cuSPARSE path cast to float32 on the add;
- the rotate/decay (`z·exp(λ+iω)`), in float32 to match the membrane dtype;
- the Mg-free **zero-crossing** detection → spike-step / fired update (per-neuron, no cross-thread race).

State is **double-buffered** (`re_in`/`re_out` swap each step) so the matvec reads a consistent pre-step state;
`prev_im`/`fired`/`spike_step` accumulate in place. The whole step is **one kernel launch** instead of ~15.

## Validation

- **Bridge-level bit-identity** (`tests/test_rf_megakernel.py`, 3 passed): megakernel == loop phase read for a
  **bind** (permutation, 1 nonzero/row) AND a **bundle** (strided sum, L nonzeros/row — exercises the matvec
  accumulation), max phase diff < 1e-2 (float32 + the crossing-step is an int, so boundary neurons are the only
  source of any diff); and default-off uses the loop.
- **Answer-identical end-to-end:** a composer with `enable_rf_cudagraph=True` returns the SAME who/what answers as
  the loop composer AND the ground truth, with the no-confab moat consistent (abstention preserved). The cleanup
  is argmax, so the tiny float32 phase differences never change an answer.
- **~17× faster:** query 798.9 ms → 46.1 ms (the megakernel collapses the per-step launches AND compounds with the
  CYCLE-152 batched scan). **Contention-skewed** (measured while a game shared the GPU); the clean quiet-GPU number
  is the follow-up, but the order is decisive and exceeds the 11× prototype (which only graphed; this also fuses
  the matvec + element-wise into one kernel).

## Honest scope + next

- **Default OFF.** The production agent does not yet opt in — flip `enable_rf_cudagraph` (composer/agent) only
  after the FULL conversational suite passes with it on (incl. `test_embedded_clause`, multi-turn, reconsolidation)
  and a clean quiet-GPU speedup is banked. The bridge-level + end-to-end checks here are the foundation; the
  full-suite-on gate is the adoption step.
- **CUDA-graph capture is now a smaller follow-on** (design doc step 3): the megakernel already reduces the op to
  208 launches (1/step); graph-capturing that loop removes the residual per-step launch overhead for a final
  increment on top of the 17×. Optional — the megakernel alone is the bulk of the win.
- The matvec uses the general CSR (not a structured-elementwise special-case), so it covers bind / unbind / bundle
  / clauses uniformly — no per-op dispatch.

## Reproduce
```bash
SIM_BACKEND=cupy python -u -m pytest tests/test_rf_megakernel.py -q       # bit-identity (GPU)
# end-to-end: RFPhasorComposer(enable_rf_cudagraph=True) == default answers, ~17x faster query
```
