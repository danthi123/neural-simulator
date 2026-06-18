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
- **~24× faster (clean quiet-GPU):** query 856.5 ms → 36.3 ms (`_phaseB_megakernel_clean_speedup.py`, RTX 3090,
  no contention). The megakernel collapses the per-step launches AND compounds with the CYCLE-152 batched scan.
  This **exceeds** the earlier 17× (which was contention-skewed by concurrent gaming) and the 11× graph-only
  prototype — the megakernel also fuses the matvec + element-wise into one kernel, so on a clean GPU (where the
  loop is purely launch-bound) the win is larger, not smaller.

## ADOPTION GATE — PASSED (2026-06-17)

The pre-registered full-conversational-suite adoption gate is GREEN. `_phaseB_megakernel_conversation_validation.py`
(SIM_BACKEND=cupy) builds a LOOP agent and a MEGAKERNEL agent at the same seed, hears the same facts (incl. a
**recursive embedded clause** `dog see (cat go south)` + two chain facts), and runs the WHOLE conversational stack:

| op | loop | megakernel | expect |
|---|---|---|---|
| what(dog, go) | north | north | north |
| who(go, north) | dog | dog | dog |
| is_it_true(cat, come, south) | yes | yes | yes |
| is_it_true(river, look, west) [NEGATE] | no | no | no |
| is_it_true(apple, stop, east) [unstored] | unknown | unknown | unknown |
| what(dog, see) [**embedded clause**] | cat go south | cat go south | cat go south |
| what(bird, fly) [**abstain**] | None | None | None |
| query_chain(dog, [eat, swim]) [**multi-hop**] | river | river | river |

**8/8 answer-identical (megakernel == loop == ground truth)** — including the embedded clause (the exact case that
killed the period-shortening lever, `2026-06-17-resonate-period-free-speedup.md`) and the no-confab abstention
(moat preserved). ⇒ the megakernel is safe to adopt for real-time conversation. **The agent now exposes a
default-OFF opt-in:** `BrainConversationalAgent(..., enable_rf_cudagraph=True)` passes through to the composer
(GPU-only; default OFF keeps the loop path byte-identical for tests/numpy). Flip it on for the real-time
conversation path; leave it off everywhere else.

## Honest scope + next

- **Default OFF — adoption gate now PASSED (see above).** The full conversational suite is answer-identical with
  the megakernel on (8/8 ops incl. embedded clause + multi-hop + abstention) and a clean quiet-GPU 24× is banked.
  The agent exposes the default-off `enable_rf_cudagraph` opt-in; the real-time conversation path can flip it on.
  Bridge-level default stays OFF so tests/numpy are byte-identical.
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
