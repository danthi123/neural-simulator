# Phase 1 GPU-port benchmark — contaminated result, needs clean re-run

**Date:** 2026-05-05 ~07:35 EDT
**Trigger:** post_chain_bench.ps1 ran the bench harness immediately after
the bio_three_factor sweep finished, but graded-DA probe at parallel=6
launched 3 minutes later — so the bench measured under heavy GPU
contention.

---

## Raw bench output (CONTAMINATED, not for decision-making)

| Config | sec/event | Speedup |
|---|---|---|
| baseline_fp32_cpu_eligibility | 1724.6 ms | 1.00× |
| phase1_gpu_eligibility | 2720.8 ms | **0.63× (1.6× slower)** |
| phase2_gpu_eligibility_fp16 | 2721.0 ms | 0.63× |

## Why this isn't trustworthy

Timeline:
- 07:33:18 — bio_three_factor sweep finished
- 07:33:49 — bench harness started
- 07:36:13 — graded-DA probe launched at parallel=6 (6 concurrent procs)
- 07:33-08:21 — bench ran while graded-DA's 6 procs hammered the GPU

So:
- **baseline** ran while graded-DA was just starting (lower contention)
- **phase1** ran while graded-DA was mid-flight (peak contention)
- **phase2** ran while graded-DA was still running (peak contention)

The "speedup" comparison is therefore mostly measuring GPU contention
differences, not Phase 1 vs baseline algorithmic differences.

## What might still be true under contention

There IS a possible real effect: under GPU contention, GPU-port might
be slower than CPU-port because:
- CPU-port: weight updates on host (no GPU dispatch contention)
- GPU-port: every weight update fights other procs for CUDA cores

If so, the recommendation depends on usage pattern:
- Single-process runs: gpu_eligibility=True wins
- Parallel=N sweeps: gpu_eligibility=False might win

This needs CLEAN measurement before defaulting either way.

## Plan

1. Wait for full chain to finish: graded-DA (~50 min) + maybe validation (~1.5 hr)
2. When GPU is genuinely free, re-run `bench_perf_stack.py --quick`
3. If Phase 1 is still neutral or negative under no-contention conditions:
   - Revert `gpu_eligibility=True` default to `False`
   - Keep the flag for opt-in (still useful for future single-process workloads)
   - Update `2026-05-05-perf-roadmap.md` with honest finding
4. If Phase 1 IS faster when GPU is free, recommend usage-pattern-aware
   defaults: parallel=1 enables it, parallel>=3 disables it

## What this teaches about benchmarking discipline

- Benchmarks must run with NO concurrent GPU work
- Auto-bench scripts should poll for GPU idle, not just chain completion
- The post_chain_bench.ps1 wrapper should add a 5-10 min idle-GPU wait
  after the chain completes before starting the bench

Future benchmarks should:
1. Verify nvidia-smi reports <10% GPU util before starting
2. Run sequentially (parallel=1 in bench harness) to measure single-process
3. Optionally run a separate parallel=N sweep AFTER for "throughput vs
   parallelism" measurements

## What this DOESN'T affect

- **Phase 2 FP16**: separately validated via `tests/test_fp16_drift.py`.
  Voltage trace identical (<1mV) FP32 vs FP16 over 1000 steps. The
  drift test runs in isolation (small 64-neuron arch, ~9 sec). FP16
  doesn't break anything; whether it's a perf WIN is unmeasured.
- **Parallel=6**: directly visible from VRAM/utilization, doesn't
  need bench harness validation. User confirmed parallel=2 was wasting
  resources; parallel=6 is a real cheap win.
- **Cloud H100 deploy**: independent of these results.

## Honest takeaway

The Phase 1 perf claim was speculative ("expected 2× from eliminating
round-trip"). The contaminated bench suggests it might be worth less
than that, or even neutral. We need clean measurements before taking
a position. Don't ship the default change until we have the data.
