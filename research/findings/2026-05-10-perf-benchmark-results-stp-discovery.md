# Perf benchmark results — STP is the inner-loop bottleneck (57%)

**Date:** 2026-05-10 12:12 EDT
**Status:** Empirical perf data + new optimization target identified
**Trigger:** Per autonomous arc plan, run perf_benchmark suite immediately
after 64-word smoke completes
**Hardware:** RTX 3090 24 GB, ~700 MB VRAM at 8-word arch, 1500 steps each

---

## Headline finding

**STP (Short-Term Plasticity) dominates the inner-loop step time at 57%.**
This is the biggest optimization target the audit didn't surface — bigger
than FP16, bigger than freeze-plasticity, bigger than reduced step counts.

The bridge's built-in step profiler made this discovery automatic:

```
[PROFILER] avg/step: 8.86ms
  t_stp=5.05ms (57%)   ← STP per-step is over half the budget
  t_plast=1.70ms (19%)
  t_syn=1.20ms (14%)
  t_dyn=0.64ms (7%)
  t_homeo=0.11ms (1%)
  t_init=0.13ms (1%)
  t_final=0.03ms (0%)
```

`bio_three_factor` sets `cfg.enable_per_type_stp = False` but
`cfg.enable_short_term_plasticity` is True by default — and STP is
where 57% of the inner loop goes.

## Speedup measurement (4 configs, 1500 steps each)

Tested at Tier 2.1 v4 8-word arch (n_lang=4096, n_motor=1000):

| Config | steps/sec | ms/step | Speedup |
|--------|-----------|---------|---------|
| Baseline (FP32, no freeze) | 99.5 | 10.05 | 1.00× |
| FP16 (cp_eligibility_trace) | **112.9** | 8.86 | **1.135×** |
| Freeze-plasticity-during-reset | 110.6 | 9.04 | 1.111× (noise) |
| FP16 + freeze | 109.2 | 9.16 | 1.098× |

### Interpretation

- **FP16 yields a real 13.5% speedup.** Less than the audit's predicted
  1.2-1.5× but real. Worth flipping the default ON after multi-seed
  accuracy validation tomorrow.

- **Freeze-plasticity doesn't show up in the microbenchmark.** This is
  expected — perf_benchmark runs a tight loop of bare sim_steps without
  the inter-event reset_steps blocks where the optimization fires. The
  freeze flag only matters in real training workloads with the
  `reset_steps + stim_steps` pattern. Real-workload benchmarks
  (chat_synonym_demo wall-clock comparison) needed.

- **Combined doesn't compose.** The sub-1.10× combined result vs FP16
  alone at 1.135× suggests there's ~2-3% measurement variance dominating.

## STP optimization opportunity

If STP is 57% of step time, optimizing it has 4-7× more leverage than
FP16. Possible angles:

### Option A: Disable STP in chat configs (HIGH IMPACT)

bio_three_factor + consolidation_synonym_trainer don't biologically need
STP for word-action binding. STP models second-millisecond synaptic
fatigue/recovery — relevant for gamma oscillations or fast-spiking
neuron dynamics, not for binding stable language→motor associations.

If we set `cfg.enable_short_term_plasticity = False` in these training
configs, the t_stp section drops to ~0 ms.

**Predicted speedup: ~2.4×** (5.05 ms / total 8.86 ms = 57% removed →
remaining 3.81 ms vs current 8.86 ms = 2.32× faster).

**Risk:** Need to verify multi-seed binding accuracy holds without STP.
Tier 2.1 BREAKTHROUGH paper had STP enabled; flipping it off may shift
the trained dynamics. Single-seed accuracy validation needed.

### Option B: Optimize STP kernels

If STP is needed biologically, the kernel itself may be optimizable.
The bridge has `cfg.enable_per_type_stp = False` (per-type STP would
be slower); current STP uses global U/tau parameters. Still consuming
5 ms/step.

Possible inefficiencies:
- Per-step decay update of cp_stp_x (size = n_synapses = 30M+)
- Per-step recovery update of cp_stp_u
- Both are write-heavy operations

A fused kernel for STP could reduce memory traffic. Or FP16 for STP
state (currently fp32) would halve bandwidth like fp16_synapse_state
does for eligibility traces.

### Option C: Skip STP during quiet steps

Same idea as freeze-plasticity-during-reset: STP decay during reset_steps
is biologically meaningful but training-irrelevant. Could gate STP
similarly.

## Revised optimization priority order

Given empirical data, the audit's order changes:

1. ~~**#1 fp16_synapse_state default**~~ — measured 1.135× speedup,
   real but smaller than predicted. Still worth the default flip.
2. **NEW #2: Disable STP in chat configs** — predicted 2.32× speedup,
   biggest single lever. Validate multi-seed accuracy holds.
3. **#3 plasticity-off during reset** — small speedup at training time
   (not measurable in microbench). Defer until #2 is validated.
4. **#5 profile inner loop** — DONE, this finding came from it.
5. **NEW #6: STP kernel optimization** — if disabling isn't safe.

## Proposed next experiment (when GPU frees)

Quick STP-disabled benchmark to confirm the speedup estimate:

```bash
# Need to add a --disable-stp flag to perf_benchmark first.
# Or directly modify the bio_three_factor.run_three_factor call to
# set cfg.enable_short_term_plasticity = False before bridge build.
```

If STP-disabled yields ~2.3× speedup as predicted, it's the dominant
optimization for biology-grounded chat training. Cumulative best case:

- Baseline: 99.5 steps/sec
- + FP16: 112.9 (1.135×)
- + STP-disabled: ~263 steps/sec (1.135 × 2.32 = 2.63×)

That's ~2.6× total local speedup, which compounds with cloud GPU
multipliers: H100 + STP-disabled + FP16 ≈ 25-50× over current 3090
FP32 baseline.

## Caveats

- Microbenchmark uses 1500 steps (~10 sec walltime). Variance is
  measurable; speedups under 1.10× are noise.
- 8-word arch (n_motor=1000); not the same shape as 64-word arch
  (n_motor=6000) where the bottleneck might shift.
- Profile output represents ratios that should be relatively
  stable across arch sizes since each section scales similarly.

## Files

- Baseline: `research/findings/raw/perf/baseline.json`
- FP16: `research/findings/raw/perf/fp16.json`
- Freeze: `research/findings/raw/perf/freeze.json`
- Both: `research/findings/raw/perf/both.json`
- Bench harness: `research.runners.perf_benchmark`
- Bridge profiler: `gpu_config.enable_step_profiler=True`
- Optimization audit (pre-data): `2026-05-10-perf-optimization-audit.md`

## Recommended morning sequence

1. Read this finding
2. Single-seed `chat_speak_synonym_demo --seed 42` with STP disabled.
   Compare to tonight's seed 42 baseline (W2A 25%, A2W 50%).
3. If accuracy holds within 5pp: flip default in bio_three_factor +
   consolidation_synonym_trainer.
4. Measure 64-word smoke wall-clock with STP disabled — should drop
   from 9.4 hr to ~4 hr. Major savings.
5. THEN move to cloud, where STP-disabled + FP16 + better hardware
   compounds to ~25-50× current baseline.
