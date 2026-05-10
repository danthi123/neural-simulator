# Optimization arc runbook — tomorrow's session

**Goal:** Validate which optimizations from the perf audit actually
yield speedup, with empirical data, and flip defaults for those that
work cleanly. Pre-cloud-deploy preparation.

**Estimated time:** 4-6 hrs.

**Pre-staged tonight:**
- `bridge.set_global_plasticity_gain(value)` API + 8 tests
- `freeze_plasticity_during_reset` opt-in flag in `bio_three_factor.run_three_factor`
- `research.runners.perf_benchmark` harness (auto-enables built-in step profiler)
- `cfg.fp16_synapse_state` validated 2026-05-05, default still False
- Bridge has built-in step profiler (`gpu_config.enable_step_profiler=True`)
  that times init/stp/syn/dyn/plast/homeo/final every 500 steps

---

## Step 1: Baseline benchmark (10 min)

Establish the "before" timing on the architecture we'll use most.
Tier 2.1 v4 scaled (n_lang=4096, n_motor=1000) is a good representative.

```bash
python -m research.runners.perf_benchmark \
    --steps 2000 \
    --vocab-size 8 \
    --n-lang-input 4096 \
    --n-motor-per-action 1000 \
    --n-motor-fs-per-action 120 \
    --out research/findings/raw/perf/baseline.json
```

Expected output:
```
=== Results ===
  steps/sec:   ~XX.X
  ms/step:     ~YY.YY
  VRAM peak:   ~ZZZZ MB

[PROFILER] avg/step: A.AAms | t_init=B (P%) | t_stp=C (P%) | t_syn=D (P%) | ...
```

Record steps/sec + dominant profiler section (highest %). That's the
optimization target.

## Step 2: FP16 isolated (10 min)

```bash
python -m research.runners.perf_benchmark \
    --steps 2000 --fp16 \
    --vocab-size 8 --n-lang-input 4096 \
    --n-motor-per-action 1000 --n-motor-fs-per-action 120 \
    --out research/findings/raw/perf/fp16.json
```

Compare steps/sec to baseline. Expected 1.2-1.5× faster.

**If speedup ≥ 1.2×:** flip default in bio_three_factor + consolidation_synonym_trainer.
**If < 1.1× or accuracy concern:** keep opt-in only.

## Step 3: freeze-plasticity-during-reset isolated (10 min)

```bash
python -m research.runners.perf_benchmark \
    --steps 2000 --freeze-plasticity-during-reset \
    --vocab-size 8 --n-lang-input 4096 \
    --n-motor-per-action 1000 --n-motor-fs-per-action 120 \
    --out research/findings/raw/perf/freeze_reset.json
```

**Critical caveat:** the perf_benchmark uses a tight loop of bare
sim_steps — there are no "events" with reset_steps in the
benchmark. So the freeze-reset flag SHOULD have no effect in
this microbenchmark. To measure real-world speedup, instead compare
chat_synonym_demo wall-clock with vs without the flag:

```bash
# Baseline
time python -m research.runners.chat_synonym_demo \
    --seed 42 --n-events-per-direction 50 --quiet

# With opt enabled — manual edit of train_chat_bridge call needed,
# OR add --freeze-plasticity-during-reset to chat_synonym_demo CLI
# (one-line change to argparse).
```

## Step 4: Combined FP16 + freeze-reset (10 min)

```bash
python -m research.runners.perf_benchmark \
    --steps 2000 --fp16 --freeze-plasticity-during-reset \
    --vocab-size 8 --n-lang-input 4096 \
    --n-motor-per-action 1000 --n-motor-fs-per-action 120 \
    --out research/findings/raw/perf/fp16_freeze.json
```

Expected: combined speedup multiplicative (e.g., 1.3 × 1.4 = 1.8×).

## Step 5: Multi-seed accuracy validation (~3 hr)

Speedup means nothing if accuracy degrades. Run a 3-seed validation
of chat_speak_synonym_demo (the tightest accuracy benchmark we have)
WITH the optimizations enabled. Compare to tonight's 6-seed baseline:

| Seed | Baseline A2W | Optimized A2W | Pass? |
|------|--------------|---------------|-------|
| 42   | 50%          | ?             | ≥45%  |
| 43   | 75%          | ?             | ≥70%  |
| 44   | 100%         | ?             | ≥90%  |

Decision criteria:
- All 3 seeds within 10pp of baseline → flip default ON
- Any seed >10pp below baseline → keep opt-in, investigate

(Skip seeds 100/101/102 if baseline already at 100% — they don't add
discrimination signal.)

## Step 6: Profile-guided next-target identification (30 min)

Run perf_benchmark with --profile flag on the SLOWEST architecture
(64-word at n_motor=6000):

```bash
python -m research.runners.perf_benchmark \
    --steps 500 \
    --vocab-size 64 \
    --n-lang-input 4096 \
    --n-motor-per-action 6000 \
    --n-motor-fs-per-action 720 \
    --profile \
    --out research/findings/raw/perf/profile_64word.json
```

Read the `[PROFILER] avg/step:` line. The dominant section (e.g.,
t_plast=70%) is the next optimization target. Possible findings:
- `t_syn` dominates → CSR matrix-vector multiply optimization
- `t_plast` dominates → STDP kernel optimization (FP16 throughout?)
- `t_dyn` dominates → neuron dynamics kernel (probably already fused)
- `t_homeo` dominates → homeostasis (rarely the bottleneck)

The Python-side profile (cProfile output) will show if any wrapper
overhead is unexpectedly high.

## Step 7: Document findings + commit (30 min)

Write `research/findings/2026-05-10-perf-optimization-results.md` with:
- Baseline numbers (steps/sec at each arch)
- Per-optimization speedup measured
- Combined speedup
- Multi-seed accuracy preserved (or not)
- Decision: which defaults flipped, which stay opt-in
- Next target identified by profile

## Step 8 (deferred to next iteration): items 4-6 from audit

Per priority order:
4. Sparser cross-region density (1-2 days, full revalidation needed)
5. Profile inner loop (DONE in step 6 above — already shipped infra)
6. FP16 throughout — needs spike-train divergence analysis

## Cloud deploy gating

Once steps 1-7 complete:
- 3090 local FP32 baseline = 1× (current state)
- 3090 local FP16 + freeze-reset = ~2-2.7× (target)
- A100 80GB FP16 + same opts = ~6-8× (cloud)
- H100 80GB FP16 + same opts = ~12-20× (cloud)

For sweep workloads (multiple seeds, multiple configs), cloud at H100
becomes ~16× faster than current local 3090 FP32. A nominal 10-hour
local sweep becomes ~38 min on H100, costing ~$3 ($4-8/hr × 0.6 hr).

**Decision rule:** if local optimization yields ≥2× speedup, deploy
to cloud at that point. If <2× speedup, debug profile output to
identify the real bottleneck before adding cloud cost on top.

## Pre-staged scripts + presets

All ready to fire after baseline benchmark:

| Use case | Path |
|----------|------|
| Benchmark harness | `python -m research.runners.perf_benchmark` |
| Multi-seed wrapper | `scripts/multiseed_chat_speak_synonym_demo.sh` |
| Path A overnight chain | `scripts/chain_path_a_overnight.sh` |
| Vocab tier dispatcher | `text_eval.get_synonym_groups(N)` for N ∈ {8,12,16,24,32,48,64,96,128,256} |
| Axis-decoupling presets | `consolidation_synonym_64word_encoding_scale_smoke` etc. (4 variants) |

## Provenance

- Audit: `research/findings/2026-05-10-perf-optimization-audit.md`
- Global plasticity API: `sim/bridge.py:2403` (set_global_plasticity_gain)
- Tests: `tests/test_global_plasticity_gain.py`
- Benchmark harness: `research/runners/perf_benchmark.py`
- Built-in step profiler: `sim/bridge.py:4287` (_run_one_simulation_step)
