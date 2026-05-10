# Find-the-ceiling discoveries (overnight 2026-05-10)

**Trigger:** User directive earlier in the arc — "start very high on the
scale to test for failure, then bring it down as needed to fit in local
compute."

**Headline:** Initial extrapolations were WRONG by significant margins.
Reality is much more capable than predicted, BUT wall-clock cost grows
faster than VRAM cost.

---

## Predictions vs reality

The original capacity-rule extrapolation predicted 64-word @ n_motor=6000
would consume ~28 GB VRAM (OOM on 24 GB 3090). Actual: ~16 GB.

| Vocab | n_motor | Predicted VRAM | Actual VRAM | Status |
|-------|---------|----------------|-------------|--------|
| 16    | 2000    | ~7 GB          | ~7 GB       | matches |
| 64    | 6000    | **~28 GB (OOM)** | **~16 GB** | predicted-OOM was wrong by 12 GB |

**Why the prediction was wrong:** the linear-extrapolation assumed dense
connectivity. Actual connectivity is sparse (`density=0.1` per pathway).
With sparse connectivity, total synapses scale roughly as
N_lang × N_motor × density × N_pathways, NOT N×N. So a 3× motor pool
increase doesn't 9× the synapse count — it 3×s it.

## The real bottleneck is wall-clock, not VRAM

Per-chunk pacing observed:
- 16-word @ n_motor=2000: ~145s per chunk+sleep cycle
- 64-word @ n_motor=6000: ~700s per chunk+sleep cycle (~5× slower)

Architecture is 3× bigger but per-step is 5× slower. The slow factor
is per-synapse work in the inner loop. On a 24 GB 3090, we have plenty
of VRAM headroom but limited wall-clock for find-the-ceiling experiments.

**Implication:** the practical ceiling is set by acceptable wall-clock
per smoke (~30-45 min for fast iteration), not VRAM.

## Pre-staged tiers ready for testing

Per user directive:
- Vocab tiers 24/32/48/64/96/128/256 shipped
- 64-word at n_motor=6000 in flight (validates VRAM)
- 96/128/256 use numbered-variant fallback (north_05, ...) for
  encoding-collision wall testing

## Three axes of optimization not yet decoupled

User flagged this insight: I'd been scaling MOTOR pool and vocab together,
which conflates the two axes. The three independent axes are:

1. **n_motor** (motor capacity) — 2000 / 4000 / 6000 / 12000
2. **n_lang_input** (encoding capacity) — 4096 / 8192 / 16384
3. **vocab size** — 8 / 16 / 32 / 64 / ...

Pre-staged decoupling experiments (all smoke ~30-60 min each):

| Preset | n_lang | n_motor | vocab | Tests |
|--------|--------|---------|-------|-------|
| `consolidation_synonym_64word_encoding_scale_smoke` | **8192** | 2000 | 64 | encoding alone |
| `consolidation_synonym_64word_lang_balanced_smoke` | **8192** | 4000 | 64 | balanced aspect |
| `consolidation_synonym_256word_big_encoding_smoke` | **16384** | 2000 | 256 | encoding wall at extreme vocab |
| `consolidation_synonym_16word_big_motor_smoke` | 4096 | **12000** | 16 | motor capacity excess |

Together these will isolate WHICH of the three axes is rate-limiting.

## Hypothesis to validate

Per the 16-word smoke result tonight (primary 50%, ASCII synonyms 0%,
Unicode arrows often work), the encoding-collision wall is real and
visible at 16-word. Increasing n_motor past the rule's floor doesn't
help if encoding is the bottleneck.

**Predicted ranking** (binding accuracy, hypothesis):
- `64word_encoding_scale_smoke` (n_lang=8192, n_motor=2000): SHOULD bind
  better than current 64-word @ n_motor=6000 if encoding is real bottleneck
- `64word_scaled_smoke` (current, n_lang=4096, n_motor=6000): in flight
- `64word_lang_balanced_smoke` (n_lang=8192, n_motor=4000): SHOULD be
  best of the three if both axes contribute

If `encoding_scale` outperforms `scaled` by >10pp, the architecture
strategy pivots: scale encoding first, motor second.

## Compute tomorrow

Tonight's perf audit identified up to 2.7× speedup from 3 optimizations
(items #1-3 in `2026-05-10-perf-optimization-audit.md`). All shipped
as opt-in flags tonight; benchmark suite ready in `perf_benchmark.py`.
Validating tomorrow could:
- Run all axis-decoupling tests at 1.5-2.7× current pace
- Test 96-word and beyond without 3-hour smokes per tier

## Cloud-deploy gating (re-confirmed)

Local optimization first, cloud second. The compounding factor:
- 3090 local FP32 baseline = 1×
- 3090 + tonight's opt-stack = 2-2.7× (if validated)
- A100 80GB FP16 + opt-stack = 6-12×
- H100 80GB FP16 + opt-stack = 12-24×

A find-the-ceiling sweep that takes ~12 hours locally on FP32 baseline
finishes in ~30-60 min on cloud H100 with optimizations. **Once
optimizations validate, the cloud path becomes cheap.**

## Recommended morning sequence

1. Read 64-word smoke result + write findings (already in flight)
2. Run perf_benchmark suite — get speedup numbers
3. Single-seed accuracy validation if speedup ≥1.2×
4. Decide which optimizations to flip on by default
5. Launch encoding-scale 64-word smoke (axis-decouple #1)
6. Eventually queue lang-balanced + big-encoding 256-word + big-motor 16-word

If steps 1-4 land within 90 min of waking: cloud is the next decision.
If not: profile output identifies what's actually slow.

## Provenance

- Optimization audit: `research/findings/2026-05-10-perf-optimization-audit.md`
- Optimization runbook: `docs/plans/2026-05-10-optimization-arc-runbook.md`
- Overnight summary: `research/findings/2026-05-10-overnight-arc-summary.md`
- Perf benchmark harness: `research/runners/perf_benchmark.py`
- Bridge step profiler: `sim/bridge.py:4287` (`gpu_config.enable_step_profiler=True`)
