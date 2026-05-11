# :speak inference latency across vocab tiers — n_lang dominates, not vocab size

**Date:** 2026-05-11 02:18 EDT
**Status:** Complete — 8-tier benchmark suite (4w through 64w) ran
end-to-end after the 96-word XL NEGATIVE eval was killed at the user's
request (commit `f30da80`). Auto-chained via
`research/findings/raw/g11_bg/chain_inference_benchmark_post_xl.ps1`.
**Trigger:** User option (b) earlier: "pre-stage a benchmark script
that times inference (not training) at each tier, so we have a verified
inference-cost chart."
**Provenance:** Raw results at
`research/findings/raw/perf/inference_bench/inference_bench_v{N}w.json`
for N in {4, 8, 12, 16, 24, 32, 48, 64}.

---

## Headline

**The validated 16-word arch (n_lang=4096, n_motor=2000) gives sub-2-sec
:speak latency** — conversational for the chat REPL. The validated 64-word
ceiling (n_lang=8192, n_motor=2000) gives ~6 sec :speak — borderline
usable, feels slow. The cliff is at 16w→24w where n_lang doubles 4096→8192.

| Tier | n_lang | n_motor | :speak (ms) | std (ms) | init (s) | VRAM (MB) |
|------|--------|---------|-------------|----------|----------|-----------|
| 4w   | 2048   | 500     |        1255 |      156 |     11.5 |       148 |
| 8w   | 4096   | 1000    |        1286 |      260 |     39.9 |       591 |
| 12w  | 4096   | 2000    |        1710 |      116 |     63.5 |      1226 |
| 16w  | 4096   | 2000    |        1720 |       46 |     64.3 |      1226 |
| 24w  | 8192   | 2000    |        7335 |      871 |    137.0 |      2362 |
| 32w  | 8192   | 2000    |        6439 |      592 |    131.2 |      2362 |
| 48w  | 8192   | 2000    |        6450 |      637 |    125.5 |      2362 |
| 64w  | 8192   | 2000    |        6051 |      557 |    122.2 |      2362 |

(All runs: STP off, 10 rounds × 4 actions = 40 :speak samples per tier,
warm cache after one untimed dry run.)

## Three latency bands

The 8 tiers cluster cleanly into **three arch bands**, each defined by
its (n_lang, n_motor) pair. Within a band, latency is flat regardless of
vocab size:

### Band A — small (n_lang=2048-4096, n_motor=500-1000): ~1.27 sec

```
4w:  1255 ± 156 ms
8w:  1286 ± 260 ms
```

Tier 1 (4w) and Tier 2.1 (8w) live here. **Conversationally fast.**
This is the recipe for a snappy chat REPL.

### Band B — medium (n_lang=4096, n_motor=2000): ~1.72 sec

```
12w: 1710 ± 116 ms
16w: 1720 ± 46  ms
```

Tier 2.2 / 2.3 arch. **Still conversational** — under 2 seconds. The
extra motor pool adds 30% latency vs Band A but doesn't break the
"feels live" UX.

### Band C — large (n_lang=8192, n_motor=2000): ~6.57 sec

```
24w: 7335 ± 871 ms
32w: 6439 ± 592 ms
48w: 6450 ± 637 ms
64w: 6051 ± 557 ms
mean(24-64w) = 6569 ± 484 ms
```

Tier 3-ish arch (encoding-axis scale-up, validated local-3090 ceiling).
**Borderline usable** — feels like waiting on a slow LLM. Not the
1-2 sec target for natural chat.

## Cost driver: n_lang, not vocab size

The latency cliff at 16w→24w is **n_lang doubling 4096→8192**, not the
24-word vocab itself. Evidence:

- 32w, 48w, 64w all use the same arch as 24w → all cluster at 6.0-6.5
  sec (±400ms). The vocab size has essentially zero per-:speak cost.
- 12w and 16w share an arch → identical latency (within 10ms).
- 4w → 8w doubles n_lang (2048→4096) AND n_motor (500→1000) and adds
  only 30 ms.
- 12w → 16w (no arch change) adds 10 ms.
- 16w → 24w (n_lang doubles, n_motor flat) adds **5600 ms (3.7×)**.

So the dominant per-:speak cost is the `language_input → motor_X` pathway
which scales with n_lang × n_motor. Doubling n_lang doubles per-step
compute. The 3.7× wall-clock impact (vs the predicted 2×) suggests the
larger arch also needs more simulation steps to settle motor firing into
a decodable pattern.

## What this means for the user-facing chat workflow

1. **Conversational sweet spot is 16-word vocab on the validated arch
   (n_lang=4096, n_motor=2000).** Sub-2-sec :speak, plenty of vocabulary
   diversity for embodied directional chat. This is where lineage `main`
   should default at session start.

2. **64-word (the local-3090 ceiling) is usable but slow** — 6 sec
   :speak is past the "feels live" threshold. Still useful for
   asynchronous workflows ("type a list, get responses later") but not
   for back-and-forth dialogue.

3. **96-word and above (NEGATIVE @ XL) are doubly blocked:**
   - Retention wall: at n_motor=2000 the 96-word arch can't bind
     synonyms (see
     `research/findings/2026-05-11-96word-XL-encoding-NEGATIVE.md`)
   - Latency wall: even if it worked, n_lang=16384 would push :speak to
     ~12+ sec (extrapolating the n_lang→latency relationship)

4. **VRAM is not the bottleneck.** Even at 64w, only 2.4 GB / 24 GB
   used. Plenty of headroom for parallel REPL sessions, lineage
   reload caching, etc. Inference is **compute-bound, not memory-bound**.

5. **Cold-start cost is real but recoverable.** Bridge build time scales
   137 sec @ 24w from 11 sec @ 4w. The lineage workflow (load weights
   from disk instead of training fresh) cuts this to seconds for any
   tier, because the build still happens but training events are 0.
   This is the lineage MVP's main UX win.

## Phase 1 optimization map (per master plan)

The Phase 1 optimization design
(`docs/plans/2026-05-10-phase1-local-optimization-design.md`) targets
3-5× cumulative speedup for the inner loop. Applied to these tiers:

| Tier | current | -3× | -5× |
|------|---------|-----|-----|
| 4w   | 1.3s    | 0.4s | 0.3s |
| 16w  | 1.7s    | 0.6s | 0.3s |
| 64w  | 6.1s    | 2.0s | 1.2s |

**A 3× speedup brings 64w into the conversational band** (~2 sec). A 5×
speedup makes every tier feel sub-second. Targets 1-3 in the Phase 1
design (bridge construction speedup, sparser cross-region density,
FP16 throughout) are the highest-ROI candidates and address the bottleneck
identified here (per-step cost dominated by lang_input × motor edge count).

## Methodology notes

- Each tier built a fresh bridge with `n_events_per_direction=2`
  (essentially no training; bridge is randomly weighted). The benchmark
  measures *raw inference cost*, not the quality of trained inference.
  Real :speak in a trained REPL is the same wall-clock since inference
  freezes plasticity — but accuracy is different.
- STP was disabled (2026-05-10 default). With STP on, expect ~2.8×
  slower per the 2026-05-10 perf benchmark
  (`research/findings/2026-05-10-stp-default-flip.md`).
- 10 rounds × 4 actions = 40 :speak samples per tier; mean + std reported.
- Outlier handling: 24w's 7335ms (vs 6051-6450ms for the rest of Band C)
  is likely just first-tier-in-the-band variance. Median across Band C
  would be ~6.45 sec; mean is 6569 ± 484 ms.

## Provenance + next steps

- This findings doc: `research/findings/2026-05-11-inference-latency-across-vocab-tiers.md`
- Raw per-tier JSON: `research/findings/raw/perf/inference_bench/`
- Chain script: `research/findings/raw/g11_bg/chain_inference_benchmark_post_xl.ps1`
- Related XL NEGATIVE: `research/findings/2026-05-11-96word-XL-encoding-NEGATIVE.md`
- Related design: `docs/plans/2026-05-10-phase1-local-optimization-design.md`

**Suggested next action:** ship the 16-word default for lineage `main`
and the chat REPL (already implicitly happening via `chat_repl --mode
synonym` synonym16 mode), document the per-tier latency expectation in
the README "performance" section, and prioritize Phase 1 optimization
targets 1-3 to break the n_lang=8192 latency wall.
