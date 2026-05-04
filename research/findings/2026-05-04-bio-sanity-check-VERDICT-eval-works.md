# Bio-scale eval sanity check — VERDICT: eval works

**Date:** 2026-05-04 13:16 EDT
**Chain:** Stage 1 of 2026-05-04 biological-scale test plan
**Source:** `research/findings/raw/g11_bg/text_eval_sanity_check_bio_*_seed*.json` (24 runs)
**Aggregated to:** `research/findings/2026-05-04-bio-sanity-check-results.md`

---

## TL;DR

Hand-built perfect language→motor weights at biological scale (lang=2048,
motor=500/action, recurrent E + E/I balance + NMDA bistability) give
aligned ≥ 4/6 seeds in BOTH `density 0.30` AND `density 1.0` perfect
conditions. The eval methodology was never broken.

**The 2026-05-04 minimal-arch B1 finding ("eval is broken") was actually
"architecture too minimal".** Stripping the cortical canon (recurrence,
E/I balance, NMDA) along with the cascade left the motor pools unable
to hold a representation of which pool to fire, even with hand-coded
perfect weights. Adding back the canon at biological scale restores the
eval to working order.

## Implication for the broader investigation

The 18-day W→A 0/N alignment streak — across v2 architecture variants,
SWR replay, fundamentals sweep, biology sweep — may be entirely a
small-scale artifact. Every prior W→A eval used motor pools without
recurrence + NMDA. Population coding requires populations; signal
amplification requires recurrence; sustained representations require
NMDA bistability. None of those were present.

The PoC stage (auto-fired at 13:16:52, PID 15708) tests whether full
STDP training can find the right weights at biological scale — i.e.,
whether the architecture *learns* the mapping when given correct
biology. 12 runs at parallel=3 = ~5 hours. Two conditions:
- `bio_baseline`: cortical canon alone, no biology fix
- `bio_topo_fs`: cortical canon + Pulvermüller topographic prior +
  Vogels PV-FSI

## What the sanity check measured

Six seeds × four modes = 24 runs at biological scale, no training,
hand-built weights, single-process (parallelism=1).

Conditions that aligned ≥ 4/6 seeds:
- `bio perfect, density 0.30` — perfect weights at standard density
- `bio perfect, density 1.0` — perfect weights at full connectivity

Control conditions (expected NOT to align):
- `bio wrong-mapping`: rotated weights — eval correctly rejects (no
  alignment with TRUE labels)
- `bio random weights`: U[0, 8.0] — chance accuracy

Memory peak: ~2.4 GB GPU per single-process run (vs 22 GB headroom on
RTX 3090). Runtime: 24 runs × ~2.5 min = ~60 min total.

## Compared to minimal-arch B1 (2026-05-04 ~10:00 EDT)

| Test | Architecture | Perfect mode aligned |
|---|---|---|
| Minimal B1 | lang=256, motor=25, no recurrence, no NMDA | 0-1/6 |
| Bio sanity | lang=2048, motor=500, recurrence + E/I + NMDA | ≥ 4/6 (BOTH densities) |

Same `evaluate_word_to_action` code. Same eval pipeline. Same hand-built
weights logic (just at larger N). Different result entirely.

## What's running next

`bio_proof_of_concept` at parallel=3 / 6 seeds = 12 runs.
- Started: 13:16:52
- ETA: ~18:30 EDT
- Auto-aggregator + autonomous decision (B3 fallback if STDP fails)
  via `scripts/orchestrate_bio_post_sanity.ps1` (PID 9832).

## Lessons for future experimentation

1. **Always test the minimum biological scale, not below it.** The
   "minimal architecture" was minimal in a NON-biological way (no
   recurrence, no E/I, no NMDA). It should have been "minimal cascade,
   maximal canon" instead. Future "minimum" architectures should keep
   cortical canon and only strip the cascade-specific structures.

2. **Sanity-check the eval at biological scale, not at investigative
   scale.** All prior W→A investigations used motor pools without
   recurrence. The bio_sanity_check confirms the eval works when given
   biological dynamics — meaning all those prior 0/N alignment results
   need re-evaluation at biological scale before they can be trusted as
   "the architecture/learning rule failed."

3. **Hardware capacity is real.** Bio scale uses ~2.4 GB / 24 GB VRAM —
   4× larger than minimum scale, still 10x of headroom. Future
   experiments should default to biologically-faithful scales since
   the hardware permits it.
