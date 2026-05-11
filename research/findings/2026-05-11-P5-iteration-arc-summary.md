# P5 ventral semantic stream — autonomous iteration arc summary

**Date:** 2026-05-11
**Context:** Autonomous overnight arc per user mandate. Goal:
make sim conversational by closing the P5 word↔meaning gap.

## Headline (honest)

Four P5 iterations completed at seed 42. ALL FAIL. The architecture
has a hard floor at apple_self_cosine ~0.22 (target >0.5).
Diagnosis is increasingly clear: semantic_cortex lacks attractor
dynamics in the current parameter space.

| Iteration | Change | apple_self | naming | Wall |
|---|---|---|---|---|
| Original (3 seeds) | default params, raw cosine | 0.216-0.347 | 0.88-1.07x | 8-15 min |
| Iter A | engram-tag methodology + 300 events | 0.227 | 1.08x | 5 min |
| Iter B | + strict two-stage gating + lang drive replay | 0.226 | 1.08x | 5 min |
| Iter C | + scale wernicke 2x + semantic_cortex 2x | 0.207 (WORSE) | 0.99x | 6 min |
| Iter D | + attractor tuning (density 0.25, weight 4.0) | RUNNING | RUNNING | ~10 min ETA |

Per superpowers:systematic-debugging Phase 4.5 iron law:
"if 3+ fixes fail, question architecture." We're at attempt 4.

## What we learned

1. **Test methodology matters** (iter A). Engram-tag (P1 D.13
   pattern) gives cleaner signal than raw spike-count cosine.
   First time same-concept > cross-concept consistently.

2. **Strict gate timing doesn't matter** (iter B). McClelland
   1995 CLS strict wake/sleep separation didn't move the
   needle. Iter A vs iter B numbers identical to within 0.001.

3. **Size DOES NOT matter** (iter C). Scaling wernicke 200→400
   and semantic_cortex 500→1000 made things WORSE. This rules
   out the wernicke bottleneck hypothesis. More neurons = more
   noise without proper attractor formation.

4. **Pattern is consistent, not seed variance**. apple_self
   stays at 0.22 ± 0.02 across iter A, B, C — same seed, same
   training, same test methodology. The architecture has a
   structural ceiling.

## Diagnosis

semantic_cortex doesn't form stable point attractors. The
recurrent connectivity (density 0.10, weight 1.0) is 3-4x WEAKER
than the feedforward input (lang→wernicke weight 3.0,
wernicke→semantic weight 4.0). Every new drive overwrites the
existing pattern — there's no basin-of-attraction to "snap back
to" the trained ensemble.

Biology source: Wang 2002 PFC NMDA bistability requires recurrent
weights COMPARABLE TO or STRONGER than feedforward. Real cortex
has ~20-30% recurrent connectivity. Patterson 2007 ATL hub theory
similarly requires attractor dynamics.

## What's working (validated multi-seed)

| Capability | Multi-seed | Catalog | Wall |
|---|---|---|---|
| P1 trisynaptic loop | 3/3 BIOLOGY PASS | D.03+D.12+D.13 | ~3 min/seed |
| P2 engram tagging API | 12 unit tests pass | D.14 | API |
| P3.1 concept replay | 5 unit tests pass | D.19 | API |
| P4.1 positional binding | 3/3 PASS | D.01+D.02+D.11 | ~5 min/seed |

These four are real, biology-grounded wins. The architectural
substrate works.

## What's NOT working (this arc)

| Capability | Status |
|---|---|
| P5 ventral semantic (apple/river) | 0/4 iterations at seed 42 |
| Liu 2012 causal recall (4-direction) | 0/3 multi-seed — test methodology issue |

The Liu 2012 result is a test methodology issue (single-word
paired-stim doesn't overcome seed-specific structural bias), NOT
a P2 API failure. P2 is independently validated.

The P5 result is a real architectural limitation.

## Three paths forward (decision pending iter D)

**Path A: iter D + iter E (in-progress)**
- iter D: attractor tuning (recurrent_density 0.25,
  recurrent_weight 4.0, drive_steps 300) — RUNNING
- iter E: re-launch with weight-inspection diagnostic (just
  shipped on bac3f26) — measures if STDP actually learned
  the binding, separately from whether dynamics produce
  attractors. Distinguishes "trained but noisy dynamics" from
  "didn't train at all".

**Path B: pragmatic pivot to multi-pool semantic**
- Replicate Tier 1 architecture (which PASSED multi-seed for
  motor pools 5/6 at 4-word, 5/6 at 8-word) for abstract
  concept pools
- Each concept gets its own 500-neuron pool with FS lateral
  inhibition
- Trains via contrastive 4+ concept paired-drive
- Doesn't match ATL hub theory but matches what works

**Path C: hybrid**
- Keep P5 substrate, train using Tier 1 paradigm at 4-direction
  level. Tests whether the substrate works under contrastive
  multi-concept training.

If iter D PASSES at seed 42, Path A continues (launch 43/44).
If iter D FAILS, run iter E next (weight inspection) — if
weights look right but dynamics noisy, dynamics need MORE
attractor work. If weights also flat, training paradigm needs
contrastive multi-concept (Path C or B).

## Strategic context

The user's goal is conversational sim. Tier 1 (2026-05-06)
ALREADY achieves bidirectional word↔motor binding at 4-word
vocab (5/6 seeds). Tier 2.1 (2026-05-06) extends to 8-word
synonym vocab (5/6, A→W 6/6, mean 64%). These ARE conversational
capabilities for direction-word commands.

P5 was supposed to ADD non-motor concepts (apple, river — not
tied to directions). That's the architectural gap. The arc has
clarified that the ATL-hub-style scaffolding doesn't work
out-of-the-box; either dynamics need fundamental rework (iter
D/E and beyond) OR a different scaffolding (Path B multi-pool).

## Code shipped this arc

- `research/runners/validate_ventral_semantic.py`: 5 CLI flags
  (--strict-two-stage, --drive-lang-during-replay,
  --semantic-cortex-recurrent-density/weight, --drive-steps,
  --lang-to-wernicke-weight, --wernicke-to-semantic-weight)
- `research/runners/aggregate_ventral_semantic_seeds.py`: P5
  multi-seed aggregator (supports any iter prefix)
- `research/runners/aggregate_causal_recall_seeds.py`: Liu 2012
  multi-seed aggregator
- Weight inspection diagnostic: `weight_diagnostics` field in
  all P5 results going forward
- 5 findings docs (iter A FAIL, iter B FAIL, iter C FAIL, Liu
  2012 multi-seed, this summary)
- Liu 2012 Unicode arrow fix

## Wall clock budget

This arc: ~2 hours of autonomous work
- 4 P5 iterations × 5-6 min/seed = ~25 min compute
- 3 Liu 2012 seeds × 2 min = 6 min compute
- Total compute: ~30 min
- Rest: code, docs, diagnostics

Per the autonomous-runs principle: hardware-bound estimates
are reliable, plan-time estimates are NOT. So this is faster
than expected on compute.
