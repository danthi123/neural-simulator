---
title: "Generative replay BEATS the bounded buffer (0.69 vs 0.52) but doesn't match flat (0.95) — the GENERATOR ITSELF forgets, and it bounds STORAGE not per-step COMPUTE"
date: 2026-08-09
type: finding
status: contributing
lane: memory-continual-learning
seeds: [42, 43, 44, 45, 46, 47]
---

# Generative replay: the right direction, two honest residuals (the generator's own forgetting + compute still O(N))

## Claim

<!--derived-->

Following the CLS bounded-buffer NEGATIVE (`0c7531785`: a raw-engram buffer + eviction bounds replay COVERAGE to F
and forgets evicted facts), the biological/SOTA fix — a **FIXED-SIZE neural generator that re-dreams ALL learned
facts** (van de Ven 2020) — was built and tested. It **works in the right direction but is a PARTIAL**: generative
replay (0.692) clearly **BEATS the bounded buffer (0.517, +0.175)** — regenerating all facts is better than storing
a recent window — but does **NOT match the flat O(N) store (0.950)**. Two honest residuals: **(1) the generator
ITSELF forgets** (regeneration fidelity degrades 1.00→0.80–0.90 as N grows, which the shared slow readout
amplifies); **(2) it bounds STORAGE but NOT per-step COMPUTE** (still regenerates + replays O(N) facts per sleep).

## Data (N=20, 6-seed, byte-identical, verify-confirmed)

<!--derived-->

| arm | N=20 retention | note |
|---|---|---|
| flat (O(N) replay) | 0.950 | the target (unbounded store) |
| **generative (fixed generator)** | **0.692** | per-seed 0.55/0.80/0.80/0.90/0.70/0.40 |
| bounded_buffer (F=5) | 0.517 | the CLS negative it must beat |

Generative degrades 1.00@N=10 → 0.692@N=20 (still tracks N). The generator's trained store is **1344 floats,
CONSTANT in N** (asserted; vs the raw buffer's O(N)), and stores **0 raw patterns** (regeneration is a spiking
forward pass, not a lookup — verified). Raw: `research/findings/raw/fm_generative_replay_decisive.json`.

## Read — the answer to "does a year of data scale?", honestly and in layers

<!--derived-->

- **STORAGE can be bounded.** The fixed generator (1344 floats) regenerates all N facts vs the O(N) raw buffer.
  This is real and verified (0 stored patterns; genuinely generative; fixed-size across N and across two builds).
- **But RETENTION is not yet solved, because the generator itself forgets.** generative 0.692 « flat 0.950. The
  −0.258 gap traces to regeneration fidelity falling 1.00→0.80–0.90 as N grows (capacity is NOT the issue — H_gen=96
  ≫ N=20). This is the RECURSION the de-risk was built to test, and it FOUND it: bounding the store just moves the
  catastrophic-forgetting problem *into* the generator.
- **And per-step COMPUTE is NOT bounded.** Even with a fixed store, the generative arm still regenerates + replays
  O(N) patterns every sleep — sleep-consolidation cost still grows with lifetime. So this does not, by itself,
  answer the owner's speed concern; it addresses storage growth only.

**Net:** a year of data does NOT yet scale with bounded cost on this substrate — but the two failures precisely name
the two remaining mechanisms, both of which biology uses: **(a) a non-forgetting generator** (generative replay
"all the way down", or consolidating the generator into a stable cortical model), and **(b) sparse / prioritized
replay** (the brain does not replay every memory every night — recency/salience/schema-gated replay bounds
per-sleep compute). Neither is a capability wall; both are named next levers.

## Rigor / anti-cheats (build + adversarial verify, CONFIRMED)

- generative BEATS buffer and is below flat, re-run byte-identical (verify, maxdiff 0.0); NOT a single-seed claim.
- Generator GENUINELY fixed-size (trained-param count constant across N=10/N=20 and across two builds) and GENUINELY
  generative (`generator_stored_raw_patterns==0`; regeneration is a spiking Izhikevich reservoir + synaptic readout
  via a local delta rule, not a host lookup / not an O(N) pattern buffer in disguise).
- flat + bounded_buffer MEASURED in-run on the SAME reservoir/seed/env (not imported). cfg.seed byte-identical;
  de-clamped bdsp_wmax=1e9; no `sim/` edit; backend recorded.
- Declared honest scaffolding (scrutinized): the class query is a deterministic sparse code of the class INDEX
  (regenerated from the index, not stored); a fixed anchor (0.5, a world constant) supplies the bias dof. Neither
  stores per-fact information.

## Next levers (named, not deferred)

1. **The generator's own recursion** (load-bearing): keep the fixed generator current without forgetting — van de
   Ven self-replay of its OWN regenerations (partially implemented; strengthen), or consolidate the generator into
   the stable slow cortex so regeneration reads a non-drifting model.
2. **Sparse / prioritized replay** to bound per-sleep COMPUTE (recency/salience/schema-gated), so cost decouples
   from lifetime even though coverage stays complete.

NO-EXTERNAL-NEEDED: <!--derived--> van de Ven 2020 (doi:10.1038/s41467-020-17866-2) is the recorded external grounding; sparse/
prioritized replay (Mattar-Daw prioritized replay; schema-gated Tse) is the newly-named companion mechanism.
