# 2026-05-24 overnight primitives synthesis -- three biology-grounded mechanisms algebra-validated

**Date:** 2026-05-24 (work continuing through morning + day)
**Status:** Synthesis of completed pieces; Direction A substrate result still in flight
**Frozen bars:** 0.80 multi-seed; NEVER tuned
**No-confab moat:** 7/7 green throughout
**Protected set:** byte-empty diff e8a99a2..HEAD

## Owner-directed autonomous continuous-work pattern

User mandated continuous autonomous overnight work towards conversational
capabilities while staying biology-grounded. Per the standing discipline
(no idle waiting; no promise-stalls; falsify-cheaply-first; honest
propagation every outcome to both remotes).

## What's been validated today

### Pillar n=103: Direction E theta-gamma ALGEBRA VALIDATED

Lisman-Idiart theta-gamma multiplexing (catalog N.16) -- the catalog-
flagged load-bearing primitive for sequence storage -- tested in numpy
algebra. Multi-seed (42/43/44), 300 trials/load, N_DIM=256, N_VOCAB=16,
PHASE_NOISE_STD=0.05 (matches resonate-and-fire biologization probe).

| Load | Mean acc | Per-seed |
|------|----------|----------|
| 2    | 1.000    | [1.000, 1.000, 1.000] |
| 3    | 1.000    | [1.000, 1.000, 1.000] |
| 5    | 1.000    | [1.000, 1.000, 1.000] |
| 7    | 1.000    | [1.000, 1.000, 1.000] (N_GAMMA cap) |

**Adversarial review CLEAR** (independent fresh-agent reviewer; RAN
every probe + 14 exploit-class checks):
- Permutation control 0.198 == 1/LOAD chance 0.200 (decoder uses
  slot phase, not pattern recognition alone)
- No-window control 0.193 == 1/LOAD chance 0.200 (slot windowing
  load-bearing)
- High-overlap (measured 0.32) holds at 1.000 (overlap-robust)
- Noise stress through sigma=5.0 (100x biological) at active_frac=0.05

Capability_status pillar recorded; substrate biologization design
written (docs/plans/2026-05-24-direction-E-theta-gamma-substrate-
design.md); substrate clock implementation simplified via pirazzini
step-index phase pattern reuse (~half-day build instead of original
~2 days).

### Direction F cross-bridge: precise abstention bound + familiarity-gate fix

Three cheap-first numpy probes (5 sec total wall):

| Variant | Test I (abstention) | Test II (discrim) | Verdict |
|---------|---------------------|-------------------|---------|
| Trivial (no interference) | 1.000 | 1.000 | Uninformative |
| Realistic (shared substrate; cosine-threshold) | **0.712** | 0.996 | ABSTENTION_FAILS |
| Familiarity-gate fix | **0.999** | 0.996 | RESOLVED |

**Generalizable biology-translatable insight:** abstention always
requires a SEPARATE familiarity / match-strength signal (perirhinal
cortex novelty + hippocampus + locus coeruleus norepinephrine), never
a single threshold on the identification score. Same principle that
RESOLVED FHRR shortcut-3 (2026-05-22). Applies to ANY cross-region
or cross-bridge composition.

### Direction E+F INTEGRATED algebra probe = PASS at G.20 "age-5" scale

Combines theta-gamma multiplexing + cross-bridge composition + familiarity
gate. At 160-concept vocab (5 bridges x 32 concepts each), 5-slot
sequence, 2-bridges-per-slot interference:

| Metric | Mean | Per-seed |
|--------|------|----------|
| Active concept-ID | 0.997 | [0.992, 1.000, 1.000] |
| Inactive abstain | 1.000 | [1.000, 1.000, 1.000] |
| Combined | 0.999 | [0.997, 1.000, 1.000] |

The complete conversational-primitive algebra (sequence storage +
cross-bridge composition + selective abstention) is sufficient at the
G.20 "age-5" target.

### Direction A ec_context (in flight): seed 42 single-seed PASS

Substrate test of ec_context-based sequence storage (project catalog
D.01+D.02+D.11) closing the (c) generative-replay arc's REPLAY_DOESNT_
REACTIVATE bound. CRITICAL BUG caught in flight and corrected per
falsify-cheaply-first (encoding-smoke verified before relaunch). Seed
42 result: 0.875 (7/8) ABOVE the 0.80 bar.

Per-sequence breakdown:
- 7/8 sequences correctly retrieved slot-3 word in top-3
- 1 failure: ['small','big','north'] -> top-3 ['small','big','west']
  (positional cue selected other in-sequence words instead of slot-3)

Multi-seed (43, 44) still in flight; full result + smell test + dedicated
adversarial review pending. The runner is being reviewed in parallel
by a fresh-agent adversarial reviewer (background task).

## How the three primitives compose (the standing target)

```
        +-----------+ +-----------+ +-----------+
        | ec_context| | theta-    | |  cross-   |
        | spatial   | | gamma     | |  bridge   |
        | position  | | temporal  | |  with     |
        | code      | | phase     | |  fam      |
        | (D.01-11) | | (N.16)    | |  gate     |
        +-----------+ +-----------+ +-----------+
              |             |             |
              v             v             v
        +-----------------------------------------+
        |  Substrate sequence-storage primitive    |
        |  capable of K-slot multi-bridge          |
        |  conversational retrieval                |
        +-----------------------------------------+
              |
              v
        +-----------------------------------------+
        |  Working sim chat REPL with sequence     |
        |  understanding (sentences not just      |
        |  one-word retrieval)                     |
        +-----------------------------------------+
```

The algebra clears the bar for the integrated primitive at 160-concept
vocab. The substrate implementations of each primitive (ec_context tested
NOW in flight; theta-gamma queued; cross-bridge familiarity-gate
demonstrated in algebra) are independent next-direction work paths.

## Honest scope

- Pillar n=103 is ALGEBRA only; substrate biologization is the next
  decisive step that turns this into a substrate-validated capability
- Direction A is substrate; single-seed PASS at 0.875 is preliminary
  (multi-seed pending; smell test + adversarial review required)
- Direction F is algebra; substrate cross-bridge composition is a
  separate build (the algebra demonstrates the mechanism is sound)
- Integrated E+F is algebra; full substrate integration is a larger
  build (queued after each independent primitive is substrate-validated)

This is NOT fluent open-ended language, NOT an LLM, NOT a complete
conversational sim. It IS a clean propagation of three biology-grounded
mechanisms validated at the algebra level + one in-flight substrate test
of the spatial code primitive, all under the project's strict scientific
discipline.

## What's next (autonomous chain)

1. **Direction A multi-seed completion** (ETA ~15:30 EDT 2026-05-24)
   - If >= 0.80 multi-seed: smell test (3 anti-cheat controls) +
     dedicated fresh-agent adversarial review + pillar n=104 if all
     clear + capacity sweep (slot_count 3..7)
   - If < 0.80: characterize the bound + pivot to theta-gamma substrate
     biologization (Direction E Task 1+)

2. **Direction E substrate biologization** (queued; ~half-day build)
   - Step-index phase pattern from pirazzini (reuse byte-equivalent)
   - NeuromodulatorConfig synaptic_gain for per-slot drive gating
   - Smoke -> Task 4 multi-seed decisive run -> review -> pillar

3. **Cross-bridge substrate composition** (queued; deferred until
   single-bridge substrate primitives are validated)

4. **Chat REPL integration** (deferred; combines substrate sequence
   storage + cross-bridge composition into user-facing sim chat with
   sentence understanding)

Standing autonomy: continuous work; no idle waiting; falsify-cheaply
first; reuse-DRY; ASCII; both-remote propagation; no hand-back; no
declare-unfit.
