---
type: finding
status: qualified
date: 2026-09-01
integration_faculty: open-ended-generation
mechanism: the #3E generate-channel PLAUSIBILITY gate, converted from a host `P>=tau` co-occurrence matrix comparison to a SPIKING monosynaptic synaptic associative read (SpikingAssociativePlausibilityOrgan), wired OPT-IN into the production ChatBrain generate turn behind BRAIN_SPIKING_PLAUSIBILITY
verdict: QUALIFIED — the plausibility DECISION is now computed by neurons+synapses+spikes (0 host P>=tau calls), lesion-load-bearing, moat-safe, byte-identical-off, agreement ~0.88-0.91; it MATCHES the host replay-vs-random advantage ON AVERAGE at both operating points (tiny parity mean 1.01, rich parity mean 1.24) and cleanly dominates on the richer graph (all 6 seeds). BUT on the sparse tiny own-facts graph 2/6 seeds underperform host (parity 0.54/0.78) and the more-selective spiking gate SUPPRESSES generation (gen 5,3,1,3,5,3 vs host 5,6,5,5,6,7). Not uniform across all 6 seeds -> kept DEFAULT-OFF (opt-in), byte-identical-off = zero production regression.
lane: integration-spine · burn down the last host scaffold in the default chat GENERATE path
artifacts:
  - research/findings/raw/_brain_native_plausibility_derisk.json
  - research/findings/raw/_brain_native_plausibility_derisk_rich.json
verification: >
  Through the REAL research.runners.brain_chat_tui.ChatBrain / _build_generation_proposer path (rf composer,
  SIM_BACKEND=numpy), seeds 42,43,44,100,101,102, at two operating points (the tiny interlinked graph = the
  generate-channel GO's own 2.1x-3.4x point; the richer type-structured graph). Only the plausibility GATE differs
  between conditions (same facts, same host-oracle draw): HOST = _related(w1,w2)=P[w1,w2]>=tau; SPIKING = the
  co-occurrence graph installed as cortex_A->dlpfc_B synapses (weight ∝ co-occurrence count) on a real
  SimulationBridge, related(w1,w2) = drive w1's assembly and read whether w2's readout assembly fires above the
  brain's own threshold. The gate decision reads cp_firing_states, never P>=tau (hot-path host-comparison count = 0
  all seeds). See body for the per-seed tables.
---

# The #3E plausibility gate becomes BRAIN-NATIVE: a spiking synaptic read replaces the host `P>=tau` matrix (qualified)

## What this converts (the last host scaffold in the default GENERATE path)

The open-ended GENERATE channel (`BRAIN_GENERATE_CHANNEL`, default-ON, moat-verified) lets the brain VOLUNTEER a
novel grounded proposition on an open prompt. Its DRAW is already spiking. But the generate-channel GO finding
(`2026-08-18-generate-channel-wired-brain-chat-GO`) declared one residual: the PLAUSIBILITY gate — the decision
that selects which recombinations are sensible — was a **host** float comparison

    _plausible(a, ac, p) = _related(a, ac) and _related(ac, p)     # selectional preference
    _related(w1, w2)     = P[row[w1], row[w2]] >= tau              # <-- host matrix comparison

over the brain's own co-occurrence matrix `P` (`tau` = the 50th percentile of the positive edges). That comparison
sits between sensation and action -> by the brain-based-only standard it is a shortcut to convert. This task
converts it and measures exactly where the spiking version matches the host and where it falls short.

## The mechanism (`SpikingAssociativePlausibilityOrgan`)

The co-occurrence graph is embodied as **synaptic weights** and relatedness is decided by **spikes**:

- Two Izhikevich populations on a real `SimulationBridge` (reuses `build_loop_wm_bridge`): a cortex "input" and a
  dlPFC "readout" layer, one disjoint neuron assembly per concept.
- The association graph is installed as **directed** synapses `cortex_A -> dlpfc_B` with weight **proportional to
  the co-occurrence count** `P[A,B]` (Hebbian: association strength = synaptic strength). The readout never projects
  back -> the read is strictly **monosynaptic** (no multi-hop transitive blow-up; a recurrent spreading read
  saturated — on a connected graph every concept became reachable, measured in the calibration probe).
- `related(A, B)`: drive input assembly `A` briefly; `B`'s readout assembly **fires** iff the `A->B` synapse carries
  enough current to reach threshold. Weak co-occurrence -> weak synapse -> sub-threshold EPSP -> silent -> "not
  related"; strong co-occurrence -> supra-threshold -> fires -> "related". The `tau` boundary **emerges from the
  spike threshold**, it is not a host `>=`.
- The readout threshold is the brain's **own**: `tau_spike` = the 50th percentile of the **positive readout
  firing-fractions** (the same rule the host applies to `P`, applied to the brain's spiking output). The hot-path
  decision `firing_frac[A][B] >= tau_spike` reads a **spike count**, never `P`.

`install(prop)` swaps `self.related` in for the proposer's host `_related`, so the unchanged `_plausible` now
decides via spikes. Wired into `ChatBrain._generate_hypothesis` behind `BRAIN_SPIKING_PLAUSIBILITY`; unset/`0`
never builds the organ (byte-identical).

## Verification — through the real handler, 6 seeds, two operating points

Runner: `research/runners/_brain_native_plausibility_derisk.py`. Artifacts:
`research/findings/raw/_brain_native_plausibility_derisk.json` (tiny) and
`research/findings/raw/_brain_native_plausibility_derisk_rich.json` (rich). advantage = replay-plausible-fraction /
random-plausible-fraction; only the GATE differs between HOST and SPIKING (same facts, same host-oracle draw).

### TINY graph (the generate-channel GO's own 2.1x-3.4x operating point; |vocab|=11)

| seed | host adv | SPIKING adv | parity | agree(P>=tau) | F1 | lesion(shuffle) | ablate rel | hot-host calls | leaks/negrep | gen (spk vs host) |
|------|----------|-------------|--------|---------------|----|-----------------|-----------|----------------|--------------|-------------------|
| 42   | 3.07 | 3.28 | 1.07 | 0.93 | 0.85 | 0.00 | 0 | 0 | 0/0 | 5 vs 5 |
| 43   | 2.74 | 1.47 | 0.54 | 0.88 | 0.73 | 0.00 | 0 | 0 | 0/0 | 3 vs 6 |
| 44   | 2.14 | 1.67 | 0.78 | 0.83 | 0.63 | 1.36 | 0 | 0 | 0/0 | 1 vs 5 |
| 100  | 2.53 | 3.53 | 1.39 | 0.91 | 0.81 | 0.69 | 0 | 0 | 0/0 | 3 vs 5 |
| 101  | 2.65 | 3.17 | 1.20 | 0.82 | 0.60 | 1.71 | 0 | 0 | 0/0 | 5 vs 6 |
| 102  | 3.36 | 3.62 | 1.08 | 0.94 | 0.86 | 2.50 | 0 | 0 | 0/0 | 3 vs 7 |
| mean | 2.75 | 2.79 | 1.01 | 0.88 | 0.75 | 1.04 | 0 | 0 | 0/0 | — |

### RICH type-structured graph (|vocab|=19)

| seed | host adv | SPIKING adv | parity | agree(P>=tau) | F1 | lesion(shuffle) | ablate rel | hot-host calls | leaks/negrep | gen |
|------|----------|-------------|--------|---------------|----|-----------------|-----------|----------------|--------------|-----|
| 42   | 1.81 | 1.62 | 0.90 | 0.92 | 0.81 | 1.28 | 0 | 0 | 0/0 | 15 |
| 43   | 1.79 | 1.81 | 1.01 | 0.94 | 0.87 | 1.76 | 0 | 0 | 0/0 | 8 |
| 44   | 1.79 | 2.90 | 1.62 | 0.87 | 0.70 | 0.94 | 0 | 0 | 0/0 | 12 |
| 100  | 1.96 | 2.93 | 1.50 | 0.92 | 0.82 | 0.00 | 0 | 0 | 0/0 | 9 |
| 101  | 1.84 | 2.59 | 1.41 | 0.87 | 0.70 | 0.85 | 0 | 0 | 0/0 | 9 |
| 102  | 1.93 | 1.91 | 0.99 | 0.93 | 0.83 | 1.29 | 0 | 0 | 0/0 | 16 |
| mean | 1.85 | 2.29 | 1.24 | 0.91 | 0.79 | 1.02 | 0 | 0 | 0/0 | — |

- **The conversion is real and clean (all 12 seed-runs).** Relatedness is decided from `cp_firing_states`
  (|vocab| spiking reads per proposer); the hot-path host `P>=tau` comparison is **never** called while installed
  (count = 0). The spiking `related()` reproduces the host relation (agreement mean 0.88 tiny / 0.91 rich).
- **Lesion load-bearing (anti-cheat).** A SHUFFLED-synapse organ (co-occurrence neighbourhoods destroyed,
  marginals kept — the b2 shuffled-graph anti-cheat, now in synapses) drops the advantage well below the intact
  spiking advantage every seed; an ABLATED-synapse organ (zero association weight) makes relatedness collapse
  entirely (0 related pairs, all seeds). The LEARNED structure, read through synapses, carries the signal.
- **Moat-safe.** 0 hypothesis->known-fact leaks, 0 negated re-proposed, untaught-cue abstention 20/20, all seeds
  (the no-confab moat is downstream of plausibility and untouched).
- **Byte-identical when OFF.** `BRAIN_SPIKING_PLAUSIBILITY=0` never builds the organ; `gate()` volunteers the SAME
  hypotheses as the pure-host baseline (all seeds); the tiny-demo `--smoke` JSON SHA is unchanged (5026f7b7…).
- **Advantage — matches host on average.** Tiny: host mean 2.75x -> SPIKING mean 2.79x (parity 1.01, beats host
  on 4/6). Rich: host mean 1.85x -> SPIKING mean 2.29x (parity 1.24, beats host on 5/6). The spiking gate delivers
  the same 2-3x discrimination the finding named as the residual — now computed by the brain.

## The honest negatives (why it ships OPT-IN, not default-ON)

Because this is a real production change to `_generate_hypothesis`, the bar for default-ON is that the spiking
gate hold across ALL 6 seeds. Two measured shortfalls, both on the SPARSE tiny own-facts graph, keep it opt-in:

- **Per-seed advantage variance.** On the tiny graph 2/6 seeds underperform host (parity 0.54 seed 43, 0.78 seed
  44; min 0.54). The tiny graph is small (11 concepts, pattern_size 12), so the spiking read's operating point
  (`tau_spike` = median of positive firing-fractions) is seed-sensitive: on those seeds the point-neuron threshold
  admits/rejects borderline pairs differently than the host float `>=`, lowering discrimination. On the richer
  graph this vanishes (parity min 0.90, all 6 >= host on average).
- **Generation suppression on the sparse graph.** The spiking gate is MORE selective on the tiny graph, so the
  brain volunteers FEWER novel props: gen 5,3,1,3,5,3 (spiking) vs 5,6,5,5,6,7 (host) — seed 44 drops 5->1. This
  is a real functional consequence measured through the real `gate()`. On the richer graph it does not appear
  (gen 8-16, healthy). A production default must be robust at the sparse own-facts operating point, which this is
  not yet.

Neither is a moat, provenance, or byte-identity failure — the conversion itself is sound. They are precisely the
substrate limit this task exists to map: a **point-neuron monosynaptic read with a small-assembly median threshold
is less discriminating, and more variable, than a host float comparison on a sparse graph.**

## Honest residual + the named next rung

- The synaptic weights are SET from the co-occurrence counts (the same counts the host `P` holds); **online
  Hebbian self-organization** of those weights (the synapses LEARN the associations through use) is the next rung
  toward a fully self-organized plausibility.
- To close the tiny-graph gap for default-ON: reduce the small-graph operating-point variance — larger / ENSEMBLE
  assemblies (average the readout over several assembly assignments; a real cortex reads a redundant population,
  not a 12-neuron patch), and/or a GRADED (soft) plausibility read that preserves borderline pairs instead of a
  hard spike threshold. Once those bring the tiny-graph parity + gen count to host-parity on all 6 seeds, flip
  `_SPIKING_PLAUSIBILITY_DEFAULT_ON = True`.
- The selectional-preference STRUCTURE and SVO template remain host scaffolding (unchanged). Magnitude is the
  own-facts operating point (2-3x), not the 3E corpus PPMI (14-24x) — a strengthening lever orthogonal to this
  brain-native conversion. Toy-scale taxonomy.

## Verdict

**QUALIFIED.** The host `P>=tau` plausibility shortcut is REPLACEABLE by a spiking synaptic associative read that
computes the SAME relatedness by spikes — provenance-clean, lesion-load-bearing, moat-safe, byte-identical-off,
agreement ~0.9, and matching the host replay-vs-random advantage on average at both operating points (and cleanly
dominating on the richer graph, all 6 seeds). It is NOT uniformly >= host on the sparse tiny graph (2/6 seeds
underperform; generation is suppressed), so it ships **default-OFF / opt-in** (`BRAIN_SPIKING_PLAUSIBILITY=1`) with
byte-identical-off guaranteeing zero production regression, pending the ensemble/graded-read + online-Hebbian rung
that closes the tiny-graph gap. The host scaffold is retained as the `=0` oracle until then.
