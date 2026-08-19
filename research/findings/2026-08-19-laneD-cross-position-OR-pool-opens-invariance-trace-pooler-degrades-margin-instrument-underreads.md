---
type: finding
status: mixed
date: 2026-08-19
lane: perception
mechanism: invariance-from-temporal-continuity
runner: research/runners/_laneD_v1_pooler_trace_invariance_derisk.py
artifacts:
  - research/findings/raw/lanes/perception/xpos_off_6seed.json
  - research/findings/raw/lanes/perception/xpos_or_win4_6seed.json
  - research/findings/raw/lanes/perception/xpos_or_win6_6seed.json
---

# Cross-position OR-pooling on the lane-D trace-pooler front end (board #44): the innate complex-cell pool DOES make identity position-tolerant (held decode 0.11 -> 0.92, 6-seed, scramble-controlled) — CONFIRMING the sibling closure — but routed through the existing trace pooler it is a NO-GO, and the 2026-08-18 pairwise cosine margin UNDER-READS the invariance the pool provides

**One-line verdict.** The named representation-side move — a cross-position complex-cell OR-pool
UPSTREAM of the trace pooler (Hubel & Wiesel 1962; `--cross-pos-pool or_local`, default OFF =
byte-identical) — makes identity position-tolerant on the lane-D front end: held-out-position
category decode off the pooled V1-complex rises from `0.1111` (raw, BELOW chance `0.3333` — position
DOMINATES) to `0.9236` (win 4) / `0.8889` (win 6) over 6 seeds, with pixel-scramble at chance. This
CONFIRMS the sibling 2026-08-19 closure
([`vision-pooling-invariance-topology-not-learning-NOGO`](2026-08-19-vision-pooling-invariance-topology-not-learning-NOGO.md),
[`vision-hmax-hierarchy-...-load-bearing`](2026-08-19-vision-hmax-hierarchy-composed-pooling-solves-position-invariance-learning-not-load-bearing.md))
on the specific lane-D runner. Two honest results ride on top, both NEW: (1) routed through the
EXISTING trace pooler the pipeline is a **NO-GO** — the pooler's own gate reads TRACE-ROUTED-NOGO
(win 6) / PARTIAL-1/6 (win 4), because its sparse top-k binding DEGRADES the pooled code (trace
decode `0.5625`/`0.375` << the frozen-pool topology `0.75`/`0.8403`); learning downstream is not
load-bearing and is actively harmful here. (2) The 2026-08-18 root-cause instrument — the pairwise
same-category/cross-POSITION vs cross-category cosine margin — UNDER-READS this invariance: it lifts
only from `-0.0877` to `+0.0041` (win 6) while the centroid decode reaches `0.8889`, so "margin ~
0.000" was a property of the stringent pairwise instrument, NOT proof pooling cannot deliver
invariance. No `sim/` edit; additive default-off flag; `SIM_BACKEND=numpy`; 6 seeds.

`EXTERNAL-SEARCH-RAN:` complex-cell OR/MAX-pooling across position = Hubel & Wiesel 1962 (complex
cell = OR-pool over simple cells of one orientation across a retinotopic neighbourhood); the
composed-hierarchy surpass = Riesenhuber & Poggio 1999 (HMAX). Both are the canonical "pool across
position for translation tolerance" mechanisms and were verified against the sibling runners here.
This finding EXTENDS, and does not contradict, the two sibling findings above.

## What this adds over the sibling closure (why it is not a re-derivation)

The sibling established the capability + the "topology not learning" root cause on NEW runners
(`_vision_pooling_invariance_derisk`, `_vision_hmax_hierarchy_derisk`) and DECODE readouts. This
finding is on the runner the board named (`_laneD_v1_pooler_trace_invariance_derisk`) and contributes
three things absent from the sibling record:

1. **The 2026-08-18 cosine-margin instrument, measured under OR-pooling.** Neither sibling runner
   computes the same-cat/cross-pos vs cross-cat margin. Measured here it lifts `-0.0877 -> +0.0041`
   (win 6) — marginally across zero — even as decode reaches `0.8889`. The invariance lives in
   CENTROID structure (nearest-cosine-centroid decode), which the pairwise margin barely registers.
   **The instrument is part of the emulation:** the 2026-08-18 "margin ~ 0" is an instrument limit.
2. **The OR-pool -> existing trace-pooler COMPOSITION.** The sibling decoded directly off the pooled
   code. Feeding the pooled features to the lane-D trace pooler DEGRADES them (top-k sparsification
   discards the pooled invariance) — so the existing binding stage is the WRONG downstream for a
   position-pooled representation. The correct downstream is the sibling's HMAX C2 global-max, not a
   sparse competitive pooler.
3. **The `--cross-pos-pool` mechanism is now in the named runner**, default-off byte-identical
   (verified: every pre-existing metric/field identical to pristine main; only additive keys appear).

## Result — 6 seeds (42/43/44/100/101/102), chance 0.3333, 24 held images

Held-out-position category decode (mean over 6 seeds), and the invariance-margin readout, per arm:

| front end | inv-margin (pre->post) | OR/V1-direct decode | frozen-pool (topology) | trace-pooler | pixel-scramble |
|---|---|---:|---:|---:|---:|
| OFF (raw V1-complex)   | `-0.0877` -> `-0.0877` | `0.1111` | `0.2014` | `0.3194` | 0.3472 <!--derived--> |
| OR-pool win 4, stride 2 | `-0.0877` -> `-0.0222` | `0.9236` | `0.75`   | `0.5625` | `0.3194` |
| OR-pool win 6, stride 2 | `-0.0877` -> `+0.0041` | `0.8889` | `0.8403` | `0.375`  | 0.2778 <!--derived--> |

Per-seed OR-pool-direct decode is unanimous above chance (win 4: 0.79/0.96/1.00/0.88/0.92/1.00;
win 6: 0.75/0.83/0.92/0.92/0.92/1.00). The raw front end is BELOW chance (0.04-0.25), i.e.
anti-invariant — position dominates identity. `--complex-norm none` here (no orientation
z-competition), which is why the raw baseline is lower than the sibling's normalized `0.35`.

## The anti-cheats — what each returned

1. **Held-out positions (THE bar).** Held positions are never in training; OR-pool-direct decode
   `0.9236`/`0.8889` (6/6 above chance `0.3333`) vs raw `0.1111`. Position invariance is real.
2. **Pixel-scramble does not decode.** Per-image scramble sits at chance for every arm (`0.3194`,
   0.2778 <!--derived-->, 0.3472 <!--derived-->) — the invariance is not a low-level pixel artifact.
3. **Like-for-like.** The OR-pool is applied identically to every arm and the V1-direct baseline, so
   the lift is the POOL, not a per-arm asymmetry.
4. **Default-off byte-identical.** `--cross-pos-pool off` reproduces pristine main field-for-field
   (verified by stash/diff); only additive keys (`invariance_margin`, `cross_pos_pool`) appear.

## Root cause — topology carries it, learning does not, and the trace pooler is the wrong downstream

The frozen-pool arm (innate window, no learning) already reads `0.75`/`0.8403` — essentially all of
the OR-pool-direct decode. Learning adds nothing (this replicates the sibling's
`scaffold_trace_minus_frozen` ~ 0). What is NEW: the trace pooler, given the SAME pooled input,
decodes `0.5625`/`0.375` — BELOW the frozen topology. Its `k_win`-of-`n_col` top-k selection is a
competitive SPARSIFIER; on an already-invariant pooled code it throws away the very axis that was
pooled. So the runner's own gate (which requires the TRACE pooler to beat its controls) reads
TRACE-ROUTED-NOGO / PARTIAL-1/6: the named "upstream pool -> trace pooler" pipeline is a NO-GO, not
because the pool fails but because the downstream binder un-does it.

This unifies with the sibling: the invariance-carrying quantity is the innate OR-pooling TOPOLOGY
(retinotopic, Hubel-Wiesel), and the correct composition is a global MAX over the pool (HMAX C2),
which PRESERVES the pooled invariance, rather than a sparse competitive pooler, which discards it.

## NOT-A-WALL — the next mechanism (already named and de-risked)

The capability is settled: position-invariant identity emerges from innate cross-position OR-pooling
(this finding + the two siblings, all 6-seed). The correct downstream is the sibling's HMAX S->C
stack (held decode 0.5972 <!--derived--> on the harder histogram-matched CONFIGURAL task where a flat pool is forced
to chance — see the HMAX finding), NOT the lane-D trace pooler. The genuine open work is therefore
NOT another rate pooling variant (that route is mapped) but board #72: a SPIKING S->C stack wired
into the production ventral path. The pairwise-cosine margin should be RETIRED as the invariance
instrument in favour of centroid held-position decode (it under-reads by an order of magnitude).

## Reproduce

```bash
SIM_BACKEND=numpy OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -u -m \
  research.runners._laneD_v1_pooler_trace_invariance_derisk --seeds 42 43 44 100 101 102 \
  --n-categories 3 --n-train-pos 4 --n-held-pos 2 --n-ex 4 --cross-pos-pool or_local \
  --or-pool-win 6 --or-pool-stride 2 \
  --out research/findings/raw/lanes/perception/xpos_or_win6_6seed.json
# baseline: drop --cross-pos-pool (default off) -> byte-identical to pristine main.
```

## Sources

- Hubel, D. H. & Wiesel, T. N. (1962). Receptive fields, binocular interaction and functional
  architecture in the cat's visual cortex. *J. Physiol.* 160:106-154. (complex cell = OR-pool over
  position)
- Riesenhuber, M. & Poggio, T. (1999). Hierarchical models of object recognition in cortex. *Nat.
  Neurosci.* 2:1019-1025. (HMAX S->C composed pooling — the correct downstream)
