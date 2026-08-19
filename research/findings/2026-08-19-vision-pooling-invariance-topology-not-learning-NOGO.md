---
type: finding
status: negative
date: 2026-08-19
mechanism: invariance-from-temporal-continuity
runner: research/runners/_vision_pooling_invariance_derisk.py
artifacts:
  - research/findings/raw/lanes/perception/vision_pooling_invariance_6seed.json
---

# Position-invariant object recognition (board #44): complex-cell POOLING opens the invariance, but its POOLING TOPOLOGY is carried by innate retinotopy — a learned-from-continuity pool is a 6-seed NO-GO, and given the topology the trace rule is INERT

**One-line verdict.** On the deployed Gabor/V1 front end, the missing mechanism the two prior NO-GOs
named — a complex-cell POOLING across position (not competition, not decorrelation) — DOES open
held-position invariance: a hand-wired oracle pool beats V1-direct by **+0.42** and an innate local
retinotopic-window pool by **+0.26** (6/6 seeds), and the pooled code genuinely discards position
(object 0.59 vs position 0.27 ≈ chance). BUT the invariance is carried by the pooling TOPOLOGY
(which simple cells feed one complex unit across space), which is retinotopic/pre-wired — NOT by
learning. Learning the pooling topology from a moving-object continuity stream (Földiák trace rule)
is a clean **6-seed NO-GO** (held decode 0.2673, BELOW V1-direct's 0.3472 and its own frozen null),
and once the topology is a fixed local window the trace rule adds only **+0.03** over a frozen
random pool (learning inert, and the effect survives temporal shuffle → static artifact). This
UNIFIES all four levers on this wall (competition, decorrelation, learned-global-pool, SFA) under
one root cause and names the next mechanism. No `sim/` edit; additive runner; `SIM_BACKEND=numpy`.

`EXTERNAL-SEARCH-RAN:` Földiák 1991 (trace rule) + Wiskott & Sejnowski 2002 (Slow Feature Analysis,
the principled temporal-slowness objective the trace rule online-approximates) + Hubel & Wiesel
1962 (complex cell = OR-pool over simple cells of one orientation across position) — all are the
canonical "learn invariance from temporal continuity / pool across position" mechanisms; both the
online (trace) and closed-form (SFA) versions were run here. This finding also fills the exact gap
[`research/biology/invariance-from-temporal-continuity.md`](../biology/invariance-from-temporal-continuity.md)
flags: its trace-rule GO was for CATEGORY membership and it states "POSITION ... invariance ... were
NOT tested." Position is now tested.

## Design — one shared V1 front end, four arms, one GO gate, 6 seeds

Task: oriented bars; identity = orientation (4 classes), retinal position = the nuisance to pool
out. 8 positions spanning ±11 px along x; INTERLEAVED held-out — train on {0,2,4,6}, decode at
{1,3,5,7} NEVER seen in training (anti-cheat 1). Front end (all arms): pixels → deployed Gabor bank
(`sim/visual_cortex.build_v1_simple_weights`, reused by import, NOT edited) → V1-complex (orient ×
position, pooled over frequency) → local orientation competition (z-score across orientation
columns per position — lateral inhibition; applied IDENTICALLY to every arm so it cannot be the
differentiator). Decode = nearest-cosine-centroid, centroids trained on train-position codes.

- **A · V1-DIRECT** — no pooling (the position-specific code the two NO-GOs also read).
- **B · LEARNED-GLOBAL** — the target emergent mechanism: 12 units, `g_i = W_i·x` over the full
  feature vector, trace-based competition (winner on the leaky-integrated activity + duty-cycle
  boosting), the cross-position pooling TOPOLOGY learned from a moving-object continuity stream by
  the Földiák trace rule (`dW = lr·ȳ·x`). Controls: B_shuffled (order destroyed), B_frozen (lr=0).
- **C · SCAFFOLD-LOCAL** — innate LOCAL retinotopic pooling windows (a FLAGGED developmental
  complex-cell RF scaffold, per the task's "if you must scaffold the pooling topology, flag it");
  within each window the trace rule learns the orientation preference. Controls: C_frozen (random
  prefs), C_shuffle (continuity destroyed).
- **D · ORACLE-GLOBAL** — hand-wired sum over ALL positions per orientation channel (the pooling
  ceiling / headroom reference; fully host-designed topology).

## Result — 6 seeds (42/43/44/100/101/102), chance 0.25

Artifact: `research/findings/raw/lanes/perception/vision_pooling_invariance_6seed.json`. Held-out-
position object decode, per-seed and mean:

| arm | 42 | 43 | 44 | 100 | 101 | 102 | mean | vs V1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A V1-direct (no pool)          | 0.38 | 0.29 | 0.40 | 0.29 | 0.31 | 0.42 | **0.3472** | — |
| B learned-global trace pool    | 0.25 | 0.35 | 0.25 | 0.21 | 0.27 | 0.27 | **0.2673** | −0.08 |
| B_shuffled (control)           | 0.29 | 0.35 | 0.31 | 0.27 | 0.25 | 0.40 | 0.3125 | — |
| B_frozen (lr=0 control)        | 0.19 | 0.21 | 0.23 | 0.27 | 0.31 | 0.29 | 0.25 | — |
| C scaffold-local, trace-learn  | 0.62 | 0.44 | 0.67 | 0.67 | 0.73 | 0.73 | **0.6424** | +0.30 |
| C_frozen (random prefs)        | 0.62 | 0.40 | 0.62 | 0.67 | 0.69 | 0.67 | **0.6111** | +0.26 |
| C_shuffle (continuity killed)  | —    | —    | —    | —    | —    | —    | 0.625 | +0.28 |
| D oracle global-per-orient     | 0.77 | 0.79 | 0.83 | 0.67 | 0.83 | 0.73 | **0.7708** | +0.42 |

Verdict fractions (all 6 seeds unanimous): **B_emergent_go 0/6**; pooling_capability (oracle beats
V1) 6/6; scaffold_topology_beats_v1 6/6; **learning_load_bearing (C_trace beats C_frozen) 0/6**.

## The four anti-cheats — what each returned

1. **Held-out positions.** The bar. B (emergent learned pool) does NOT clear it (0.2673 ≈ chance,
   below V1-direct). Oracle/scaffold pooling DO (+0.42 / +0.26) — pooling as a capability is real.
2. **Temporal continuity load-bearing.** FAILS to be load-bearing where invariance actually
   appears: C_trace 0.6424 ≈ C_shuffle 0.625 ≈ C_frozen 0.6111. The invariance survives temporal
   shuffle → it is a STATIC artifact of the pooling window, not the trace rule. (For ARM B, where
   the trace does drive the topology, there is no invariance to be load-bearing FOR.)
3. **Position pooled out.** HOLDS for the pool that achieves invariance (scaffold): object decode
   off the held code 0.5903 (≫ chance 0.25) while position decode off the SAME units 0.2708 (≈ chance-
   position 0.25). Genuine invariance — position information is discarded, identity retained.
4. **6 seeds + nulls.** Unanimous across seeds; B_frozen (random projection) and a label-shuffle
   null both sit at chance; determinism verified (identical re-run).

## Root cause — quantified, and it unifies all four levers

Pooling HAS headroom: oracle +0.42, and the orientation-competition normalization is shared by
every arm (V1-direct includes it and still reads 0.3472), so the +0.42 is the POOL, not the norm.
The `attributable_to` decomposition (runner-computed, pooled means) is unambiguous: of the scaffold
pool's held invariance, only **4.9%** is attributable to LEARNING (95.1% is already in the frozen
control), while **43.2%** is attributable to the POOLING TOPOLOGY (over the shared V1-direct
baseline); ARM B's held decode is **−16.9%** attributable to temporal continuity (the shuffled
control EXCEEDS grouped). The failure is specifically that a LEARNED-from-scratch pool cannot
capture the headroom:

- **A learned pool can only place weight where it saw activity** — trained positions plus RF-overlap
  tails. Held positions that are genuinely unseen get ~zero weight, so the learned unit cannot
  respond to the object there. The oracle beats V1 precisely because its topology is pre-wired to
  span ALL positions (including unseen ones); the scaffold beats V1 because its window is pre-wired
  to span the local neighbourhood the held positions fall in. Remove the pre-wiring (ARM B) and the
  pool cannot exceed V1-direct, which already exploits the same RF-overlap tolerance.
- **Given a pre-wired topology, feature-learning is inert on this task** (C_trace − C_frozen =
  +0.03), because identity here is a single LOCAL feature (orientation) that any projection of the
  8 orientation channels preserves — there is nothing for the trace rule to bind. The trace rule's
  feature-grouping only becomes load-bearing when identity is a CONJUNCTION a random projection
  would NOT preserve (a configural object), which oriented bars are not.

This is the same wall the prior two NO-GOs hit, now decomposed: competition (2026-08-11 harder-kWTA)
and decorrelation (2026-08-18 learned-decorrelation) "separate but do not pool"; the learned-global
pool here DOES pool but cannot pre-wire its own retinotopic extent; SFA (closed-form slowness,
tested in the scoping sweep) ties V1-direct for the identical reason. The invariance-carrying
quantity is the pooling TOPOLOGY, and on this front end it is retinotopic/developmental, not learned.

## NOT-A-WALL — the next mechanism (named, with the headroom it must capture)

The biology is explicit and consistent with this result: complex-cell pooling RFs are developmentally
pre-structured (retinotopic OR-pooling over a local neighbourhood — Hubel-Wiesel), and temporal-
continuity learning (Földiák/SFA) tunes WHICH features within that neighbourhood bind, not the spatial
extent from scratch. Two concrete next levers, in order:

1. **Accept innate local retinotopic pooling topology as biology** (a developmental complex-cell RF,
   not a host shortcut — it is a real genetic/activity-dependent given), and make the trace rule
   load-bearing by moving to **configural objects** (multi-part shapes / conjunctions) where a random
   projection FAILS and learned feature-binding is required. The headroom the learned binding must
   capture is the oracle's +0.42 minus what a frozen pool gives on that harder task.
2. **A hierarchical S→C stack** (HMAX / Wallis-Rolls VisNet): stacked local-pool layers compose
   local shift-tolerance into global invariance, so no single layer must weight unseen positions.

An honest verdict settles the op-point: at this op-point (z-competition, rf=2, span ±11), the
learned-from-scratch pool is refuted for position invariance, so the invariance-carrying topology is
banked as retinotopic/scaffolded and the residual is a missing configural task + hierarchy, not a
missing learning rule for a single flat pooling layer.

## Reproduce

```bash
SIM_BACKEND=numpy OMP_NUM_THREADS=2 .venv/bin/python -u -m \
  research.runners._vision_pooling_invariance_derisk --seeds 42 43 44 100 101 102 \
  --out research/findings/raw/lanes/perception/vision_pooling_invariance_6seed.json
```

GO gate (ARM B, per seed): held-object decode ≥ chance+0.15; beats V1-direct, B_shuffled, B_frozen by
0.10; pixel-scramble does not decode; object decodable while position not decodable off the same
units. Separately, learning_load_bearing = C_trace beats C_frozen by 0.10.
