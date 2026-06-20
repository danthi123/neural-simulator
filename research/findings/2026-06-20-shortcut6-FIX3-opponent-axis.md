# Shortcut #6 — FIX 3 opponent-axis push-pull (2026-06-20)

**VERDICT: FIX 3 is NEGATIVE — it RE-BIASES to N.** The margin-SNR residual is now a CONFIRMED deeper
structural sub-blocker (a persistent N-S surplus at the commit/selection pools), not a tie-resolution
artifact. The next gated step is a focused **deep-research scoping of the SC-margin / N-S-surplus**, NOT more
read-out variants. #6 re-orient stays PARTIAL (the FIX-1 state: dom tracks, SCRAM collapses, but the
post-change Σ is well above HOST).


**Type:** GPU experiment (grid-32, `SIM_BACKEND=cupy`), the prescribed next mechanism after the cascade
North-bias FIX (`2026-06-20-cascade-north-bias-FIX.md`, `138f7d59`). FIX 1 (stochastic tie-break) CONVERTED
the host North-bias shortcut to spikes and made the SC orienting decode load-bearing (SCRAM collapses 3/3),
but the full re-orient stays PARTIAL (post-change Σ 59–83 vs HOST ~1.6) — a **margin-SNR residual** (the
grid-32 SC orienting margin is tiny → weak-margin phases random-walk). The simple margin-amplification
(stronger SC drive) was NEGATIVE (it re-biases N→W). This doc tests the scoping's prescribed **FIX 3 —
opponent-axis push-pull** (`2026-06-20-cascade-north-bias-scoping.md` §FIX 3).

**Owner standard (load-bearing):** BRAIN-BASED-ONLY; grid-32 IS the verdict (never grid-8); a boundary is not
an exit. NO `sim/` edit. The no-confab moat is array-disjoint from the nav cascade and untouched.

---

## FIX 3 — the mechanism (point-neuron, NO `sim/` edit)

The four cardinals form two natural opponent axes (N↔S, E↔W). The flat 4-way argmax read-out lets a faint
position-correct SC margin on ONE axis be a sub-threshold contributor lost in the global race (and the N-first
ordering decides the residual). FIX 3 makes the decision a **signed push-pull per axis**: `axis_NS =
count[N] − count[S]`, `axis_EW = count[E] − count[W]`; pick the axis with the larger `|signed margin|`, then
the sign within that axis. A faint W-over-E surplus (far-W goal) becomes a DECISIVE axis winner rather than a
weak 4-way contender. When BOTH axes are within `sc_opponent_axis_eps` (a genuine tie), fall through to the
FIX-1 tie-break draw (so FIX 3 sharpens the weak-but-nonzero per-axis margins that FIX 1's whole-set tie-break
could not resolve). Biology: the superior-colliculus motor map's opponent/push-pull organization (distinct
populations encode OPPOSING movement directions, balanced E/I preventing a directional bias; Nature Comms
2023, Comms Biol 2025; catalog H.25). Runner kwarg `sc_opponent_axis` (env `SC_OPPONENT_AXIS=1`), **default
OFF = byte-identical** (the flat argmax path is unchanged). PURE READ-OUT (reads the same sel_X/commit_X
counts the flat argmax reads). Reports `opponent_axis_fraction` (the anti-cheat: a GO needs the axis margin
load-bearing, not a fall-through to the tie-break).

---

## The per-phase Σ table (seed 42, grid-32, n=1800, warmup-600)

Goal schedule: phase0 NE `(30,30)`, phase1 far-W `(1,30)`, phase2 SW `(1,1)`, phase3 SE `(30,1)` (3
re-orients). Per-phase `final_quarter_mean_distance` (lower = better):

| arm | ph0 (NE) | ph1 (far-W) | ph2 (SW) | ph3 (SE) | Σ post-change | dom per phase | tie-frac | opp-axis-frac |
|---|---|---|---|---|---|---|---|---|
| **FIX1** (tie-break) | 26.29 | 0.98 | 31.16 | 50.99 | **83.13** | N, W, W, E (tracks) | 0.329 | — |
| **FIX1+3** (+ opponent-axis) | 29.67 | 0.81 | 30.85 | 53.94 | **85.60** | **N, N, N, N (re-biased)** | 0.136 | 0.864 |
| **HOST** (ceiling) | 0.50 | 0.50 | 0.58 | 0.50 | **1.57** | E, W, S, E (tracks) | 0.000 | — |
| **SCRAM(FIX1+3)** (lesion) | _pending_ | _pending_ | _pending_ | _pending_ | **_pending_** | _pending_ | _pending_ | _pending_ |

_(FIX1 / HOST / SCRAM(FIX1) reference rows reproduced from `2026-06-20-cascade-north-bias-FIX.md` seed-42.)_

### What the table shows — FIX 3 RE-BIASES (the decisive negative)

1. **No improvement, and the North-bias RE-EMERGES.** FIX1+3 post-change Σ = **85.60** ≈ FIX1-only's 83.13
   (statistically the same, slightly worse). But the dom flips back from FIX1's `N,W,W,E` (3 distinct,
   tracks) to **`N,N,N,N` (stuck-N every phase)** — FIX 3 re-introduced the exact North-bias FIX 1 removed.
   The N-fraction is back up to 0.34–0.36 every phase (vs FIX1's balanced 0.24–0.28).
2. **The opponent axis IS load-bearing — but it carries the bias.** `opponent_axis_fraction = 0.864` (the
   axis margin decided 86% of steps; the tie-break fell to 0.136). So FIX 3 is doing what it was built to do
   (the read-out is dominated by the signed axis margin), but the axis it selects is the contaminated one.

### The decisive diagnostic — a structural N-S surplus at the commit pools

The per-quarter commit-count axis margins (`sum count[N]−count[S]` vs `count[E]−count[W]`, the exact
quantities FIX 3's argmax compares) reveal the failure mechanism:

| phase (goal) | N | E | S | W | axis N−S | axis E−W | picks |
|---|---|---|---|---|---|---|---|
| ph0 (NE) | 15769 | 14349 | 12117 | 14827 | **+3652** | −478 | N-S → **N** |
| ph1 (far-W) | 15040 | 10807 | 13530 | 10693 | **+1510** | +114 | N-S → **N** |
| ph2 (SW) | 14366 | 14063 | 11553 | 15144 | **+2813** | −1081 | N-S → **N** |
| ph3 (SE) | 13913 | 14430 | 10951 | 14341 | **+2962** | +89 | N-S → **N** |

- **There is a LARGE, persistent, goal-INVARIANT N-over-S surplus at the commit pools** (N−S = +1510 to
  +3652, ~2800 average). This is the structural North-bias, now visible as a raw count surplus at the
  selection stage — it is NOT a tie-resolution artifact (the counts genuinely differ).
- **The position-bearing E-W margin is TINY by comparison** (E−W = −1081 to +114). Even at the far-W goal
  (ph1) E−W = **+114** (slightly E-favoring!), and the SW goal (ph2) E−W = −1081 (W-favoring = correct, but
  ~3× smaller than the N-S noise).
- **`|N−S| ≥ |E−W|` in ALL 4 phases** → the opponent-axis read-out ALWAYS selects the N-S axis and, being
  positive, always commits **N**. The position-correct E-W signal is real but swamped by the structural N-S
  surplus before the axis comparison even sees it.

**Why FIX 1 worked but FIX 3 fails on the same data:** FIX 1's stochastic tie-break treated the near-equal
4-way counts as ties and randomized them — which *masked* the N-S surplus (a coin-flip among `[N,E,S,W]`
ignores the systematic +2800 on the N-S axis as long as no single pool is the strict max). FIX 3's hard
signed-difference does the OPPOSITE: it reads the systematic N-surplus as decisive "N-S axis evidence" and
commits to N every step. **The signed push-pull amplifies the very structural N-surplus FIX 1 was
randomizing away.** This is exactly the margin-amplification failure mode the scoping flagged ("it
re-biases").

### The surplus is STRUCTURAL across ALL arms (the clincher that it's upstream of the read-out)

The TOTAL commit-count axis surplus (whole-episode, all 1800 steps) across the arms — this is the raw
selection-stage count, independent of the read-out:

| arm | total N | total S | **N−S surplus** | total E | total W | E−W | dom |
|---|---|---|---|---|---|---|---|
| BASELINE (no fix) | 60990 | 50203 | **+10787** | 54772 | 57908 | −3136 | N,N,N,N |
| FIX1 (tie-break) | 54596 | 45040 | **+9556** | 58648 | 60729 | −2081 | N,W,W,E |
| FIX1+2 (+ sel homeo) | 56084 | 46973 | **+9111** | 57068 | 67721 | −10653 | W,N,W,W |
| FIX1+3 (opponent-axis) | ~58000 | ~48000 | **~+10800** | — | — | — | N,N,N,N |

- **The N-S surplus (~+9000 to +10800) is present in EVERY arm**, including the ones that track the goal
  (FIX1, FIX1+2). It is a structural property of the spiking selection pools, NOT a read-out artifact.
- **FIX 1 "works" by NOT reading the surplus as decisive** — it randomizes the near-ties, so the systematic
  N-over-S count never becomes a hard N at the read-out (the surplus is still +9556, but the dom tracks
  because the read-out ignores it on ties). It MASKS, it does not REMOVE.
- **FIX 2 (per-pool homeostasis at the sel stage) barely dented it** (−9%: +9556 → +9111) — the per-region
  threshold-adapt does NOT equalize the N-S count offset, which is why FIX1+2 did not improve the score over
  FIX1. (FIX1+2's E−W = −10653 shows the homeostasis perturbed the E-W axis, not removed the N-S surplus.)
- **FIX 3 reads the surplus as decisive** and re-biases. ⇒ The surplus lives UPSTREAM of the read-out, at the
  spiking selection stage, and is NOT removed by per-pool homeostasis. This is the precise, confirmed deeper
  sub-blocker the deep-research scoping must target.

**Verified rig facts (controller-checkable):** the rig uses `sc_popvector_readout=True` (line 161 of the
de-risk runner) — the SYMMETRIC pop-vector cosine decode, so the deployed half-plane-ramp's S-suppression
asymmetry (`max(0,−sy_offset)` darkening cortex_S) is NOT the active source here. The rig also uses
`enable_cluster_e_topography=True` (line 121) — the N corner is placed at the geometric TOP of the unit
square (N=(0.5,1.0), S=(0.5,0.0)); whether this topographic prior, the shared-STN→GPi common baseline, or the
SC→cortex drive is the dominant source of the +9000 N-S count surplus is the precise question for the gated
deep-research scoping (a single-region firing-rate probe down the cortex→str→GPi→thal→sel chain would
localize it).

---

## The FIX-3 edit summary

`g11_bg_runner.py`:
- New kwarg `sc_opponent_axis: bool = False` (+ `sc_opponent_axis_eps: int = 0`, env `SC_OPPONENT_AXIS`). A
  persistent `_opp_axis_count` tracker + `_sc_opponent_axis` flag added at episode start.
- `_argmax_action(counts)` extended: when `_sc_opponent_axis` is on, compute the signed margin per opponent
  axis (N−S, E−W); if EITHER axis exceeds `sc_opponent_axis_eps`, return the sign of the larger-`|margin|`
  axis; only when BOTH axes tie does it fall through to the FIX-1 tie-break / N-first ordering. **Default
  (flag off) skips the new branch entirely → byte-identical.** Reports `opponent_axis_count` /
  `opponent_axis_fraction` in the result JSON.
- `_nav_sc_popvector_readout_derisk.py`: `--fix3` / `--opponent-axis-eps` CLI flags + the opponent-axis
  fraction surfaced per arm.

NO `sim/` edit (reuse-by-import + a read-out formula). The default-off path is byte-identical.

---

## Anti-cheat table (seed 42)

| anti-cheat | requirement | result | pass? |
|---|---|---|---|
| Per-phase per-cardinal dom (THE discriminator) | dom must SHIFT to track the moving goal (not re-biased to a new fixed cardinal) | FIX1+3 dom = **N,N,N,N** (re-biased back to stuck-N) | ❌ FIX 3 fails this |
| 4-cardinal symmetry / S reachable | no cardinal structurally dominant; S reachable | FIX1+3 N-frac 0.34–0.36 (N dominant again); commit-pool N−S surplus ~2800 goal-invariant | ❌ N structurally dominant |
| Host ceiling | host re-orients, anchors the gap | HOST post-change Σ 1.57, dom tracks every phase | ✅ |
| Regime fidelity = grid-32 (NOT grid-8) | the verdict is grid-32/1800/warmup-600 | all arms grid-32/1800/warmup-600 | ✅ |
| Scramble / lesion MUST collapse | SCRAM(FIX1+3) must collapse relative to intact | _pending (running; lower value on a NEGATIVE arm — see note)_ | _pending_ |
| Opponent axis is load-bearing | report opponent_axis_fraction (axis margin deciding, not the tie-break) | FIX1+3 opponent_axis_fraction = **0.864** (the axis margin IS load-bearing — but it carries the bias) | ✅ measured (the tell) |
| FIX-3-OFF == byte-identical | the flag guard | the default-off `_argmax_action` skips the new branch; FIX1-only run reproduces (dom N,W,W,E, opp_axis_frac n/a) | ✅ by construction |
| No-confab moat untouched | nav cascade array-disjoint from the composer's complex synapses | no conversational regions in these nav runs | ✅ |

---

## VERDICT — FIX 3 NEGATIVE; the margin-SNR residual is a CONFIRMED deeper sub-blocker → deep-research next

**FIX 3 (opponent-axis push-pull read-out) is NEGATIVE at faithful grid-32 (seed 42).** Post-change Σ = 85.60
(no improvement over FIX1-only 83.13), and the dom flips back to **N,N,N,N** — the opponent-axis read-out
RE-BIASES to N. The commit-count axis-margin diagnostic is decisive: there is a **large, persistent,
goal-invariant N-over-S surplus at the commit/selection pools** (N−S ≈ +2800 every phase) that dwarfs the
tiny position-bearing E-W margin (E−W ≈ −1081…+114). FIX 3's signed push-pull always picks the larger-|margin|
N-S axis and commits N; it AMPLIFIES the very structural surplus that FIX 1's stochastic tie-break was
randomizing away (`opponent_axis_fraction = 0.864` → the axis margin is load-bearing, but it carries the
bias). This is exactly the re-bias failure the scoping flagged for margin amplification.

**The honest finding: the margin-SNR residual is a CONFIRMED deeper structural sub-blocker — a persistent
N-S surplus at the selection pools — NOT a tie-resolution artifact.** Three independent read-out levers now
converge on this: (1) FIX 1 (stochastic tie-break) MASKS it (randomizes the near-ties) → tracks but leaves a
margin-SNR residual; (2) stronger SC drive (the margin-amplification screen) RE-BIASES (N→W); (3) FIX 3
(opponent-axis) RE-BIASES (back to N). The residual lives UPSTREAM of the read-out, at the spiking selection
stage — no read-out reorganization can fix a contaminated count surplus the read-out merely reads.

**Per the owner's HARD rule (no boundary → stop) and the task directive (FIX 3 NEGATIVE → do NOT grind
variants → recommend a focused deep-research scoping), the next gated step is a deep-research scoping of the
N-S surplus / SC-margin-resolution problem.** The precise question: *why do the commit/sel pools carry a
goal-invariant ~2800 N-over-S surplus, and what removes it at the spiking selection stage* (candidates the
scoping should rank: (a) the SC far-blob resolution at grid-32 — a dim/small far goal-blob in the 16×16
`sc_map` gives a tiny E-W margin that the N-S structural surplus swamps → sharper SC RF / more SC neurons /
foveation; (b) the source of the N-S surplus itself — is it the `sc_map→cortex` half-plane geometry, the
sel_X/commit_X accumulator wiring, or a residual cluster-A/E topographic prior that favors the N corner; (c)
per-axis baseline subtraction at the selection stage — a homeostatic / divisive-norm operation that removes
the common-mode N-S offset BEFORE the axis comparison, so the opponent-axis sees only the differential
position signal). The deep-research gate fires here under condition (a) (a confirmed boundary + the next move
is a mechanism to overcome it) and (f) (≥2 distinct read-out approaches — FIX 1 partial, FIX 3 + margin-amp
NEGATIVE — to the same goal).

**#6 status: re-orient stays PARTIAL** (the FIX-1 state remains the best: dom tracks the goal, the SC decode
is load-bearing — SCRAM collapses 3/3, far-W reached at host-level finalQ — but the post-change Σ 59–83 ≫
HOST ~1.6). FIX 3 did NOT close it and did NOT improve it. This is an honest NEGATIVE for the opponent-axis
mechanism and a SHARPENED diagnosis of the residual, NOT a "closed boundary" and NOT a stop.

**Net:** the opponent-axis push-pull read-out re-biases to N because the structural N-S surplus at the
selection pools is a count-level contamination upstream of the read-out; the residual #6 margin-SNR is a
confirmed deeper sub-blocker requiring a focused deep-research scoping of the N-S-surplus / SC-margin
problem (the precise next gated step). Point-neuron, default-off byte-identical, NO `sim/` edit. The
no-confab moat is untouched (the nav cascade is `cp_*` nav state, array-disjoint from the composer's complex
`cp_rf_w_*` synapses).

_GPU (`SIM_BACKEND=cupy`). The FIX-3 edit + the FIX1+3 arm JSON committed the moment they landed (anti-rest),
pushed to both remotes. grid-32 IS the verdict (never grid-8). NO `sim/` edit. The moat is untouched._
