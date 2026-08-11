---
type: finding
status: qualified
date: 2026-08-11
lane: laneC
mechanism: source-monitor-popcode-homeostasis
runner: research/runners/_laneC_source_monitor_popcode_homeostasis_gate.py
seeds: [700, 701, 702, 703, 704, 705]
verdict: NO-GO (5/6) — honest negative; no-harm boundary structurally resolved, floor-clearing not universal
artifacts:
  - research/findings/raw/parallel_gates/source_monitor_popcode_homeostasis_calibration_217-230.json
  - research/findings/raw/parallel_gates/source_monitor_popcode_homeostasis_decisive_700-705.json
---

# The source-monitor no-harm boundary is an OPERATING-POINT (f-I saturation of under-provisioned pools), not a competition problem: independent per-source POPULATION-CODED pools with NO shared budget clear the floor and preserve strong sources where V2's shared-budget competition could not

## The boundary being surpassed

<!--derived-->
V2 local fast-spiking competition
(`2026-08-03-source-monitor-coresidency-v2-calibration-NO-GO`) cleared the `0.15`
source-margin floor but, on seed 217, WEAKENED the already-strong self source by
`0.0092` versus its competition lesion — a no-harm failure. The cause is a SHARED
BUDGET: each source's interneurons inhibit the other two pools, so lifting the two
weak sources drains the strong rival. That is a conservation effect. Per THE LAW a
conservation boundary is an undiscovered mechanism, not a wall.

## What the de-risk found (NumPy backend, seeds 205-259 as OBSERVED exploration)

<!--derived-->
Because the episode patterns are disjoint, at each source's recall the rivals fire
~0, so a source's margin is essentially its own pool rate. The weak-source deficit
is therefore NOT a rival-suppression problem, and two recall-time GAIN levers were
each measured NO-GO here:

- <!--derived--> Fixed per-source recurrent self-excitation (same weight all
  sources, no cross-pool budget) is non-monotone and unstable: at one weight the
  strong self source COLLAPSES (margin +0.234 → −0.088, dominance lost). A fixed
  recurrent gain bursts/blocks the pool at some weights; it is not clean no-harm.
- <!--derived--> One-shot up-only homeostatic scaling of a 12-neuron pool's own
  feedforward weights makes the weak margin WORSE, not better (e.g. a heard pool
  scaled ×1.6 falls 0.110 → 0.087): over-driving a saturated pool pushes it into
  its f-I / refractory / depolarization-block ceiling, where more drive LOWERS the
  net rate. This reproduces, in the DISJOINT regime, the same ceiling the overlap
  arc's own-gain hit
  (`2026-08-06-source-monitor-fair-inhibition-and-own-gain-both-NO-GO-margin-is-a-shared-cell-pattern-separation-problem`)
  — so here it is intrinsic to the pool's rate dynamics, not the shared cells.

The unifying diagnosis is CLAUDE.md's deepest lesson: the weak margin is an
OPERATING-POINT property. A 12-neuron pool driven at the V1 point sits at its f-I
ceiling, so every recall-time gain either over-drives it (floor fail) or, via a
shared budget, drains a rival (V2's no-harm fail). <!--derived--> Across seeds
205-259, 10/55 (~18%) fail the floor at baseline n=12; the worst is −0.048 (a
source that loses its own recall).

## The mechanism: population-coded independent pools + up-only homeostatic scaling

Provision each source pool as a population code (n_source_memory 12 → 24) and keep
V1's independent per-source pools with NO cross-pool competition. <!--derived-->
At baseline, with NO mechanism at all, this clears the floor on 8/10 of the worst
observed seeds and 8/8 easy seeds — so no-harm is structural: nothing is
redistributed. On the now non-saturated pool, an UP-ONLY one-shot per-source
homeostatic synaptic-scaling consolidation (Turrigiano multiplicative scaling of
each pool's OWN learned episode→source synapses toward a fixed rate set-point,
`R_target=0.20`, capped) adds guard-band headroom while holding no-harm.

No-harm is STRUCTURAL, not tuned: the scaling is up-only (no weight is ever scaled
down) and each source's scaling touches only synapses whose post-neuron is in its
own source-memory pool. No source's drive can be reduced and no cross-source
synapse is touched — there is no shared budget to conserve. <!--derived--> Across
the 18 observed hard+easy seeds the scaling held no-harm 18/18.

## Calibration result (2 observed seeds; decisive sweep preregistered, not run)

Table and inline values below are rounded from the cited calibration artifact.
<!--derived-->
Artifact:
`research/findings/raw/parallel_gates/source_monitor_popcode_homeostasis_calibration_217-230.json`.
Both were hard at n=12 (V2-class failures) and both pass all controls here:

| seed | small-pool (n=12) min | mechanism ON min | homeostasis min-gain | self factor | status |
|---:|---:|---:|---:|---:|:---:|
| 217 | 0.1408 | 0.1692 | −0.0033 | 1.0000 | CALIBRATION_PASS |
| 230 | 0.1200 | 0.2071 | −0.0033 | 1.0000 | CALIBRATION_PASS |

Seed 217 is the exact seed V2 weakened self on. Here the strong self source is
UNTOUCHED (factor `1.0`), the floor is cleared (`0.1692 ≥ 0.15`), and the same
seed fails the floor at n=12 (`0.1408`) — establishing that population coding, not
a recall-time mechanism, does the work. The `−0.0033` min-gain is single-shot
recall state drift (measured ~0.003–0.005), inside the documented `0.01` no-harm
tolerance; the primary no-harm guarantee is the structural up-only/disjoint proof,
which the runner asserts as a precondition.

## Verdict — the preregistered decisive 6-seed sweep: 5/6 PASS → NO-GO (HONEST NEGATIVE, coordinator-run)

NO-EXTERNAL-NEEDED: this is a METHOD verdict (recall-time provisioning is not a universal floor-clearer), NOT a
fundamental-limit / different-paradigm claim — the capability stays OPEN with a concrete, biologically-standard next
mechanism named (heterosynaptic LTD / competitive commitment at the ENCODING step, a textbook synaptic mechanism),
and the boundary is convergent with our own prior fair-inhibition NO-GO reached from the overlap side. The next
mechanism's de-risk is where the external deep-read belongs; banking this negative needs no new literature.

<!--derived-->
The fresh unobserved 6-seed sweep (seeds 700–705, SIM_BACKEND=numpy) was RUN by the coordinator
(`research/findings/raw/parallel_gates/source_monitor_popcode_homeostasis_decisive_700-705.json`, `AGGREGATE 5/6 PASS
-> NO-GO`). Per-seed: 700 DEVELOPMENT_PASS (min_margin +0.2125), 701 DEVELOPMENT_PASS (+0.1883), **702 DEVELOPMENT_FAIL**,
703 HELD_OUT_PASS (+0.1958), 704 HELD_OUT_PASS (+0.2342), 705 HELD_OUT_PASS (+0.1575). Across the passing seeds the
structural no-harm control holds (homeo_min_gain ≥ −0.004; the strong source's scaling factor stays 1.0 — nothing
redistributed), so **the no-harm boundary IS genuinely resolved** and population coding clears the `0.15` floor on 5/6.

**Why 702 fails — a SECOND residual mode, distinct from the one predicted.** Seed 702's population-coded margin is
+0.1963 (ABOVE floor), but its SMALL-pool control also clears the floor (`small_pool_min=+0.1542 > 0.15`), so the
mechanism (n=12→24) is not shown to be CAUSALLY necessary on that seed — a causal-attribution miss, not a below-floor
miss. Combined with the predicted weak-encoding residual (~2/55 seeds like 244/259 that stay below floor at every pool
size), the honest read is: **recall-time provisioning (bigger independent pools + up-only homeostatic scaling) is
NECESSARY-and-sufficient for the no-harm half of the problem, but is NOT a universal floor-clearer** — 1/6 fresh seeds
escapes it.

**Banked as an honest negative that maps the boundary + names the next mechanism** (per THE LAW — a NO-GO defers a
METHOD, not the capability). Recall-time gain has now failed from BOTH the overlap side (fair-inhibition NO-GO) and the
disjoint side (this sweep). The convergent next mechanism is **COMPETITIVE ENCODING** — heterosynaptic LTD /
target-source commitment at the LEARNING step — so the weakest source's representation is strengthened (and made
causally floor-dependent on its own pool) BEFORE recall, rather than rescued afterward.

Reproduce:
```
PYTHONPATH=. SIM_BACKEND=numpy .venv/bin/python -u \
  -m research.runners._laneC_source_monitor_popcode_homeostasis_gate \
  --seeds 700 701 702 703 704 705 \
  --json research/findings/raw/parallel_gates/source_monitor_popcode_homeostasis_decisive_700-705.json
```

<!--derived--> The named residual: ~2/55 observed seeds (e.g. 244, 259) stay below
the floor at EVERY pool size and are not rescued by up-only scaling — a source
whose single-source Hebbian encoding is genuinely too weak, which no recall-time
gain can lift off the f-I ceiling. If a decisive seed lands in that class the sweep
returns 5/6 (a NEGATIVE), and the next mechanism is COMPETITIVE ENCODING —
heterosynaptic LTD / target-source commitment at the learning step to strengthen
the weakest source's representation before recall — the same encoding-side
prescription the fair-inhibition NO-GO reached from the overlap side, now reached
from the disjoint side too.

## Scope and scaffolds

Refuted levers acknowledged and not re-proposed: V2 cross-pool competition
(shared-budget no-harm fail), self-normalised "fair" inhibition and own-gain (both
already tested-and-NEGATIVE in the fair-inhibition finding). Standing scaffolds
(unchanged): caller-supplied sparse episode activity, physical source-afferent
identity, an externally timed learning window, host spike-count evaluation. NEW
scaffold: the homeostatic scaling is host-computed and host-timed (a one-shot
consolidation). The biology it stands in for — synaptic scaling to a firing
set-point — is real; its spiking / astrocytic slow-loop implementation is deferred
and named. No language, confidence scalar, or response policy is claimed.
