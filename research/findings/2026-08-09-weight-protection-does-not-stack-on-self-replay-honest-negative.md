---
type: finding
status: negative
date: 2026-08-09
mechanism: synaptic-importance-gated-plasticity (EWC/SI/Zenke) + fixed-orthogonal-readout-targets (PS-SNN)
lane: H-memory (continual learning / sleep-replay retention)
---

# Weight protection does NOT stack on top of self-replay — an honest negative with teeth

**2026-08-09.** Target: raise sequential-teacher retention PAST the sleep-replay plateau (~0.55 frac-recalled
6-seed mean at extreme sequential interference; `2026-06-30-100M-C2-scaleup`, main `8d2510d3`) with a PROVEN
continual-learning mechanism — synaptic-importance-gated plasticity (EWC Kirkpatrick 2017 / Zenke SI 2017 / the
project Phase-1.4 gate-freeze that got 103% retention) — NOT more replay (replay MAGNITUDE is already REFUTED as
the lever, `8d2510d3`). We also tested the PS-SNN fixed-orthogonal-target readout (Hu et al. 2026, Sci.Rep.).

**Verdict: NEGATIVE with teeth.** On the brain's own readout synapses, layered on top of the brain's own
self-replay, the importance gate gives **no structured retention benefit over replay alone**, and orthogonal
distributed target codes **hurt**. The reason is a mechanism insight, not a tuning miss: **self-replay already
does the job weight protection would do** — it reconsolidates the whole readout from the stored engrams each
night, so freezing the readout is redundant. The residual is engram-store fidelity, exactly as `8d2510d3`
concluded.

## What was built (brain-based, no sim/ edit, additive)

`research/runners/_teacher_loop_weight_protect_derisk.py` (reuse-by-import: the teacher-loop scaling net-build +
world, the corrective-acquire `ReferentEnv`, and the sleep-replay `Hippocampus`). The readout is the substrate's
own H_last→out synapses (`cp_connections` at the Bellec leaky-readout edges): read at build, moved by the readout
gradient, **committed back into `cp_connections` each fact**. The hidden layers are the FIXED spiking reservoir
(the port's own framing), so all interference AND all protection live on the readout synapses — the SPEC's target.

- **Importance = the brain's own diagonal Fisher** `F_jk = E[(r_j·δ_k)²]` over the hippocampal store's replayed
  activity (the readout-loss curvature per synapse), computed from `r` (substrate features) and `W` only — **no
  host freeze-list, no label lookup**. Gate: `gain_jk = 1/(1 + λ·F̂_jk)` — high-importance (consolidated)
  synapses resist change; new-fact plasticity flows to the low-importance ones. Applied host-side in the readout
  trainer (the substrate's `cp_plasticity_rate_gain` field is **unusable** — allocating it perturbs the forward
  pass 60→36 spikes because it activates a bridge-side plasticity path; not byte-identical-when-off, so rejected).
- **Fixed orthogonal targets** (PS-SNN / DG decorrelation): rows of a random orthonormal K×K matrix, predefined
  once, never learned; NLMS regression to the code, nearest-centre classification.

Four arms, one net-build / seed / epoch-budget / replay-schedule — the ONLY difference is the gate (and, for the
ortho arm, the fixed code): `replay` (baseline, no gate), `protect` (importance gate, λ=8), `scramble` (SAME gate
amount, importance permuted across synapses — the structure control), `orthoprot` (orthogonal targets + gate).

## Result — 6 seeds (42–47), N=10, chance 0.10

Aggregate artifact (all seed means + deltas below): `research/findings/raw/teacher_loop_weight_protect_aggregate.json`
(per-seed: `research/findings/raw/teacher_loop_weight_protect_s42.json` … `_s47.json`).

<!--derived-->

| arm | frac-recalled@N=10 (6-seed mean ± sd) | vs replay |
|-----|----------------------------------------|-----------|
| replay (self-replay baseline) | **0.683 ± 0.134** | — |
| protect (EWC/SI importance gate) | **0.700 ± 0.153** | **+0.017** (1/6 seeds) |
| scramble (same protection, wrong synapses) | 0.700 ± 0.153 | +0.017 |
| orthoprot (PS-SNN orthogonal codes) | 0.483 ± 0.195 | **−0.200** |

- **(a) NO rise.** protect 0.700 ≈ replay 0.683 (+0.017, inside 1 sd; protect>replay in only 1/6 seeds; reaches
  ≥0.8 in 2/6). The target 0.8+ is not met.
- **(b) importance STRUCTURE is NOT load-bearing.** `protect == scramble` (0.700 == 0.700; margin +0.000). The
  negligible effect is a *global plasticity brake*, indistinguishable from protecting random synapses — the
  Fisher structure (which synapses) carries nothing here. This is the decisive tooth: the mechanism's whole
  premise (protect the RIGHT synapses) has no measurable purchase.
- **(c) immediate acquisition preserved** (the one thing the gate does do): protect immediate-acq 0.959 vs
  replay 0.991 (EWC new-learning tradeoff −0.032 — small, as EWC predicts). So the gate does not *block* new
  learning; it simply adds no retention.
- **orthogonal codes HURT** (−0.200): fixed distributed codes are harder for this small-perceptual-category
  readout to hit than one-hot targets, and the drifting softmax was never the bottleneck.

**Harder regime (N=20, hidden=20, 3 seeds), where replay drops to 0.45 — below the 0.55 wall, with real
headroom:** protect 0.467 vs replay 0.450 (+0.017); **protect < scramble** (−0.017). The negative holds where the
wall bites hardest — it is not an artifact of a saturated easy operating point.

## Why (the mechanism, load-bearing)

<!--derived-->
Self-replay **re-derives the readout from the stored engrams every night** (interleaved reconsolidation of every
fact). Weight protection tries to *keep* the readout from moving; replay makes keeping-it unnecessary because it
*rebuilds* it. The two are redundant, so protection adds nothing on top of replay — and can only conflict (which
is why gating replay itself, in an earlier lever, was slightly worse than gating wake only). The retention
ceiling (~0.68 at N=10) is set by the **lossy hippocampal engram** (a single prototype per fact), NOT by readout
plasticity. This **reinforces `8d2510d3`**: the load-bearing lever is engram-store fidelity, not the readout and
not replay magnitude — and now, not weight protection either.

## Teeth summary

<!--derived-->
(a) retention rise vs replay: **FAIL** (+0.017, 1/6). (b) importance structure load-bearing: **FAIL** (protect ==
scramble). (c) immediate acquisition high + tradeoff measured: **PASS** (0.959; −0.032). (d) capacity holds:
PASS (protect mean-retained-acc ~0.55). Orthogonal-target teeth: **FAIL** (−0.200, honest negative on PS-SNN
codes in this regime). The controls (scramble; the N=20 harder regime) are what give the negative its teeth —
the mechanism is not merely unhelpful, its causal premise (protect the important synapses) is falsified here.

## Reproduce

Single-seed smoke → 6-seed (numpy; the net is tiny — 41 neurons — so numpy avoids GPU contention; SIM_BACKEND=cupy
for the 3090):
```
for s in 42 43 44 45 46 47; do SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m research.runners._teacher_loop_weight_protect_derisk --seed $s \
  --milestones 1 5 10 --n-max 10 --epochs 40 --replay-epochs 24 --n-draws 32 --lam 8.0 \
  --out research/findings/raw/teacher_loop_weight_protect_s$s.json & done; wait
```
Artifacts: `research/findings/raw/teacher_loop_weight_protect_s{42..47}.json`.

## What to try next (the capability is NOT abandoned — the METHOD is)

The negative points the same direction `8d2510d3` did: **biologize the engram store**. Raise retention by making
the hippocampal trace richer than a single lossy prototype (multi-trace / pattern-separated multi-exemplar
engrams; a generative engram that samples the fact's manifold, van de Ven 2020) so replay reconsolidates a
higher-fidelity readout — rather than trying to freeze a readout that replay is already rebuilding. Weight
protection may yet matter in a regime WITHOUT replay (protection as a *replacement* for replay, not an
add-on) or on the HIDDEN reservoir if it were plastic — both untested here and worth a separate lever.
