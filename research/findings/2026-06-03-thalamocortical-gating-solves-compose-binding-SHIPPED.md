# Thalamocortical gating solves the "compose-pathways went silent" binding problem — SHIPPED — 2026-06-03

**One line:** The deep-research Track-2 lever (Logiaco-Abbott-Escola 2021 thalamocortical dynamical gating) is
implemented in the spiking bridge and **solves the v16 compose-binding problem that defeated STDP-grown
weights** — gated routing binds the verb→motor mapping **4/4 deterministically (multi-seed)** and re-binds to
any mapping with **zero weight change**, where grown weights were 5/20 seed-fragile and could not re-bind.

## The problem it solves

The project's compose-pathway work (v16 `verb_pool_X → motor_Y`, grown by STDP from zero-init) "went silent":
an additive weight that must be silent-when-unbound and strong-when-bound is exactly what STDP cannot reach
from zero (vanishing signal). Direct-drive forensics showed the pathway essentially silent after
compose-training; the permuted-mapping anti-cheat showed 5/20 PASS with the true mapping seldom uniquely best
— seed-fragile, not real binding. The deep research diagnosed the root cause: **we tried to store the
composition as grown static weights; biology binds by dynamical gating.**

## What shipped (the primitive)

A per-pathway **multiplicative transmission gate** in `sim/bridge.py`, mirroring the existing plasticity-gate
machinery but scaling synaptic **CURRENT** instead of weight updates (the complement CLAUDE.md flagged as
unimplemented):
- `RegionPathway.transmission_gate: str` (sim/regions.py) — tag a pathway; propagated through the wiring plan.
- `cp_transmission_gain` — per-synapse multiplier applied to `effective_connections_matrix` in
  `_run_one_simulation_step` (fresh matrix, never mutates `cp_connections`; `None` / zero-overhead when unused).
- `bridge.set_transmission_gate(name, value)` — open/close a route at runtime.

12 surgical touch points; regression-clean (53 core CPU tests). Validated in spikes
(`tests/test_transmission_gate.py`): closed → target silent (no current despite non-zero weight); open →
target fires; re-binding reroutes the same source with zero weight change.

## The payoff (the v16 compose problem, solved)

`research/runners/gated_compose_demo.py` + `tests/test_gated_compose.py`: 4 verb pools (GO/COME/STOP/LOOK) +
4 motor pools (N/E/S/W) + 16 verb→motor routes, pre-wired with a **fixed** weight and held **CLOSED**.
Binding `(go, north)` just **opens** gate `g_GO_N`; driving "go" alone then drives `motor_N` through the
dynamics, not a grown weight.

| | Result |
|---|---|
| Bind `{GO:N, COME:S, STOP:W, LOOK:E}` → drive each verb alone | **4/4 deterministic**, seeds 42/43/44 |
| Re-bind `{GO:S, COME:W, STOP:E, LOOK:N}` (a different mapping) | **4/4** for the new mapping |
| Weight change across the re-binding | **0** (sum\|W\| unchanged) |
| STDP-grown weights (the prior approach), for comparison | 5/20, seed-fragile, could not re-bind |

Binding = which gate is open. It is deterministic (you bind exactly what you gate), re-bindable on command,
interference-free (orthogonal gates), and one-shot (no training pass) — the four properties the
thalamocortical literature predicts and that grown static weights lack.

## Honest scope

- This is the **routing/binding** primitive (Option A of the research's design: a per-pathway multiplicative
  gate). It validates that binding-by-gating works in the spiking substrate on the compose vocabulary. The
  full **low-rank effective-connectivity gate** (`J_eff = J_cc + Σ s_k u_k vₖᵀ`, Logiaco Option C) needed for
  *sequencing* bound primitives is the further step, not done here.
- The gate is *set by a bind command* here (an external controller opens the right gate). In the full
  thalamocortical loop the **basal ganglia** select which gate via disinhibition (`gpi → thal → cortex`); the
  `g11_bg` cascade already has that skeleton, so wiring the gate to BG disinhibition is the biology-completing
  next step.
- "Deterministic 4/4" is by construction (you bind what you gate) — which is the *point*: it converts a
  seed-fragile grown-weight binding into a reliable, controllable one. The scientific content is that this
  works in genuine spiking dynamics with zero weight change, not that argmax-of-an-open-gate is surprising.

## Verdict

**SHIPPED.** The transmission-gate primitive is in the bridge, validated in spikes, and solves the v16
compose-binding problem (4/4 deterministic + re-binding, zero weight change) that STDP-grown weights could
not. The biology-faithful fix for "compose-pathways went silent" is real. Next: BG-driven gate selection
(close the thalamocortical loop) and Option C (low-rank gate) for sequencing.
