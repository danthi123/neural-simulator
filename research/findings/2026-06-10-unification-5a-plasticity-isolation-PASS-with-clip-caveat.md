# Unification de-risk 5a — plasticity-update isolation PASSES; the global weight CLIP is the one ungated gap (2026-06-10)

**Roadmap step 2 (consolidate navigation + conversational onto one brain), cheapest-first de-risk 5a**
(`docs/plans/2026-06-10-nav-conv-single-instance-unification-design.md` §5a). This is the REQUIRED first
move on BOTH the looser and the strict (RF co-resident) merge paths, because the all-Izhikevich
nav + parser + dlPFC merge depends on it either way.

## The question

On ONE shared bridge that runs the navigation brain's learning (reward-modulated STDP + a global dopamine
neuromodulator whose `plasticity_rate` target has scope="all") AND the parser's Hebbian learning (both are
GLOBAL config flags), does a per-synapse plasticity gate held at 0.0 keep a fixed conversational population
byte-identical while the ungated populations still learn — and does a navigation-length burst corrupt a later
conversational read of the frozen slice?

The 2026-06-04 conversational unification already proved the gate freezes a fixed population against global
**Hebbian** (`2026-06-04-unified-bridge-plasticity-isolation.md`). 5a adds the navigation stressor the earlier
test never exercised: reward-modulated STDP + the dopamine scope="all" plasticity-rate multiplier, plus
step-coexistence.

Probe: `research/runners/derisk_unification_5a_plasticity_step_isolation.py` (GPU/CuPy, ~210-neuron
brain-region-framework bridge: nav_ctx→nav_d1 ungated reward-STDP learner; conv_a→conv_b frozen
[`plasticity_gate=0.0`]; parser_a→parser_b ungated Hebbian control). Forensics:
`_derisk_5a_diag.py`, `_derisk_5a_isolate.py`.

## Result — a two-part answer

**(1) The per-synapse plasticity gate ISOLATES weight UPDATES — PASS.** With the clip bounds set above the
frozen population's weight (`--clip-max 20`, frozen conv weight ≈6):

| Check | Result |
|---|---|
| (a) conv_frozen weights byte-identical after the nav burst | **True** (max\|dw\| = 0.000) |
| (b) nav_learn changed (control non-vacuous: reward-STDP live) | True (max\|dw\| = 1.52) |
| (c) parser_learn changed (control non-vacuous: Hebbian live) | True (max\|dw\| = 0.71) |
| (d) conv read identical before vs after the nav burst (step coexistence) | **True** (read1 == read2 exactly) |
| dopamine end-concentration (confirms the scope="all" stressor ran) | 1.11 (baseline 0.5) |

Every weight-UPDATE path is correctly gated by `cp_plasticity_rate_gain`: Hebbian potentiation
(`sim/bridge.py:6157`), Hebbian decay (`:6171`), STDP delta (`:6269-6270`), and the reward
eligibility→weight conversion (`:6456-6457`). gain=0 ⇒ each multiplies to zero. The frozen slice is
byte-stable under the combined nav reward-STDP + dopamine + Hebbian stressor, and a navigation burst does not
corrupt a subsequent conversational read.

**(2) The global weight CLIP is NOT gated — the one real gap.** The first 5a run FAILED with the frozen slice
drifting by **8.718** — about as much as the ungated controls. Root cause (systematic-debugging, not a guess):
two `cp.clip(self.cp_connections.data, w_min, w_max)` calls clip the WHOLE weight array to the active rule's
bounds, **regardless of the plasticity gate**:
- `sim/bridge.py:6175` (Hebbian block) — clips to `[hebbian_min_weight, hebbian_max_weight]` (defaults
  `[0.05, 1.0]`).
- `sim/bridge.py:6480` (reward block) — clips to `[w_min, w_max]` where `w_max = stdp_w_max if enable_stdp
  else hebbian_max_weight` (defaults `[0.0, 2.0]`).

A frozen synapse whose weight lies OUTSIDE the active rule's clip bounds is moved by the clip, not by an
update. The probe's frozen weight (6.0) exceeds the default `hebbian_max_weight=1.0`, so whenever Hebbian ran
the clip slammed it to 1.0 (peak conv ≈9.7 → 1.0 ⇒ dw ≈8.7, the exact observed number). Isolating each rule
confirmed it: "stdp only" and "reward+stdp" FROZE the slice (conv=6 fits inside `stdp_w_max=30`), while
"hebbian only" and "reward-only (stdp off)" MOVED it (the clip uses the `[0.05, 1.0]` Hebbian bounds). Running
the probe with `--clip-max 1.0` reproduces the drift deterministically; `--clip-max 20` (bounds above the
frozen weight) gives the clean PASS above.

## Why this matters for the merge — and the mitigation

The design's §4.3 trust-but-verify checked the four weight-UPDATE gating sites (all correct) but not the clip.
5a closes that gap. The frozen real-valued conversational populations on the shared `cp_connections` (the
parser's fixed role-routes ≈300, the dlPFC fixed graph edges, and a rate-composer bind population ≈320 if one
is co-resident) have weights ABOVE the navigation's clip bounds (`stdp_w_max=150` for the actor;
`hebbian_max_weight=1.0`). On a naive merge their weights would be clipped during navigation reward steps (to
`stdp_w_max`) and during any Hebbian step (to `hebbian_max_weight`).

**Key mitigations for the merge (task 12):**
1. **Raise the clip bounds above the frozen population's max real-valued weight** (`stdp_w_max` and
   `hebbian_max_weight` ≥ the composer/parser/dlPFC max). Simplest; tension is it loosens the navigation's
   STDP soft-bound, so verify the nav actor does not over-grow (it asymptotes well below `w_max` by design).
2. **Keep frozen conversational real-valued weights within the nav's existing bounds** (scale role-routes to
   ≤ `stdp_w_max` with compensating drive).
3. **Gate the clip in `sim/`** (skip / preserve gain=0 synapses) — cleanest, and it can be folded into the
   already-required protected RF-coexistence edit (one byte-review).

**Important narrowing for the STRICT (RF co-resident) path:** the FHRR resonate-and-fire composer's BINDING
weights are COMPLEX (`cp_rf_w_re` / `cp_rf_w_im`), array-disjoint from `cp_connections`. The clip only touches
`cp_connections.data`, so the RF composer's binding weights are **immune** to this gap. Only the real-valued
frozen conversational populations (parser role-routes, dlPFC fixed edges) are exposed.

## Verdict

**5a PASS on its load-bearing claim** (the plasticity gate isolates weight updates + step coexistence holds),
with one precisely-characterized, fully-mitigable caveat (the ungated global weight clip). The merge is not
blocked; it carries a concrete, documented requirement: the frozen conversational real-valued weights must sit
within the shared bridge's clip bounds (raise the bounds, scale the weights, or gate the clip). This is the
measured cost the cheap-first de-risk was meant to find — surfaced before any protected-module build.
