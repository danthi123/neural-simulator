# Spiking unified agent — building the brain analogue (pure-biology backlog #1) — 2026-06-04

**One line:** The owner chose (c) — realize the unified agent in genuine spikes. Stage 2a SHIPPED: the **flat
robust core** (flat SVO facts + who/what Q&A + abstention) runs in spikes at the benchmark's 320-concept vocab
and **reproduces the numpy benchmark at 100%** (40/40, 2 seeds), including the no-confabulation moat. Each
operation is a population of spiking-phasor neurons; the benchmark is the spec and gate.

## Context

The unified-agent benchmark runs as numpy phasor algebra (fast, but not the brain analogue). Backlog item #1 is
to run the *same* agent in genuine spikes. A scoping pass (`docs/plans/2026-06-04-spiking-unified-agent-scoping.md`)
found the spiking substrate largely already exists and is validated; a cheap de-risk
(`_spiking_pattern_completion_probe.py`, RESOLVES) confirmed the one new load-bearing piece — pattern-completion
cleanup — works in spikes. So this is an integration build, staged + benchmark-gated.

## Stage 2a — the flat robust core, SHIPPED

`research/runners/spiking_unified_agent.py` (`SpikingUnifiedAgent`): stores flat subject-verb-object facts and
answers who/what + abstains, entirely on the validated `spiking_phasor_fhrr` substrate (Orchard & Jarvis 2023):
- a fact = `bundle(role_AGENT ⊗ agent, role_ACTION ⊗ action, role_PATIENT ⊗ patient)` — phase-sum neurons bind
  each role-filler, phase-midpoint bundles them;
- a query unbinds a role (phase-subtraction neurons) and cleans up to the vocabulary (winner-take-all by
  spike-phase similarity + an abstention threshold).

Driven by the benchmark's frozen test set (so it is comparable category-for-category), `N_dim=512`:

| category | spikes | numpy benchmark |
|---|---|---|
| flat | **16/16 = 100%** | 100% |
| who-query | **12/12 = 100%** | 100% |
| abstain | **12/12 = 100%** | 100% |

OVERALL flat core: **40/40 = 100%** (2 seeds). The no-confabulation moat (abstain on an in-vocabulary but
never-stored pair) holds in genuine spikes. `tests/test_spiking_unified_agent.py` guards it.

## Stage 2b — one-attribute composition, SHIPPED

The patient as adjective⊗noun ("red ball"), decoded in spikes by a **two-factor enumeration factoring**: unbind
the patient role, then for each adjective unbind it (phase-subtraction neurons) and clean up to the nouns — the
adjective whose unbind yields the best clean noun is the attribute. Enumeration (not the iterative resonator)
sidesteps the resonator's "problem of 2" (F=2 limit cycles) and is robust for two factors; it is genuinely
spiking (unbind + clean-up populations). Flat-vs-attributed is auto-detected by comparing the flat-noun clean-up
similarity against the best adjective-factoring similarity. Flat and one-attribute facts are stored together, so
the decode must discriminate.

Result (`N_dim=512`, 2 seeds), full robust core:

| category | spikes | numpy benchmark |
|---|---|---|
| flat | 16/16 = 100% | 100% |
| **one-attribute** | **12/12 = 100%** | 100% |
| who-query | 12/12 = 100% | 100% |
| abstain | 12/12 = 100% | 100% |

OVERALL robust core: **52/52 = 100%** (2 seeds). The full robust core — fact memory, two-factor attribute
composition, who/what Q&A, and the no-confabulation moat — now runs in genuine spikes and matches the numpy
benchmark category-for-category. No spurious attributes on flat facts, no missed attributes on attributed facts.

## Stage 2c — the BIOLOGICAL resonate-and-fire substrate, RESOLVES

Stages 2a/2b run on `spiking_phasor_fhrr` (genuine time-stepped spiking, but Orchard's function-first *integrator*
neurons). Stage 2c re-runs the SAME robust core on `resonate_fire_fhrr` — the genuine biological
**resonate-and-fire** neuron model (Izhikevich 2001 / Frady-Sommer 2019): bind/unbind/bundle on resonate-and-fire
neurons, and clean-up by `ResonateFireTPAM`, a complex-valued **attractor network** whose stable fixed points are
the stored concepts. Abstention is a basin-of-attraction property (the recurrent drive collapses below threshold
for an ungroundable input) — the no-confabulation moat as network dynamics, not a list-argmax threshold.

`research/runners/_rf_unified_agent_probe.py` (`RFUnifiedAgent`), at the frozen test set's CORE vocabulary
(30 nouns / 15 verbs / 12 adjectives):

| category | biological substrate |
|---|---|
| flat | 8/8 = 100% |
| one-attribute | 6/6 = 100% |
| who-query | 6/6 = 100% |
| abstain | 6/6 = 100% |

OVERALL: **26/26 = 100%** — **RESOLVES**. The robust core reproduces on the genuine biological neuron model with
the biological attractor-network cleanup. Honest scope: the resonate-and-fire substrate steps a ~1000-step cycle
per operation and the TPAM settles recurrently, so this validates the substrate at the core vocabulary on CPU;
the reduced vocabulary is an EASIER clean-up than 320, so this is a substrate-works validation, not a
difficulty-matched one. The full-320 resonate-and-fire run is the reserved-GPU build (stage 3).

## Stage 3 — full benchmark in spikes (two-attribute + clauses)

Two-attribute patients ("big red ball" = adj1 ⊗ adj2 ⊗ noun) need the F=3 factoring resonator. Two things made
it work in the spiking agent:

1. **Membrane-state bundle + crosstalk subtraction.** The pure-phase midpoint bundle discards magnitude, and the
   recovered patient's crosstalk (~0.1 similarity to the true product) defeats the F=3 resonator. Keeping the
   complex-SUM bundle (the neuron's subthreshold membrane state, magnitude intact) lets the known agent + action
   role-bindings be subtracted EXACTLY (predictive explaining-away) → the clean patient phasor (similarity
   **1.000** to the true product). The resonator then factors a clean product. Biologically: the spike reads out
   the phase, but the magnitude is the subthreshold membrane potential, and explaining-away is predictive coding.
2. **Dimension + selection.** The F=3 resonator (60 adjectives sharing one codebook) needs **D=2048** (correct at
   2048, sim 0.97; fails at ≤1024). Model selection is a parsimony *upgrade* (flat → one → two, each only if it
   beats the running best by a margin), because for a two-attribute patient BOTH flat and one-attribute sit at
   the noise floor and only two-attribute scores high — a nested "two only if one beats flat" cascade wrongly
   defaults to flat.

**Embedded clauses** ("duck see (cat chase bird)") are decoded by recursive unbinding: a filler is detected as a
clause when unbinding its ACTION role yields a confident verb; the inner agent + action are cleaned up and
explained away (the same crosstalk subtraction), and the inner patient is decoded by recursion. The clause filler
is stored as the spike-phase of its role-binding superposition (the magnitude lives in the membrane-state bundle).
A speed optimization runs the expensive resonator only when neither flat nor one-attribute already explains the
patient cleanly.

**Full benchmark in spikes (`N_dim=2048`, 2 seeds):**

| category | spikes | numpy benchmark |
|---|---|---|
| flat | 16/16 = 100% | 100% |
| one-attribute | 12/12 = 100% | 100% |
| two-attribute | 10/10 = 100% | 100% |
| **depth-1 clause** | **10/10 = 100%** | 100% |
| who-query | 12/12 = 100% | 100% |
| abstain | 12/12 = 100% | 100% |

**OVERALL: 72/72 = 100%.** The spiking agent reproduces **every benchmark category the numpy agent does** —
fact memory, one- and two-attribute composition, embedded clauses, who/what Q&A, and the no-confabulation moat —
in genuine spikes. (Depth-2 clause-in-clause is the documented ceiling in BOTH the numpy and spiking agents.)
Remaining: the full-320 / resonate-and-fire scale-up on GPU (the reserved-GPU run).

## Honest scope + the path to full pure-biology

- **Substrate (stage 2a):** `spiking_phasor_fhrr` is genuine time-stepped spiking, but its bind/unbind are
  Orchard's *function-first integrator* neurons (hand-built counter circuits) — spiking, not yet a biological
  neuron *model*. The project also has `resonate_fire_fhrr` — the **biological** resonate-and-fire neuron
  (Izhikevich 2001 / Frady-Sommer 2019) with `rf_bind/rf_unbind/rf_bundle` AND `ResonateFireTPAM`, a genuine
  complex-valued **attractor-network** cleanup (the biological CA3 pattern-completion the (b) result needs, with
  abstention as a basin-of-attraction property). Migrating the agent onto that substrate + TPAM is the
  pure-biology refinement (**stage 2c — DONE above, 26/26 at core vocab**).
- **One-attribute composition (stage 2b):** SHIPPED (above) via spiking enumeration factoring — 12/12.
- **Two-attribute + clauses at 320 in spikes (stage 3):** the larger, GPU-worthy part (the reserved-GPU run);
  two-attribute needs the F=3 factoring, clauses the recursive decode.

## Verdict

**Stages 2a + 2b + 2c + 3 SHIPPED.** The unified agent runs in genuine spikes and reproduces the numpy benchmark
**category-for-category**: the robust core (flat / who / abstain) at 100% on BOTH the engineering-scaffold
substrate AND the biological resonate-and-fire neuron model + attractor cleanup; and the **FULL benchmark**
(flat + one-attribute + two-attribute + depth-1 clause + who + abstain) at **72/72 = 100%** (2 seeds, N_dim=2048)
on the scaffold substrate. The brain analogue does fact memory, one- and two-attribute composition, embedded
clauses, who/what Q&A, and the no-confabulation moat (as a basin-of-attraction property) — entirely in spikes.
Each stage gated by the same benchmark. Remaining: the full-320 + resonate-and-fire scale-up on GPU (the
reserved-GPU run); depth-2 clause-in-clause is the documented ceiling in both the numpy and spiking agents.
