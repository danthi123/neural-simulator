# Brain-analogue unification: the spiking unified agent on real-V1-sensory-grounded codes — 2026-06-04

**One line:** The two validated brain-analogue threads — the GENUINE-SPIKES unified agent (Orchard-2023 phasor
populations: fact memory + who/what + abstention + 1/2-attribute composition + embedded clauses) and SENSORY
GROUNDING (real V1 Gabor receptive fields + ventral-hierarchy decorrelation) — are unified end-to-end: the
spiking agent runs the full frozen benchmark on concept codes derived from a real biological V1 bank instead of
random/constructed codes — at **72/72 = 100% multi-seed (2 seeds), constructed parity in genuine spikes**, with no
spiking-quantization cost on the core categories.

## Why this is the capstone of the session

The cheat-removal arc resolved two pieces independently this session:
- The full composition agent runs in GENUINE SPIKES at 320 concepts (`spiking_unified_agent`, Orchard-2023
  substrate) — the brain-analogue, not numpy algebra.
- Real Gabor-V1 sensory grounding + ventral-hierarchy decorrelation feeds the NUMPY agent at **constructed
  parity (92.3%, 6-category core 100%)** — cheat-backlog #4 (ungrounded codes), resolved for the visual subset.

This unifies them: the brain-analogue (spikes) on the biology-grounded codes (V1 features) — the most complete
brain-analogue conversational artifact in the project: every operation a population of spiking-phasor integrator
neurons, every concept code derived from a real receptive-field bank.

## Setup

`research/runners/spiking_unified_agent_grounded.py` (reuse-by-import). Each of the 320 benchmark concepts →
a distinct synthetic visual stimulus → the real V1 Gabor bank (`sim/visual_cortex.py`, 8192 simple cells) → V1
response → ZCA decorrelation (ventral-hierarchy stand-in) → a fixed complex projection → phase angles → converted
to [0,1) and realized as integer spike steps (`phases_to_spikes`) = the concept's spiking-phasor symbol. The
`SpikingUnifiedAgent` gained a backward-compatible `external_phases` hook (default None = random symbols; existing
3 tests still pass) so grounded codes drop in. Then the same frozen test set runs (flat / 1-attr / 2-attr /
clause-depth1 / who / abstain), multi-seed.

## Result

| substrate / codes | flat | 1-attr | 2-attr | clause-d1 | who | abstain | overall |
|---|---|---|---|---|---|---|---|
| numpy + grounded + decorrelate (reference) | 100% | 100% | 100% | 100% | 100% | 100% | 92.3%* |
| **spiking + grounded + decorrelate (2 seeds)** | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** | **72/72 = 100%** |

\*numpy overall 92.3% = 6-category core 100% with clause-depth2 the documented ceiling (clause-depth2 not run in
the spiking core-benchmark harness, which covers flat/1-attr/2-attr/clause-depth1/who/abstain — the comparison is
the numpy 6-category core 100% vs the spiking 72/72 = 100%).

The genuine-spikes brain-analogue reproduces the full core benchmark on real-V1-grounded codes at **100%**,
identical to its result on constructed codes — so the spike-step quantization (`CYCLE_STEPS`) costs nothing on the
core categories, and the sensory grounding feeds the spiking agent exactly as it fed the numpy agent. The
ventral-hierarchy decorrelation makes the grounded V1 codes constructed-quality (low inter-code coherence), which
the spiking phasor populations handle natively (bind = phase-sum neurons, unbind = phase-subtraction neurons,
clean-up = winner-take-all by spike-phase similarity, the F=3 resonator for two attributes). This is the session's
capstone: fact memory + who/what Q&A + abstention + one/two-attribute composition + embedded clauses, every
operation a population of spiking-phasor integrator neurons, every concept code derived from a real biological V1
receptive-field bank — the most complete brain-analogue conversational artifact in the project.

## Honest scope

- The grounding pipeline is real V1 Gabor; the per-concept stimuli are synthetic distinct textures (no natural
  images for abstract words — the embodied-cognition limit). The decorrelation is a ZCA stand-in for the ventral
  hierarchy.
- The spiking substrate quantizes phases to integer spike steps (`CYCLE_STEPS`), so a small accuracy cost vs the
  continuous numpy algebra is expected and is itself the honest measurement of spiking-realization fidelity.
- Reuse-by-import; the only edit was the backward-compatible `external_phases` hook on the spiking agent (its 3
  tests still pass).

## Files

- `research/runners/spiking_unified_agent_grounded.py` — the unification runner.
- `research/runners/spiking_unified_agent.py` — gained the `external_phases` hook (+ `run_core_benchmark`
  passthrough).
