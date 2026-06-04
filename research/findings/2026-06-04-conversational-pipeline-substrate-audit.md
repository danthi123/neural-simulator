# Conversational pipeline substrate audit — what runs on the core sim vs bolted-on modules — 2026-06-04

**One line:** The core `SimulationBridge` already has VALIDATED spiking realizations of nearly every conversational
capability (bind/unbind, relational KB + Q&A, abstention, negation, learned parser, dlPFC content-selection,
grounding, generation) — but the "unified" agents (`nested_composition_agent`, `spiking_unified_agent`, the
benchmark, the demos I shipped this session) **bypass them** and run on two **bolted-on standalone numpy-spiking
abstractions** (`spiking_phasor_fhrr`, `resonate_fire_fhrr`). The one capability with NO core-sim realization
anywhere is the **F=3 two-attribute resonator**. Consolidation = wire the validated core-sim primitives into one
bridge-based agent and retire the abstractions.

## Three substrates

- **(A) CORE SIM** = the real `SimulationBridge` (`sim/bridge.py`): Izhikevich/HH neurons, `_run_one_simulation_step`,
  `cp_firing_states`, `inject_explicit_wiring`, the BrainRegion framework. This is "integrated into the core sim."
- **(B) PHASOR-SPIKING ABSTRACTION** = `research/runners/spiking_phasor_fhrr.py` + `resonate_fire_fhrr.py` —
  standalone numpy time-stepped "spiking" (phase-sum/subtraction integrators; resonate-and-fire damped oscillators
  in `np.exp`). NOT the bridge. These are bolted-on duplicate spiking simulators.
- **(C) PURE NUMPY** = complex/matrix algebra, no spiking (`nested_composition_agent` bind = elementwise multiply).

## Capability map

| capability | unified agent uses | core-sim (A) exists? | core-sim file |
|---|---|---|---|
| concept grounding (V1 Gabor / word encoder) | A (this session) | **yes** | `sim/visual_cortex.py`, `sim/text_embeddings.py` |
| syntactic parser / role assignment | C (numpy) | **yes** | `_insubstrate_parser_stdp_probe.py` (Hebbian conj→role, 6/6) |
| bind (role⊗filler) | B | **yes** | `_insubstrate_bind_unbind_probe.py` (coincidence banks, ≥0.80 multi-seed) |
| unbind | B | **yes** | same |
| bundle / superposition | B | **yes** | `_insubstrate_relational_memory_probe.py` |
| cleanup (codebook WTA) | B | **yes** | `_insubstrate_bind_unbind_probe.py` + `abstention_gate.py` |
| **resonator (F=3 two-attribute)** | B/C | **NO** | — *(only numpy `_resonator2/3` or the rf abstraction)* |
| KB / relational fact memory | B/C | **yes** | `_insubstrate_relational_memory_probe.py` (multi-seed RESOLVES) |
| who/what Q&A | B/C | **yes** | same + `_insubstrate_qa_probe.py` |
| abstention / no-confab moat | B/C | **yes** | substrate cleanup-confidence (`_insubstrate_*`) |
| negation / yes-no | (not in unified) | **yes** | `_insubstrate_negation_probe.py` (4-role polarity, ≥0.80) |
| dialogue planning / content-selection | C (`content_selection.py`) | **yes** | `content_selection_spiking.py` (dlPFC NMDA WM on the bridge) |
| generation | B + C | **yes (partial)** | `generative_replay_loop.py` (SWR replay on the bridge) |

**11 of 13 capabilities have a validated core-sim (A) realization. The lone gap is the F=3 two-attribute
resonator** — it exists only as numpy phasor algebra (`nested_composition_agent._resonator3`) or the
resonate-and-fire abstraction.

## The bolted-on modules to consolidate

1. **`spiking_phasor_fhrr.py`** — function-first integrator neurons (numpy), the substrate the capstone + numpy
   agents actually compute on. Imported by `nested_composition_agent`, `spiking_unified_agent`.
2. **`resonate_fire_fhrr.py`** — the "biologized" resonate-and-fire variant (numpy), still not the bridge.
3. The **unified agents** (`nested_composition_agent`, `spiking_unified_agent`, `unified_agent_benchmark`, and the
   `unified_agent_*` runners I shipped this session) — clean agents, but on (B)/(C), not the bridge.

## What's already on the core sim (and validated)

The core-sim realizations EXIST and are multi-seed validated — but they live in `research/findings/raw/` as
`_insubstrate_*` probes (archived), used only by the owner-facing DEMOS (`compose_spiking_bind_demo`,
`compose_relational_memory_demo`, `compose_conversation_repl`, `compose_live_text_kb_demo`), NOT by the unified
agents. So there are TWO parallel tracks: a validated-but-archived core-sim track (demos) and a clean-but-bolted-on
abstraction track (unified agents). `compose_conversation_repl.py` is already an interactive core-sim REPL with
bind + negation.

## Honest scope of consolidation

To make the sim clean + self-contained:
1. **Promote** the validated `_insubstrate_*` core-sim primitives from `findings/raw/` into proper modules.
2. **Build ONE core-sim unified agent** on those primitives (bind/unbind/cleanup/KB/Q&A/abstention/negation/parser/
   grounding/content-selection) — replacing the numpy/abstraction unified agents.
3. **The F=3 resonator** is the real open decision: either build a core-sim two-attribute mechanism (the
   enumeration factoring the spiking agent used for 1-attribute IS just bind/unbind+cleanup = bridge-able; F=3 is
   harder), or scope two-attribute composition out, or keep a clearly-labelled numpy reference for it only.
4. **Retire / relabel** `spiking_phasor_fhrr` + `resonate_fire_fhrr` as numpy *reference* implementations, not the
   production substrate.

This is a real multi-step arc; the pieces (except the resonator) are validated and present, so it's assembly +
promotion + an honest resonator decision, not new research.
