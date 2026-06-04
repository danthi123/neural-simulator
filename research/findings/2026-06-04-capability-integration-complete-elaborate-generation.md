# Capability integration complete: dialogue-planning first-classed + generation on the core sim — 2026-06-04

**One line:** Stage 1 of the post-consolidation spine. The last functional conversational capability —
dialogue planning (`elaborate`) — is validated at production vocabulary and made first-class on the agent, and
**generation** is folded in. The whole conversational loop now runs through `SimulationBridge`s with no bolted-on
numpy simulator: comprehend → store → recall (who/what) → abstain → negate → clauses → attributes → **generate** →
**plan what to say next**.

## What changed

- **`elaborate` validated at V=320** (`_brain_agent_elaborate320_probe.py`): with the agent built on the production
  320-concept codes and a connected set of stored facts, `elaborate(topic)` returns an **on-topic associate** — and
  not just any neighbor, the **strongest** one (the concept that co-occurs most with the topic), chosen by the dlPFC
  spiking content-selection Control (`SpikingSpreadingController`, a real 2-region `SimulationBridge`). **Seeds
  42/43/44: 4/4 each** (multi-seed). (`elaborate` spreads over the association graph built from the agent's own facts, so its difficulty tracks the
  number of facts, not the vocabulary size — this confirms the capability is wired and working in the consolidated
  agent.)
- **`elaborate` first-classed:** the dlPFC Control is now built lazily and **cached**, rebuilt only when the
  association graph changes — keyed on the graph **content**, not the fact count (a length key would return a stale
  Control for a different fact set of the same size; caught by the smell-test and pinned by a regression test).
- **Generation added:** `composer.render_fact(agent)` / `agent.describe(agent)` produce a sentence about a known
  subject from the **spiking memory** (`'dog go north'`, with the action + patient decoded from the spiking unbind,
  not the stored labels), and **abstain (None)** on an unknown subject — the no-confab moat extends to generation.

## Tests

12/12 on-brain tests pass (`tests/test_brain_conversational_agent.py` 7 + `tests/test_core_sim_composition.py` 5),
including two new ones: `test_generation_describe` (generate + abstain) and
`test_elaborate_cache_invalidates_on_graph_change` (the content-keyed cache regression guard). No regression.

## Where this leaves the pipeline (the honest substrate map)

Functionally complete on the core sim. The remaining items are the post-consolidation spine the owner approved:
- **1.5** measure the real captured-code correlation (cheap, de-risks 2 and 3);
- **2** migrate the load-bearing numpy off the composer — the **cleanup** (`argmax` → spiking attractor cleanup,
  de-risked by the rf TPAM at 320) and then the linear **bundling / ON-OFF opponency** — each matrix-gated, a
  capacity regression reported as the honest cost;
- **3** the fully-grounded run (capture 320 concept-pool activities so the codes are the substrate's own);
- **B** collapse the three functional bridges into one bridge with all regions;
- later: nested-sentence parsing (a new recursive-parser capability, not a substrate migration).

## Files

- `research/runners/core_sim_composition.py` (`render_fact`)
- `research/runners/brain_conversational_agent.py` (`describe`, dlPFC caching)
- `research/findings/raw/_brain_agent_elaborate320_probe.py` + `_brain_agent_elaborate320_s{42,43,44}.json`
- `tests/test_brain_conversational_agent.py` (+2 tests)
