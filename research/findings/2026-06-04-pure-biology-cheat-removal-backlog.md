# Pure-biology backlog: cheats/shortcuts to remove (standing directive) — 2026-06-04

**Owner directive (2026-06-04):** *"When possible (no other high-priority work remaining), I'd like to remove
any remaining cheats/shortcuts to get the sim back into a pure-biology-backed state."*

This is a **standing backlog**, not immediate work — it runs when no higher-priority task remains. It serves
the top-line goal directly (artificial life with a *proper* brain analogue; biology-translatable insights;
honest negatives under strict biology are the deliverable). This doc is the honest, growing audit of where
the current system still departs from pure biology, so the departures are explicit and removable rather than
silent.

## Current departures from pure biology (honest audit, this arc)

| # | Shortcut | Where | Pure-biology target |
|---|---|---|---|
| 1 | **Composition runs as numpy *algebra*, not spikes.** The phasor FHRR composition agent (nesting, resonator decode, bind/unbind) computes with complex-number arithmetic; the brain analogue is the *spiking* realization. | `nested_composition_agent.py`, `phasor_associative_memory.py`, `phasor_chat.py` | The resonate-and-fire spiking substrate (validated in pieces: `_spiking_*_probe.py`; learning de-risked `_spiking_stdp_phasor_learn_probe.py`). Assemble the full agent in spikes. |
| 2 | **BG gate selection is a stand-in.** Thalamic gate pools are driven *directly* with input current as a proxy for basal-ganglia disinhibition. | `gated_compose_bg_demo.py` (`bind_via_bg`) | Wire the disinhibition to the real `g11_bg` GPi→thal pathway so the BG (via striatum→GPi→thal) genuinely selects which gate opens. |
| 3 | **Gating binding is *commanded*, not learned/emergent.** "4/4 deterministic" is by construction (you bind exactly what you gate). | gated-compose demos | A mechanism where the *right* gate is selected by upstream dynamics/learning (the Logiaco Option-C low-rank gate; or learned BG action selection), not an external bind command. |
| 4 | **Codes are random/constructed or learned-but-not-grounded.** Concept codes are random phasors or STDP-learned from a hashed word encoder, not from real sensory grounding. | phasor agents; `vocab_to_drive_pattern` | Sensory-grounded codes (the visual-cortex / Gabor pipeline, Cluster K v2) feeding the concept representations. |
| 5 | **Phasor binding is a biologically-grounded *hypothesis*, not an established brain mechanism.** | the whole FHRR substrate | Not removable per se — but keep it labelled as a hypothesis; strengthen the biological grounding (resonate-and-fire, theta-gamma phase coding, the high-Q/Ih insight). |
| 6 | **Generation baseline uses a non-biological transformer teacher** (Generator-F) for the fluency reference. | generator arc | Acknowledged secondary/labelled; the from-scratch spiking generation is a documented wall. Keep clearly separated as "what spiking *inference* can do given non-biological weights." |
| 7 | **Older navigation/RL cheats** (heuristic, direct (x,y)/(gx,gy) access) — documented long-standing. | `g11_bg_runner.py` flags | The perception-arc closures already removed most (4 of 5 cheats closed, 2026-04-27); audit what remains and whether it's still used. |

## Process when this backlog is worked

For each item: (a) confirm it is actually still load-bearing (some may already be vestigial); (b) build the
pure-biology replacement; (c) re-validate the capability under the stricter regime — **a capability that only
worked with the shortcut, and honestly fails without it, is itself a finding** (the deliverable is honest
negatives under strict biology). Do NOT quietly keep a shortcut to preserve a number.

This list grows as more departures are noticed. It is the explicit ledger that keeps "proper brain analogue"
honest.
