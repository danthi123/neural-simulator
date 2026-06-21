# World & Sensory Scaling Roadmap

> Forward roadmap for sensory acuity, world richness, and artificial-life scaling.
> **GATED behind the active shortcut-closure arc + one-brain consolidation + conversational parity.**
> This is not active work — it structures "what's next once the brain is clean, fully spiking,
> hardware-portable, and conversationally capable." Captured 2026-06-21 from the retina-resolution
> and rich-world scoping discussion.

## Where this sits — the sequencing gate

Deliberately downstream. The order (owner, 2026-06-21):

0. **(ACTIVE) Shortcut-closure arc** → a fully-spiking, hardware-portable one-brain (every cognitive
   op spiking at runtime AND with on-substrate structure). *This is the current and only priority.*
0.5. **Core-capability fleshing-out + conversational parity** — the conversational stack matching a
   tiny/small LLM.
1. **Foveated visual acuity scaling** (Phase 1) — the near-term, 3090-fitting sensory upgrade the owner
   specifically wants.
2. **Larger, richer 2D world** (Phase 2) — the artificial-life on-ramp.
3. **Artificial-life / "Thronglets" capstone** (Phase 3) — the project's stated actual goal.
4. **3D embodiment** (Phase 4) — explicitly deferred; a genuine research chapter for later.

Maps onto the existing roadmap tiers (Tier 3 = artificial-life capstone). The scaling phases (1–2) are
the bridge from conversational parity to the a-life capstone.

## The load-bearing principle: the world is the environment; the brain stays spiking

Under the project's brain-based-only standard, host code is legitimate for exactly two things: the
**environment** (world state + rendering the agent's sensory view) and the **body** (acting on motor
output). Everything cognitive in between stays spiking. So every phase below is built the same way:

- The **world** can be an arbitrarily rich engine (host) — a bigger grid, a physics sim, many agents.
- The **brain** interfaces through two narrow ports: sensory-in (the retina) and motor-out (which pools fire).
- Scaling the world does NOT mean a new brain — it means a richer environment plus (where needed) scaling
  or extending the existing spiking mechanisms.

This is *why the shortcut arc comes first*: building a richer world around a clean, fully-spiking brain is
"more of the same"; building it around a brain riddled with host shortcuts would compound the mess. And the
hardware-port property (neuromorphic-compatible) is what eventually makes the heaviest phases
(many-agent a-life, 3D) feasible.

---

## Phase 1 — Foveated visual acuity (near-term, 3090-fitting)

**Goal:** much higher *effective* visual acuity without the uniform-resolution compute explosion, staying
comfortably inside the RTX 3090's 24 GB.

**Why foveation, not a uniform bigger sheet:** uniform 1080p is ~2,000× the current 32×32 → ~20M visual
neurons + ~1B synapses → blows past the 3090. The brain doesn't do that — it foveates: a tiny high-acuity
fovea + low-acuity periphery + saccades to point the fovea where attention is. That buys high effective
resolution *where it matters* at roughly the cost of a modest uniform bump, and it is strictly more
biology-faithful and more neuromorphic-friendly.

**Components:**
- *(host/env)* **Foveated sampling** of the rendered scene — full acuity in a small movable fovea window,
  log-polar / coarse downsampling in the periphery. Keeps total "pixels" modest.
- *(brain)* **Cortical-magnification V1** — more V1 cells per degree in the fovea, fewer in the periphery
  (the biological cortical magnification factor).
- *(brain)* **Saccadic control** — the agent moves its fovea to salient locations. **This reuses the
  superior-colliculus orienting machinery being closed right now in the #6 work** — the SC that orients
  toward salience *is* the saccade generator. A direct payoff from the current arc.
- *(brain)* the existing V1→V2→IT hierarchy consumes the foveated input (object recognition already lives here).

**What's built:** the 32×32 retina + V1 + the visual hierarchy; the SC orienting (#6, in closure now); the
B1 self-organized receptive fields (just de-risked GO — the RF bank scales without host-designed Gabors).
**What's new:** the foveated / log-polar sampling, the fovea↔periphery V1 split, saccade-driven re-sampling
(gaze moves → re-render the fovea), and integrating across saccades (trans-saccadic memory — a known mechanism).

**Cost / 3090-fit:** foveation is the whole point — the fovea stays small and the periphery is coarse, so the
visual front-end grows from ~10K to maybe ~50–150K neurons — a modest bump that fits the 3090 with room to
spare. Start small (e.g. a 64×64 fovea), grow as headroom allows. (Contrast: a *uniform* 128×128 is ~16× /
~300K visual neurons — still 3090-fittable, a legitimate simpler fallback if foveation proves fiddly.)

**Gates:** (a) a fine-detail discrimination task the 32×32 retina fails but the foveated retina passes;
(b) the SC saccades land the fovea on task-relevant regions; (c) stays in 24 GB; (d) the self-organized RFs
(B1) scale to the larger V1 without host Gabors.

---

## Phase 2 — Larger, richer 2D world (the a-life on-ramp)

**Goal:** a bigger 2D world with terrain, objects, resources, hazards, and — crucially — multiple agents:
the substrate artificial life needs.

**Components:**
- *(host/env)* **The world engine** — a bigger grid with terrain, objects, food/resources, hazards; cheap
  to run (2D is nearly free); a-life/game engineering, not a research problem.
- *(brain)* **Scaled place/grid code** — more place cells to tile the bigger world; grid cells for path
  integration over larger distances. A scaling of the existing mechanism.
- *(brain)* **More actions** — beyond N/E/S/W: forage, eat, interact, etc. The BG action-cascade is already
  per-action; this adds channels.
- *(brain)* **Drives / homeostasis** — hunger, fatigue, foraging motivation (the homeostatic-agent work is
  partly built).
- *(brain)* **Multiple agents** — the multi-bridge infrastructure already supports multiple brains.
- *(brain)* **Inter-agent communication** — the conversational stack (parse / compose / no-confab moat /
  multi-turn), built for human↔sim, repurposed for agent↔agent. The elegant part: emergent communication
  between creatures falls almost directly out of what already exists.

**What's built:** the place code, the BG cascade, homeostasis (partial), the multi-bridge infra, the full
conversational stack, the visual hierarchy.
**What's new:** the richer world engine, the place-code + action scaling, the drive system fleshed out, the
inter-agent comms wiring.

**Cost:** gentle — world-richness scales the brain *sub-*linearly (a bigger world needs more place cells, but
the agent's local view + action set do not explode). ~1M neurons for a rich single-agent 2D brain; ×N for N
agents. Single-agent fits the 3090; multi-agent leans on the tiering.

**Gates:** an agent navigates + forages + survives in the rich world; two agents share it and communicate via
the conversational channel.

---

## Phase 3 — Artificial life / "Thronglets" capstone

**Goal:** persistent, evolving, communicating digital creatures — the project's stated actual goal
(artificial life with a proper brain analogue).

**Components (and how much already exists):**
- **Persistence + continual learning** — the lineage system (learn across sessions, no catastrophic
  forgetting). BUILT.
- **Drives / survival** — homeostasis + foraging (from Phase 2). BUILT / extended.
- **Multiple agents** — multi-bridge. BUILT.
- **Emergent inter-agent communication** — the conversational stack as the comms substrate (from Phase 2).
  BUILT (repurposed).
- **Reproduction / evolution** — a genome + mutation + selection layer over the brain's developmental
  parameters. NEW (an a-life layer, not yet built).
- **Emergent dynamics** — communication conventions, "culture," division of behavior arising between agents.
  The open, unpredictable frontier — and the whole point.

**What's new:** the evolution/reproduction layer + the emergent multi-agent dynamics. Most other pieces are
assembly.

**Cost:** ×N agents (tiering / a GPU step-up for many agents). The neuromorphic-port property (from the
shortcut arc) is what makes large colonies eventually tractable.

**Gates:** creatures persist, reproduce, evolve, and develop communication over generations.

---

## Phase 4 — 3D embodiment (deferred)

Explicitly deferred (won't be pursued for a while). It is a genuine new chapter because it needs *new*
spiking mechanisms — all biology-grounded, but real builds:
- 3D engine (host — MuJoCo / PyBullet / Godot).
- 3D place / grid / head-direction cells (the bat-3D-navigation literature).
- Depth perception (the retina stays a 2D projection; depth / occlusion / 3D-shape extraction is new).
- Continuous motor control (population-vector read-out vs discrete N/E/S/W).
- ~few-million-neuron brain + a GPU step-up.

Parked as the long-horizon target; Phase 1's foveated retina + Phase 2's richer perception are partial
down-payments on it.

---

## Cross-cutting: the compute / hardware path

| Phase | Brain size (rough) | Hardware |
|---|---|---|
| Now | ~50–200K neurons | RTX 3090 |
| 1 (foveated retina) | +~50–150K | 3090 (comfortable) |
| 2 (rich 2D, single agent) | ~1M | 3090 with care |
| 2–3 (N agents) | ~N×1M | tiering / GPU step-up |
| 4 (3D) | ~few M | GPU step-up |

The CPU-RAM/SSD synapse tiering (`TieredSynapseStore`, already built) is the lever for the memory-heavy
phases; a single-GPU step-up (A100-class) or multi-GPU for the heaviest. The brain-based-only standard holds
throughout: the world is host, the brain is spiking.

## The recommended on-ramp (when the time comes)

The cheapest first concrete step that exercises the most of this at once: a **larger 2D world with two
creatures sharing it, communicating via the conversational channel, each with a foveated retina and hunger
drives.** That single step touches Phase 1 (foveation) and Phase 2 (rich world, multi-agent, drives,
inter-agent comms) without paying the 3D research cost — and it is recognizably the Thronglets seed.

## Built-vs-new summary

| Capability | Status |
|---|---|
| Spiking brain (perception, place code, action selection, conversation) | BUILT (being finalized in the shortcut arc) |
| Continual learning / persistence | BUILT |
| Multi-agent infrastructure | BUILT |
| Homeostasis / drives | PARTIAL |
| Self-organized V1 RFs (acuity scaling enabler) | DE-RISKED (B1, 2026-06-21) |
| SC saccade control (foveation enabler) | being closed now (#6) |
| Foveated retina sampling + cortical magnification | NEW (Phase 1) |
| Rich 2D world engine | NEW (Phase 2) |
| Inter-agent communication wiring | NEW (Phase 2, reuses the conversational stack) |
| Reproduction / evolution | NEW (Phase 3) |
| 3D spatial / depth / continuous-motor | NEW (Phase 4, deferred) |

---

*This roadmap intentionally specifies direction and dependencies, not bite-sized tasks — those get a
dedicated implementation plan per phase when that phase is actually scheduled (after the shortcut arc and
conversational parity). The key durable claims: (1) the world is host / the brain is spiking, so these are
environment + scaling work, not a new brain; (2) foveation, not uniform resolution, is the compute-efficient
and biology-faithful path to acuity, and it reuses the SC work; (3) the a-life capstone is mostly assembly of
already-built pieces plus an evolution layer; (4) 3D is the one genuinely new-mechanism chapter, deferred.*
