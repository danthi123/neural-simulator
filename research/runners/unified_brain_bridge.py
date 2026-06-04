"""One-bridge unification — Step 1: the conversational PARSER (`BridgeParser`) and COMPOSER
(`CoreSimComposer`) merged onto ONE `SimulationBridge`, their neurons as disjoint index slices.

Per the plan (`docs/plans/2026-06-04-one-bridge-unification-step1-implementation.md`): the two regions
that were two separate bridges now live on ONE bridge while staying capability-equivalent. The parser
slice (6 conjunction units + 3*R role-ensemble neurons, R=40 → 126 neurons) occupies indices 0..125; the
composer slice (8*proj_dim coincidence neurons) starts at index 126. The cross-region hand-off stays
Python-orchestrated in step 1 (the gated synaptic route is step 2; the dlPFC merge is step 3).

Terms (defined once, owner standing requirement — no undefined acronyms):
  * bridge          = one `sim.bridge.SimulationBridge` (a network of simulated Izhikevich neurons).
  * region / slice  = a contiguous block of neuron indices used for one function (parser vs composer).
  * plastic         = synapses whose weights change with learning; fixed = weights never change.
  * Hebbian learning = a co-activation weight-update rule — here, the parser's only learning.
  * coincidence wiring = the composer's FIXED bind/unbind circuit (computes the ±1 Hadamard product).
  * plasticity gate = a per-synapse multiplier (`cp_plasticity_rate_gain`) on weight updates; 0.0 freezes
    BOTH the Hebbian potentiation and the Hebbian weight-decay term over the gated synapses.

THE LOAD-BEARING ISOLATION (verified in Task 1, see
`research/findings/2026-06-04-unified-bridge-plasticity-isolation.md`): on a shared bridge with GLOBAL
`enable_hebbian_learning=True`, declaring a population `plastic=False` does NOT freeze it — the ungated
Hebbian weight-decay term still drifts its weights. The working fix (no `sim/` edit) is to ALSO tag the
fixed population with a `plasticity_gate` in its `inject_explicit_wiring` plan and call
`bridge.set_plasticity_gate("<name>", 0.0)` after wiring. The composer's `"bind"` population is therefore
gated to 0.0 here (the parser's `"parse"` population stays ungated / fully plastic). On the composer's OWN
standalone bridge Hebbian is OFF, so no gate is needed there — that default path is kept byte-identical.

WIRING ACCUMULATION (why `merge_population_into_shared_bridge` exists): `inject_explicit_wiring(plan)`
REPLACES `bridge.cp_connections` wholesale on every call (it rebuilds the CSR from the plan it is given and
resets the gate maps). Two separate calls — one for the parser, one for the composer — would have the
second clobber the first. So both regions' populations are accumulated into one plan on the bridge and the
UNION is (re-)injected; every zeroed plasticity gate is re-applied after each injection. The parser's
weights are written by training, which `UnifiedBrainBridge` runs AFTER both populations are wired (a
re-injection would otherwise reset the trained `"parse"` weights to their initial design values).
"""
from __future__ import annotations

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel

# Parser slice layout (mirrors BridgeParser): 6 conjunction units + 3 role ensembles of R neurons each.
PARSER_R = 40
PARSER_SLICE_SIZE = 6 + 3 * PARSER_R          # 126


def merge_population_into_shared_bridge(bridge, plan, gates_to_zero=()):
    """Accumulate `plan`'s populations into the shared bridge and (re-)inject the UNION of everything wired
    onto it so far, then re-apply every zeroed plasticity gate.

    `inject_explicit_wiring` is a wholesale replacement of `cp_connections` (see module docstring), so each
    region cannot inject independently without clobbering the other. This helper keeps the running union in
    `bridge._unified_wiring_plan` and the set of gate names to hold at 0.0 in `bridge._unified_gates_zero`,
    re-injects the union, and re-zeros those gates.

    Args:
        bridge: the shared SimulationBridge (already `_initialize_simulation_data`-d).
        plan: dict of {population_name: population_spec} to add (same schema as inject_explicit_wiring).
        gates_to_zero: iterable of plasticity_gate names whose per-synapse gain must be held at 0.0
                       (the FIXED composer population). The parser's plastic population passes none.
    """
    running = getattr(bridge, "_unified_wiring_plan", None)
    if running is None:
        running = {}
        bridge._unified_wiring_plan = running
    zeroed = getattr(bridge, "_unified_gates_zero", None)
    if zeroed is None:
        zeroed = set()
        bridge._unified_gates_zero = zeroed

    for name, spec in plan.items():
        if name in running:
            raise ValueError(
                f"population '{name}' already wired onto this shared bridge — index/name collision")
        running[name] = spec
    for g in gates_to_zero:
        zeroed.add(g)

    # Re-inject the full union (rebuilds cp_connections + gate maps from scratch).
    bridge.inject_explicit_wiring(running)
    # Re-apply every gate that must be held frozen (gate maps were just rebuilt → default gain 1.0).
    for g in zeroed:
        bridge.set_plasticity_gate(g, 0.0)


def build_unified_bridge(seed=42, proj_dim=64):
    """Build ONE SimulationBridge sized for both regions: (6 + 3*PARSER_R) parser neurons + 8*proj_dim
    composer neurons. Config matches the parser's (Izhikevich, GENERIC_UNSTRUCTURED, dt=1ms, global Hebbian
    ON, STDP/STP/structural/homeostasis/reward/Watts-Strogatz OFF, OU noise 20 pA) — the composer's FIXED
    wiring is protected by a plasticity gate, not by the global Hebbian flag. Returns the bridge (no wiring).
    """
    total = PARSER_SLICE_SIZE + 8 * int(proj_dim)
    cfg = CoreSimConfig()
    cfg.num_neurons = total
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True            # ON for the parser (the composer's fixed pop is gate-frozen)
    cfg.hebbian_max_weight = 400.0
    cfg.hebbian_learning_rate = 0.005
    for f in ("enable_short_term_plasticity", "enable_structural_plasticity", "enable_homeostasis",
              "enable_reward_modulation", "enable_watts_strogatz"):
        setattr(cfg, f, False)
    cfg.ou_std_current_pA = 20.0

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


class UnifiedBrainBridge:
    """The PARSER and COMPOSER on ONE shared SimulationBridge — their neurons as disjoint index slices.
    `parser_slice` is the parser's neuron index range (0..125); `composer_offset` is the first composer neuron
    index (126); `self.bridge` is the single shared bridge that holds both regions.

    The conversational API the `BrainConversationalAgent` uses is delegated here (`parse`, `store`,
    `query_patient`, `query_agent`, `ask_yes_no`, `describe`, `render_fact`), plus the read-through attributes
    `kb`, `words`, `concepts`. Dialogue planning (`elaborate`) stays on its own dlPFC bridge for now — out of
    scope until step 3.

    Build ORDER is load-bearing (see `merge_population_into_shared_bridge`): the composer's FIXED `"bind"`
    population is wired first, then the parser's plastic `"parse"` population (each wiring re-injects the
    accumulated union, resetting every weight to its DESIGN value), and the parser is TRAINED LAST. Training is
    deferred (`defer_train=True`) so it runs after all populations are wired — a later re-injection would
    otherwise reset the trained `"parse"` weights. The composer's gated bind weights stay frozen throughout
    (plasticity gain 0.0), so the parser's global-Hebbian training cannot drift them (Task 1 isolation)."""

    def __init__(self, seed=42, proj_dim=64, concepts=None):
        """`concepts` (optional): a {word: code} codebook for the composer. When None, the composer loads its
        default substrate `denoise64` concept codes (requires the cache; raises FileNotFoundError if absent).
        Passing a small synthetic codebook keeps a unit build cache-independent."""
        # Defer the import to here to avoid a construction-time import cycle (these modules import this one
        # for `merge_population_into_shared_bridge`).
        from research.runners.brain_conversational_agent import BridgeParser
        from research.runners.core_sim_composition import CoreSimComposer

        self.seed = int(seed)
        self.proj_dim = int(proj_dim)
        self.bridge = build_unified_bridge(seed=self.seed, proj_dim=self.proj_dim)
        self.parser_slice = range(0, PARSER_SLICE_SIZE)     # 0..125
        self.composer_offset = PARSER_SLICE_SIZE            # 126

        # 1) Composer first: wire the FIXED "bind" coincidence population at the offset (gated to 0.0).
        self.composer = CoreSimComposer(seed=self.seed, proj_dim=self.proj_dim, concepts=concepts,
                                        shared_bridge=self.bridge, index_offset=self.composer_offset)
        # 2) Parser next: wire the plastic "parse" population at offset 0; DEFER training (re-injection above/here
        #    resets weights, so we train only once everything is wired).
        self.parser = BridgeParser(seed=self.seed, shared_bridge=self.bridge, index_offset=0, defer_train=True)
        # 3) Train the parser LAST — no further wiring/re-injection follows, so the trained weights persist; the
        #    gated composer bind weights stay frozen under this global-Hebbian training (Task 1 isolation).
        self.parser.train()

    # --- read-through attributes the agent reads ---
    @property
    def kb(self):
        return self.composer.kb

    @kb.setter
    def kb(self, value):
        self.composer.kb = value

    @property
    def words(self):
        return self.composer.words

    @property
    def concepts(self):
        return self.composer.concepts

    # --- delegated conversational API (comprehend → store/recall/compose on the ONE shared bridge) ---
    def parse(self, sentence, voice="active"):
        """Comprehend an SVO sentence -> {role: word}. Accepts a string ('dog go north') or a 3-word list."""
        words = sentence.split() if isinstance(sentence, str) else list(sentence)
        return self.parser.parse(words, voice)

    def store(self, agent, action, patient, polarity=None):
        """Store an SVO fact in the composer's spiking memory (patient may be a concept, an attributed entity,
        or an embedded Clause; `polarity` AFFIRM/NEGATE is optional for yes/no facts)."""
        return self.composer.store(agent, action, patient, polarity=polarity)

    def query_patient(self, agent, action):
        """'what does <agent> <action>?' -> patient, or None (abstention)."""
        return self.composer.query_patient(agent, action)

    def query_agent(self, action, patient):
        """'who <action> <patient>?' -> agent, or None."""
        return self.composer.query_agent(action, patient)

    def ask_yes_no(self, agent, action, patient):
        """'does <agent> <action> <patient>?' -> 'yes'/'no'/'unknown' via the bound polarity tag."""
        return self.composer.ask_yes_no(agent, action, patient)

    def render_fact(self, agent):
        """Generation: render a stored sentence whose agent matches `agent` (decoded from the spiking unbind),
        or None if no fact's agent matches (the no-confab moat)."""
        return self.composer.render_fact(agent)

    def describe(self, agent):
        """Alias of render_fact (matches BrainConversationalAgent.describe)."""
        return self.composer.render_fact(agent)
