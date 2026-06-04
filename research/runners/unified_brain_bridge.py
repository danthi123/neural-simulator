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
from sim.backend import get_backend, to_host

# Parser slice layout (mirrors BridgeParser): 6 conjunction units + 3 role ensembles of R neurons each.
PARSER_R = 40
PARSER_SLICE_SIZE = 6 + 3 * PARSER_R          # 126

# ── Step 2 (hear_synaptic) synaptic-route operating point ────────────────────────────────────────────────
# The parser→gate→composer route that replaces the Python {role: word} hand-off. Per role, a dedicated
# `role_src` pool of D neurons drives the composer's role bank (role_ON/OFF) through a TOPOGRAPHIC route
# (role_src[i] → role_ON[i] when roles[R][i]>0, → role_OFF[i] when <0 — exactly the binary ±1 mask the
# Python path's `hadamard_spiking` applies as a direct current). The route is held closed by a transmission
# gate coupled to the parser's role ensemble; when the parser assigns a word to role R, ensemble[R] fires →
# gate `role_route_<R>` opens → that role's pattern reaches the role bank. The word's concept code drives
# the fill bank directly (role-independent, ungated). Values mirror the Task-1 de-risk + the composer's own
# drive: route weight 300, role_src drive 2500 pA (= the composer's ROLE_DRIVE), gate threshold 0.05.
ROLE_ROUTE_WEIGHT = 300.0
ROLE_SRC_DRIVE_PA = 2500.0
ROLE_ROUTE_GATE_PLASTICITY = "step2_role_route_fixed"   # one plasticity gate over ALL role-src routes (held 0.0)
ROLE_GATE_THRESHOLD = 0.05                              # couple_gate_to_pool default; parser ensembles fire above it
SYNAPTIC_ROUTE_ROLES = ("agent", "action", "patient")  # the parser's three roles (the composer also has polarity/
#                                                        attribute roles, but the parser only assigns these three)


def couple_gate_to_indices(bridge, gate_name, control_idx, threshold=ROLE_GATE_THRESHOLD, alpha=0.3,
                           open_value=1.0):
    """Append an activity-driven gate↔pool coupling using RAW neuron indices (the parser's role ensemble),
    reusing the shipped per-step `_apply_gate_couplings` hook UNCHANGED.

    This is EXACTLY the coupling dict that `bridge.couple_gate_to_pool` builds internally — only the
    name→indices resolution is different. The public `couple_gate_to_pool` resolves the control pool by REGION
    NAME and REQUIRES the brain-region framework (`region_manager`), raising RuntimeError when it is None. The
    unified bridge is built WITHOUT that framework — its parser/composer are `inject_explicit_wiring`
    populations addressed by raw indices, and `region_manager` is None — so the name-based entry point does not
    fit. The per-step gating LOGIC is identical and index-based (`_apply_gate_couplings` reads `control_idx`
    every step), so this helper appends the same dict shape with the parser's raw `role_idx[role]` indices.
    This is a runner-side wiring helper, NOT a `sim/` edit (the gating primitives are public; only the
    wire-time name→indices resolution that needs `region_manager` is replaced by indices we already hold).
    (Validated in the Task-1 de-risk `research/findings/raw/_step2_gated_route_probe.py`.)
    """
    if gate_name not in bridge._transmission_gate_to_synapses:
        raise KeyError(f"No transmission gate named '{gate_name}'. "
                       f"Known: {list(bridge._transmission_gate_to_synapses.keys())}")
    xp, _ = get_backend()
    bridge._gate_couplings.append({
        "gate_name": gate_name,
        "control_idx": xp.asarray(np.asarray(control_idx, dtype=np.int64)),
        "threshold": float(threshold), "alpha": float(alpha), "open_value": float(open_value),
        "ema": 0.0, "last_value": None,
    })


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


def build_unified_bridge(seed=42, proj_dim=64, enable_synaptic_route=False):
    """Build ONE SimulationBridge sized for both regions: (6 + 3*PARSER_R) parser neurons + 8*proj_dim
    composer neurons. Config matches the parser's (Izhikevich, GENERIC_UNSTRUCTURED, dt=1ms, global Hebbian
    ON, STDP/STP/structural/homeostasis/reward/Watts-Strogatz OFF, OU noise 20 pA) — the composer's FIXED
    wiring is protected by a plasticity gate, not by the global Hebbian flag. Returns the bridge (no wiring).

    `enable_synaptic_route`: when True, allocate `len(SYNAPTIC_ROUTE_ROLES) * proj_dim` extra neurons past the
    composer slice for the per-role `role_src` pools that drive the composer's role bank through the parser-
    gated route (Step 2 `hear_synaptic`). The default (False) keeps the bridge byte-identical to before so the
    Python-hand-off path and the step-1 tests are unaffected.
    """
    total = PARSER_SLICE_SIZE + 8 * int(proj_dim)
    if enable_synaptic_route:
        total += len(SYNAPTIC_ROUTE_ROLES) * int(proj_dim)
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

    def __init__(self, seed=42, proj_dim=64, concepts=None, enable_synaptic_route=False):
        """`concepts` (optional): a {word: code} codebook for the composer. When None, the composer loads its
        default substrate `denoise64` concept codes (requires the cache; raises FileNotFoundError if absent).
        Passing a small synthetic codebook keeps a unit build cache-independent.

        `enable_synaptic_route` (Step 2): when True, also wire the parser→gate→composer SYNAPTIC route that
        `hear_synaptic` uses (per-role `role_src` pools + topographic gated routes into the composer's role
        bank, each gate coupled to the parser's role ensemble). The routes are wired BEFORE the parser trains
        (a later re-injection would reset the trained `"parse"` weights — see `merge_population_into_shared_bridge`),
        and their weights are plasticity-gated to 0.0 so the parser's global Hebbian training cannot drift them.
        The default (False) keeps the bridge byte-identical to the step-1 build; `hear_synaptic` then raises a
        clear error directing the caller to enable the route."""
        # Defer the import to here to avoid a construction-time import cycle (these modules import this one
        # for `merge_population_into_shared_bridge`).
        from research.runners.brain_conversational_agent import BridgeParser
        from research.runners.core_sim_composition import CoreSimComposer

        self.seed = int(seed)
        self.proj_dim = int(proj_dim)
        self.synaptic_route_enabled = bool(enable_synaptic_route)
        self.bridge = build_unified_bridge(seed=self.seed, proj_dim=self.proj_dim,
                                           enable_synaptic_route=self.synaptic_route_enabled)
        self.parser_slice = range(0, PARSER_SLICE_SIZE)     # 0..125
        self.composer_offset = PARSER_SLICE_SIZE            # 126

        # 1) Composer first: wire the FIXED "bind" coincidence population at the offset (gated to 0.0).
        self.composer = CoreSimComposer(seed=self.seed, proj_dim=self.proj_dim, concepts=concepts,
                                        shared_bridge=self.bridge, index_offset=self.composer_offset)
        # 2) Parser next: wire the plastic "parse" population at offset 0; DEFER training (re-injection above/here
        #    resets weights, so we train only once everything is wired).
        self.parser = BridgeParser(seed=self.seed, shared_bridge=self.bridge, index_offset=0, defer_train=True)
        # 3) Synaptic route (opt-in) BEFORE training: per-role role_src pools + topographic gated routes into the
        #    composer role bank, coupled to the parser ensembles. Wired here (not lazily) because every wiring
        #    re-injects the union and resets weights — so it must precede the parser's (final) training.
        self._role_src = None
        if self.synaptic_route_enabled:
            self._wire_synaptic_route()
        # 4) Train the parser LAST — no further wiring/re-injection follows, so the trained weights persist; the
        #    gated composer bind weights (and the role-route weights) stay frozen under this global-Hebbian
        #    training (Task 1 isolation).
        self.parser.train()

    # --- Step 2: the synaptic parser→gate→composer route (replaces the Python {role: word} hand-off) ---
    def _wire_synaptic_route(self):
        """Wire, once, the parser→gate→composer SYNAPTIC route used by `hear_synaptic`.

        For each parser role R (agent/action/patient): a dedicated `role_src[R]` pool of D neurons drives the
        composer's role bank through a TOPOGRAPHIC route — `role_src[R][i] → composer.role_ON[i]` when
        `composer.roles[R][i] > 0`, else `→ composer.role_OFF[i]`. This reproduces, as synaptic current, the
        exact binary ±1 mask that the Python path's `hadamard_spiking` applies as a direct current
        (`role_on = (role_vec>0)*ROLE_DRIVE`, `role_off = (role_vec<0)*ROLE_DRIVE`). Each route is tagged
        `transmission_gate="role_route_<R>"` (held closed until the parser fires R) AND
        `plasticity_gate=ROLE_ROUTE_GATE_PLASTICITY` held 0.0 (FIXED weights on a global-Hebbian bridge — the
        Task-1 isolation finding — so Hebbian decay cannot drift them). The gate is then coupled to the
        parser's role ensemble (raw indices, via `couple_gate_to_indices`): when the parser assigns a word to
        role R, ensemble[R] fires → the gate opens → that role's pattern reaches the role bank.
        """
        xp, _ = get_backend()
        D = self.composer.D
        roles = self.composer.roles                 # {role: ±1 D-vector}
        idx = self.composer.idx                      # offset composer banks (role_ON/OFF host below)
        role_on_h = to_host(idx["role_on"]).astype(np.int64)
        role_off_h = to_host(idx["role_off"]).astype(np.int64)

        # role_src pools live past the composer slice (the extra capacity build_unified_bridge allocated).
        src_base = self.composer_offset + 8 * self.proj_dim
        self._role_src = {}
        plan = {}
        for ri, r in enumerate(SYNAPTIC_ROUTE_ROLES):
            base = src_base + ri * D
            src = np.arange(base, base + D, dtype=np.int64)
            self._role_src[r] = xp.asarray(src)
            pre, post = [], []
            sign = np.asarray(roles[r])             # the composer's ±1 pattern for this role
            for i in range(D):                       # topographic: src[i] → role_ON[i] or role_OFF[i]
                dst = role_on_h[i] if sign[i] > 0 else role_off_h[i]
                pre.append(int(src[i])); post.append(int(dst))
            plan[f"role_route_{r}"] = {
                "pre_indices": pre, "post_indices": post,
                "initial_weights": np.full(len(pre), ROLE_ROUTE_WEIGHT, dtype=np.float32),
                "plastic": False, "plasticity_gate": ROLE_ROUTE_GATE_PLASTICITY,
                "transmission_gate": f"role_route_{r}", "conn_type": "E_TO_E", "count": len(pre),
            }
        merge_population_into_shared_bridge(self.bridge, plan, gates_to_zero=(ROLE_ROUTE_GATE_PLASTICITY,))

        # Couple each route's gate to its parser role ensemble (raw indices — the unified bridge has no
        # region_manager, so the name-based couple_gate_to_pool does not apply; see couple_gate_to_indices).
        for r in SYNAPTIC_ROUTE_ROLES:
            couple_gate_to_indices(self.bridge, f"role_route_{r}", self.parser.role_idx[r])
        # Close every route gate now (re-injection left transmission gains at the 1.0 inject default = OPEN);
        # the per-step coupling re-opens only the parser-selected role's gate during comprehension.
        for r in SYNAPTIC_ROUTE_ROLES:
            self.bridge.set_transmission_gate(f"role_route_{r}", 0.0)

    def hear_synaptic(self, sentence, voice="active", polarity=None):
        """Comprehend an SVO sentence and STORE the fact via the SYNAPTIC route — the parser's role selection
        drives the composer's bind through transmission gates, NOT via a Python `{role: word}` dict (which is
        what `hear`/`store` use). The cross-region hand-off is then synaptic, not Python.

        Per word W at position P (active voice 0, passive 1; conjunction index k = P*2 + voice):
          1. Drive the parser conjunction (P, voice) so the parser's role ensemble for W's role fires.
          2. That firing opens the gate `role_route_<R>`, routing role R's ±1 pattern (via `role_src[R]`) into
             the composer's role bank — while W's concept code drives the fill bank directly (ungated). Only
             the parser-selected role's gate is open, so the role bank carries the parser-chosen role's pattern.
          3. Run the coincidence window; read the 4 AND banks → accumulate into (bound_ON, bound_OFF), exactly
             as the composer's `bind_fact`/`_op` do, but with the role drive arriving via the gated route.
        After the 3 words, `onoff(bound_ON - bound_OFF)` is the stored fact; it is appended to the composer's kb
        (with the parser-comprehended `{role: word}` fact dict, used only to ROUTE rendering — the bound VECTOR
        is the synaptic product), so query_patient / query_agent / ask_yes_no / abstention work unchanged.

        Requires the bridge to have been built with `enable_synaptic_route=True`. Returns the comprehended
        `{role: word}` dict (so callers can inspect the parse), matching `BrainConversationalAgent.hear`."""
        if not self.synaptic_route_enabled:
            raise RuntimeError(
                "hear_synaptic requires the synaptic route — construct UnifiedBrainBridge(enable_synaptic_route=True). "
                "The default build wires only the Python-hand-off path (hear/store).")
        from research.runners.core_sim_composition import onoff, _scale_to_current, FILL_DRIVE

        words = sentence.split() if isinstance(sentence, str) else list(sentence)
        assert len(words) == 3, "this minimal parser handles 3-word SVO sentences"
        v = 0 if voice in (0, "active") else 1

        # The comprehended fact dict (parser-selected roles) — used to ROUTE rendering + to confirm the parse;
        # the STORED content is the synaptic bound vector, not these labels.
        fact = {}

        xp, _ = get_backend()
        comp = self.composer
        bound_on = np.zeros(comp.D); bound_off = np.zeros(comp.D)

        for pos in range(3):
            word = words[pos]
            k = pos * 2 + v                          # parser conjunction index for this (position, voice)
            role = self.parser.role_of(pos, voice)   # the parser's spiking role assignment for this position
            fact[role] = word

            # Drive this word's concept code into the fill bank (role-independent, ungated — same as the
            # Python path's fill drive), the parser conjunction into the parser (opens the selected gate), and
            # ALL role_src pools (the open gate selects which role's pattern reaches the role bank).
            c_on, c_off = onoff(comp.concepts[word])
            fon, foff = _scale_to_current(c_on, c_off, FILL_DRIVE)
            bound_on_i, bound_off_i = self._op_synaptic(k, fon, foff)
            bound_on += bound_on_i; bound_off += bound_off_i

        if polarity is not None:
            # Polarity (yes/no) is bound the Python way (the parser does not assign a polarity role); this keeps
            # ask_yes_no working. The agent/action/patient binding above is fully synaptic.
            c_on, c_off = onoff(comp.concepts[polarity])
            fon, foff = _scale_to_current(c_on, c_off, FILL_DRIVE)
            o, f = comp._op(comp.roles["polarity"], fon, foff)
            bound_on += o; bound_off += f
            fact["polarity"] = polarity

        comp.kb.append((fact, onoff(bound_on - bound_off)))
        return fact

    def _op_synaptic(self, conj_k, fill_on_cur, fill_off_cur):
        """One spiking bind step with the role drive arriving via the GATED route (not an orchestrated role
        current). Drives: the parser conjunction `conj_k` (→ the parser fires the role ensemble → opens that
        role's gate), ALL `role_src` pools (the open gate selects which one reaches the role bank), and the
        fill bank with the word's code; then reads the 4 coincidence banks over the window. Mirrors the
        composer's `hadamard_spiking` EXCEPT the role bank is driven synaptically through the gate.

        Returns (out_on, out_off) = (rates[A]+rates[B], rates[C]+rates[D]) — identical readout to `_op`.
        """
        xp, _ = get_backend()
        bridge = self.bridge; comp = self.composer; idx = comp.idx
        from research.runners.core_sim_composition import RESET_STEPS

        bridge.cp_external_input_current[:] = 0.0
        for _ in range(RESET_STEPS):
            bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
            bridge._run_one_simulation_step()

        cur = xp.zeros(bridge.core_config.num_neurons, dtype=xp.float32)
        cur[self.parser.conj_arr[conj_k]] = self.parser.drive          # parser drive → opens the selected gate
        for r in SYNAPTIC_ROUTE_ROLES:                                  # drive all role_src; the gate selects one
            cur[self._role_src[r]] = ROLE_SRC_DRIVE_PA
        cur[idx["fill_on"]] = xp.asarray(fill_on_cur.astype(np.float32))
        cur[idx["fill_off"]] = xp.asarray(fill_off_cur.astype(np.float32))
        for bank in ("A", "B", "C", "D"):
            cur[idx[bank]] = comp.coinc_bias
        bridge.cp_external_input_current[:] = cur

        acc = {b: xp.zeros(comp.D, dtype=xp.float64) for b in ("A", "B", "C", "D")}
        for _ in range(comp.run_steps):
            bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
            bridge._run_one_simulation_step()
            for b in ("A", "B", "C", "D"):
                acc[b] += bridge.cp_firing_states[idx[b]].astype(xp.float64)
        bridge.cp_external_input_current[:] = 0.0
        rates = {b: to_host(acc[b]) / comp.run_steps for b in ("A", "B", "C", "D")}
        return rates["A"] + rates["B"], rates["C"] + rates["D"]

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
