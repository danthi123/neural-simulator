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

# ── Step 2 gate PRE-WARM (resolves the seed-42 patient-readout regression) ───────────────────────────────────
# The parser-coupled gate `role_route_<R>` opens via an EMA (alpha 0.3, threshold 0.05) of the parser role
# ensemble's firing. After each op's RESET_STEPS of zero input the EMA has decayed to ~0; and the parser
# ensemble fires at a LOW, BURSTY rate (mean ~0.042 of 40 neurons) so its EMA hovers right AT the 0.05
# threshold — the gate FLICKERS open/closed and is open on only ~1/3 of a cold readout window. The role bank
# then fires at ~1/7 the Python path's direct-role rate, thinning the cleanup margin enough to mis-decode
# borderline patients ("come") at the correlated V=16 codes on seed 42 (the documented regression; diagnosis
# in 2026-06-04-one-bridge-unification-step2-...-REGRESSION.md). The FAITHFUL fix (timing, not magnitude — no
# weight/current change, the gate is NOT set by hand): drive the parser conjunction for a PRE-WINDOW so the
# parser FIRES and (via the coupling) OPENS the gate, THEN run the readout window holding the parser-opened
# gate — the biologically correct order (comprehend → latch the route → compose). Measured: the parser opens
# the gate at step ~24-27 (well under the cap); with the parser-opened gate held, the gate reads 1.0 on all
# 150 readout steps and seed-42 what-recall returns to 6/6 (= the Python path). Validated in
# `research/findings/raw/_step2_synaptic_holdopen_validate.py`.
ROLE_GATE_PREWARM_CAP_STEPS = 60   # max pre-window steps to wait for the parser to open its role gate


# ── Step 3 (enable_dlpfc) dlPFC dialogue-planning loop operating point ─────────────────────────────────────
# The third conversational region: the dlPFC `cortex_ctx ↔ dlpfc_wm` reverberatory working-memory loop that
# `BrainConversationalAgent.elaborate` uses for dialogue planning (spreading activation over the agent's own
# association graph → the next on-topic associate). Today it builds a THROWAWAY bridge per `elaborate` call;
# Step 3 brings the loop onto the unified bridge as further index slices (cortex_ctx + dlpfc_wm), at the
# parser/composer's dt=1.0.
#
# THE TWO LOAD-BEARING DESIGN CALLS (both from the Task-1 de-risk, research/findings/2026-06-04-step3-dlpfc-dt-survives.md):
#   (1) dt=1.0 — the de-risk proved the dlPFC's NMDA-dependent working-memory latch SURVIVES dt=1.0 (263–513%
#       of the dt=0.5 rate, still NMDA-dependent). So the loop joins the unified bridge at dt=1.0, no separate
#       timestep needed.
#   (2) self-attractor weight ≈30, NOT the module's saturated 50 — at weight 50 the "persistence" is trivial
#       AMPA ping-pong that survives even NMDA-off (the WRONG mechanism); at weight 30 the loop sits in the
#       genuinely NMDA-DEPENDENT regime (the real WM latch). The merge therefore wires the self-attractors at
#       30 AND runs NMDA on the dlPFC slice only.
#
# THE SECOND CRUX (per-region NMDA): one bridge has ONE global `cfg.enable_nmda`, but the dlPFC needs NMDA
# while parser+composer must stay NMDA-OFF. Resolved by the cluster-G per-neuron NMDA MASK
# (`bridge.cp_nmda_neuron_mask` — see sim/bridge.py: NMDA current is multiplied by this 0/1 mask, so only
# masked neurons receive NMDA). The mask is set 1.0 on the dlPFC slice and 0.0 on parser+composer, with
# `cfg.enable_nmda=True`. (The bridge's auto-mask build is gated on `region_manager is not None`; the unified
# bridge has none — it is an `inject_explicit_wiring` bridge — so the mask is set DIRECTLY here, the same
# public attribute the cluster-G code populates. No `sim/` edit.)
#
# THE CSR-SAFE WIRING (why the dlPFC edges are PRE-ALLOCATED at weight 0): the dlPFC loop attractors +
# association-graph edges live in the SAME `cp_connections` CSR as the parser/composer. The graph is built
# Python-side from the agent's facts and CHANGES as facts arrive (exactly as the separate path rebuilds its
# Control). Installing graph edges at `elaborate` time via `set_pathway_weights(add_missing=True)` would, if
# the edges were NEW, trigger a CSR rebuild that RESORTS the matrix and INVALIDATES the composer's
# `composer_bind_fixed` plasticity-gate→synapse-index map (the Task-1 isolation). So ALL dlPFC edges (every
# word's self-attractor + every directed word-pair graph edge over the composer's full vocabulary) are
# PRE-ALLOCATED at construction (weight 0, tagged `plasticity_gate=DLPFC_FIXED_GATE` held 0.0 so global
# Hebbian cannot drift them) as part of the union plan. At `elaborate` time the borrowed
# `SpikingSpreadingController._install_graph_edges` calls `set_pathway_weights(add_missing=True)` and finds
# every edge ALREADY present → it only OVERWRITES `.data` IN PLACE (no rebuild, no resort, gate maps intact).
DLPFC_PATTERN_SIZE = 50            # per-concept assembly size (matches SpikingLoopContextBuffer default)
DLPFC_ATTRACTOR_WEIGHT = 30.0      # self-attractor weight — the de-risk's NMDA-DEPENDENT regime (NOT 50)
DLPFC_EDGE_SCALE = 60.0            # graph-edge scale (matches the validated SpikingSpreadingController default)
DLPFC_FIXED_GATE = "dlpfc_fixed"   # one plasticity gate over ALL dlPFC loop+graph edges (held 0.0)


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


def build_unified_bridge(seed=42, proj_dim=64, enable_synaptic_route=False, dlpfc_n=0, role_wta_n=0,
                         reservoir_n=0):
    """Build ONE SimulationBridge sized for both regions: (6 + 3*PARSER_R) parser neurons + 8*proj_dim
    composer neurons. Config matches the parser's (Izhikevich, GENERIC_UNSTRUCTURED, dt=1ms, global Hebbian
    ON, STDP/STP/structural/homeostasis/reward/Watts-Strogatz OFF, OU noise 20 pA) — the composer's FIXED
    wiring is protected by a plasticity gate, not by the global Hebbian flag. Returns the bridge (no wiring).

    `enable_synaptic_route`: when True, allocate `len(SYNAPTIC_ROUTE_ROLES) * proj_dim` extra neurons past the
    composer slice for the per-role `role_src` pools that drive the composer's role bank through the parser-
    gated route (Step 2 `hear_synaptic`). The default (False) keeps the bridge byte-identical to before so the
    Python-hand-off path and the step-1 tests are unaffected.

    `dlpfc_n` (Step 3): when > 0, allocate `2 * dlpfc_n` extra neurons past everything else for the dlPFC
    dialogue-planning loop's two regions (`cortex_ctx` + `dlpfc_wm`, `dlpfc_n` each), and turn on `cfg.enable_nmda`
    so the dlPFC slice can carry its NMDA-dependent working-memory latch. The per-neuron NMDA mask
    (`cp_nmda_neuron_mask`) — set by `UnifiedBrainBridge` after wiring — restricts NMDA CURRENT to the dlPFC
    slice ONLY, so parser+composer stay NMDA-free despite the global flag (the second crux). The default
    (dlpfc_n=0) keeps `cfg.enable_nmda=False` and the bridge byte-identical to the step-1/2 build.
    """
    total = PARSER_SLICE_SIZE + 8 * int(proj_dim)
    if enable_synaptic_route:
        total += len(SYNAPTIC_ROUTE_ROLES) * int(proj_dim)
    if dlpfc_n:
        total += 2 * int(dlpfc_n)
    if role_wta_n:
        # RUNG B-1b: `role_wta_n` extra neurons past everything else for the on-bridge spiking WTA that elects the
        # word's thematic role (3 excitatory role ensembles + one shared inhibitory pool). Needs num_traits=2 (an
        # inhibitory trait). We force every trait to 0 (excitatory) after init so the parser/composer stay
        # excitatory; the WTA's inhibitory pool is flipped to trait 1 by inject_explicit_wiring's
        # output_inhibitory_indices when the runner wires it. Default (0) keeps num_traits=1 -> byte-identical.
        total += int(role_wta_n)
    if reservoir_n:
        # RUNG B-1c: `reservoir_n` extra neurons past everything else for the on-bridge SPIKING reservoir (a
        # recurrent Izhikevich liquid-state machine, co-resident with the parser/composer/WTA). Its fixed-random
        # recurrence + W_in input drive + Ws_shifted read-out synapses (reservoir -> the 3 WTA ensembles) are all
        # wired RUNNER-SIDE via set_pathway_weights(add_missing=True), mirroring the WTA wiring. The reservoir's
        # inhibitory subset is flipped to trait 1 (needs num_traits=2, same as the WTA). Default (0) keeps the
        # bridge byte-identical to the step-1/2/B-1b build.
        total += int(reservoir_n)
    cfg = CoreSimConfig()
    cfg.num_neurons = total
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.num_traits = 2 if (role_wta_n or reservoir_n) else 1   # trait 1 = WTA inh pool (B-1b) / reservoir inh (B-1c)
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True            # ON for the parser (the composer's fixed pop is gate-frozen)
    cfg.hebbian_max_weight = 400.0
    cfg.hebbian_learning_rate = 0.005
    for f in ("enable_short_term_plasticity", "enable_structural_plasticity", "enable_homeostasis",
              "enable_reward_modulation", "enable_watts_strogatz"):
        setattr(cfg, f, False)
    cfg.ou_std_current_pA = 20.0
    # Step 3: global NMDA ON so the dlPFC slice can carry its working-memory latch; the per-neuron NMDA mask
    # (set by UnifiedBrainBridge after wiring) confines the NMDA current to the dlPFC slice, so parser+composer
    # remain NMDA-free (the second crux — one global flag, but NMDA isolated to the dlPFC via the mask).
    cfg.enable_nmda = bool(dlpfc_n)

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    if role_wta_n or reservoir_n:
        bridge.cp_traits[:] = 0     # force all excitatory; WTA/reservoir inh pools are flipped to trait 1 at wire time
    return bridge


class _SharedDlpfcContext:
    """A shared-bridge-backed stand-in for `content_selection_spiking.SpikingLoopContextBuffer` — the object a
    `SpikingSpreadingController` reads as `self.ctx`. It exposes EXACTLY the attributes the controller's
    borrowed methods touch (`.bridge`, `._cpat`, `._dpat`, `._psize`, `.B`), but its bridge is the UNIFIED
    bridge and its concept assemblies are sparse subsets of the dlPFC slice's `cortex_ctx`/`dlpfc_wm` index
    ranges, laid out byte-IDENTICALLY to how `SpikingLoopContextBuffer` lays them out (same `n`, same
    `pattern_size`, same `seed`-derived permutation, same word ordering) so the merged dialogue planning
    reproduces the separate path. It does NOT build its own bridge (the whole point of Step 3) — and it does
    NOT install loop attractors via a CSR-rebuilding `set_pathway_weights(add_missing=True)` (the unified-bridge
    attractor edges are PRE-ALLOCATED at construction; see `UnifiedBrainBridge._wire_dlpfc`).

    Layout faithfulness (oracle parity): the separate `SpikingSpreadingController` builds
    `SpikingLoopContextBuffer(vocab, n, pattern_size=50, seed)` → `perm = np.random.default_rng(seed).permutation(n)`,
    concept i (in SORTED vocab order) → assembly `perm[i*ps:(i+1)*ps]` indexing the region's cortex/dlPFC index
    arrays. Here `cortex_ctx` = global indices `[cortex_base : cortex_base+n)` and `dlpfc_wm` =
    `[dlpfc_base : dlpfc_base+n)`, so the SAME permutation over the SAME word ordering picks the SAME
    (offset-shifted) assemblies. The spiking dynamics differ only by the bridge's neuron heterogeneity (a
    larger total-N RNG draw), not by the assembly/graph structure.
    """

    def __init__(self, bridge, words, cortex_base, dlpfc_base, n, pattern_size, seed):
        import sim.backend as B
        self.B = B
        self.xp, _ = B.get_backend()
        self.bridge = bridge
        self._psize = int(pattern_size)
        self._n = int(n)
        # Word ordering MUST match the controller's `self._vocab = sorted(...)` so assembly i is the same word.
        self._words = sorted(words)
        cidx = np.arange(cortex_base, cortex_base + n, dtype=np.int64)
        didx = np.arange(dlpfc_base, dlpfc_base + n, dtype=np.int64)
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(n)
        self._cpat = {}
        self._dpat = {}
        self._cpat_host = {}
        self._dpat_host = {}
        for i, w in enumerate(self._words):
            p = perm[i * pattern_size:(i + 1) * pattern_size]
            cpat, dpat = cidx[p], didx[p]
            self._cpat[w] = self.xp.asarray(cpat)
            self._dpat[w] = self.xp.asarray(dpat)
            self._cpat_host[w] = cpat
            self._dpat_host[w] = dpat


class UnifiedBrainBridge:
    """The PARSER and COMPOSER on ONE shared SimulationBridge — their neurons as disjoint index slices.
    `parser_slice` is the parser's neuron index range (0..125); `composer_offset` is the first composer neuron
    index (126); `self.bridge` is the single shared bridge that holds both regions.

    The conversational API the `BrainConversationalAgent` uses is delegated here (`parse`, `store`,
    `query_patient`, `query_agent`, `ask_yes_no`, `describe`, `render_fact`), plus the read-through attributes
    `kb`, `words`, `concepts`. Dialogue planning (`elaborate`, Step 3) joins the SAME bridge when
    `enable_dlpfc=True`: the dlPFC `cortex_ctx`/`dlpfc_wm` loop is wired as further index slices and `elaborate`
    drives that shared slice (instead of building a throwaway bridge per call), matching the separate-dlPFC
    `BrainConversationalAgent.elaborate` behavior.

    Build ORDER is load-bearing (see `merge_population_into_shared_bridge`): the composer's FIXED `"bind"`
    population is wired first, then the parser's plastic `"parse"` population (each wiring re-injects the
    accumulated union, resetting every weight to its DESIGN value), and the parser is TRAINED LAST. Training is
    deferred (`defer_train=True`) so it runs after all populations are wired — a later re-injection would
    otherwise reset the trained `"parse"` weights. The composer's gated bind weights stay frozen throughout
    (plasticity gain 0.0), so the parser's global-Hebbian training cannot drift them (Task 1 isolation)."""

    def __init__(self, seed=42, proj_dim=64, concepts=None, enable_synaptic_route=False, enable_dlpfc=False,
                 role_wta_n=0, reservoir_n=0):
        """`concepts` (optional): a {word: code} codebook for the composer. When None, the composer loads its
        default substrate `denoise64` concept codes (requires the cache; raises FileNotFoundError if absent).
        Passing a small synthetic codebook keeps a unit build cache-independent.

        `enable_synaptic_route` (Step 2): when True, also wire the parser→gate→composer SYNAPTIC route that
        `hear_synaptic` uses (per-role `role_src` pools + topographic gated routes into the composer's role
        bank, each gate coupled to the parser's role ensemble). The routes are wired BEFORE the parser trains
        (a later re-injection would reset the trained `"parse"` weights — see `merge_population_into_shared_bridge`),
        and their weights are plasticity-gated to 0.0 so the parser's global Hebbian training cannot drift them.
        The default (False) keeps the bridge byte-identical to the step-1 build; `hear_synaptic` then raises a
        clear error directing the caller to enable the route.

        `enable_dlpfc` (Step 3): when True, also bring the dlPFC dialogue-planning loop (`cortex_ctx`/`dlpfc_wm`
        regions) onto the SAME bridge as further index slices, with NMDA isolated to that slice (the per-neuron
        NMDA mask), and route `elaborate` through it. All dlPFC loop+graph edges are PRE-ALLOCATED at weight 0
        (gated `dlpfc_fixed` held 0.0) over the composer's full vocabulary BEFORE the parser trains, so that
        `elaborate`-time graph-edge installs only overwrite weights in place (no CSR rebuild → the composer's
        bind-gate isolation is preserved). The default (False) keeps the bridge byte-identical to the step-1/2
        build; `elaborate` then raises a clear error directing the caller to enable the dlPFC."""
        # Defer the import to here to avoid a construction-time import cycle (these modules import this one
        # for `merge_population_into_shared_bridge`).
        from research.runners.brain_conversational_agent import BridgeParser
        from research.runners.core_sim_composition import CoreSimComposer

        self.seed = int(seed)
        self.proj_dim = int(proj_dim)
        self.synaptic_route_enabled = bool(enable_synaptic_route)
        self.dlpfc_enabled = bool(enable_dlpfc)
        self.role_wta_n = int(role_wta_n)      # RUNG B-1b: extra neurons for the on-bridge role-WTA (0 = off)
        self.reservoir_n = int(reservoir_n)    # RUNG B-1c: extra neurons for the on-bridge spiking reservoir (0 = off)

        # The dlPFC region size must be known BEFORE the bridge is sized. It mirrors the separate
        # SpikingSpreadingController: n = max(600, 60*len(vocab)) per region, over the composer's full
        # vocabulary. The vocab is the concept words — read from the codebook (if given) or the denoise64 cache.
        self._dlpfc_n = 0
        self._dlpfc_words = None
        if self.dlpfc_enabled:
            self._dlpfc_words = self._resolve_vocab_words(concepts, self.seed)
            self._dlpfc_n = max(600, 60 * len(self._dlpfc_words))

        self.bridge = build_unified_bridge(seed=self.seed, proj_dim=self.proj_dim,
                                           enable_synaptic_route=self.synaptic_route_enabled,
                                           dlpfc_n=self._dlpfc_n, role_wta_n=self.role_wta_n,
                                           reservoir_n=self.reservoir_n)
        # RUNG B-1b: base index of the on-bridge role-WTA slice (past the composer, role_src, and dlPFC slices).
        _post_composer = (PARSER_SLICE_SIZE + 8 * self.proj_dim
                          + (len(SYNAPTIC_ROUTE_ROLES) * self.proj_dim if self.synaptic_route_enabled else 0)
                          + 2 * self._dlpfc_n)
        self.role_wta_base = _post_composer if self.role_wta_n else None
        # RUNG B-1c: base index of the on-bridge spiking-reservoir slice (past the WTA slice, if any).
        self.reservoir_base = (_post_composer + self.role_wta_n) if self.reservoir_n else None
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
        # 3b) dlPFC loop (opt-in) BEFORE training: pre-allocate the cortex_ctx<->dlpfc_wm self-attractors + all
        #     directed graph edges (weight 0, gated 0.0) and set the per-neuron NMDA mask to the dlPFC slice
        #     only. Also wired here (not lazily) for the same reason — every wiring re-injects the union and
        #     resets weights, so it must precede the parser's (final) training.
        self._dlpfc_ctx = None
        self._dlpfc_controller = None
        self._dlpfc_graph_key = None
        if self.dlpfc_enabled:
            self._wire_dlpfc()
        # 4) Train the parser LAST — no further wiring/re-injection follows, so the trained weights persist; the
        #    gated composer bind weights (and the role-route + dlPFC edges) stay frozen under this global-Hebbian
        #    training (Task 1 isolation).
        self.parser.train()

    @property
    def dlpfc_bridge(self):
        """The bridge the dlPFC loop runs on — the unified bridge itself when `enable_dlpfc=True` (the Step-3
        invariant: the dlPFC shares the bridge), else None."""
        return self.bridge if self.dlpfc_enabled else None

    @staticmethod
    def _resolve_vocab_words(concepts, seed):
        """The concept words the dlPFC sizes its assemblies for — the codebook keys when a {word: code} dict is
        given, else the sorted words of the `seed`'s denoise64 cache (the composer's default vocabulary). Loading
        the cache words here (a tiny read, not a bridge build) lets the dlPFC region be sized before the bridge
        is constructed; it raises FileNotFoundError exactly as the composer would if the cache is absent."""
        if concepts is not None:
            return sorted(concepts.keys())
        # Mirror CoreSimComposer's default load: read the denoise64 cache's word list for this seed.
        from research.runners.core_sim_composition import CACHE
        d = np.load(CACHE % int(seed))
        return sorted(k[5:] for k in d.files if k.startswith("obs__"))

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

        GATE PRE-WARM (faithful timing, not magnitude — resolves the seed-42 patient regression; see
        ROLE_GATE_PREWARM_CAP_STEPS): the readout is split into two windows over the SAME drive current —
          (1) a PRE-WINDOW that runs until the parser FIRES and the coupling OPENS the selected role's gate
              (capped at ROLE_GATE_PREWARM_CAP_STEPS), accumulating NOTHING. The gate genuinely opens from the
              parser's firing here — it is not set by hand.
          (2) the READOUT window, run holding the parser-opened gate (the per-step gate coupling is paused for
              this window so the gate RETAINS the value the parser's comprehension produced — the biologically
              correct order: comprehend → latch the route → compose). The coincidence banks are accumulated
              here, with the gate at the parser-opened value (1.0) for the whole window instead of flickering
              at the EMA threshold and starving the role bank.
        The coupling + the closed-gate default are restored at the end so the NEXT op starts clean (this op is
        self-contained, exactly as before). No synaptic weight or drive magnitude is changed.

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

        # (1) PRE-WINDOW: run (no accumulation) until the parser opens one of its role gates, capped. The gate
        # opens via the coupling from the parser's firing — purely the parser's doing, not a hand-set value.
        role_gate_names = [f"role_route_{r}" for r in SYNAPTIC_ROUTE_ROLES]

        def _any_role_gate_open():
            for gn in role_gate_names:
                syn = bridge._transmission_gate_to_synapses.get(gn)
                if syn is not None and bridge.cp_transmission_gain is not None:
                    if float(bridge.cp_transmission_gain[syn].mean()) >= 0.99:
                        return True
            return False

        for _ in range(ROLE_GATE_PREWARM_CAP_STEPS):
            bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
            bridge._run_one_simulation_step()
            if _any_role_gate_open():
                break

        # (2) READOUT: hold the parser-opened gate by pausing the per-step coupling for this window (the gate
        # keeps the value the parser's comprehension set), then accumulate the coincidence banks.
        saved_couplings = bridge._gate_couplings
        bridge._gate_couplings = []
        try:
            acc = {b: xp.zeros(comp.D, dtype=xp.float64) for b in ("A", "B", "C", "D")}
            for _ in range(comp.run_steps):
                bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
                bridge._run_one_simulation_step()
                for b in ("A", "B", "C", "D"):
                    acc[b] += bridge.cp_firing_states[idx[b]].astype(xp.float64)
        finally:
            # Restore the coupling and re-close every role gate + reset its EMA so the next op starts clean
            # (each op holds the gate closed at entry and re-opens only the parser-selected role).
            bridge._gate_couplings = saved_couplings
            for r in SYNAPTIC_ROUTE_ROLES:
                bridge.set_transmission_gate(f"role_route_{r}", 0.0)
                cpl = next((c for c in bridge._gate_couplings if c["gate_name"] == f"role_route_{r}"), None)
                if cpl is not None:
                    cpl["ema"] = 0.0
                    cpl["last_value"] = None
            bridge.cp_external_input_current[:] = 0.0

        rates = {b: to_host(acc[b]) / comp.run_steps for b in ("A", "B", "C", "D")}
        return rates["A"] + rates["B"], rates["C"] + rates["D"]

    # --- Step 3: the dlPFC dialogue-planning loop on the shared bridge ---
    def _wire_dlpfc(self):
        """Wire, once at construction, the dlPFC `cortex_ctx ↔ dlpfc_wm` loop onto the unified bridge:

          1. Lay the two dlPFC regions as contiguous index slices PAST the composer (and any role-src) slice.
          2. Build the shared-slice context (`_SharedDlpfcContext`) whose per-concept assemblies are
             byte-identically laid out to the separate `SpikingLoopContextBuffer` (oracle parity).
          3. PRE-ALLOCATE every dlPFC edge into the union plan (so no later `set_pathway_weights` rebuilds the
             CSR and breaks the composer's bind-gate isolation):
               * c2d: cortex_A → dlpfc_B for ALL (A, B) word pairs (V² pairs). The DIAGONAL (A==B) is the
                 forward self-attractor (weight DLPFC_ATTRACTOR_WEIGHT); the off-diagonal pairs are the
                 association-graph edge SLOTS, pre-allocated at weight 0 and overwritten at `elaborate` time.
               * d2c: dlpfc_C → cortex_C for all C (the backward self-attractor, weight DLPFC_ATTRACTOR_WEIGHT).
             All dlPFC edges are tagged `plasticity_gate=DLPFC_FIXED_GATE` and held at 0.0 so the parser's
             global Hebbian training (and any elaborate-time firing) cannot drift them.
          4. Set the per-neuron NMDA mask 1.0 on the dlPFC slice, 0.0 elsewhere — NMDA current reaches ONLY the
             dlPFC (parser+composer stay NMDA-free despite the global `cfg.enable_nmda=True`).

        Self-attractor weight is DLPFC_ATTRACTOR_WEIGHT (≈30), the de-risk's genuinely NMDA-DEPENDENT regime
        (the module's 50 would be trivial AMPA ping-pong — the wrong mechanism); see the module-level note.

        NOTE on scale: pre-allocating V² c2d pairs is fine at the validated probe vocabulary (V=16). Scaling the
        dlPFC to the production V≈320 vocabulary would need a sparser pre-allocation (only the realizable
        association pairs) — a documented step-4 concern, out of Task-2 scope.
        """
        xp, _ = get_backend()
        words = sorted(self.composer.words)
        V = len(words)
        n = self._dlpfc_n
        ps = DLPFC_PATTERN_SIZE

        # dlPFC slices live PAST the composer (+ role-src route pools, if wired).
        base = self.composer_offset + 8 * self.proj_dim
        if self.synaptic_route_enabled:
            base += len(SYNAPTIC_ROUTE_ROLES) * self.proj_dim
        self._dlpfc_cortex_base = base
        self._dlpfc_dlpfc_base = base + n

        # Shared-slice context: assemblies laid out IDENTICALLY to SpikingLoopContextBuffer (same n, ps, seed,
        # word ordering) so the borrowed spreading-activation methods reproduce the separate path.
        self._dlpfc_ctx = _SharedDlpfcContext(self.bridge, words, self._dlpfc_cortex_base,
                                              self._dlpfc_dlpfc_base, n, ps, self.seed)

        # --- Pre-allocate the dlPFC edges (vectorized outer products) into one gated population. ---
        cpat = {w: self._dlpfc_ctx._cpat_host[w] for w in words}   # cortex assembly indices per word
        dpat = {w: self._dlpfc_ctx._dpat_host[w] for w in words}   # dlPFC assembly indices per word

        c2d_pre = []; c2d_post = []; c2d_w = []
        for a in words:                        # cortex_A -> dlpfc_B for ALL (A, B): diagonal = self-attractor
            preA = np.repeat(cpat[a], ps)      # (ps*ps,) each cortex_A neuron -> every dlpfc_B neuron
            for b in words:
                c2d_pre.append(preA)
                c2d_post.append(np.tile(dpat[b], ps))
                w0 = DLPFC_ATTRACTOR_WEIGHT if a == b else 0.0     # self-attractor on the diagonal, slot elsewhere
                c2d_w.append(np.full(ps * ps, w0, dtype=np.float32))
        d2c_pre = []; d2c_post = []; d2c_w = []
        for c in words:                        # dlpfc_C -> cortex_C (backward self-attractor only)
            d2c_pre.append(np.repeat(dpat[c], ps))
            d2c_post.append(np.tile(cpat[c], ps))
            d2c_w.append(np.full(ps * ps, DLPFC_ATTRACTOR_WEIGHT, dtype=np.float32))

        pre = np.concatenate(c2d_pre + d2c_pre).astype(np.int64)
        post = np.concatenate(c2d_post + d2c_post).astype(np.int64)
        ww = np.concatenate(c2d_w + d2c_w).astype(np.float32)
        plan = {"dlpfc_loop": {
            "pre_indices": pre, "post_indices": post, "initial_weights": ww,
            "plastic": False, "plasticity_gate": DLPFC_FIXED_GATE, "conn_type": "E_TO_E", "count": int(pre.size),
        }}
        merge_population_into_shared_bridge(self.bridge, plan, gates_to_zero=(DLPFC_FIXED_GATE,))

        # --- Per-region NMDA: mask 1.0 on the dlPFC slice ONLY (parser+composer stay NMDA-free). ---
        nmask = xp.zeros(self.bridge.core_config.num_neurons, dtype=xp.float32)
        nmask[self._dlpfc_cortex_base:self._dlpfc_dlpfc_base + n] = 1.0
        self.bridge.cp_nmda_neuron_mask = nmask

    def _assoc_graph(self):
        """An association graph (concept -> {concept: weight}) built from the agent's OWN stored facts (the
        composer's kb): the agent/action/patient of each fact co-occur. Clause patients are skipped (their inner
        concepts are structural). Identical to `BrainConversationalAgent._assoc_graph` — the graph the dialogue-
        planning Control spreads over."""
        graph = {}
        for fact, _ in self.composer.kb:
            cs = [fact.get(r) for r in ("agent", "action", "patient")]
            cs = [c for c in cs if isinstance(c, str)]
            for x in cs:
                for y in cs:
                    if x != y:
                        graph.setdefault(x, {})[y] = graph.get(x, {}).get(y, 0.0) + 1.0
        return graph

    def elaborate(self, topic):
        """Dialogue planning on the SHARED bridge: bring up the next on-topic concept about `topic`, chosen by
        the dlPFC spiking spreading-activation Control over the agent's own association graph — driving the
        unified bridge's dlPFC slice (NOT a throwaway bridge). Returns an associate concept, or None if `topic`
        is unconnected (the abstention / no-confab moat). Matches `BrainConversationalAgent.elaborate`.

        The Control is a `SpikingSpreadingController` whose `.ctx` is swapped for the shared-slice context
        (`_SharedDlpfcContext`): its validated methods (`_install_graph_edges`, `relevance_by_latency`,
        `turn_latency`, `_reset_wm`) run UNCHANGED against the shared dlPFC slice — reuse-by-import, no edit to
        `content_selection_spiking`. The association graph stays Python-built (scope clamp). The Control is cached
        and rebuilt only when the graph CONTENT changes (the off-diagonal c2d graph-edge weights are overwritten
        in place — no CSR rebuild — so the composer's bind isolation is preserved)."""
        if not self.dlpfc_enabled:
            raise RuntimeError(
                "elaborate requires the dlPFC loop — construct UnifiedBrainBridge(enable_dlpfc=True). The default "
                "build wires only the parser+composer (comprehend/store/recall); dialogue planning is opt-in.")
        from research.runners.content_selection_spiking import SpikingSpreadingController
        from research.runners.content_selection import SaidTrace

        graph = self._assoc_graph()
        if topic not in graph:
            return None
        # cache key = the graph CONTENT (not kb length: different fact sets can share a length -> stale Control).
        key = tuple(sorted((k, tuple(sorted(v.items()))) for k, v in graph.items()))
        if self._dlpfc_controller is None or self._dlpfc_graph_key != key:
            # Build a SpikingSpreadingController WITHOUT its __init__ (which would build a throwaway bridge): set
            # its attributes by hand against the shared-slice context, then install the graph edges (in place,
            # over the pre-allocated slots) via its OWN validated method. This reuses the validated method bodies
            # verbatim — no edit to content_selection_spiking.
            ctrl = object.__new__(SpikingSpreadingController)
            ctrl.graph = graph
            ctrl._vocab = sorted(set(graph) | {a for v in graph.values() for a in v})
            ctrl.ctx = self._dlpfc_ctx
            ctrl.said = SaidTrace(decay=0.9)                      # the validated SpikingSpreadingController default
            ctrl._install_graph_edges(DLPFC_EDGE_SCALE)           # overwrites the pre-allocated c2d slots IN PLACE
            self._dlpfc_controller = ctrl
            self._dlpfc_graph_key = key
        # The dlPFC's VALIDATED dialogue-planning config (content_selection_spiking, 6/6 seeds 2026-06-03) runs
        # OU OFF: OU background noise tips the bistable concept attractors into spurious ON states (Hopfield
        # spurious states), corrupting the latency-ranked selection. The unified bridge runs OU ON for the
        # parser+composer; here we toggle it off ONLY for the dlPFC spreading-activation read (elaborate drives
        # and reads the dlPFC slice alone — parser+composer are not active during it), matching the validated
        # regime. `cfg.enable_ou_process` is read dynamically each step (sim/bridge.py), so the toggle is clean.
        prev_ou = self.bridge.core_config.enable_ou_process
        self.bridge.core_config.enable_ou_process = False
        try:
            return self._dlpfc_controller.turn_latency([topic])
        finally:
            self.bridge.core_config.enable_ou_process = prev_ou

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
    def hear(self, sentence, voice="active", polarity=None):
        """Comprehend an SVO statement and store it (parse -> store), mirroring `BrainConversationalAgent.hear`
        exactly. This is the transparent-API entry point so the conversational agent runs on the unified bridge
        unchanged. `sentence` is 'agent action patient' (or its passive frame). The step-2 spiking
        comprehend->compose route is the separate `hear_synaptic`; this is the default parse-then-store
        comprehension that `elaborate`'s association graph (built from stored facts) reads from."""
        roles = self.parse(sentence, voice)
        self.store(roles["agent"], roles["action"], roles["patient"], polarity=polarity)
        return roles

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
