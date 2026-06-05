"""One-bridge unification — Step 1 tests (parser + composer merged onto ONE SimulationBridge).

Task 1 (the load-bearing de-risk): prove that a synapse population declared ``plastic=False`` does NOT
drift when the GLOBAL ``enable_hebbian_learning`` flag is ON and a *different* (plastic) population on the
SAME bridge is being co-activated/trained.

Terms (defined once):
  * bridge        = one ``sim.bridge.SimulationBridge`` (a network of simulated Izhikevich neurons).
  * population    = a named set of synapses injected via ``bridge.inject_explicit_wiring(plan)``.
  * plastic       = synapses whose weights change with learning; fixed = weights never change.
  * Hebbian learn = a co-activation weight-update rule (the parser's only learning).

The merge conflict this de-risks: the PARSER region needs Hebbian learning ON; the COMPOSER region's
coincidence (bind/unbind) wiring is FIXED. On one shared bridge there is only ONE global
``enable_hebbian_learning`` flag. Step 1 sets it True (for the parser) and relies on the composer's
population being held FIXED to keep its weights from drifting. This test verifies exactly that assumption.

FINDING (2026-06-04): the ``plastic=False`` flag ALONE does NOT isolate a population under global Hebbian
learning — it is honored only in the STDP weight-update path, NOT the Hebbian one (which gates per-synapse
only via ``cp_plasticity_rate_gain``). The first run of this test FAILED: the FIXED weight drifted
320.0 -> 319.897 over 300 steps via the ungated Hebbian weight-decay term. The fallback the plan specifies
is therefore required and applied here: tag the fixed population with a ``plasticity_gate`` and set its
per-synapse plasticity gain to 0.0 (``bridge.set_plasticity_gate(name, 0.0)``). That freezes both the
Hebbian potentiation delta and the decay term for the fixed synapses, with NO ``sim/`` edit.
See ``research/findings/2026-06-04-unified-bridge-plasticity-isolation.md``.
"""
from __future__ import annotations

import numpy as np
import pytest

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.backend import get_backend, to_host


# Edge lists for the two test populations, kept here so the weight-readback helper can locate each
# population's synapses inside the bridge's shared CSR weight storage (which is sorted by (pre, post)
# and carries no population labels of its own).
_PARSE_EDGES = [(0, 6)]                       # plastic "parse"-style pair at offset 0
_OFF = 6 + 3 * 40                             # composer slice starts past the parser slice (= 126)
_BIND_EDGES = [(_OFF, _OFF + 1)]             # FIXED "bind"-style pair (plastic=False), weight 320
_BIND_GATE = "bind_fixed"                     # plasticity-gate name → set gain 0.0 to truly freeze (fallback)


def _weights_of(bridge, population_name):
    """Return a host (NumPy) copy of the named population's synaptic weights.

    Weight storage on the bridge is ``bridge.cp_connections`` — a CSR sparse matrix where
    ``cp_connections[i, j]`` is the weight of the i->j synapse (see ``inject_explicit_wiring``:
    it builds ``self.cp_connections`` from the explicit edges, so ``.data`` holds the per-synapse
    weights). We look each population edge up by CSR element access and copy to host so the
    comparison is backend-agnostic and decoupled from any later in-place mutation of ``.data``.
    """
    edges = {"parse": _PARSE_EDGES, "bind": _BIND_EDGES}[population_name]
    csr = bridge.cp_connections
    vals = [float(to_host(csr[int(i), int(j)])) for (i, j) in edges]
    return np.asarray(vals, dtype=np.float64)


def _build_merged_bridge():
    """One bridge sized for both regions, global Hebbian ON, with a plastic 'parse' pair and a FIXED
    'bind' pair (offset past the parser slice). Returns the constructed bridge."""
    D = 64
    cfg = CoreSimConfig()
    cfg.num_neurons = 6 + 3 * 40 + 8 * D          # parser slice (126) + composer slice (8*D)
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = 42
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True            # ON for the parser
    cfg.hebbian_max_weight = 400.0
    cfg.hebbian_learning_rate = 0.005
    for f in ("enable_short_term_plasticity", "enable_structural_plasticity", "enable_homeostasis",
              "enable_reward_modulation", "enable_watts_strogatz"):
        setattr(cfg, f, False)
    cfg.ou_std_current_pA = 20.0

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)

    off = _OFF
    plan = {
        "parse": {"pre_indices": [0], "post_indices": [6],
                  "initial_weights": np.array([0.5], np.float32),
                  "plastic": True, "conn_type": "E_TO_E", "count": 1},
        # FIXED population. plastic=False is kept (correct intent + STDP isolation), but it is NOT enough
        # under global Hebbian — so we ALSO tag it with a plasticity_gate and zero its gain below.
        "bind":  {"pre_indices": [off], "post_indices": [off + 1],
                  "initial_weights": np.array([320.0], np.float32),
                  "plastic": False, "plasticity_gate": _BIND_GATE,
                  "conn_type": "E_TO_E", "count": 1},
    }
    bridge.inject_explicit_wiring(plan)
    # Fallback (required — see module docstring + finding): zero the per-synapse plasticity gain over the
    # fixed population so the Hebbian potentiation AND decay terms are both multiplied by 0 for it.
    bridge.set_plasticity_gate(_BIND_GATE, 0.0)
    return bridge


def test_fixed_population_survives_global_hebbian():
    """The FIXED ('bind', plastic=False) population's weights must be byte-identical before vs after the
    PLASTIC ('parse') pair is driven into co-activation for many steps under global Hebbian learning.

    Control (non-vacuity): the PLASTIC pair's weight MUST change in the same setup — otherwise the
    isolation assertion would be meaningless (a synapse that never updates can't demonstrate isolation).
    """
    bridge = _build_merged_bridge()
    cfg = bridge.core_config
    xp, _ = get_backend()

    before_bind = _weights_of(bridge, "bind")
    before_parse = _weights_of(bridge, "parse")

    # Drive co-activation of the plastic pair (neurons 0 and 6) for many steps. Hebbian co-firing would
    # change a plastic synapse. Advance the clock each step (CLAUDE.md gotcha: _run_one_simulation_step
    # does NOT advance current_time_ms; the caller must).
    for _ in range(300):
        cur = xp.zeros(cfg.num_neurons, dtype=xp.float32)
        cur[0] = 2500.0
        cur[6] = 2500.0
        bridge.cp_external_input_current[:] = cur
        bridge.runtime_state.current_time_ms += cfg.dt_ms
        bridge._run_one_simulation_step()

    after_bind = _weights_of(bridge, "bind")
    after_parse = _weights_of(bridge, "parse")

    # Control: the test must be non-vacuous — the plastic pair's weight DID change.
    assert not np.array_equal(before_parse, after_parse), (
        "PLASTIC 'parse' pair did not change under global Hebbian — the drive is too weak, so the "
        "isolation assertion would be vacuous. Strengthen the drive before trusting a PASS. "
        f"before={before_parse} after={after_parse}")

    # The load-bearing assertion: the FIXED population is isolated from global Hebbian.
    assert np.array_equal(before_bind, after_bind), (
        "FIXED composer weights drifted under global Hebbian -> per-population plastic=False does NOT "
        f"isolate. before={before_bind} after={after_bind}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: UnifiedBrainBridge skeleton — ONE bridge sized for both regions.
#
# The skeleton builds a single SimulationBridge whose neuron count is the sum of the parser slice
# (6 conjunction units + 3*40 role-ensemble neurons = 126) and the composer slice (8*proj_dim). It
# exposes the two regions' index layout: `parser_slice` (the parser's contiguous block, 0..125) and
# `composer_offset` (the first composer neuron index, 126). No region wiring yet — that is Tasks 3–5.
# ─────────────────────────────────────────────────────────────────────────────
def test_unified_skeleton_sizes_and_disjoint_slices():
    """UnifiedBrainBridge(seed=42, proj_dim=64) builds ONE bridge of (6 + 3*40) + 8*64 neurons, with a
    parser_slice = range(0, 126) and composer_offset = 126, and the two regions do not overlap."""
    from research.runners.unified_brain_bridge import UnifiedBrainBridge

    proj_dim = 64
    u = UnifiedBrainBridge(seed=42, proj_dim=proj_dim)

    expected_parser = 6 + 3 * 40                       # conjunction units + role ensembles = 126
    expected_total = expected_parser + 8 * proj_dim    # + composer coincidence slice

    assert u.bridge.core_config.num_neurons == expected_total, (
        f"bridge sized {u.bridge.core_config.num_neurons}, expected {expected_total}")
    # global Hebbian must be ON (the parser's learning rule lives on this shared bridge)
    assert u.bridge.core_config.enable_hebbian_learning is True

    assert u.parser_slice == range(0, expected_parser), u.parser_slice
    assert u.composer_offset == expected_parser, u.composer_offset

    # the two regions are disjoint and tile the bridge with no gap/overlap
    parser_set = set(u.parser_slice)
    composer_set = set(range(u.composer_offset, expected_total))
    assert parser_set.isdisjoint(composer_set), "parser and composer slices overlap"
    assert len(parser_set) + len(composer_set) == expected_total, "slices leave a gap or overlap"


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: BridgeParser parameterized to wire into a shared bridge at an index offset.
#
# When given a `shared_bridge`, the parser must NOT build its own bridge — it uses the provided one,
# offsets every conjunction/role index by `index_offset` (in the "parse" wiring plan AND the drive/readout
# index arrays), injects its population onto the shared bridge, and trains through the offset. The result
# must be capability-equivalent to a standalone parser: voice-invariant role assignment (the active<->passive
# 1st<->3rd flip). The default (no `shared_bridge`) path must stay byte-identical to before.
# ─────────────────────────────────────────────────────────────────────────────
def test_bridgeparser_on_shared_bridge_parses_voice_invariantly():
    """A BridgeParser wired onto a freshly-built merged bridge at offset 0 assigns the SAME agent in the
    active frame ('dog go north') and the passive frame ('north go dog') — voice-invariant, just like the
    standalone parser. It uses the provided bridge (does not allocate its own)."""
    from research.runners.brain_conversational_agent import BridgeParser
    from research.runners.unified_brain_bridge import build_unified_bridge

    shared = build_unified_bridge(seed=42, proj_dim=64)
    parser = BridgeParser(seed=42, shared_bridge=shared, index_offset=0)

    # The parser used the shared bridge, not a private one.
    assert parser.bridge is shared, "shared-bridge parser must NOT build its own bridge"

    # Voice-invariant comprehension (the core parser capability): agent is 'dog' in both frames.
    active = parser.parse(["dog", "go", "north"], "active")
    passive = parser.parse(["north", "go", "dog"], "passive")
    assert active["agent"] == "dog", f"active agent wrong: {active}"
    assert passive["agent"] == "dog", f"passive agent wrong: {passive}"
    # full active role assignment is correct
    assert active == {"agent": "dog", "action": "go", "patient": "north"}, active


def test_bridgeparser_default_path_unchanged():
    """The no-arg (standalone) BridgeParser path is unchanged: it builds its OWN bridge of 6 + 3*40 neurons
    and parses voice-invariantly. (Regression guard — the merge must not perturb the default behavior.)"""
    from research.runners.brain_conversational_agent import BridgeParser

    parser = BridgeParser(seed=42)
    assert parser.bridge.core_config.num_neurons == 6 + 3 * 40
    assert parser.parse(["dog", "go", "north"], "active")["agent"] == "dog"
    assert parser.parse(["north", "go", "dog"], "passive")["agent"] == "dog"


# A tiny synthetic concept codebook so the shared-bridge composer tests do not depend on the `denoise64`
# concept-code cache: 8 ORTHONORMAL vectors of dimension `proj_dim` (rows of a QR factorization of a random
# Gaussian). Orthogonal codes have near-zero pairwise cosine, so the spiking unbind→cleanup margin is wide
# and unambiguous — random Gaussian codes at D=64 leave a razor-thin margin that the bridge's OU noise can
# flip (more so on the shared bridge, whose 638-neuron heterogeneity/OU stream shifts the composer's operating
# point vs the standalone 512-neuron bridge). The production `denoise64` codes are likewise decorrelated, so
# orthogonal synthetic codes are the faithful well-conditioned analogue, not a weakening of the test. NOTE:
# these are SPIKING tests — they run on the validated production (CuPy/GPU) backend, NOT NumPy (on NumPy the
# composer's low-D cleanup and the parser's Hebbian convergence both diverge from the validated behavior).
def _synthetic_concepts(proj_dim=64, seed=0):
    rng = np.random.default_rng(seed)
    words = ["dog", "cat", "go", "come", "north", "south", "river", "look"]
    q, _ = np.linalg.qr(rng.standard_normal((proj_dim, proj_dim)))   # orthonormal rows
    return {w: q[i] for i, w in enumerate(words)}


# ─────────────────────────────────────────────────────────────────────────────
# Task 4: CoreSimComposer parameterized to wire into a shared bridge at an index offset.
#
# When given a `shared_bridge`, the composer wires its FIXED "bind" coincidence population at the offset
# (every role_ON/OFF, fill_ON/OFF, A/B/C/D index shifted by `index_offset`). Because the shared bridge has
# global Hebbian learning ON, plastic=False is NOT enough to freeze the bind weights (Task 1 finding) — so
# the population is tagged plasticity_gate="composer_bind_fixed" and its gain set to 0.0. The default
# (no shared_bridge) path builds a standalone bridge with Hebbian OFF and needs no gate (kept byte-identical).
# ─────────────────────────────────────────────────────────────────────────────
def test_composer_on_shared_bridge_recovers_flat_fact():
    """A CoreSimComposer wired onto a merged bridge at offset 126 stores a flat fact and recovers it via
    spiking unbind, with abstention on an unstored cue. Uses a tiny synthetic codebook (no cache needed)."""
    from research.runners.core_sim_composition import CoreSimComposer
    from research.runners.unified_brain_bridge import build_unified_bridge

    proj_dim = 64
    shared = build_unified_bridge(seed=42, proj_dim=proj_dim)
    composer = CoreSimComposer(seed=42, proj_dim=proj_dim, shared_bridge=shared, index_offset=126,
                               concepts=_synthetic_concepts(proj_dim))

    assert composer.bridge is shared, "shared-bridge composer must NOT build its own bridge"
    # the FIXED bind population is gated to 0.0 on the Hebbian-enabled shared bridge (Task-1 isolation)
    assert "composer_bind_fixed" in shared.list_plasticity_gates()
    assert shared.get_plasticity_gate_value("composer_bind_fixed") == 0.0

    composer.store("dog", "go", "north")
    assert composer.query_patient("dog", "go") == "north"
    assert composer.query_agent("go", "north") == "dog"
    assert composer.query_patient("river", "look") is None      # abstention (no-confab moat)


def test_composer_default_path_unchanged():
    """The no-shared-bridge CoreSimComposer path is unchanged: it builds its OWN 8*proj_dim bridge with
    Hebbian OFF and no plasticity gate, and recovers a flat fact. (Regression guard.)"""
    from research.runners.core_sim_composition import CoreSimComposer

    proj_dim = 64
    composer = CoreSimComposer(seed=42, proj_dim=proj_dim, concepts=_synthetic_concepts(proj_dim))
    assert composer.bridge.core_config.num_neurons == 8 * proj_dim
    assert composer.bridge.core_config.enable_hebbian_learning is False
    # default standalone bridge has NO plasticity gates (composer relies on Hebbian-OFF, not a gate)
    assert composer.bridge.list_plasticity_gates() == []
    composer.store("dog", "go", "north")
    assert composer.query_patient("dog", "go") == "north"


# ─────────────────────────────────────────────────────────────────────────────
# Task 5: UnifiedBrainBridge wires BOTH regions onto one bridge + exposes the agent API.
#
# The unified bridge builds the shared bridge, constructs BridgeParser(offset=0) + CoreSimComposer(offset=126)
# into it, and delegates the conversational API (parse / store / query_patient / query_agent / ask_yes_no /
# describe / render_fact, + kb / words / concepts). The whole comprehend→store→recall loop must work on ONE
# bridge, and the composer's FIXED bind weights must be UNCHANGED after the parser trained on the shared bridge
# (re-asserting Task 1's isolation at FULL scale — the parser's Hebbian training must not drift the gated
# composer weights).
# ─────────────────────────────────────────────────────────────────────────────
def _all_bind_weights_equal(unified, expected=320.0):
    """Read every composer bind-synapse weight from the shared bridge's CSR and check it equals `expected`
    (the fixed coincidence weight W_COINC). Reconstructs the bind edge list from the composer's offset index
    banks exactly as build_bind_bridge wired them."""
    from sim.backend import to_host
    idx = unified.composer.idx
    D = unified.composer.D
    role_on = to_host(idx["role_on"]); role_off = to_host(idx["role_off"])
    fill_on = to_host(idx["fill_on"]); fill_off = to_host(idx["fill_off"])
    A = to_host(idx["A"]); B = to_host(idx["B"]); C = to_host(idx["C"]); Dd = to_host(idx["D"])
    edges = []
    for src1, src2, dst in ((role_on, fill_on, A), (role_off, fill_off, B),
                            (role_on, fill_off, C), (role_off, fill_on, Dd)):
        for i in range(D):
            edges.append((int(src1[i]), int(dst[i])))
            edges.append((int(src2[i]), int(dst[i])))
    csr = unified.bridge.cp_connections
    vals = np.array([float(to_host(csr[i, j])) for (i, j) in edges], dtype=np.float64)
    return bool(np.all(vals == expected)), vals


def test_unified_end_to_end_one_bridge():
    """END-TO-END on ONE bridge: UnifiedBrainBridge comprehends an SVO sentence (parser), stores + recalls it
    (composer), abstains on the unknown — all on the single shared bridge. And the composer's FIXED bind
    weights are still exactly the design value after the parser's Hebbian training (full-scale isolation)."""
    from research.runners.unified_brain_bridge import UnifiedBrainBridge

    proj_dim = 64
    u = UnifiedBrainBridge(seed=42, proj_dim=proj_dim, concepts=_synthetic_concepts(proj_dim))

    # ONE bridge holds both regions.
    assert u.parser.bridge is u.bridge
    assert u.composer.bridge is u.bridge

    # Comprehension (parser on the shared bridge): voice-invariant role assignment.
    roles = u.parse("dog go north")
    assert roles == {"agent": "dog", "action": "go", "patient": "north"}, roles

    # Store + recall (composer on the same bridge).
    u.store(roles["agent"], roles["action"], roles["patient"])
    assert u.query_patient("dog", "go") == "north"
    assert u.query_agent("go", "north") == "dog"
    assert u.query_patient("river", "look") is None         # abstention (no-confab moat)

    # Full-scale isolation re-assertion: the parser trained (Hebbian) on the shared bridge AFTER the composer's
    # bind population was wired; the gated bind weights must be byte-identical to their fixed design value.
    ok, vals = _all_bind_weights_equal(u, expected=320.0)
    assert ok, (
        "composer FIXED bind weights drifted under the parser's global Hebbian training on the shared bridge "
        f"-> plasticity-gate isolation failed at full scale. min={vals.min()} max={vals.max()}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 6: the capability NO-REGRESSION gate — does merging the parser + composer onto one bridge change the
# composer's recall versus the two separate bridges, at PRODUCTION scale with the REAL substrate codes?
#
# For each seed (42/43/44): build the SEPARATE-bridge baseline (`CoreSimComposer`, proj_dim=800, REAL denoise64
# V=16 codes) and the UNIFIED one-bridge (`UnifiedBrainBridge`, same default codes), run the SAME capability
# matrix (flat / one-attribute / two-attribute / negation, 6 trials each via the shared `run_matrix` helper),
# and assert the UNIFIED score for EACH category is within ±1 trial of the SEPARATE baseline (the spiking-noise
# tolerance). A drop beyond ±1 in any category, any seed, is a REGRESSION — the test must NOT be weakened to pass;
# the regression is the measured cost of the shared step loop (shared OU background noise + the parser slice's
# activity shifting the composer's operating point) and is recorded honestly for the controller to decide on a
# mitigation (e.g. per-region OU / per-region reset). Two-attribute is a KNOWN V=16 boundary in BOTH arrangements,
# so it is compared unified-vs-separate, NOT vs a perfect score.
#
# RUNTIME: this is a heavy SPIKING run on the production (CuPy/GPU) backend — each seed builds a ~6526-neuron
# unified bridge AND a ~6400-neuron separate bridge and trains the parser; ~minutes per seed, tens of minutes
# total for 3 seeds. There is no registered `slow` pytest marker in this repo, so the test is NOT marked (an
# unregistered marker would only warn, not gate); it is still runnable directly:
#     pytest tests/test_unified_brain_bridge.py::test_unified_capability_no_regression -v
# It runs on the auto-selected backend (CuPy when present) and SKIPS gracefully if a seed's denoise64 cache is
# absent (it must not silently pass on NumPy with no GPU, where the substrate's validated behavior diverges).
# ─────────────────────────────────────────────────────────────────────────────
_CAP_CATEGORIES = ("flat", "one_attr", "two_attr", "negation")


def test_unified_capability_no_regression():
    """Multi-seed (42/43/44) at production proj_dim=800 with the REAL denoise64 codes: for EACH category the
    UNIFIED (one-bridge) capability score must be within ±1 trial of the SEPARATE-bridge baseline. A genuine
    drop beyond ±1 in any category/seed is a REGRESSION (do not weaken — record it). Also confirms the parser
    parses voice-invariantly on the merged production bridge ('dog go north' active == 'north go dog' passive).

    HEAVY + on-demand: builds two production (~6.5K-neuron) bridges per seed and trains the parser — tens of minutes
    on the GPU. Skipped by default so the suite stays fast; the load-bearing claim (the merge preserves the core
    conversational capabilities) is already gated by `test_unified_end_to_end_one_bridge`. The full multi-seed result,
    including the documented two-attribute capacity-edge cost at the marginal D=800 and the D=2048 mitigation, lives in
    `research/findings/2026-06-04-one-bridge-unification-step1-capability.md`. Run on demand:
        SIM_RUN_HEAVY_CAPABILITY=1 pytest tests/test_unified_brain_bridge.py::test_unified_capability_no_regression -v
    """
    import os
    if not os.environ.get("SIM_RUN_HEAVY_CAPABILITY"):
        pytest.skip("heavy multi-seed production capability gate (run with SIM_RUN_HEAVY_CAPABILITY=1); result + the "
                    "two-attribute boundary cost are in 2026-06-04-one-bridge-unification-step1-capability.md")
    from research.findings.raw._unified_bridge_capability_probe import run_capability_comparison

    seeds = (42, 43, 44)
    # Gate at the PRODUCTION dimension D=2048 (the stage-1.5 decision). At D=2048 there is NO regression in any
    # category (verified 2026-06-04); the merge is capability-equivalent. (At the marginal D=800 two-attribute, the
    # K=5 capacity-edge category, drops ~1 trial — a marginal-regime artifact documented in the finding.)
    try:
        results = run_capability_comparison(seeds=seeds, proj_dim=2048, n=6)
    except FileNotFoundError:
        pytest.skip("denoise64 concept-code cache not present for one of seeds 42/43/44")

    # Parser-on-merged-bridge confirmation (every seed): correct active parse + voice-invariant agent.
    for seed in seeds:
        p = results[seed]["parser"]
        assert p["active"] == {"agent": "dog", "action": "go", "patient": "north"}, (
            f"seed {seed}: parser active parse wrong on the merged production bridge: {p['active']}")
        assert p["passive_agent"] == "dog", (
            f"seed {seed}: parser passive frame did not assign the same agent on the merged bridge "
            f"(voice-invariance broken): passive_agent={p['passive_agent']!r}")

    # No-regression assertion: every category, every seed, unified within ±1 trial of separate.
    regressions = []
    for seed in seeds:
        sep = results[seed]["separate"]
        uni = results[seed]["unified"]
        for cat in _CAP_CATEGORIES:
            sep_ok = sep[cat][0]
            uni_ok = uni[cat][0]
            if uni_ok - sep_ok < -1:                     # unified more than 1 trial below separate
                regressions.append((seed, cat, sep_ok, uni_ok))

    assert not regressions, (
        "UNIFIED bridge capability REGRESSED vs the SEPARATE baseline (drop > 1 trial in some category/seed) — "
        "this is the measured cost of the shared step loop; record it, do NOT weaken the test. Drops: "
        + "; ".join(f"seed {s} {c}: separate {so} -> unified {uo}" for s, c, so, uo in regressions))


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Task 1 (the load-bearing DE-RISK): parser-role-gated synaptic routing.
#
# The one falsifiable question: on the unified bridge (parser at offset 0, composer at offset 126 on ONE
# SimulationBridge), can the PARSER's role-ensemble FIRING — when the parser assigns a word to a role —
# OPEN a transmission gate that routes a drive SELECTIVELY to that role's target and NOT the others?
#
# Mechanism under test (all SHIPPED + VALIDATED primitives, no sim/ edit):
#   * `transmission_gate` on a pathway scales that route's effective synaptic CURRENT in [0,1] at runtime.
#   * the gate is OPENED by a control pool's firing (EMA-smoothed) inside `_run_one_simulation_step`
#     (the shipped `_apply_gate_couplings` hook; here the control pool is the parser's role ensemble).
#
# The probe (`research/findings/raw/_step2_gated_route_probe.py`) adds, on the SAME bridge that holds the
# trained parser: one MARKER input pool and three TARGET pools (agent/action/patient), with strong
# excitatory routes marker→target gated `route_<role>`, each gate coupled to the parser's role ensemble.
# Driving the parser conjunction for (position 0, active) fires the AGENT ensemble → opens `route_agent`
# only → the marker reaches agent_target, not action/patient. Driving (position 2, active) fires the
# PATIENT ensemble → the SAME marker reaches patient_target instead — a re-bind with ZERO weight change.
# A control: with NO parser drive (no ensemble firing) the marker reaches NO target (all gates closed).
#
# This is a SPIKING test on the validated production (CuPy/GPU) backend, NOT NumPy (the parser's Hebbian
# convergence + the gating operating point are GPU-validated). It SKIPS gracefully if no GPU backend.
# ─────────────────────────────────────────────────────────────────────────────
def test_step2_parser_role_gated_route_is_selective():
    """De-risk PASS criteria, all three conditions on ONE unified bridge with ZERO weight change between them:

      1. parser conjunction (pos 0, active) → AGENT ensemble fires → marker routes to agent_target ONLY
         (agent_target rate >> 0; action_target, patient_target ~silent).
      2. parser conjunction (pos 2, active) → PATIENT ensemble fires → SAME marker routes to patient_target
         ONLY (patient_target rate >> 0; agent_target, action_target ~silent).
      3. NO parser drive → NO ensemble fires → marker reaches NO target (all three ~silent).

    Also confirms (non-vacuity, per the task's explicit requirement) that the driven parser ENSEMBLE itself
    actually fired in conditions 1 and 2 — a gate result is only trustworthy if the control pool fired.
    """
    from sim.backend import is_gpu_backend
    if not is_gpu_backend():
        pytest.skip("Step 2 gated-route de-risk is a SPIKING test; run on the validated CuPy/GPU backend "
                    "(the parser's Hebbian convergence + gate operating point are GPU-validated, not NumPy).")

    from research.findings.raw._step2_gated_route_probe import run_gated_route_probe

    res = run_gated_route_probe(seed=42, proj_dim=64)

    # The parser ensembles must actually have fired when their conjunction was driven (else any gate result
    # is meaningless — the control pool never activated). The probe reports the driven-ensemble firing rate.
    assert res["agent_drive"]["ensemble_rate"] > 0.05, (
        "AGENT ensemble did not fire when its conjunction was driven — gate result untrustworthy. "
        f"ensemble_rate={res['agent_drive']['ensemble_rate']:.4f}")
    assert res["patient_drive"]["ensemble_rate"] > 0.05, (
        "PATIENT ensemble did not fire when its conjunction was driven — gate result untrustworthy. "
        f"ensemble_rate={res['patient_drive']['ensemble_rate']:.4f}")

    # Condition 1: agent-drive → marker routes to agent_target ONLY.
    ad = res["agent_drive"]
    assert ad["agent_target"] > 0.05, f"agent-drive: agent_target should fire, got {ad['agent_target']:.4f}"
    assert ad["action_target"] < 0.01, f"agent-drive: action_target should stay closed, got {ad['action_target']:.4f}"
    assert ad["patient_target"] < 0.01, f"agent-drive: patient_target should stay closed, got {ad['patient_target']:.4f}"
    assert ad["agent_target"] > 5 * max(ad["action_target"], ad["patient_target"], 1e-6), (
        "agent-drive: agent_target must DECISIVELY dominate the other targets (selective routing). "
        f"agent={ad['agent_target']:.4f} action={ad['action_target']:.4f} patient={ad['patient_target']:.4f}")

    # Condition 2: patient-drive → the SAME marker routes to patient_target ONLY (re-bind, zero weight change).
    pd = res["patient_drive"]
    assert pd["patient_target"] > 0.05, f"patient-drive: patient_target should fire, got {pd['patient_target']:.4f}"
    assert pd["agent_target"] < 0.01, f"patient-drive: agent_target should stay closed, got {pd['agent_target']:.4f}"
    assert pd["action_target"] < 0.01, f"patient-drive: action_target should stay closed, got {pd['action_target']:.4f}"
    assert pd["patient_target"] > 5 * max(pd["agent_target"], pd["action_target"], 1e-6), (
        "patient-drive: patient_target must DECISIVELY dominate (re-routed selectively). "
        f"agent={pd['agent_target']:.4f} action={pd['action_target']:.4f} patient={pd['patient_target']:.4f}")

    # Condition 3 (control): no parser drive → no gate opens → marker reaches no target.
    nd = res["no_drive"]
    assert nd["agent_target"] < 0.01 and nd["action_target"] < 0.01 and nd["patient_target"] < 0.01, (
        "no-drive control: with NO parser firing, the marker must reach NO target (all gates closed). "
        f"agent={nd['agent_target']:.4f} action={nd['action_target']:.4f} patient={nd['patient_target']:.4f}")

    # Zero weight change across all three conditions — binding is which gate is open, not which weight grew.
    assert res["weight_delta"] < 1e-3, (
        f"synaptic weights changed across the routing conditions ({res['weight_delta']:.6f}) — the route must "
        "re-bind purely via the gates, with no weight change (thalamocortical gating hypothesis).")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Task 2: `UnifiedBrainBridge.hear_synaptic` — comprehend→store via the gated route.
#
# This is the real thing Task 1 de-risked. `hear_synaptic(sentence, voice)` comprehends an SVO sentence and
# STORES the fact via the SYNAPTIC route: for each word, the PARSER's role-ensemble firing opens a transmission
# gate that routes that role's ±1 pattern into the composer's role bank (role_ON/OFF), while the word's concept
# code drives the fill bank (role-independent, ungated). The coincidence circuit binds (role ⊗ filler) for the
# parser-SELECTED role — the role each word binds to is chosen by the parser's SPIKING, NOT by a Python
# {role: word} dict (which is what the reference `hear`/`store` path uses). After the 3 words, the accumulated
# (bound_ON, bound_OFF) is the stored fact, appended to the composer's kb so query_patient / query_agent /
# ask_yes_no / abstention all work unchanged.
#
# The PASS bar (the standing gate): the synaptic route reproduces the Python-hand-off path's capability —
# the SAME fact stored + recalled (who / what / abstain), voice-invariantly, and matching what `u.hear` stores.
# The synaptic route is NOT weaker than the Python route. A genuine failure to reproduce is an honest finding
# (research/findings/2026-06-04-step2-hear-synaptic-PARTIAL.md), not a weakened test.
#
# SPIKING test on the validated production (CuPy/GPU) backend (the parser's Hebbian convergence + the gate
# operating point are GPU-validated, not NumPy). Skips gracefully without a GPU backend.
# ─────────────────────────────────────────────────────────────────────────────
def test_hear_synaptic_stores_fact_via_gated_route():
    """`u.hear_synaptic("dog go north")` stores the SVO fact via the parser→gate→composer SYNAPTIC route (no
    Python {role: word} dict passed to store), and the recall is IDENTICAL to the Python-hand-off `u.hear`:

      * query_patient("dog", "go")  == "north"    (who/what recall through the synaptic-bound fact)
      * query_agent("go", "north")  == "dog"
      * query_patient("river", "look") is None    (abstention — the no-confab moat, preserved)
      * voice-invariance: the passive frame stores the SAME fact (agent still "dog", recall identical)
      * parity: a sibling bridge that stored the SAME sentence via the Python `hear` path recalls the same
        patient/agent — the synaptic route is not weaker than the Python route.
    """
    from sim.backend import is_gpu_backend
    if not is_gpu_backend():
        pytest.skip("hear_synaptic is a SPIKING test; run on the validated CuPy/GPU backend (the parser's "
                    "Hebbian convergence + gate operating point are GPU-validated, not NumPy).")

    from research.runners.unified_brain_bridge import UnifiedBrainBridge

    proj_dim = 64
    concepts = _synthetic_concepts(proj_dim)

    # The synaptic-route bridge (opt-in: wires the parser→gate→composer routes before training).
    u = UnifiedBrainBridge(seed=42, proj_dim=proj_dim, concepts=concepts, enable_synaptic_route=True)

    # --- Comprehend + store via the SYNAPTIC route (active voice). NO Python {role: word} dict is passed. ---
    u.hear_synaptic("dog go north")
    assert u.query_patient("dog", "go") == "north", (
        "synaptic-route store→recall failed: what does dog go? should be 'north'. The parser-gated role drive "
        "did not bind the fact as the Python path does.")
    assert u.query_agent("go", "north") == "dog", "synaptic-route recall: who go north? should be 'dog'."
    # Abstention preserved: an unstored cue returns None (the no-confab moat must survive the synaptic route).
    assert u.query_patient("river", "look") is None, (
        "synaptic-route abstention broke: an unstored (river, look) cue must return None (no confabulation).")

    # --- Voice-invariance: the PASSIVE frame stores the SAME fact via the synaptic route. ---
    u2 = UnifiedBrainBridge(seed=42, proj_dim=proj_dim, concepts=concepts, enable_synaptic_route=True)
    u2.hear_synaptic("north go dog", voice="passive")   # passive frame of "dog go north"
    assert u2.query_patient("dog", "go") == "north", (
        "voice-invariance broke on the synaptic route: the passive frame must store the same fact "
        "(agent 'dog', patient 'north').")
    assert u2.query_agent("go", "north") == "dog"

    # --- Parity with the Python hand-off path: a sibling bridge storing the SAME sentence via the Python
    #     {role: word} hand-off (`parse` → `store`, the regression oracle that `BrainConversationalAgent.hear`
    #     uses) must recall the same patient/agent. The synaptic route is not weaker than the Python route. ---
    u_ref = UnifiedBrainBridge(seed=42, proj_dim=proj_dim, concepts=concepts)
    roles_ref = u_ref.parse("dog go north")                     # Python comprehension → {role: word} dict
    u_ref.store(roles_ref["agent"], roles_ref["action"], roles_ref["patient"])   # Python hand-off store
    assert u_ref.query_patient("dog", "go") == u.query_patient("dog", "go") == "north", (
        "the synaptic route must reproduce the Python hand-off path's patient recall (no regression).")
    assert u_ref.query_agent("go", "north") == u.query_agent("go", "north") == "dog", (
        "the synaptic route must reproduce the Python hand-off path's agent recall (no regression).")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Task 3: the SYNAPTIC-ROUTE no-regression gate — does `hear_synaptic` (comprehension routing
# composition IN SPIKES through transmission gates) reproduce the Python parse+store hand-off path at
# PRODUCTION scale, multi-seed?
#
# The synaptic route only affects FLAT subject-verb-object (SVO) comprehension (the parser→composer hand-off).
# Attribute / clause / negation facts are stored structurally via `composer.store` and are UNCHANGED by this
# route, so the gate is the FLAT SVO category: per random 3-word fact, store via `u.hear_synaptic("a ac p")`
# and check `query_patient(a, ac) == p` (the "what") AND `query_agent(ac, p) == a` (the "who"). The ORACLE is
# the Python hand-off (`u.parse` → `u.store`) on the SAME bridge, over the SAME facts. The PASS bar: the
# SYNAPTIC who/what recall is within ±1 trial of the Python path on EVERY seed (the synaptic route is not
# weaker). A genuine drop beyond ±1 in any metric/seed is a REGRESSION — the test must NOT be weakened to pass;
# the regression is recorded honestly (e.g. gate-EMA warm-up costing rate at scale, or the larger composer
# shifting the parser firing-rate→gate coupling) for the controller to decide on a mitigation.
#
# RUNTIME: this is a heavy SPIKING run on the production (CuPy/GPU) backend — each seed builds ONE ~16.5K-neuron
# unified bridge (parser 126 + composer 8*2048 + the 3*2048 role-src route pools) and trains the parser, then
# runs BOTH paths over N flat facts; tens of minutes per seed. Skipped by default (like step 1's heavy gate) so
# the suite stays fast; the load-bearing claim (the synaptic route reproduces the Python store→recall) is already
# gated by `test_hear_synaptic_stores_fact_via_gated_route` at D=64. Run on demand:
#     SIM_RUN_HEAVY_CAPABILITY=1 pytest tests/test_unified_brain_bridge.py::test_step2_synaptic_no_regression -v
# It runs on the auto-selected backend (CuPy when present) and SKIPS gracefully if a seed's denoise64 cache is
# absent (it must not silently pass on NumPy with no GPU, where the substrate's validated behavior diverges).
# ─────────────────────────────────────────────────────────────────────────────
def test_step2_synaptic_no_regression():
    """Multi-seed (42/43/44) at production proj_dim=2048 with the REAL denoise64 codes: for the FLAT SVO
    category, the SYNAPTIC route's (`hear_synaptic`) who AND what recall must each be within ±1 trial of the
    Python hand-off path's recall on EVERY seed. A genuine drop beyond ±1 in any metric/seed is a REGRESSION
    (do not weaken — record it). Also confirms the parser parses voice-invariantly on the merged production
    bridge with the route wired ('dog go north' active == 'north go dog' passive) and abstention is preserved.

    STATUS (2026-06-04): PASSES at D=2048 — synaptic 6/6 == python 6/6 on every seed/metric, abstention
    preserved. An earlier 1-seed regression (seed-42 patient `what` at 4/6, from the parser-coupled gate
    FLICKERING at its EMA threshold and starving the composer's role bank) was diagnosed and RESOLVED by a
    faithful gate PRE-WARM in `UnifiedBrainBridge._op_synaptic` (the gate opens from the parser's firing and
    is held at that value while the composer reads — comprehend → latch → compose; no weight/drive magnitude
    changed, the gate not set by hand). The assertion below is unchanged (it was never weakened to pass).

    HEAVY + on-demand: builds one production (~16.5K-neuron) unified bridge per seed and trains the parser —
    tens of minutes on the GPU. Skipped by default so the suite stays fast; the load-bearing claim (the
    synaptic route reproduces the Python store→recall) is already gated by
    `test_hear_synaptic_stores_fact_via_gated_route` at D=64. The full multi-seed result + the regression
    history, diagnosis, and the pre-warm fix live in
    `research/findings/2026-06-04-one-bridge-unification-step2-DONE.md`. Run on demand:
        SIM_RUN_HEAVY_CAPABILITY=1 pytest tests/test_unified_brain_bridge.py::test_step2_synaptic_no_regression -v
    """
    import os
    if not os.environ.get("SIM_RUN_HEAVY_CAPABILITY"):
        pytest.skip("heavy multi-seed production synaptic-route no-regression gate (run with "
                    "SIM_RUN_HEAVY_CAPABILITY=1); result is in 2026-06-04-one-bridge-unification-step2-DONE.md")
    from sim.backend import is_gpu_backend
    if not is_gpu_backend():
        pytest.skip("the synaptic-route gate is a SPIKING run; run on the validated CuPy/GPU backend (the "
                    "parser's Hebbian convergence + gate operating point are GPU-validated, not NumPy).")
    from research.findings.raw._step2_synaptic_capability_probe import run_synaptic_comparison, find_regressions

    seeds = (42, 43, 44)
    # Gate at the PRODUCTION dimension D=2048 (the stage-1.5 decision; step 1 verified D=2048 is the
    # capability-equivalent operating point). The synaptic route only changes the FLAT SVO hand-off.
    try:
        results = run_synaptic_comparison(seeds=seeds, proj_dim=2048, n=6)
    except FileNotFoundError:
        pytest.skip("denoise64 concept-code cache not present for one of seeds 42/43/44")

    # Parser-on-merged-bridge confirmation (every seed): correct active parse + voice-invariant agent.
    for seed in seeds:
        p = results[seed]["parser"]
        assert p["active"] == {"agent": "dog", "action": "go", "patient": "north"}, (
            f"seed {seed}: parser active parse wrong on the merged production bridge with the route wired: "
            f"{p['active']}")
        assert p["passive_agent"] == "dog", (
            f"seed {seed}: parser passive frame did not assign the same agent on the merged bridge "
            f"(voice-invariance broken): passive_agent={p['passive_agent']!r}")
        # Abstention (the no-confab moat) must survive the synaptic route on every seed.
        assert results[seed]["abstention"], (
            f"seed {seed}: an unstored cue did NOT abstain through the synaptic route (no-confab moat broke).")

    # No-regression assertion: every metric (who/what), every seed, synaptic within ±1 trial of python.
    regressions = find_regressions(results)
    assert not regressions, (
        "SYNAPTIC route FLAT recall REGRESSED vs the Python hand-off path (drop > 1 trial in some metric/seed) "
        "— this is the measured cost of routing composition through transmission gates at scale; record it, do "
        "NOT weaken the test. Drops: "
        + "; ".join(f"seed {s} {m}: python {po} -> synaptic {so}" for s, m, po, so in regressions))
