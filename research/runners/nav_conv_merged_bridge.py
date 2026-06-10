"""Navigation + Conversational single-instance merge (roadmap step 2) — builder + ports.

Per `docs/plans/2026-06-10-nav-conv-merge-implementation-design.md`. The brain-region framework IS a
wrapper around `inject_explicit_wiring` (`sim/bridge.py:1514-1526`), so the merge appends the parser +
dlPFC as framework `BrainRegion`/`RegionPathway` to the navigation lists; everything is wired by one
init-time injection of `region_manager.build_wiring_plan(...)`.

This file starts with the PARSER PORT (the highest implementation risk 4.1): `BridgeParser` cannot be a
drop-in `shared_bridge=` because its merge path re-injects and would clobber navigation, and it assumes a
contiguous parser block. So we re-express the parser as framework regions and PORT its drive/train/read to
raw framework slice indices. `--microcheck` builds a parser-only framework bridge (behind a navigation-stub
region that forces a non-zero offset, exercising the slice arithmetic) and validates that the parser learns
the voice-invariant role map on framework slices under the merged config — retiring risk 4.1 before the full
STEP 2a build.

Reuse-by-import; no `sim/` edit.
"""
from __future__ import annotations

import argparse

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion, RegionPathway
from sim.backend import get_backend, to_host

# parser ground truth (from brain_conversational_agent): conjunction index k = position*2 + voice
# (voice 0=active, 1=passive); each k teacher-binds to one role.
PARSER_R = 40
ROLES = ["agent", "action", "patient"]
_GT = {0: "agent", 1: "patient", 2: "action", 3: "action", 4: "patient", 5: "agent"}

# the parser's gate name on the merged bridge (frozen to 0.0 after the train pass)
PARSER_GATE = "parser_fixed"


# ── parser as framework regions/pathways ─────────────────────────────────────────────────────────────────
def parser_regions_pathways(R: int = PARSER_R):
    """The parser's two framework regions (separate slices) + the all-to-all plastic conj->role pathway.

    parse_conj : 6 conjunction units. parse_role : 3*R role-ensemble neurons (agent|action|patient blocks).
    The conj->role pathway is all-to-all (density 1.0), init weight 0.5 exactly (jitter 0), plastic, tagged
    `parser_fixed` so it can be frozen after the Hebbian train pass.
    """
    regions = [
        BrainRegion(name="parse_conj", n_neurons=6, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="parse_role", n_neurons=3 * R, exc_fraction=1.0, internal_density=0.0),
    ]
    pathways = [
        RegionPathway(from_region="parse_conj", to_region="parse_role",
                      density=1.0, weight_mean=0.5, weight_jitter=0.0,
                      plastic=True, plasticity_gate=PARSER_GATE),
    ]
    return regions, pathways


def _parser_index_arrays(bridge, R: int = PARSER_R):
    """Resolve the parser's conjunction + per-role index arrays from the FRAMEWORK slices (any offset)."""
    rm = bridge.region_manager
    conj = list(rm.indices("parse_conj"))            # 6 contiguous global indices
    role_base_list = list(rm.indices("parse_role"))  # 3*R contiguous global indices
    role_idx = {r: role_base_list[i * R:(i + 1) * R] for i, r in enumerate(ROLES)}
    xp, _ = get_backend()
    conj_arr = xp.asarray(conj, dtype=xp.int64)
    role_arr = {r: xp.asarray(v, dtype=xp.int64) for r, v in role_idx.items()}
    return conj_arr, role_arr


def _step_reset(bridge, reset: int = 20):
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset):
        bridge._run_one_simulation_step()


def train_parser_on_slices(bridge, conj_arr, role_arr, n_epochs: int = 30, train_steps: int = 120,
                           drive: float = 2500.0):
    """Port of BridgeParser._train onto framework slices: for each conjunction k, co-drive conj[k] + the
    teacher role ensemble _GT[k]; Hebbian co-firing strengthens conj[k]->correct-role. Caller must have
    Hebbian ON + STDP/reward OFF for this pass."""
    xp, _ = get_backend()
    n = bridge.core_config.num_neurons
    for _ in range(n_epochs):
        for k in range(6):
            _step_reset(bridge)
            cur = xp.zeros(n, dtype=xp.float32)
            cur[conj_arr[k]] = drive
            cur[role_arr[_GT[k]]] = drive
            bridge.cp_external_input_current[:] = cur
            for _ in range(train_steps):
                bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0


def role_of_on_slices(bridge, conj_arr, role_arr, position: int, voice="active",
                      test_steps: int = 80, drive: float = 2500.0):
    """Port of BridgeParser.role_of: drive the (position, voice) conjunction ALONE; the role ensemble that
    fires most is the learned role."""
    xp, _ = get_backend()
    n = bridge.core_config.num_neurons
    k = position * 2 + (0 if voice in (0, "active") else 1)
    _step_reset(bridge)
    cur = xp.zeros(n, dtype=xp.float32)
    cur[conj_arr[k]] = drive
    bridge.cp_external_input_current[:] = cur
    rates = {r: 0.0 for r in ROLES}
    for _ in range(test_steps):
        bridge._run_one_simulation_step()
        for r in ROLES:
            rates[r] += float(to_host(bridge.cp_firing_states[role_arr[r]].astype(xp.float64).mean()))
    bridge.cp_external_input_current[:] = 0.0
    return max(rates, key=rates.get)


def parse_on_slices(bridge, conj_arr, role_arr, words, voice="active", test_steps: int = 80, drive: float = 2500.0):
    assert len(words) == 3, "this minimal parser handles 3-word SVO sentences"
    return {role_of_on_slices(bridge, conj_arr, role_arr, pos, voice, test_steps, drive): words[pos]
            for pos in range(3)}


# ── the parser-on-framework-slices micro-check (risk 4.1) ────────────────────────────────────────────────
def _build_parser_microcheck_bridge(seed: int, R: int, nav_stub: int, ou: float):
    """A framework bridge with [nav_stub, parse_conj, parse_role] under the MERGED config. The nav_stub forces
    a non-zero offset so the parser's slice arithmetic is exercised (the real merged condition)."""
    regions, pathways = parser_regions_pathways(R)
    if nav_stub > 0:
        regions = [BrainRegion(name="nav_stub", n_neurons=nav_stub, exc_fraction=1.0,
                               internal_density=0.0)] + regions
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.dt_ms = 1.0
    cfg.seed = int(seed)
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    # MERGED config: Hebbian for the parser pass, STDP/reward OFF, the merged clip bounds (5a mitigation).
    cfg.enable_hebbian_learning = True
    cfg.hebbian_learning_rate = 0.005
    cfg.hebbian_max_weight = 400.0
    cfg.stdp_w_max = 400.0
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_parameter_heterogeneity = False
    if ou > 0:
        cfg.enable_ou_process = True
        cfg.ou_std_current_pA = float(ou)
    else:
        cfg.enable_ou_process = False
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def microcheck(seed: int = 42, R: int = PARSER_R, nav_stub: int = 50, ou: float = 0.0,
               n_epochs: int = 30, train_steps: int = 120):
    xp, backend = get_backend()
    print(f"[parser-microcheck] backend={backend} seed={seed} nav_stub={nav_stub} ou={ou}")
    bridge = _build_parser_microcheck_bridge(seed, R, nav_stub, ou)
    rm = bridge.region_manager
    conj_base = rm.indices("parse_conj")[0]
    role_base = rm.indices("parse_role")[0]
    print(f"[parser-microcheck] {len(bridge.core_config.brain_regions)} regions, "
          f"{bridge.core_config.num_neurons} neurons, {int(bridge.cp_connections.nnz)} synapses; "
          f"conj_base={conj_base} role_base={role_base}")

    conj_arr, role_arr = _parser_index_arrays(bridge, R)

    # train pass on the framework slices, then FREEZE the parser gate (5a isolation)
    train_parser_on_slices(bridge, conj_arr, role_arr, n_epochs=n_epochs, train_steps=train_steps)
    bridge.set_plasticity_gate(PARSER_GATE, 0.0)

    # comprehension: active "dog go north" and the passive frame must both call dog the agent
    active = parse_on_slices(bridge, conj_arr, role_arr, ["dog", "go", "north"], voice="active")
    passive = parse_on_slices(bridge, conj_arr, role_arr, ["north", "go", "dog"], voice="passive")
    print(f"[parser-microcheck] active  parse: {active}")
    print(f"[parser-microcheck] passive parse: {passive}")

    ok_active = active.get("agent") == "dog" and active.get("action") == "go" and active.get("patient") == "north"
    ok_voice_inv = passive.get("agent") == "dog"   # voice-invariant agent assignment
    passed = ok_active and ok_voice_inv

    print(f"\n[parser-microcheck] active SVO correct      : {ok_active}")
    print(f"[parser-microcheck] voice-invariant agent    : {ok_voice_inv}  (passive 'north go dog' -> agent=dog)")
    print(f"[parser-microcheck] {'PASS' if passed else 'FAIL'} — the parser "
          f"{'learns the voice-invariant role map on framework slices (risk 4.1 retired)' if passed else 'did NOT port cleanly; see parses above'}")
    return passed


def main():
    ap = argparse.ArgumentParser(description="Nav+Conv merge builder (microcheck mode for now)")
    ap.add_argument("--microcheck", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nav-stub", type=int, default=50)
    ap.add_argument("--ou", type=float, default=20.0,
                    help="OU noise pA for the parser train pass (validated: 20 PASSES, 0=off FAILS — degenerate "
                         "WTA readout; the merge enables OU only for the pass, then restores OU-off for nav)")
    ap.add_argument("--n-epochs", type=int, default=30)
    ap.add_argument("--train-steps", type=int, default=120)
    args = ap.parse_args()
    if args.microcheck:
        ok = microcheck(seed=args.seed, nav_stub=args.nav_stub, ou=args.ou,
                        n_epochs=args.n_epochs, train_steps=args.train_steps)
        raise SystemExit(0 if ok else 1)
    ap.error("only --microcheck is implemented so far (STEP 2a build is next)")


if __name__ == "__main__":
    main()
