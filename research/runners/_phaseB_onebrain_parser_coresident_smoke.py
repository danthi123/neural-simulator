"""ROADMAP PHASE 2 (the real "one brain"), GAP B construction SMOKE -- the load-bearing risk before STEP B1: can the
Izhikevich Hebbian PARSER and the resonate-and-fire (RF) composer registers co-exist on ONE bridge, each un-regressed?

GAP B (the parser front-end) drives the composer's operand from the PARSER's neural role decision instead of a host
`{role: word}` dict. The genuinely-new link is narrow (read which role the parser fires -> select the bind), but it
requires the Izhikevich parser (state in v/u as VOLTAGE, stepped by `_run_one_simulation_step`) to co-reside with RF
registers (state in v/u as a COMPLEX phasor, stepped by the masked `rf_resonate_steps`). The merged nav+conv bridge
already proved this regime (step 2b); this smoke confirms it for a MINIMAL parser+RF bridge before the full de-risk.

Two checks, single seed (a construction smoke, not the multi-seed gate):
  (1) PARSER un-regressed co-resident: a `BridgeParser` on slice [0:P] of a bridge that ALSO has an RF slice trains +
      comprehends "dog go north" / passive correctly (== the standalone parser's ground-truth roles).
  (2) RF un-regressed co-resident: a masked `rf_kick` + bind on the RF slice [P:P+7D] recovers a filler == a standalone
      RF bind (the parser slice's incidental Izhikevich firing does not corrupt the RF op; the masked RF op does not
      corrupt the parser slice).
PASS => the co-residence construction is sound; build the full STEP B1 (comprehend -> store -> query on one bridge).
Reuse-by-import (BridgeParser + _build_rf_bridge + RFPhasorComposer + masked rf_kick); NO sim/ edit. GPU.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_parser_coresident_smoke --seed 42 --D 64
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "cupy")

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.config import CoreSimConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402
from research.runners.brain_conversational_agent import BridgeParser  # noqa: E402
from research.runners.rf_phasor_composer import RFPhasorComposer, _build_rf_bridge  # noqa: E402

VOCAB = ["dog", "cat", "bird", "river", "apple", "go", "come", "look", "stop", "swim", "north", "east", "south", "west"]


def build_coresident_bridge(seed, P, n_rf):
    """ONE Izhikevich bridge sized P (parser) + n_rf (RF registers). Hebbian ON (the parser needs it); the RF slice
    has no cp_connections wiring (its memory is in cp_rf_w_re/im), so global Hebbian has nothing to touch there."""
    cfg = CoreSimConfig()
    cfg.num_neurons = P + n_rf
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed); cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0; cfg.num_traits = 1
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_max_weight = 400.0
    cfg.hebbian_learning_rate = 0.005
    for f in ("enable_short_term_plasticity", "enable_structural_plasticity", "enable_homeostasis",
              "enable_reward_modulation", "enable_watts_strogatz"):
        setattr(cfg, f, False)
    cfg.ou_std_current_pA = 20.0
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def check_parser_coresident(seed, n_rf):
    """(1) Build a co-resident bridge, put the parser on slice [0:P], train it, and comprehend active + passive."""
    R = 40
    P = 6 + 3 * R
    b = build_coresident_bridge(seed, P, n_rf)
    parser = BridgeParser(seed=seed, R=R, shared_bridge=b, index_offset=0)   # wires + trains on [0:P]
    active = parser.parse(["dog", "go", "north"], voice="active")
    passive = parser.parse(["north", "go", "dog"], voice="passive")
    # active: pos0->agent, pos1->action, pos2->patient ; passive flips 1st<->3rd
    ok_active = (active.get("agent") == "dog" and active.get("action") == "go" and active.get("patient") == "north")
    ok_passive = (passive.get("agent") == "dog" and passive.get("action") == "go" and passive.get("patient") == "north")
    print(f"  [parser co-resident] active={active} ok={ok_active} | passive={passive} ok={ok_passive}", flush=True)
    return ok_active and ok_passive, P, b


def check_rf_coresident(seed, D, b, rf_base):
    """(2) A masked RF bind on slice [rf_base:rf_base+7D] of the SAME bridge == a standalone RF bind."""
    comp = RFPhasorComposer(seed=seed, D=D, vocab=VOCAB, period=200)
    n = b.core_config.num_neurons
    rf_mask = np.zeros(n, dtype=bool); rf_mask[rf_base:rf_base + 7 * D] = True
    # bind agent_role x "dog" into bound register, then unbind agent_role -> recover "dog", on the co-resident slice.
    za = comp._to_phasor(comp.roles["agent"]); fa = comp._to_phasor(comp.concepts["dog"])
    # registers (local within the rf slice): fill@0, bound@1, recovered@2  (offset by rf_base)
    o = rf_base
    bind = [(o + 1 * D + k, o + 0 * D + k, complex(za[k])) for k in range(D)]
    unbind = [(o + 2 * D + k, o + 1 * D + k, complex(np.conj(za[k]))) for k in range(D)]
    kick = np.zeros(n, dtype=np.complex128); kick[o:o + D] = fa
    b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
    b.rf_set_complex_weights(bind); b.rf_kick(kick, period=comp.period, lam=0.0, neuron_mask=rf_mask)
    b.rf_resonate_steps(comp.period + 8)
    b.rf_set_complex_weights(unbind); b.rf_resonate_steps(comp.period + 8)
    rec = np.asarray(b.rf_read_phases())[o + 2 * D:o + 3 * D]
    word = comp._cleanup(rec, VOCAB)
    # standalone reference on a pure-RF bridge
    bs = _build_rf_bridge(3 * D, seed)
    bind_s = [(1 * D + k, 0 * D + k, complex(za[k])) for k in range(D)]
    unbind_s = [(2 * D + k, 1 * D + k, complex(np.conj(za[k]))) for k in range(D)]
    kick_s = np.zeros(3 * D, dtype=np.complex128); kick_s[0:D] = fa
    bs.cp_membrane_potential_v[:] = 0.0; bs.cp_recovery_variable_u[:] = 0.0
    bs.rf_set_complex_weights(bind_s); bs.rf_kick(kick_s, period=comp.period, lam=0.0)
    bs.rf_resonate_steps(comp.period + 8)
    bs.rf_set_complex_weights(unbind_s); bs.rf_resonate_steps(comp.period + 8)
    word_s = comp._cleanup(np.asarray(bs.rf_read_phases())[2 * D:3 * D], VOCAB)
    ok = (word == "dog") and (word == word_s)
    print(f"  [rf co-resident] recovered='{word}' standalone='{word_s}' truth='dog' ok={ok}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--D", type=int, default=64)
    args = ap.parse_args()
    D = args.D
    n_rf = 7 * D
    print(f"[GAP-B construction smoke] parser + RF co-resident on ONE bridge, each un-regressed? (seed {args.seed}, D={D})\n",
          flush=True)
    ok_parser, P, b = check_parser_coresident(args.seed, n_rf)
    ok_rf = check_rf_coresident(args.seed, D, b, rf_base=P)
    print(f"\n{'='*92}", flush=True)
    if ok_parser and ok_rf:
        print(f"  PASS: the Izhikevich Hebbian parser + the RF composer registers co-exist on ONE bridge, each "
              f"un-regressed (parser comprehends active+passive correctly; the masked RF bind == standalone). ==> the "
              f"co-residence construction is sound; build STEP B1 (comprehend -> store -> query on one persistent "
              f"bridge, the parser's neural role decision selecting the bind).", flush=True)
    else:
        print(f"  FAIL: parser_ok={ok_parser} rf_ok={ok_rf} -- the co-residence construction needs a fix before B1 "
              f"(if parser fails: the RF slice perturbs the parser's training/readout; if rf fails: the parser's "
              f"Izhikevich step corrupts the RF op despite the mask, or the standalone differs).", flush=True)
    print(f"{'='*92}", flush=True)


if __name__ == "__main__":
    main()
