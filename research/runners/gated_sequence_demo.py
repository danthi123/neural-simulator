"""Sequencing bound primitives by gating: temporal variable binding on the thalamocortical substrate.

The genuinely-informative step beyond single-binding routing: a PLAN is an ordered list of (verb, motor)
bindings, and the basal ganglia step through it, disinhibiting one thalamic gate pool at a time -> the
bridge-internal coupling opens that cortical route gate -> the verb routes to its motor -> the next binding.
The output is the ordered motor sequence.

Critically this includes TEMPORAL VARIABLE BINDING: the SAME verb can be bound to DIFFERENT motors at
different sequence positions ("go north ... go south"). Gated re-binding handles it (open g_GO_N then later
g_GO_S); STDP-grown weights fundamentally cannot (a verb's grown weight is a constant -- it cannot be both
N and S). This is the foundation for multi-element structures (utterances are ordered sequences of bindings).

Honest scope: the SEQUENCER here is an external plan-loop (the BG selection order is given); autonomous
cortical sequence generation with preparatory transitions (Logiaco-Abbott-Escola Option C, the low-rank
effective-connectivity gate over cortical trajectories) is the further build. This demonstrates that the
gating substrate produces ordered, temporally-rebindable composition.

  SIM_BACKEND=numpy python -m research.runners.gated_sequence_demo
"""
import numpy as np

from research.runners.gated_compose_bg_demo import build_bg_gated_bridge, couple_all_route_gates
from research.runners.gated_compose_demo import MOTORS


def produce_sequence(sb, plan, settle=20, readout=40, drive_pA=1500.0):
    """Step the BG through `plan` (ordered (verb,motor) bindings); return the produced motor sequence.

    For each binding the BG disinhibits ONLY that binding's thalamic pool (others go silent -> their gates
    close via the coupling), then the verb is driven and the winning motor recorded."""
    from sim.backend import to_host
    out = []
    for verb, motor in plan:
        sb.cp_external_input_current[:] = 0.0
        sb.cp_external_input_current[np.asarray(sb.region_manager.indices(f"thal_{verb}_{motor}"))] = drive_pA
        for _ in range(settle):                      # let the gate open (and the previous one close) in-substrate
            sb._run_one_simulation_step()
        sb.cp_external_input_current[np.asarray(sb.region_manager.indices(f"verb_{verb}"))] = drive_pA
        acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
        for _ in range(readout):
            sb._run_one_simulation_step()
            acc += to_host(sb.cp_firing_states).astype(np.float64)
        out.append(max(MOTORS, key=lambda m: acc[np.asarray(sb.region_manager.indices(f"motor_{m}"))].mean()))
    return out


def main():
    print("=== sequencing bound primitives by gating (temporal variable binding) ===\n", flush=True)
    # a plan where GO is bound to DIFFERENT motors at different positions -- impossible for grown weights
    plans = {
        "go north, stop west, come south": [("GO", "N"), ("STOP", "W"), ("COME", "S")],
        "go north, look east, GO SOUTH (re-bind GO)": [("GO", "N"), ("LOOK", "E"), ("GO", "S")],
    }
    for seed in (42, 43):
        sb = build_bg_gated_bridge(seed=seed)
        couple_all_route_gates(sb)
        print(f"  seed {seed}:", flush=True)
        for label, plan in plans.items():
            produced = produce_sequence(sb, plan)
            want = [m for _, m in plan]
            ok = produced == want
            print(f"    plan [{label}] -> produced {produced}  want {want}  {'(ok)' if ok else '(X)'}", flush=True)
    print("\n  -> the BG sequences gates; the same verb (GO) re-binds to N then S across the sequence with zero", flush=True)
    print("     weight change -- temporal variable binding, the foundation for multi-element structures.", flush=True)


if __name__ == "__main__":
    main()
