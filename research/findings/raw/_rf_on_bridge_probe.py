"""RF-on-bridge de-risk GATE (Task 3): the project's compositional task (vocab N_CUES x N_FILLERS, loads 2/3/5,
frozen bar 0.80) with EVERY resonate-and-fire operation routed through the bridge's RESONATE_AND_FIRE step --
proving the FHRR composition runs on the SimulationBridge's own neurons, not the numpy reference.

Mirrors research/runners/resonate_fire_fhrr.py's ResonateFireFHRR self-test, but `rf_resonate(kick)` is replaced by
`bridge.rf_kick(kick); step period+8; bridge.rf_read_phases()`. Symbols are phase vectors in [0,1); bind/unbind/
bundle are the right complex kicks; cleanup is phase-cosine similarity. GATE: accuracy >= 0.80 at all loads AND
abstention separates (groundable sim min > ungroundable sim max) -> GO (parity with the numpy reference).

  python -u -m research.findings.raw._rf_on_bridge_probe --d 256 --trials 30 --period 1000 \
      --out research/findings/raw/_rf_on_bridge.json
"""
import argparse
import json
import numpy as np

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.enums import NeuronModel
from sim.bridge import SimulationBridge


def build_rf_bridge(D, seed=42):
    cfg = CoreSimConfig()
    cfg.num_neurons = int(D)
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
              "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
              "enable_watts_strogatz", "enable_neuromodulator_subsystem", "enable_brain_region_framework"):
        if hasattr(cfg, f):
            setattr(cfg, f, False)
    cfg.ou_std_current_pA = 0.0
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    bridge.core_config.neuron_model_type = NeuronModel.RESONATE_AND_FIRE.name
    return bridge


def _to_phasor(phase):
    return np.exp(2j * np.pi * np.asarray(phase))


def _similarity(u, v):
    # FHRR phase similarity: mean cosine of phase differences.
    return float(np.mean(np.cos(2.0 * np.pi * (np.asarray(u) - np.asarray(v)))))


def run(D, n_cues, n_fillers, loads, trials, period, seed):
    bridge = build_rf_bridge(D, seed)

    def resonate(kick):
        bridge.rf_kick(np.asarray(kick), period=period)
        for _ in range(period + 8):
            bridge._run_one_simulation_step()
        return np.asarray(bridge.rf_read_phases())

    rng = np.random.default_rng(seed)
    results = {}
    for load in loads:
        n_correct = 0
        n_total = 0
        g_sims = []
        u_sims = []
        for _ in range(trials):
            cues = [rng.uniform(0.0, 1.0, D) for _ in range(n_cues)]
            fillers = [rng.uniform(0.0, 1.0, D) for _ in range(n_fillers)]
            ci = list(rng.choice(n_cues, size=load, replace=False))
            fi = list(rng.choice(n_fillers, size=load, replace=True))
            facts = list(zip(ci, fi))
            bound = [resonate(_to_phasor(cues[c]) * _to_phasor(fillers[f])) for (c, f) in facts]
            composite = resonate(np.sum([_to_phasor(b) for b in bound], axis=0))
            for (c, f) in facts:
                rec = resonate(_to_phasor(composite) * np.conj(_to_phasor(cues[c])))
                sims = [_similarity(rec, fillers[k]) for k in range(n_fillers)]
                if int(np.argmax(sims)) == f:
                    n_correct += 1
                n_total += 1
                g_sims.append(max(sims))
            for c in range(n_cues):
                if c in ci:
                    continue
                rec = resonate(_to_phasor(composite) * np.conj(_to_phasor(cues[c])))
                sims = [_similarity(rec, fillers[k]) for k in range(n_fillers)]
                u_sims.append(max(sims))
        acc = n_correct / max(n_total, 1)
        g = np.array(g_sims)
        u = np.array(u_sims)
        abst = float(np.min(g)) > float(np.max(u))
        results[load] = {"accuracy": acc, "groundable_sim_min": float(np.min(g)),
                         "ungroundable_sim_max": float(np.max(u)), "abstention_separates": abst}
        print(f"[RF-BRIDGE] L={load}: acc={acc:.4f} g_min={np.min(g):.3f} u_max={np.max(u):.3f} "
              f"abstain={abst}", flush=True)

    bar = 0.80
    all_pass = all(results[l]["accuracy"] >= bar and results[l]["abstention_separates"] for l in loads)
    verdict = "GO" if all_pass else "NEGATIVE"
    print(f"\n[VERDICT] RF-on-bridge composer task (bar {bar}, all loads) -> {verdict}")
    return results, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--n-cues", type=int, default=8)
    ap.add_argument("--n-fillers", type=int, default=8)
    ap.add_argument("--loads", type=int, nargs="+", default=[2, 3, 5])
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--period", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results, verdict = run(args.d, args.n_cues, args.n_fillers, args.loads, args.trials, args.period, args.seed)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"params": vars(args), "results": results, "verdict": verdict}, fh, indent=2, default=float)
        print(f"[RF-BRIDGE] wrote {args.out}")


if __name__ == "__main__":
    main()
