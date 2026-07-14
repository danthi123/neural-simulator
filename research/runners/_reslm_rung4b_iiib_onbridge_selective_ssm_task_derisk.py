"""PAST-RESERVOIR Rung 4b-iii-b (end-to-end ON-BRIDGE learning): the full selective-SSM gated-conjunction task with the
SSM forward state driven through the on-bridge `cp_ssm_state` (`enable_selective_ssm_state`) — the gate (shunt) + the
eligibility trace + the read-out are trained reading the ON-BRIDGE state each step, and the learned selective gate beats a
fixed-shunt reservoir. Closes the loop: the transport-free selective SSM LEARNS while its state lives on the spiking
bridge. (Rung 4b-iii-a proved the on-bridge state is byte-equivalent to numpy; this runs the actual LEARNING loop through
it.) NO further `sim/` edit (uses the Rung-4b-ii mechanism).

Per-token, per sequence: set cp_ssm_inject = Win·E[tok], cp_ssm_shunt = softplus(w·u) (selective) / fixed (fixed_res),
step the bridge, read s = cp_ssm_state; at the read (query) step compute the read-out logits + error, update the read-out
(delta) + the gate (eligibility trace × the local read-out error). Forget-bias so lam_eff starts high (hold).

ARMS: selective (gate trained via the on-bridge state's eligibility) vs fixed_res (fixed shunt) vs chance. At Rung-4a's
full task size the on-bridge result MATCHES the numpy Rung-4a result EXACTLY (seed 42: selective 0.541, fixed_res 0.270 --
byte-identical to numpy, confirming the Rung-4b-iii-a equivalence in the LEARNING loop). GO: selective beats fixed_res
+ chance -> the on-bridge selective SSM LEARNS while its state lives on the spiking bridge.

Run: SIM_BACKEND=numpy python -m research.runners._reslm_rung4b_iiib_onbridge_selective_ssm_task_derisk --seeds 42
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import numpy as np

from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
from sim.bridge import SimulationBridge
from sim.regions import BrainRegion
from sim.enums import NeuronModel, NeuronType
from sim.backend import to_host as _host

K = 6
D_IN = 10
N_HID = 64
DEPTH = 12
N_SEQ = 900
EPOCHS = 10
LR_RO = 0.05
LR_GATE = 0.4
K_LEAK = 0.06
C_INIT = -1.2               # softplus(-1.2)~0.26 shunt -> lam_eff ~0.92 at init (hold)


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
def _softplus(z): return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def _build_bridge(seed):
    region = BrainRegion(name="ssm", n_neurons=N_HID, exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0,
                         inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                         izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)
    cfg = CoreSimConfig()
    cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed); cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True; cfg.brain_regions = [region]; cfg.region_pathways = []
    cfg.enable_stdp = cfg.enable_hebbian_learning = cfg.enable_nmda = False; cfg.fast_spike_reset = True
    for f in ("enable_homeostasis", "enable_short_term_plasticity", "enable_ou_process", "enable_conductance_noise",
              "enable_parameter_heterogeneity", "enable_structural_plasticity", "enable_coincidence_detection",
              "enable_two_compartment_dap"):
        setattr(cfg, f, False)
    cfg.enable_selective_ssm_state = True; cfg.ssm_k_leak = K_LEAK
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms); b.runtime_state.actual_seed_used = seed
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def _seqs(seed):
    rng = np.random.default_rng(seed * 11 + 5); rule = rng.integers(0, K, (K, K)); out = []
    for _ in range(N_SEQ):
        k = int(rng.integers(0, K)); q = int(rng.integers(0, K)); out.append(([k] + [K] * DEPTH + [q], int(rule[k, q])))
    return out


def _run_arm(seed, arm):
    b = _build_bridge(seed); xp = b.xp; n = N_HID
    rng = np.random.default_rng(seed * 7 + 2)
    E = np.random.default_rng(seed * 3 + 1).standard_normal((K + 1, D_IN)) * 0.8
    Win = rng.standard_normal((N_HID, D_IN)) / np.sqrt(D_IN)
    w = rng.standard_normal((N_HID, D_IN)) / np.sqrt(D_IN); c = np.full(N_HID, C_INIT)
    fixed_gsh = _softplus(np.random.default_rng(seed * 9).standard_normal(N_HID) * 0.3 + C_INIT)
    Wro = np.zeros((K, n)); seqs = _seqs(seed); ntr = int(0.7 * len(seqs))
    for _ep in range(EPOCHS):
        for (toks, y) in seqs[:ntr]:
            b.cp_ssm_state[:] = xp.asarray(np.zeros(n, np.float32))         # reset the slow state per sequence
            ew = np.zeros((N_HID, D_IN)); ec = np.zeros(N_HID); s_prev = np.zeros(n)
            for t, tok in enumerate(toks):
                u = E[tok]; inj = Win @ u
                if arm == "fixed_res":
                    gsh = fixed_gsh; a = None
                else:
                    a = w @ u + c; gsh = _softplus(a)
                b.cp_ssm_inject[:] = xp.asarray(inj.astype(np.float32))
                b.cp_ssm_shunt[:] = xp.asarray(gsh.astype(np.float32))
                b._run_one_simulation_step()
                s = _host(b.cp_ssm_state).astype(np.float64)                # READ the on-bridge state
                if arm == "selective":
                    lam = np.clip(1.0 - K_LEAK * (1.0 + gsh), 0.0, 1.0)
                    dl = -K_LEAK * _sig(a); base = (s_prev - inj) * dl
                    ew = lam[:, None] * ew + base[:, None] * u[None, :]; ec = lam * ec + base
                if t == len(toks) - 1:
                    z = Wro @ s; z -= z.max(); p = np.exp(z); p /= p.sum(); err = p.copy(); err[y] -= 1.0
                    delta = Wro.T @ err; Wro -= LR_RO * np.outer(err, s)
                    if arm == "selective":
                        w -= LR_GATE * (delta[:, None] * ew); c -= LR_GATE * (delta * ec)
                s_prev = s
    cor = tot = 0
    for (toks, y) in seqs[ntr:]:
        b.cp_ssm_state[:] = xp.asarray(np.zeros(n, np.float32))
        for t, tok in enumerate(toks):
            u = E[tok]; inj = Win @ u
            gsh = fixed_gsh if arm == "fixed_res" else _softplus(w @ u + c)
            b.cp_ssm_inject[:] = xp.asarray(inj.astype(np.float32)); b.cp_ssm_shunt[:] = xp.asarray(gsh.astype(np.float32))
            b._run_one_simulation_step()
        cor += int(np.argmax(Wro @ _host(b.cp_ssm_state).astype(np.float64)) == y); tot += 1
    return cor / tot


def run(seed):
    acc = {a: _run_arm(seed, a) for a in ("selective", "fixed_res")}
    chance = 1.0 / K
    go = bool(acc["selective"] > acc["fixed_res"] + 0.08 and acc["selective"] > chance + 0.12)
    print(f"[rung4b-iiib seed={seed}] ON-BRIDGE selective={acc['selective']:.3f} fixed_res={acc['fixed_res']:.3f} "
          f"(chance={chance:.3f}) -> {'GO' if go else 'no'}", flush=True)
    return {"seed": seed, **acc, "GO": go}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", type=str, default=None); a = ap.parse_args()
    res = [run(s) for s in a.seeds]
    print(f"[rung4b-iiib] {sum(1 for r in res if r['GO'])}/{len(res)} GO", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
