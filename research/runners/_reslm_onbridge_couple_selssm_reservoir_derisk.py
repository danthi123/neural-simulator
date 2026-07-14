"""PAST-RESERVOIR — the ON-BRIDGE (fully-spiking) COUPLING: a recurrent SPIKING reservoir region + the on-bridge SELECTIVE
channel (`cp_ssm_state`, `enable_selective_ssm_state`) CO-RESIDENT on ONE real SimulationBridge, read-out over BOTH — does
adding the on-bridge selective channel to a spiking reservoir's read-out lift a long-range CONJUNCTION the spiking reservoir
alone cannot? This is the emergence-bar realization (fully spiking, one brain) of the numpy coupling GO (frozen + joint,
transport-free, adversarially verified). All pieces are already on-bridge-validated: the spiking reservoir (EMERGE-82,
internal_density recurrence + Izhikevich + conductance synapses) and the selective channel (Rung 4b-iii-a byte-equivalent to
numpy, 4b-iii-b learns end-to-end on-bridge). Here they are combined on ONE bridge.

TASK (Rung-2/4b gated-conjunction, tractable on-bridge): [KEY, filler x DEPTH, QUERY] -> rule[KEY, QUERY]. A fixed window /
memoryless read cannot do it; the reservoir's fading memory struggles at depth; the selective channel HOLDS the distal KEY.

MECHANISM (per token, transport-free): drive the reservoir region via cp_external_input_current (W_in @ onehot) + set the
selective channel's cp_ssm_inject/cp_ssm_shunt on a DISJOINT neuron slice; step the bridge T_STEP times; read h_t = the
reservoir slice's spike-rate + c_t = cp_ssm_state[ssm_slice]. At the QUERY step, read-out over [h_t, c_t]; train the read-out
(delta) + the gate (forward-mode eligibility x FIXED RANDOM FEEDBACK -- no BPTT, no transport). Wash the bridge to post-init
between sequences.

ARMS (single variable = the on-bridge selective channel): res_only (read-out over the spiking reservoir h_t ONLY) /
res_plus_sel (read-out over [h_t, c_t], selective gate co-trained on-bridge). GO (>=2/3 first, then 6): res_plus_sel beats
res_only + chance -> the on-bridge selective channel lifts the spiking reservoir past its long-range bound, on real spikes.

Run: SIM_BACKEND=numpy python -m research.runners._reslm_onbridge_couple_selssm_reservoir_derisk --seeds 42
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
from sim.backend import to_host as _host, get_backend

K = 6                    # vocab of KEY/QUERY symbols (+1 filler)
D_IN = K + 1             # one-hot input dim
N_RES = 80               # spiking reservoir neurons
N_SSM = 64               # selective-channel neurons (disjoint slice)
DEPTH = 10               # filler length (the long-range gap)
N_SEQ = 500
EPOCHS = 6
T_STEP = 6               # bridge steps per token
LR_RO = 0.05
LR_GATE = 0.4
K_LEAK = 0.06
C_INIT = -1.2            # softplus(-1.2)~0.26 shunt -> lam_eff ~0.92 (hold)
IN_SCALE = 24.0          # reservoir input current scale (pA)
BIAS = 6.0


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
def _softplus(z): return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def _build_bridge(seed):
    """ONE bridge: a recurrent spiking reservoir region (internal_density recurrence) + a disjoint ssm region carrying the
    selective channel (enable_selective_ssm_state). Learning + OU off; the reservoir recurrence is the fixed-random liquid."""
    res = BrainRegion(name="reservoir", n_neurons=N_RES, exc_fraction=0.8, internal_density=0.2,
                      exc_weight_mean=2.5, inh_weight_mean=6.0, weight_jitter=0.3, plastic_internal=False)
    ssm = BrainRegion(name="ssm", n_neurons=N_SSM, exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0,
                      inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)
    cfg = CoreSimConfig()
    cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed); cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True; cfg.brain_regions = [res, ssm]; cfg.region_pathways = []
    cfg.enable_stdp = cfg.enable_hebbian_learning = cfg.enable_nmda = False; cfg.fast_spike_reset = True
    for f in ("enable_homeostasis", "enable_short_term_plasticity", "enable_ou_process", "enable_conductance_noise",
              "enable_parameter_heterogeneity", "enable_structural_plasticity", "enable_coincidence_detection",
              "enable_two_compartment_dap"):
        setattr(cfg, f, False)
    cfg.enable_selective_ssm_state = True; cfg.ssm_k_leak = K_LEAK
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms); b.runtime_state.actual_seed_used = seed
    b._initialize_simulation_data(called_from_playback_init=False)
    res_idx = np.asarray(b.region_manager.indices("reservoir"))
    ssm_idx = np.asarray(b.region_manager.indices("ssm"))
    snap = {k: np.asarray(_host(getattr(b, k))).copy() for k in
            ("cp_membrane_potential_v", "cp_recovery_variable_u", "cp_firing_states")}
    return b, res_idx, ssm_idx, snap


def _wash(b, snap):
    xp, _ = get_backend()
    for k, v in snap.items():
        getattr(b, k)[:] = xp.asarray(v) if xp is not None else v
    b.cp_ssm_state[:] = 0.0; b.cp_ssm_inject[:] = 0.0; b.cp_ssm_shunt[:] = 0.0
    b.cp_external_input_current[:] = 0.0


def _seqs(seed):
    rng = np.random.default_rng(seed * 11 + 5); rule = rng.integers(0, K, (K, K)); out = []
    for _ in range(N_SEQ):
        k = int(rng.integers(0, K)); q = int(rng.integers(0, K))
        out.append(([k] + [K] * DEPTH + [q], int(rule[k, q])))
    return out


def _run_arm(seed, arm):
    b, res_idx, ssm_idx, snap = _build_bridge(seed)
    xp, _ = get_backend()
    use_sel = arm == "res_plus_sel"
    rng = np.random.default_rng(seed * 7 + 2)
    E = np.eye(D_IN)                                              # one-hot token embedding
    W_in = (np.random.default_rng(seed * 7919 + 3).random((N_RES, D_IN)) * 2 - 1) * IN_SCALE
    Win_s = np.random.default_rng(seed * 31 + 9).standard_normal((N_SSM, D_IN)) / np.sqrt(D_IN)
    w = rng.standard_normal((N_SSM, D_IN)) / np.sqrt(D_IN); c = np.full(N_SSM, C_INIT)
    feat_dim = N_RES + (N_SSM if use_sel else 0)
    Wro = np.zeros((K, feat_dim))
    Bc = np.random.default_rng(seed * 191 + 11).standard_normal((N_SSM, K)) / np.sqrt(K)  # fixed random feedback (gate)
    seqs = _seqs(seed); ntr = int(0.7 * len(seqs))

    def _step_token(tok):
        """Drive reservoir + selective channel for one token; return (h_rate, c) reading the bridge's real spikes/state."""
        u = E[tok]
        drive = np.zeros(int(b.core_config.num_neurons), np.float32)
        drive[res_idx] = (W_in @ u + BIAS).astype(np.float32)
        b.cp_external_input_current[:] = 0.0
        b.cp_external_input_current[res_idx] = xp.asarray(drive[res_idx]) if xp is not None else drive[res_idx]
        inj = Win_s @ u
        gsh = _softplus(w @ u + c)
        full_inj = np.zeros(int(b.core_config.num_neurons), np.float32); full_sh = np.zeros_like(full_inj)
        full_inj[ssm_idx] = inj.astype(np.float32); full_sh[ssm_idx] = gsh.astype(np.float32)
        b.cp_ssm_inject[:] = xp.asarray(full_inj) if xp is not None else full_inj
        b.cp_ssm_shunt[:] = xp.asarray(full_sh) if xp is not None else full_sh
        counts = np.zeros(N_RES, np.float64)
        for _ in range(T_STEP):
            b._run_one_simulation_step()
            counts += np.asarray(_host(b.cp_firing_states)).astype(np.float64)[res_idx]
        h = counts / T_STEP
        c_state = np.asarray(_host(b.cp_ssm_state)).astype(np.float64)[ssm_idx]
        return h, c_state, u, inj, gsh

    for _ep in range(EPOCHS):
        for (toks, y) in seqs[:ntr]:
            _wash(b, snap); ew = np.zeros((N_SSM, D_IN)); ec = np.zeros(N_SSM); s_prev = np.zeros(N_SSM)
            for t, tok in enumerate(toks):
                h, c_state, u, inj, gsh = _step_token(tok)
                if use_sel:
                    lam = np.clip(1.0 - K_LEAK * (1.0 + gsh), 0.0, 1.0)
                    dl = -K_LEAK * _sig(w @ u + c); base = (s_prev - inj) * dl
                    ew = lam[:, None] * ew + base[:, None] * u[None, :]; ec = lam * ec + base
                    s_prev = c_state
                if t == len(toks) - 1:
                    feat = np.concatenate([h, c_state]) if use_sel else h
                    z = Wro @ feat; z -= z.max(); p = np.exp(z); p /= p.sum(); err = p.copy(); err[y] -= 1.0
                    Wro -= LR_RO * np.outer(err, feat)
                    if use_sel:
                        delta_c = Bc @ err                       # transport-free random feedback to the gate
                        w -= LR_GATE * (delta_c[:, None] * ew); c -= LR_GATE * (delta_c * ec)
    cor = tot = 0
    for (toks, y) in seqs[ntr:]:
        _wash(b, snap)
        for t, tok in enumerate(toks):
            h, c_state, u, inj, gsh = _step_token(tok)
        feat = np.concatenate([h, c_state]) if use_sel else h
        cor += int(np.argmax(Wro @ feat) == y); tot += 1
    return cor / tot


def run(seed):
    acc = {a: _run_arm(seed, a) for a in ("res_plus_sel", "res_only")}
    chance = 1.0 / K
    go = bool(acc["res_plus_sel"] > acc["res_only"] + 0.06 and acc["res_plus_sel"] > chance + 0.10)
    print(f"[onbridge-couple seed={seed}] res_plus_sel={acc['res_plus_sel']:.3f} res_only={acc['res_only']:.3f} "
          f"(chance={chance:.3f}) -> {'GO' if go else 'no'}", flush=True)
    return {"seed": seed, **acc, "GO": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    res = [run(s) for s in a.seeds]
    print(f"[onbridge-couple] {sum(1 for r in res if r['GO'])}/{len(res)} GO", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
