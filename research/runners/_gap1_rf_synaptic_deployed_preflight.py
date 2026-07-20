"""gap#1 fully-synaptic RF transduction — RUNG 3 (deployed confirmation): does the FULLY-SYNAPTIC read (RF spike ->
slow-NMDA synapse -> readout g_nmda -> log decode, NO host rf_read_phases) reproduce the DEPLOYED accumulated-state
fidelity that the deep-NLL needs? The RF-phase-encode deployed pre-flight got accumulated corr 0.998 (via rf_read_phases);
if the SYNAPTIC read matches it on the same deployed injects, the fully-synaptic path has deep-NLL parity (RUNG 2 already
showed the synaptic read is value-faithful corr 1.0 on a grid; here on the real zero-inflated deployed distribution).
NO host phase read; NO sim/ edit (inject_explicit_wiring + reads public cp_conductance_g_nmda)."""
import sys; sys.path.insert(0, "/home/dant123/Projects/sim")
import os; os.environ.setdefault("SIM_BACKEND", "cupy")
import argparse
import numpy as np
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
from sim.enums import NeuronModel
from sim.backend import to_host
from research.runners._emerge_reservoir_lm_derisk import Vocab
from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences

PERIOD = 200; PLO, PHI = 0.05, 0.95


def build_synaptic_rf(n_chan, w=30.0, seed=42):
    """2*n_chan neurons: n_chan ENCODERS [0..n) + n_chan READOUTS [n..2n); a diagonal slow-NMDA synapse enc_i->rdt_i
    carries each RF spike into rdt_i's g_nmda = w*exp(-latency/tau) = the value (RUNG 2)."""
    cfg = CoreSimConfig(); cfg.num_neurons = 2 * n_chan
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name; cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = seed; cfg.dt_ms = 1.0; cfg.connections_per_neuron = 0; cfg.num_traits = 1
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity", "enable_structural_plasticity",
              "enable_homeostasis", "enable_reward_modulation", "enable_watts_strogatz", "enable_neuromodulator_subsystem",
              "enable_brain_region_framework"):
        if hasattr(cfg, f): setattr(cfg, f, False)
    cfg.ou_std_current_pA = 0.0; cfg.enable_nmda = True
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    enc = np.arange(n_chan); rdt = np.arange(n_chan) + n_chan
    b.inject_explicit_wiring({"enc2rdt": {"pre_indices": enc.tolist(), "post_indices": rdt.tolist(),
                              "initial_weights": [w] * n_chan, "plastic": False, "conn_type": "enc2rdt",
                              "exc_receptors": ["nmda_slow"] * n_chan}})
    b.core_config.neuron_model_type = NeuronModel.RESONATE_AND_FIRE.name
    return b, enc, rdt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssm", default="bridges/wkv_ckpt/wkv_ssmU_v1000_d128_seed42.npz")
    ap.add_argument("--corpus", default="data/corpus/tinystories_train.txt")
    ap.add_argument("--n-sentences", type=int, default=40000)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    W = np.load(args.ssm, allow_pickle=True)
    V = int(W["V"]); D = int(W["d_model"]); words = list(W["words"])
    emb = W["emb.weight"].astype(np.float64); lnw = W["ln.weight"].astype(np.float64); lnb = W["ln.bias"].astype(np.float64)
    Wv = W["Wv.weight"].astype(np.float64)
    decay = float(np.exp(-np.log1p(np.exp(W["w"][0]))))
    def _ln(x):
        return (x - x.mean()) / (x.std() + 1e-5) * lnw + lnb
    Vd = np.stack([Wv @ _ln(emb[t]) for t in range(V)], 0)
    _sc0 = max(1e-6, 1.0 - decay)
    Dd = np.concatenate([np.maximum(Vd, 0.0), np.maximum(-Vd, 0.0)], 1) / _sc0   # [V, 2D] deployed injects (>=0)

    sents = load_sentences(args.corpus, args.n_sentences); ev = sents[int(len(sents) * 0.9):]
    vocab = Vocab(words[:-1]); ev_ids = [vocab.ids(s) for s in ev][: args.n_eval]
    tok = np.concatenate([np.asarray(i) for i in ev_ids if len(i)])
    flat = Dd[tok].reshape(-1)
    VMAX = float(np.percentile(flat[flat > 0], 99.8)) if (flat > 0).any() else 1.0

    n_chan = 2 * D
    b, enc, rdt = build_synaptic_rf(n_chan, w=30.0, seed=args.seed)

    def synaptic_read(inj):
        """inj [2D] >=0 -> RF phase on encoders -> resonate -> READOUT g_nmda (fully synaptic) -> value (log decode)."""
        d = np.clip(np.asarray(inj, np.float64), 0.0, VMAX)
        p = PLO + (PHI - PLO) * (d / max(VMAX, 1e-9))
        z = np.zeros(2 * n_chan, np.complex64); z[enc] = np.exp(1j * 2 * np.pi * p).astype(np.complex64)
        mask = np.zeros(2 * n_chan, bool); mask[enc] = True
        try: b.rf_kick(z, period=PERIOD, neuron_mask=mask)
        except TypeError: b.rf_kick(z, period=PERIOD)
        for _ in range(PERIOD + 8):
            b.cp_external_input_current[:] = 0.0
            b._run_one_simulation_step()
        return np.asarray(to_host(b.cp_conductance_g_nmda), np.float64)[rdt]   # the synaptic conductance (encodes value)

    # CALIBRATE the fixed log read-out g_nmda -> value ONCE on a value grid (a fixed synaptic nonlinearity; tau,w fixed)
    cal_vals = np.linspace(0.0, VMAX, n_chan)
    g_cal = synaptic_read(cal_vals)
    m = g_cal > 1e-9
    x = np.log(np.clip(g_cal[m] / max(g_cal.max(), 1e-9), 1e-12, 1.0))
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, cal_vals[m], rcond=None)
    def decode(g):
        gg = np.clip(g / max(g_cal.max(), 1e-9), 1e-12, 1.0)
        return np.clip(coef[0] * np.log(gg) + coef[1], 0.0, VMAX)
    print(f"[calib] log read-out on {int(m.sum())}/{n_chan} channels: recovered corr "
          f"{np.corrcoef(decode(g_cal[m]), cal_vals[m])[0,1]:.4f}")

    # accumulate exact vs SYNAPTIC-read state over deployed sentences
    S_ex, S_syn = [], []
    for ids in ev_ids:
        if not len(ids): continue
        s_ex = np.zeros(2 * D); s_syn = np.zeros(2 * D)
        for t in ids:
            d = Dd[t]
            g = synaptic_read(d); d_syn = decode(g)
            s_ex = decay * s_ex + d
            s_syn = decay * s_syn + d_syn
            S_ex.append(s_ex.copy()); S_syn.append(s_syn.copy())
    S_ex = np.asarray(S_ex); S_syn = np.asarray(S_syn)
    acc_corr = float(np.corrcoef(S_syn.reshape(-1), S_ex.reshape(-1))[0, 1])
    ch = np.array([np.corrcoef(S_syn[:, c], S_ex[:, c])[0, 1] if S_ex[:, c].std() > 1e-9 else np.nan for c in range(2 * D)])
    bias = float((S_syn - S_ex).mean())
    print(f"\n[FULLY-SYNAPTIC deployed] accumulated corr(s_syn, s_exact) = {acc_corr:.4f}  "
          f"(RF-phase-encode ref was 0.998; deep-NLL needs >~0.9)")
    print(f"  per-channel corr median {np.nanmedian(ch):.4f} p10 {np.nanpercentile(ch,10):.4f} | accum bias {bias:+.4f}")
    go = acc_corr > 0.9
    print(f"\n  => RUNG 3 {'GO — the FULLY-SYNAPTIC read reproduces the deployed accumulated-state fidelity (deep-NLL parity); gap#1 spiking input is fully synaptic, NO host rf_read_phases' if go else 'partial — inspect'}")


if __name__ == "__main__":
    main()
