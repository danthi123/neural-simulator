"""gap#1 RF PHASE **deployed-accumulated-state** pre-flight (the day's hardest lesson: validate on DEPLOYED inputs,
not a static value grid). The static pre-flight (_gap1_rf_phase_preflight.py) showed RF phase decode is unbiased on a
uniform [-3,3] grid. But the DEPLOYED values are the dual-nonneg injects d=relu(+/-v_t), v_t=Wv.LN(emb[x_t]) --
ZERO-INFLATED (relu floors half of them at 0), so most values are SMALL, exactly where a phase dead-zone would bite.

This measures, on REAL sentences:
  (1) the DEPLOYED value distribution (is it zero-inflated? what's the range?);
  (2) the RF-encode's per-token decode error, BY VALUE BAND (small/mid/large) -- the M0 bias-spread test on the real dist;
  (3) the ACCUMULATED-state fidelity: charge cp_ssm_state via decay-leaky recurrence over the RF-decoded injects vs the
      EXACT injects, and measure corr(s_rf, s_exact) AND per-channel accumulated-error MEAN (accumulated bias).

GO for the full build: accumulated corr > 0.85 (M0 curve GO territory) AND accumulated bias unbiased across value bands
(bias-spread small). Control-first: M1 already re-confirmed +0.874 GO on this exact checkpoint (map_corr 1.000).
NO sim/ edit (drives + reads public arrays; independent RF oscillators, connections_per_neuron=0)."""
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


def build_rf(n, seed=42):
    cfg = CoreSimConfig(); cfg.num_neurons = int(n)
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name; cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = seed; cfg.dt_ms = 1.0; cfg.connections_per_neuron = 0; cfg.num_traits = 1
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity", "enable_structural_plasticity",
              "enable_homeostasis", "enable_reward_modulation", "enable_watts_strogatz", "enable_neuromodulator_subsystem",
              "enable_brain_region_framework"):
        if hasattr(cfg, f): setattr(cfg, f, False)
    cfg.ou_std_current_pA = 0.0
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    b.core_config.neuron_model_type = NeuronModel.RESONATE_AND_FIRE.name
    return b


def rf_encode_decode(b, d_vals, VMAX, period, p_lo=0.05, p_hi=0.95):
    """Encode a vector of NON-NEGATIVE values d_vals in [0,VMAX] as RF phases in the guard-banded arc [p_lo,p_hi]
    (so no value hits the 0/1 phase wrap), resonate, read phases, decode back. Returns d_hat (same shape)."""
    d = np.clip(np.asarray(d_vals, np.float64), 0.0, VMAX)
    p = p_lo + (p_hi - p_lo) * (d / max(VMAX, 1e-9))     # value -> phase in [p_lo, p_hi]
    z = np.exp(1j * 2 * np.pi * p)                        # unit magnitude, value in PHASE
    b.rf_kick(z.astype(np.complex64), period=period)
    for _ in range(period + 8):
        b.cp_external_input_current[:] = 0.0
        b._run_one_simulation_step()
    p_read = np.asarray(to_host(b.rf_read_phases()), np.float64)   # [N] in [0,1)
    d_hat = (p_read - p_lo) / (p_hi - p_lo) * VMAX
    return np.clip(d_hat, 0.0, VMAX), p_read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssm", default="bridges/wkv_ckpt/wkv_ssmU_v1000_d128_seed42.npz")
    ap.add_argument("--corpus", default="data/corpus/tinystories_train.txt")
    ap.add_argument("--n-sentences", type=int, default=40000)
    ap.add_argument("--n-eval", type=int, default=30)          # sentences to accumulate over (cheap deployed pre-flight)
    ap.add_argument("--period", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vmax-pct", type=float, default=99.8)    # VMAX = this percentile of deployed injects (guard vs outliers)
    ap.add_argument("--json", default="research/findings/raw/_gap1_rf_phase_deployed_preflight.json")
    args = ap.parse_args()

    W = np.load(args.ssm, allow_pickle=True)
    V = int(W["V"]); D = int(W["d_model"]); words = list(W["words"])
    emb = W["emb.weight"].astype(np.float64); lnw = W["ln.weight"].astype(np.float64); lnb = W["ln.bias"].astype(np.float64)
    Wv = W["Wv.weight"].astype(np.float64)
    decay = float(np.exp(-np.log1p(np.exp(W["w"][0]))))        # exp(-softplus(w)) == the runner's uniform decay

    def _ln(x):
        mu = x.mean(); sd = x.std() + 1e-5
        return (x - mu) / sd * lnw + lnb

    # V-vector dictionary: v_t depends ONLY on the token -> 1000 distinct vectors (the "V-vector phasor dictionary").
    Vd = np.stack([Wv @ _ln(emb[t]) for t in range(V)], 0)                 # [V, D] signed
    Dd = np.concatenate([np.maximum(Vd, 0.0), np.maximum(-Vd, 0.0)], 1)    # [V, 2D] dual-nonneg injects d=relu(+/-v), >=0

    # sentences + vocab (saved words -> no rebuild mismatch)
    sents = load_sentences(args.corpus, args.n_sentences)
    n = len(sents); ev = sents[int(n * 0.9):]
    vocab = Vocab(words[:-1])
    ev_ids = [vocab.ids(s) for s in ev][: args.n_eval]

    # deployed value distribution (weighted by actual eval-token frequency)
    tok = np.concatenate([np.asarray(ids) for ids in ev_ids if len(ids)])
    d_stream = Dd[tok]                                                     # [Ntok, 2D] the actual injects seen
    flat = d_stream.reshape(-1)
    VMAX = float(np.percentile(flat[flat > 0], args.vmax_pct)) if (flat > 0).any() else 1.0
    frac_zero = float((flat == 0).mean())
    print(f"[deployed dist] tokens={len(tok)} injects={flat.size}  frac==0 (relu floor)={frac_zero:.3f}  "
          f"nonzero: mean={flat[flat>0].mean():.4f} p50={np.median(flat[flat>0]):.4f} p99.8={VMAX:.4f} max={flat.max():.4f}")

    b = build_rf(2 * D, seed=args.seed)

    # ACCUMULATE exact vs RF-decoded state through the decay-leaky recurrence, per sentence (wash between sentences).
    s_ex_all, s_rf_all = [], []
    per_tok_err_by_band = {"lo": [], "mid": [], "hi": []}      # per-token decode error, by value band
    for ids in ev_ids:
        if not len(ids): continue
        s_ex = np.zeros(2 * D); s_rf = np.zeros(2 * D)
        for t in ids:
            d = Dd[t]                                          # [2D] exact injects for this token
            d_hat, _ = rf_encode_decode(b, d, VMAX, args.period)
            e = d_hat - d
            per_tok_err_by_band["lo"].append(e[d <= 0.02 * VMAX])          # near-zero (the dominant, dead-zone risk)
            per_tok_err_by_band["mid"].append(e[(d > 0.02 * VMAX) & (d <= 0.5 * VMAX)])
            per_tok_err_by_band["hi"].append(e[d > 0.5 * VMAX])
            s_ex = decay * s_ex + d
            s_rf = decay * s_rf + d_hat
            s_ex_all.append(s_ex.copy()); s_rf_all.append(s_rf.copy())

    S_ex = np.asarray(s_ex_all); S_rf = np.asarray(s_rf_all)              # [Nstate, 2D]
    acc_corr = float(np.corrcoef(S_rf.reshape(-1), S_ex.reshape(-1))[0, 1])
    # per-channel accumulated corr (the state the read-out actually reads)
    ch_corr = np.array([np.corrcoef(S_rf[:, c], S_ex[:, c])[0, 1] if S_ex[:, c].std() > 1e-9 else np.nan
                        for c in range(2 * D)])
    acc_err = S_rf - S_ex
    acc_bias = float(acc_err.mean())
    # per-token decode bias by band (the M0 value-dependence test on the REAL distribution)
    def _mrs(lst):
        a = np.concatenate(lst) if lst and any(len(x) for x in lst) else np.array([0.0])
        return float(a.mean()), float(np.sqrt((a ** 2).mean())), int(a.size)
    lo_m, lo_r, lo_n = _mrs(per_tok_err_by_band["lo"])
    mid_m, mid_r, mid_n = _mrs(per_tok_err_by_band["mid"])
    hi_m, hi_r, hi_n = _mrs(per_tok_err_by_band["hi"])
    bias_spread = abs(lo_m - mid_m) + abs(lo_m - hi_m)                    # M0's value-dependence signature (in value units)

    print(f"\n[per-token decode bias by value band] (M0 test on the DEPLOYED zero-inflated dist):")
    print(f"    lo (d<=2%VMAX, dead-zone risk, n={lo_n}): mean {lo_m:+.4f}  rms {lo_r:.4f}")
    print(f"    mid(2%..50%VMAX,          n={mid_n}): mean {mid_m:+.4f}  rms {mid_r:.4f}")
    print(f"    hi (d>50%VMAX,            n={hi_n}): mean {hi_m:+.4f}  rms {hi_r:.4f}")
    print(f"    => per-token bias-spread across bands = {bias_spread:.4f}  (value scale VMAX={VMAX:.3f})")
    print(f"\n[ACCUMULATED state (what the read-out reads)] over {S_ex.shape[0]} states x {2*D} channels:")
    print(f"    corr(s_rf, s_exact) = {acc_corr:.4f}   (M0: >~0.85 = GO territory; M1 corr 1.000 -> +0.874)")
    print(f"    per-channel corr: median {np.nanmedian(ch_corr):.4f}  p10 {np.nanpercentile(ch_corr,10):.4f}  "
          f"min {np.nanmin(ch_corr):.4f}")
    print(f"    accumulated bias (mean err) = {acc_bias:+.4f}  (state scale ~ mean|s| {np.abs(S_ex).mean():.3f})")

    go = bool(acc_corr > 0.85 and bias_spread < 0.15 * VMAX)
    print(f"\n  => PRE-FLIGHT {'GO — build the full RF phase encode' if go else 'NO-GO — encode bias/fidelity insufficient'}"
          f"  (acc_corr {acc_corr:.3f} > 0.85 AND bias-spread {bias_spread:.4f} < {0.15*VMAX:.4f})")

    import json
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    json.dump({"runner": "gap1_rf_phase_deployed_preflight", "ssm": args.ssm, "decay": decay, "VMAX": VMAX,
               "frac_zero": frac_zero, "period": args.period, "n_eval": args.n_eval,
               "acc_corr": acc_corr, "acc_bias": acc_bias, "ch_corr_median": float(np.nanmedian(ch_corr)),
               "ch_corr_p10": float(np.nanpercentile(ch_corr, 10)), "ch_corr_min": float(np.nanmin(ch_corr)),
               "band_lo": [lo_m, lo_r, lo_n], "band_mid": [mid_m, mid_r, mid_n], "band_hi": [hi_m, hi_r, hi_n],
               "bias_spread": bias_spread, "go": go}, open(args.json, "w"), indent=2)
    print(f"-> {args.json}")


if __name__ == "__main__":
    main()
