"""2026-07-15 — emergence-engine #1 de-risk (research gate `2026-07-15-emergence-engine-research-gate-...`): does a
DELTA-RULE fast-weight CONTENT-ADDRESSABLE STORE over the frozen reservoir's states extend the memory horizon PAST the
fixed-ALIF window, where a fading reservoir provably cannot?

THE TASK (retrieve-by-content, so a POSITION-AGNOSTIC cache provably fails -- the load-bearing 2026-07-11 caution):
  [STORE k1 v1 k2 v2  f_1..f_T  PROBE k_j] -> predict v_j   (j random; k/v disjoint content symbols; f_i disjoint fillers)
The model must MATCH the probed key k_j to the stored key and retrieve its PAIRED value. A cache-bag (mean of stored
values) is at chance between v1,v2; a fading reservoir loses the bindings past the ALIF window (~5-15 tokens).

THE STORE (the synthesis spec): over the trial, maintain a fast-weight M; at each stored (k_i,v_i) WRITE with the
reservoir states as key/value (delta rule M += eta (v - M k) k^T); at PROBE READ v_hat = M q (q = probe state), and give
[q ; v_hat] to a ridge read-out (the "2-stage read-out"). ARMS: none (state only) / delta / additive (M += v k^T,
saturates under interference) / cachebag (v_hat = mean stored value, position-agnostic) / keyshuffle (write keys shuffled
-> content-addressing breaks). GATE (6-seed, T past the ALIF window): delta >> none AND delta > additive AND
delta >> cachebag AND keyshuffle collapses to none; within-window (T=5) none already works (positive control).

Run: SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u -m research.runners._reslm_deltastore_horizon_derisk
"""
import os, sys, json, argparse
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._emerge_reservoir_lm_eprop_recurrent_derisk import RateReservoir


def make_kv_task(K, F, T, n_trials, seed, n_pairs=2):
    """[STORE k1 v1 .. kP vP f_1..f_T PROBE k_j] -> v_j. n_pairs = the INTERFERENCE lever (more bindings held at once ->
    additive store saturates, delta holds). Needs K >= 2*n_pairs distinct content symbols."""
    STORE, PROBE, FILL0 = K, K + 1, K + 2
    rng = np.random.default_rng(seed)
    trials, probe_pos, targets = [], [], []
    for _ in range(n_trials):
        syms = rng.permutation(K)               # first n_pairs = keys, next n_pairs = values (disjoint)
        keys = [int(syms[i]) for i in range(n_pairs)]
        vals = [int(syms[n_pairs + i]) for i in range(n_pairs)]
        j = int(rng.integers(n_pairs))
        fillers = [FILL0 + int(rng.integers(F)) for _ in range(T)]
        body = [STORE]
        for i in range(n_pairs):
            body += [keys[i], vals[i]]
        seq = body + fillers + [PROBE, keys[j]]
        trials.append(np.asarray(seq, dtype=np.int64))
        probe_pos.append(len(seq) - 1)
        targets.append(vals[j])
    kpos = [(1 + 2 * i, 2 + 2 * i) for i in range(n_pairs)]   # (key_pos, val_pos) per stored pair
    return trials, probe_pos, targets, kpos, (STORE, PROBE, FILL0)


def _store_features(states, kpos, probe_pos, arm, eta, rng):
    """Build [q ; v_hat] for one trial. states[t] = reservoir state at t. arm selects the store variant."""
    n2 = states.shape[1]
    q = states[probe_pos]
    if arm == "none":
        return np.concatenate([q, np.zeros(n2)])
    if arm == "cachebag":
        vbag = np.mean([states[vp] for (_, vp) in kpos], axis=0)   # position-agnostic mean value (no content match)
        return np.concatenate([q, vbag])
    M = np.zeros((n2, n2))
    pairs = list(kpos)
    if arm == "keyshuffle":
        keys = [kp for (kp, _) in pairs]; rng.shuffle(keys)
        pairs = [(keys[i], pairs[i][1]) for i in range(len(pairs))]  # keys mismatched to values -> content breaks
    for (kp, vp) in pairs:
        k = states[kp]; v = states[vp]
        if arm == "additive":
            M += np.outer(v, k)
        else:                                    # delta rule (error-correcting)
            M += eta * np.outer(v - M @ k, k)
    v_hat = M @ q
    return np.concatenate([q, v_hat])


def _ridge_readout(Phi_tr, y_tr, Phi_ev, y_ev, K, lam=1.0):
    """Closed-form ridge read-out to one-hot targets; argmax accuracy on eval. (the '2-stage read-out' as a linear map.)"""
    Y = np.zeros((len(y_tr), K)); Y[np.arange(len(y_tr)), y_tr] = 1.0
    A = Phi_tr.T @ Phi_tr + lam * np.eye(Phi_tr.shape[1])
    W = np.linalg.solve(A, Phi_tr.T @ Y)         # (d x K)
    pred = np.argmax(Phi_ev @ W, axis=1)
    return float(np.mean(pred == y_ev))


def _eval_arms(res, K, F, T, n_pairs, n_train, n_eval, eta, seed, arms):
    tr, tp, ty, kpos, _ = make_kv_task(K, F, T, n_train, seed, n_pairs=n_pairs)
    ev, ep, ey, kpe, _ = make_kv_task(K, F, T, n_eval, seed + 5000, n_pairs=n_pairs)
    St = [np.asarray(res.forward_states(ids)) for ids in tr]
    Se = [np.asarray(res.forward_states(ids)) for ids in ev]
    row = {}
    for arm in arms:
        rng = np.random.default_rng(seed * 71 + hash(arm) % 997)
        Phi_tr = np.array([_store_features(St[i], kpos, tp[i], arm, eta, rng) for i in range(len(tr))])
        Phi_ev = np.array([_store_features(Se[i], kpe, ep[i], arm, eta, rng) for i in range(len(ev))])
        row[arm] = round(_ridge_readout(Phi_tr, np.array(ty), Phi_ev, np.array(ey), K), 4)
    return row


def run_one(seed, K=16, F=6, n_pool=120, n_train=800, n_eval=300, eta=0.5,
            adapt_win_hi=300.0, beta=1.0, Ts=(5, 15, 30), interfere_pairs=6):
    Veff = K + 2 + F
    res = RateReservoir(Veff, n_pool, seed=seed, alpha=0.3, spectral=1.1, alif=True, beta=beta, adapt_win_hi=adapt_win_hi)
    ARMS = ["none", "delta", "additive", "cachebag", "keyshuffle"]
    out = {"seed": seed, "K": K, "chance": round(1.0 / K, 4), "n_pool": n_pool, "eta": eta, "byT": {},
           "interfere_pairs": interfere_pairs}
    # (A) HORIZON: n_pairs=2, sweep T -> does the content-addressable store hold past the ALIF window?
    for T in Ts:
        out["byT"][T] = _eval_arms(res, K, F, T, 2, n_train, n_eval, eta, seed, ARMS)
    # (B) INTERFERENCE: past-window T, n_pairs high -> does DELTA beat ADDITIVE (error-correcting write vs saturation)?
    Tg = max(Ts)
    out["interfere"] = _eval_arms(res, K, F, Tg, interfere_pairs, n_train, n_eval, eta, seed, ["none", "delta", "additive"])
    g = out["byT"][Tg]; gi = out["interfere"]
    out["horizon_GO"] = bool(g["delta"] >= 0.6 and g["delta"] > g["none"] + 0.15
                             and g["delta"] > g["cachebag"] + 0.15 and g["keyshuffle"] < g["delta"] - 0.15)
    out["delta_beats_additive_under_interference"] = bool(gi["delta"] > gi["additive"] + 0.05)
    out["GO"] = out["horizon_GO"]          # the headline: content-addressable store extends the horizon
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-pool", type=int, default=120)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--Ts", type=int, nargs="+", default=[5, 15, 30])
    ap.add_argument("--out", default="research/findings/raw/_reslm_deltastore_horizon.json")
    a = ap.parse_args()
    rows = [run_one(s, n_pool=a.n_pool, n_train=a.n_train, n_eval=a.n_eval, eta=a.eta, Ts=tuple(a.Ts)) for s in a.seeds]
    for r in rows:
        segs = " | ".join(f"T{T}: none {r['byT'][T]['none']:.2f} DELTA {r['byT'][T]['delta']:.2f} "
                          f"bag {r['byT'][T]['cachebag']:.2f} shuf {r['byT'][T]['keyshuffle']:.2f}" for T in a.Ts)
        gi = r["interfere"]
        print(f"[deltastore s{r['seed']}] chance={r['chance']} || {segs} || interfere(P={r['interfere_pairs']}): "
              f"DELTA {gi['delta']:.2f} add {gi['additive']:.2f} none {gi['none']:.2f} "
              f"[delta>add:{r['delta_beats_additive_under_interference']}] || {'GO' if r['GO'] else 'no'}", flush=True)
    ngo = sum(x["GO"] for x in rows); nda = sum(x["delta_beats_additive_under_interference"] for x in rows)
    print(f"[deltastore] {ngo}/{len(rows)} horizon-GO (delta extends past ALIF window, cachebag+keyshuffle collapse); "
          f"{nda}/{len(rows)} delta>additive under interference (P={rows[0]['interfere_pairs']})", flush=True)
    json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
