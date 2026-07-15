"""2026-07-15 — on-substrate realization of TEST A (plan #3, the emergence-bar forward): does the SPIKING COINCIDENCE
BIND (the substrate's native multiplicative ⊙, `core_sim_composition.build_bind_bridge` + `hadamard_spiking` — 8-neuron
AND-banks on a real SimulationBridge) achieve the systematic held-out-composition extrapolation that the numpy ±1 bind
did (TEST A, 12-seed 0.96) and a from-scratch learner did not (0.50)?

RUNG 1 (this file, fixed projections): bind cat_code (as ROLE, ±1) ⊗ qt_code (as FILLER, graded ON/OFF) through the
coincidence circuit ON SPIKES; read the bound firing (bound_ON,bound_OFF); train a ridge read-out on the bound rates ->
intent; test held-out (cat,qt) COMBINATIONS. GO iff the spiking bind extrapolates >> a from-scratch MLP on [cat;q] +
permuted collapses -> the on-spike coincidence bind IS a systematicity primitive. (RUNG 2 = LEARN the projections by
the committed on-bridge BDSP rule -- the full emergence step.)

Reuse-by-import: `build_bind_bridge`/`hadamard_spiking` (core sim), `build_task`/`_ridge` (the systematicity harness),
`_train_snn`/`score_snn`/`standardize` (the learner control). NO `sim/` edit. GPU/CuPy (`SIM_BACKEND=cupy`); numpy = smoke.

Run: SIM_BACKEND=cupy python -u -m research.runners._onsubstrate_coincidence_systematicity_derisk --seeds 42
"""
import os, sys, json, argparse
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners.core_sim_composition import build_bind_bridge, hadamard_spiking
from research.runners._fixedbind_systematicity_derisk import build_task, _dataset, _ridge, N_INTENT
from research.runners._deep_eprop_binder_bundling_derisk import _train_snn, score_snn, standardize, T

RUN_STEPS = 150
COINC_BIAS = 0.0


def _fill_currents(code_pm, D):
    """A ±1 code -> graded ON/OFF filler currents for the coincidence bind (ON where +1, OFF where -1), scaled to a
    firing-driving current (the validated fill operating point uses graded ON/OFF drives)."""
    on = (code_pm > 0).astype(np.float32) * 220.0
    off = (code_pm < 0).astype(np.float32) * 220.0
    return on, off


def run_one(seed, D=None, hidden=48, epochs=120):
    a, b, cat_code, q_code, intent_of, held, Dc = build_task(seed, easy=False)
    D = D or Dc                                          # bind dimension = the code dim
    cells, y, is_held = _dataset(cat_code, q_code, intent_of, held, n_per=1, seed=seed)
    tr = ~is_held
    # IDENTITY projections (RUNG 1): drive the codes DIRECTLY as role/filler so the coincidence bind exposes a(X)b in the
    # first NB dims (the random projections scrambled the rule structure -> the bind couldn't expose it). D = code dim.
    # (RUNG 2 = LEARN the projections via BDSP; here they are the identity = the on-spike realization of TEST A's direct bind.)
    bridge, idx = build_bind_bridge(seed, D)
    B = np.zeros((len(cells), 2 * D), np.float32)
    for i, (c, q) in enumerate(cells):
        role = (cat_code[c] * 2 - 1).astype(np.float32)          # cat_code as a ±1 role vector (directly)
        fon, foff = _fill_currents(q_code[q] * 2 - 1, D)         # qt_code as graded ON/OFF filler currents (directly)
        bon, boff = hadamard_spiking(bridge, idx, role, fon, foff, D, RUN_STEPS, COINC_BIAS)
        B[i] = np.concatenate([bon, boff])
    out = {"seed": seed, "chance": round(1.0 / N_INTENT, 4), "n_held": int(is_held.sum()), "D": D}
    Btr, Bev = standardize(B[tr], B)
    pb = _ridge(Btr, y[tr], Bev, N_INTENT, lam=8.0)
    out["spikebind_train"] = round(float(np.mean(pb[tr] == y[tr])), 4)
    out["spikebind_held"] = round(float(np.mean(pb[is_held] == y[is_held])), 4)
    # control: from-scratch MLP learner on [cat;q] concat (TEST A: memorizes+fails)
    CAT = np.array([cat_code[c] for (c, q) in cells]); Q = np.array([q_code[q] for (c, q) in cells])
    C = np.concatenate([CAT, Q], axis=1); Ctr, Cev = standardize(C[tr], C)
    lay = _train_snn(Ctr, y[tr], [C.shape[1], hidden, hidden, N_INTENT], T, epochs, 0.05, 1.0, seed, credit_mode="eprop")
    _, out["mlp_held"], _ = score_snn(lay, Cev, y, is_held, 1.0); out["mlp_held"] = round(out["mlp_held"], 4)
    # anti-cheat permuted
    rp = np.random.default_rng(seed + 3); yp = y.copy(); yp[tr] = y[tr][rp.permutation(int(tr.sum()))]
    pbp = _ridge(Btr, yp[tr], Bev, N_INTENT, lam=8.0)
    out["spikebind_permuted_held"] = round(float(np.mean(pbp[is_held] == y[is_held])), 4)
    out["GO"] = bool(out["spikebind_held"] > 0.6 and out["spikebind_held"] > out["mlp_held"] + 0.15
                     and out["spikebind_permuted_held"] < out["chance"] + 0.2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", default="research/findings/raw/_onsubstrate_coincidence_systematicity.json")
    a = ap.parse_args()
    rows = [run_one(s) for s in a.seeds]
    for r in rows:
        print(f"[spikebind s{r['seed']}] chance={r['chance']} D={r['D']} || SPIKING-COINCIDENCE-BIND held={r['spikebind_held']:.3f} "
              f"(train {r['spikebind_train']:.3f}) | MLP={r['mlp_held']:.3f} | permuted={r['spikebind_permuted_held']:.3f} "
              f"|| {'GO' if r['GO'] else 'no'}", flush=True)
    ngo = sum(x["GO"] for x in rows)
    print(f"[spikebind] {ngo}/{len(rows)} GO (spiking coincidence bind extrapolates held-out compositions >> MLP; permuted collapses)", flush=True)
    json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
