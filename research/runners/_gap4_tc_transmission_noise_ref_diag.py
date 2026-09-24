"""gap#4 transport-ceiling instrument diagnostic (dev seed 7 only; no gate reads it): does the feedforward pathway of the
on-bridge BDSP net carry the input, measured against a NOISE REFERENCE and per unit? (review fix, prereg AMENDMENT 5.)

The 2026-09-24 probe (round4 diag_fullsize_ff_transmission_s7.json) compared the MEAN H1 read before and after zeroing
the input->H1 weights. Those weights are zero-mean signed Xavier draws at a tonically firing operating point, so the mean
barely moves even when transmission is real, and the probe had no intact-vs-intact reference. This diagnostic adds both.

For one variant (the 2x2 of short-term depression on the explicit feedforward synapses x the feedforward gain) it takes
five frozen reads of the same items, in order:  R1 intact, R2 intact, Z1 input->H1 zeroed, R3 intact (restored),
Z2 H2->out zeroed.  Per layer it reports:
  noise_per_unit      mean |R1 - R2| (adjacent intact reads: run-to-run noise, including state carry-over)
  effect_per_unit     mean |R2 - Z1| for H1, mean |R3 - Z2| for the output (adjacent reads again)
  effect_over_noise   effect_per_unit / noise_per_unit (about 1 => the cut is indistinguishable from noise)
  item_reliability    mean over units of the across-item correlation of R1 with R2: > 0 means each unit's
                      item-to-item variation is reproducible, i.e. stimulus-driven; about 0 means it is noise
  item_reliability_cut  the same between R2 and the cut read (the stimulus-driven pattern should vanish when cut)
  mean_intact / mean_cut  the statistic the earlier probe used, kept for comparison

Variants: legacy (STP on, ff_w_init 4, propagation 0.05) | stp_bypass (--no-ff-stp at the legacy gain) | gain (STP on,
ff_w_init 40, propagation 0.5) | stp_bypass_gain (both, the calibrated operating point). Host-only instrument;
functional read-outs only.

RUN: SIM_BACKEND=numpy python -u -m research.runners._gap4_tc_transmission_noise_ref_diag --variant legacy \
       --hidden 64 --pool-k 16 --tonic-scale 1.0 --out research/findings/raw/gap4/transport_ceiling_readout/x.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

VARIANTS = {
    "legacy": {},
    "stp_bypass": dict(no_ff_stp=True, no_structural=True),
    "gain": dict(no_structural=True, ff_w_init=40.0, propagation_strength=0.5),
    "stp_bypass_gain": dict(no_ff_stp=True, no_structural=True, ff_w_init=40.0, propagation_strength=0.5),
}


def _item_reliability(A, B):
    """Mean over units of the across-item Pearson correlation of A[:, u] with B[:, u] (units flat in either read are
    skipped). Returns (mean r, number of units used)."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    rs = []
    for u in range(A.shape[1]):
        a = A[:, u] - A[:, u].mean(); b = B[:, u] - B[:, u].mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        if den > 1e-12:
            rs.append(float((a * b).sum() / den))
    return (float(np.mean(rs)) if rs else None), len(rs)


def _layer_stats(Ra, Rb, Rcut_ref, Rcut):
    noise = float(np.abs(np.asarray(Ra) - np.asarray(Rb)).mean())
    effect = float(np.abs(np.asarray(Rcut_ref) - np.asarray(Rcut)).mean())
    rel, n_rel = _item_reliability(Ra, Rb)
    rel_cut, n_cut = _item_reliability(Rcut_ref, Rcut)
    return {"mean_intact": float(np.mean(Rcut_ref)), "mean_cut": float(np.mean(Rcut)),
            "noise_per_unit": noise, "effect_per_unit": effect,
            "effect_over_noise": (effect / noise) if noise > 0 else None,
            "item_reliability": rel, "item_reliability_units": n_rel,
            "item_reliability_cut": rel_cut, "item_reliability_cut_units": n_cut,
            "input_dependence": float(np.asarray(Ra).std(0).mean() / (np.asarray(Ra).mean() + 1e-9))}


def transmission_stats(net, X):
    """Five frozen reads (R1, R2, Z1 in->H1 cut, R3, Z2 H2->out cut) and the per-layer statistics above."""
    from sim.backend import to_host
    coo = net.br._get_cached_coo()
    row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
    w0 = np.asarray(to_host(net.br.cp_connections.data)).copy()

    def mask(li):
        pre, post = net._ff_edges[li]
        return (row >= pre[0]) & (row <= pre[-1]) & (col >= post[0]) & (col <= post[-1])

    def reads():
        with net.frozen_reads():
            return [np.asarray(a, float) for a in net._forward_batch(X)]

    def set_w(w):
        net.br.cp_connections.data[...] = net._xp.asarray(w)

    R1 = reads(); R2 = reads()
    w = w0.copy(); w[mask(0)] = 0.0; set_w(w); Z1 = reads(); set_w(w0)
    R3 = reads()
    w = w0.copy(); w[mask(len(net._ff_edges) - 1)] = 0.0; set_w(w); Z2 = reads(); set_w(w0)
    return {"h1": _layer_stats(R1[1], R2[1], R2[1], Z1[1]),
            "out": _layer_stats(R1[-1], R2[-1], R3[-1], Z2[-1]),
            "weights_restored": bool(np.array_equal(np.asarray(to_host(net.br.cp_connections.data)), w0))}


def build(variant, n_in, hidden, pool_k, k, tonic_scale, read, seed=7):
    from research.runners._gap4_transport_ceiling_readout_derisk import Gap4ReadoutNet
    kw = dict(VARIANTS[variant])
    rq = dict(read_quantity="spikes", read_window=30) if read == "spikes" else {}
    return Gap4ReadoutNet(n_in, hidden, k, seed=seed, feedback="reservoir", n_hidden_layers=2, pool_k=pool_k,
                          settle_steps=40, credit_steps=25, graded_credit=True, eval_frozen=True,
                          tonic_h_pA=450.0 * tonic_scale, tonic_o_pA=500.0 * tonic_scale, **rq, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--pool-k", dest="pool_k", type=int, default=16)
    ap.add_argument("--tonic-scale", dest="tonic_scale", type=float, default=1.0,
                    help="1.0 = the 2026-09-15 operating point (tonic 450 / 500 pA)")
    ap.add_argument("--reads", nargs="+", default=["event", "spikes"], choices=["event", "spikes"],
                    help="event = the legacy snapshot read; spikes = the window-30 spike read")
    ap.add_argument("--n-items", dest="n_items", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.seed != 7:
        raise SystemExit("REFUSED: dev diagnostic, seed 7 only")
    from research.runners._semantic_inheritance_deep_credit_derisk import make_task_semantic_inheritance
    (Xtr, _ytr, _), _te, meta, _idx = make_task_semantic_inheritance(a.seed)
    k = int(meta["k_classes"])
    X = Xtr[:a.n_items]
    out = {"probe": "gap4_tc_transmission_noise_reference_dev_s7", "seed": a.seed, "variant": a.variant,
           "variant_flags": VARIANTS[a.variant], "hidden": a.hidden, "pool_k": a.pool_k,
           "tonic_scale": a.tonic_scale, "n_items": int(len(X)), "reads": {},
           "note": "dev instrument; host-only statistics; no gate reads this"}
    for read in a.reads:
        t0 = time.time()
        net = build(a.variant, X.shape[1], a.hidden, a.pool_k, k, a.tonic_scale, read, seed=a.seed)
        st = transmission_stats(net, X)
        st["seconds"] = round(time.time() - t0, 1)
        out["reads"][read] = st
        h = st["h1"]; o = st["out"]
        print(f"[tx-noise][{a.variant} {read}] H1 effect/noise {h['effect_over_noise']:.2f} "
              f"(effect {h['effect_per_unit']:.4f}, noise {h['noise_per_unit']:.4f}) item-rel {h['item_reliability']} "
              f"-> cut {h['item_reliability_cut']} | mean {h['mean_intact']:.4f}->{h['mean_cut']:.4f} || OUT "
              f"effect/noise {o['effect_over_noise']:.2f} | {st['seconds']}s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"[tx-noise] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
