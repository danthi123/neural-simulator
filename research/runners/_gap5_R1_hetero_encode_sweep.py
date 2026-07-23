"""gap#5 candidate-#3 R1 (PRIMARY FIX, encode-only): phase-precession causal encode -> forward-ASYMMETRIC weights.

The numpy gamma-WTA GO rode start=A over a NEAR-SYMMETRIC store (adj_fwd ~= adj_rev). The fix is at ENCODING:
sharpen the forward/reverse asymmetry so a noise-ignited spiking replay travels forward regardless of start.

Cheapest sub-lever (research gate): raise btsp_hetero_dep (cfg.encode_btsp_hetero) above 0. Heterosynaptic
competition: when post B plateaus, inputs to B from non-eligible pres are DEPRESSED -> during A's plateau window the
reverse link B->A (B not eligible) is depressed while the forward A->B (A eligible) is potentiated => adj_fwd >> adj_rev.

ENCODE-ONLY: extract the between-assembly W and measure asymmetry. No readout/trials (cheap).
GO: ratio adj_fwd/adj_rev >= ~2-3x  AND  within-attractor preserved (within >= ~27, the reactivation floor).

Runs ONE (seed, hetero) so a launcher can fan out across CPU cores while the GPU trains.
"""
import argparse
import json
import os
import numpy as np
from research.runners._gap5_sequence_replay_derisk import _prepare_sequence, SEQ_CFG
from research.runners._gap5_gamma_wta_replay_derisk import _extract_W


def run_one(seed, hetero, n_mem=3, within_refresh=8, chain_fwd=24,
            chain_rule="btsp", stdp_a_plus=0.08, stdp_a_minus=0.10, stdp_tau=20.0, btsp_lr=None,
            freeze_between_refresh=False, chain_rev=0, chain_btsp_lr=None):
    cfg = dict(SEQ_CFG)
    cfg["n_mem"] = n_mem; cfg["within_events"] = 30; cfg["within_refresh"] = int(within_refresh)
    cfg["chain_fwd"] = int(chain_fwd); cfg["chain_rev"] = int(chain_rev); cfg["rank1_encode"] = True; cfg["overlap_draw"] = False
    cfg["encode_btsp_hetero"] = float(hetero)
    cfg["freeze_between_refresh"] = bool(freeze_between_refresh)
    if chain_btsp_lr is not None:
        cfg["chain_btsp_lr"] = float(chain_btsp_lr)
    cfg["chain_rule"] = str(chain_rule)
    cfg["stdp_a_plus"] = float(stdp_a_plus); cfg["stdp_a_minus"] = float(stdp_a_minus); cfg["stdp_tau"] = float(stdp_tau)
    if btsp_lr is not None:
        cfg["btsp_lr"] = float(btsp_lr)   # gap#5: the BTSP-eligibility forward mechanism's strength (default SEQ_CFG 0.02)
    prep = _prepare_sequence(int(seed), cfg)
    W = _extract_W(prep, n_mem)
    within = float(np.mean(np.diag(W)))
    adj_fwd = float(np.mean([W[i, i + 1] for i in range(n_mem - 1)]))
    adj_rev = float(np.mean([W[i + 1, i] for i in range(n_mem - 1)]))
    skip = float(np.mean([W[i, j] for i in range(n_mem) for j in range(n_mem) if j > i + 1])) if n_mem > 2 else 0.0
    ratio = adj_fwd / max(abs(adj_rev), 1e-6)
    go = (ratio >= 2.0) and (within >= 27.0) and (adj_fwd > adj_rev)
    return dict(seed=int(seed), hetero=float(hetero), n_mem=n_mem, within=within,
                adj_fwd=adj_fwd, adj_rev=adj_rev, asym=adj_fwd - adj_rev, skip_fwd=skip,
                ratio=ratio, go=bool(go), W=W.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hetero", type=float, required=True)
    ap.add_argument("--n-mem", type=int, default=3)
    ap.add_argument("--within-refresh", type=int, default=8)
    ap.add_argument("--chain-fwd", type=int, default=24)
    ap.add_argument("--chain-rule", choices=["btsp", "stdp"], default="btsp")
    ap.add_argument("--stdp-a-plus", type=float, default=0.08)
    ap.add_argument("--stdp-a-minus", type=float, default=0.10)
    ap.add_argument("--stdp-tau", type=float, default=20.0)
    ap.add_argument("--btsp-lr", type=float, default=None)
    ap.add_argument("--freeze-between-refresh", action="store_true")
    ap.add_argument("--chain-rev", type=int, default=0)
    ap.add_argument("--chain-btsp-lr", type=float, default=None, help="separate BTSP lr for the CHAIN phase (decouples forward strength from the within-attractor; e.g. within --btsp-lr 0.05 + --chain-btsp-lr 0.5)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = run_one(a.seed, a.hetero, a.n_mem, a.within_refresh, a.chain_fwd,
                a.chain_rule, a.stdp_a_plus, a.stdp_a_minus, a.stdp_tau, a.btsp_lr,
                a.freeze_between_refresh, a.chain_rev, a.chain_btsp_lr)
    r["within_refresh"] = a.within_refresh; r["chain_fwd"] = a.chain_fwd; r["chain_rule"] = a.chain_rule
    r["btsp_lr"] = a.btsp_lr; r["freeze_between_refresh"] = a.freeze_between_refresh; r["chain_rev"] = a.chain_rev
    r["chain_btsp_lr"] = a.chain_btsp_lr
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(r, f, indent=2)
    print(f"[R1] seed={r['seed']} hetero={r['hetero']:.2f} within={r['within']:.1f} "
          f"adj_fwd={r['adj_fwd']:.1f} adj_rev={r['adj_rev']:.1f} asym={r['asym']:+.2f} "
          f"ratio={r['ratio']:.2f}x => {'GO' if r['go'] else 'no'}")


if __name__ == "__main__":
    main()
