"""D · PRAGMATICS -- distinctiveness (RSA/Gricean) success-signal de-risk.

FRONTIER: the neural coincidence success s(t,u) = coincidence(belief[u], intent=t) does NOT rank the belief-ALIGNED
utterance highest for ~56% of contexts (succ_opt == aligned only 8/18 across 6 seeds). A perfect success-MAXIMISER
therefore caps at ~0.44 on "speak the ALIGNED utterance". The learning rule is correct (v3 state-value baseline passes
contingency); the REWARD is misspecified relative to the pragmatic target.

DESIGN UNDER TEST (RSA / Gricean informativeness): reward DISTINCTIVENESS, not raw recovery. An utterance is
pragmatically successful for intent t iff the listener infers t MORE than the other intents. Add a lateral /
normalizing contrast term over intents to the neural coincidence:

    s_prag(t,u) = coincidence(t,u) - mean_{t'!=t} coincidence(t',u)

coincidence(t',u) is a REAL substrate measurement (drive belief[u] + a one-hot intent probe at t', read the neural
coincidence pool). The subtraction is a subtractive/divisive normalization over the intent pools -- exactly the
lateral-inhibition contrast the RSA listener performs; here computed as a host readout over NEURAL coincidence rates
(same footing as the existing scalar readout `sum_k rate(success[k])`; a fully-neural lateral-inhibition pool is the
stated upgrade if this passes).

STEP 1 (this file, deterministic, NO training): does the distinctiveness term make succ_opt == aligned for a majority
of contexts (up from 8/18)? Report agreement WITH vs WITHOUT the term + the degeneracy control.

NO sim/ edit. Forks v2/v3 by import.
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._pragmatic_success_readback_leg2_v2_derisk import (  # noqa: E402
    build_speaker_bridge, _evaluate_success, _commit_action,
    _belief_sources, _aligned_utts, K,
)
from research.runners._recursive_tom_rsa_derisk import UTTS, STATES  # noqa: E402


def measure_full_tables(seed):
    """Deterministic. Returns:
       belief[u][t]  -- RSA L1 posterior (host, sum-1 over states) used to define `aligned`.
       S[t][u]       -- NEURAL coincidence success (substrate) for driving intent=t + listener-belief=belief[u].
       aligned[t]    = argmax_u belief[u][t]           (the pragmatic target)
    """
    belief_src = _belief_sources(seed)
    aligned = _aligned_utts(belief_src)
    belief_by_u = {ui: belief_src[u] for ui, u in enumerate(UTTS)}
    belief = np.array([[belief_by_u[u][t] for t in range(K)] for u in range(K)])  # [u][t]

    bridge, xp, idx, snap = build_speaker_bridge(seed, oracle=False)
    S = np.zeros((K, K))                                  # S[t][u]
    for t in range(K):
        for u in range(K):
            _commit_action(bridge, xp, idx, snap, t, u)
            S[t, u] = _evaluate_success(bridge, xp, idx, t, belief_by_u[u])
    return belief, S, aligned, belief_by_u


def distinctiveness_table(S):
    """s_prag[t][u] = S[t][u] - mean_{t'!=t} S[t'][u]  (contrast over INTENTS for a fixed utterance)."""
    Sp = np.zeros_like(S)
    for t in range(K):
        for u in range(K):
            others = [S[tp, u] for tp in range(K) if tp != t]
            Sp[t, u] = S[t, u] - float(np.mean(others))
    return Sp


def divisive_table(S, eps=1e-6):
    """Divisive normalization variant: s_div[t][u] = S[t][u] / (eps + sum_{t'} S[t'][u])."""
    Sd = np.zeros_like(S)
    for u in range(K):
        denom = eps + float(np.sum(S[:, u]))
        for t in range(K):
            Sd[t, u] = S[t, u] / denom
    return Sd


def _argmax_map(tab):  # tab[t][u] -> {t: argmax_u}
    return {t: int(np.argmax(tab[t])) for t in range(K)}


def _agree(mapa, mapb):
    return sum(int(mapa[t] == mapb[t]) for t in range(K))


def evaluate_seed(seed, verbose=True):
    belief, S, aligned, _ = measure_full_tables(seed)
    succ_opt_plain = _argmax_map(S)
    Sp = distinctiveness_table(S)
    Sd = divisive_table(S)
    succ_opt_prag = _argmax_map(Sp)
    succ_opt_div = _argmax_map(Sd)

    m = {
        "seed": int(seed),
        "belief_u_t": [[round(float(x), 4) for x in row] for row in belief],
        "S_t_u": [[round(float(x), 5) for x in row] for row in S],
        "Sprag_t_u": [[round(float(x), 5) for x in row] for row in Sp],
        "aligned": {str(t): int(aligned[t]) for t in range(K)},
        "succ_opt_plain": {str(t): int(succ_opt_plain[t]) for t in range(K)},
        "succ_opt_prag": {str(t): int(succ_opt_prag[t]) for t in range(K)},
        "succ_opt_div": {str(t): int(succ_opt_div[t]) for t in range(K)},
        "agree_plain_vs_aligned": _agree(succ_opt_plain, aligned),
        "agree_prag_vs_aligned": _agree(succ_opt_prag, aligned),
        "agree_div_vs_aligned": _agree(succ_opt_div, aligned),
        # degeneracy control: is the pragmatic succ_opt a CONSTANT utterance across intents?
        "prag_is_constant": bool(len(set(succ_opt_prag.values())) == 1),
        "plain_is_constant": bool(len(set(succ_opt_plain.values())) == 1),
    }
    if verbose:
        print(f"[seed {seed}] aligned={m['aligned']}", flush=True)
        print(f"          belief[u][t]:", flush=True)
        for u in range(K):
            print(f"            u={u}({UTTS[u]:>4}): " + " ".join(f"{belief[u][t]:.3f}" for t in range(K)), flush=True)
        print(f"          S[t][u] (plain coincidence):", flush=True)
        for t in range(K):
            print(f"            t={t}({STATES[t]:>4}): " + " ".join(f"{S[t][u]:.4f}" for u in range(K)), flush=True)
        print(f"          Sprag[t][u] (distinctiveness):", flush=True)
        for t in range(K):
            print(f"            t={t}({STATES[t]:>4}): " + " ".join(f"{Sp[t][u]:+.4f}" for u in range(K)), flush=True)
        print(f"    succ_opt_plain={m['succ_opt_plain']} agree_vs_aligned={m['agree_plain_vs_aligned']}/3", flush=True)
        print(f"    succ_opt_prag ={m['succ_opt_prag']} agree_vs_aligned={m['agree_prag_vs_aligned']}/3 "
              f"(constant={m['prag_is_constant']})", flush=True)
        print(f"    succ_opt_div  ={m['succ_opt_div']} agree_vs_aligned={m['agree_div_vs_aligned']}/3", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--json", type=str, default="research/findings/raw/_pragmatic_success/distinctiveness_step1.json")
    args = ap.parse_args()

    t0 = time.time()
    per_seed = [evaluate_seed(s) for s in args.seeds]
    tot = 3 * len(args.seeds)
    agg = {
        "n_contexts": tot,
        "agree_plain_vs_aligned": sum(r["agree_plain_vs_aligned"] for r in per_seed),
        "agree_prag_vs_aligned": sum(r["agree_prag_vs_aligned"] for r in per_seed),
        "agree_div_vs_aligned": sum(r["agree_div_vs_aligned"] for r in per_seed),
        "n_seeds_prag_degenerate": sum(int(r["prag_is_constant"]) for r in per_seed),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    print("\n" + "=" * 90, flush=True)
    print(f"[step1] PLAIN  succ_opt==aligned : {agg['agree_plain_vs_aligned']}/{tot}", flush=True)
    print(f"[step1] PRAG   succ_opt==aligned : {agg['agree_prag_vs_aligned']}/{tot}  "
          f"(distinctiveness contrast over intents)", flush=True)
    print(f"[step1] DIV    succ_opt==aligned : {agg['agree_div_vs_aligned']}/{tot}  (divisive-norm variant)", flush=True)
    print(f"[step1] prag degenerate (constant-utterance) seeds: {agg['n_seeds_prag_degenerate']}/{len(args.seeds)}",
          flush=True)
    out = {"runner": Path(__file__).stem, "seeds": args.seeds, "aggregate": agg, "per_seed": per_seed}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=1))
    print(f"[step1] wrote {args.json} ({agg['elapsed_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
