"""DE-RISK — Hebbian SELF-ORGANIZATION of the #3E plausibility gate's synapse weights (the "next rung" named
by 2026-09-01-plausibility-ensemble-read-host-parity-generation-all6-default-on-GO: "the synaptic weights
are still SET from the co-occurrence counts... online Hebbian self-organization of those weights remains the
next rung"). Measures whether `HebbianAssociativePlausibilityOrgan` (research/runners/hebbian_plausibility_
organ.py) reaches host `P>=tau` agreement by GROWING its cortex_ctx->dlpfc_wm synapses through REPLAY (co-
firing each stored fact's role concepts) instead of injecting a host-computed `P*gain` matrix.

Runs on a small SYNTHETIC own-facts graph (no ChatBrain -- the organ only needs P/row/facts/vocab), so this
is cheap enough to run multi-seed locally. NO sim/ edit; reuse-by-import.

  SIM_BACKEND=numpy python -u -m research.runners._hebbian_plausibility_derisk \
      --seeds 42,43,44,100,101,102 --out research/findings/raw/_hebbian_plausibility_derisk.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.hebbian_plausibility_organ import HebbianAssociativePlausibilityOrgan  # noqa: E402

FACTS = [
    ("dog", "chase", "cat"),
    ("cat", "eat", "fish"),
    ("dog", "eat", "bone"),
    ("cow", "eat", "grass"),
]


def _graph(facts):
    vocab = sorted({w for f in facts for w in f})
    row = {w: i for i, w in enumerate(vocab)}
    P = np.zeros((len(vocab), len(vocab)))
    for a, v, p in facts:
        for x in (a, v, p):
            for y in (a, v, p):
                if x != y:
                    P[row[x], row[y]] += 1.0
    return vocab, row, P


def run_seed(seed, a):
    vocab, row, P = _graph(FACTS)
    tau = float(np.percentile(P[P > 0], 50.0))
    t0 = time.time()
    org = HebbianAssociativePlausibilityOrgan(
        P, row, FACTS, vocab=vocab, seed=seed, pattern_size=a.pattern_size, n_ensemble=a.n_ensemble,
        inter_density=a.inter_density, replay_cycles=a.replay_cycles,
        hebbian_learning_rate=a.hebbian_learning_rate, hebbian_max_weight=a.hebbian_max_weight)
    build_s = time.time() - t0
    agree = org.agreement_with_host(P, row, tau)

    abl = HebbianAssociativePlausibilityOrgan(
        P, row, FACTS, vocab=vocab, seed=seed, pattern_size=a.pattern_size, n_ensemble=a.n_ensemble,
        inter_density=a.inter_density, replay_cycles=a.replay_cycles,
        hebbian_learning_rate=a.hebbian_learning_rate, hebbian_max_weight=a.hebbian_max_weight,
        lesion="ablate")
    n_related_ablate = sum(1 for x in vocab for y in vocab if x != y and abl.related(x, y))

    shuf = HebbianAssociativePlausibilityOrgan(
        P, row, FACTS, vocab=vocab, seed=seed, pattern_size=a.pattern_size, n_ensemble=a.n_ensemble,
        inter_density=a.inter_density, replay_cycles=a.replay_cycles,
        hebbian_learning_rate=a.hebbian_learning_rate, hebbian_max_weight=a.hebbian_max_weight,
        lesion="shuffle")
    shuf_agree = shuf.agreement_with_host(P, row, tau)

    row_out = {
        "seed": seed, "build_s": round(build_s, 1), "vocab_size": len(vocab),
        "n_replay_events": int(org.n_replay_events),
        "agreement": float(agree["agreement"]), "f1": float(agree["f1"]),
        "precision": float(agree["precision"]), "recall": float(agree["recall"]),
        "spk_related": int(agree["spk_related"]), "host_related": int(agree["host_related"]),
        "lesion_ablate_n_related": int(n_related_ablate),
        "lesion_shuffle_agreement": float(shuf_agree["agreement"]),
        "shuffle_below_intact": bool(shuf_agree["agreement"] < agree["agreement"]),
    }
    print(f"[hebbian seed {seed}] build={build_s:.1f}s replay_events={org.n_replay_events} "
          f"agreement={agree['agreement']:.3f} f1={agree['f1']:.3f} precision={agree['precision']:.3f} "
          f"recall={agree['recall']:.3f} | ablate_related={n_related_ablate} (want 0) | "
          f"shuffle_agreement={shuf_agree['agreement']:.3f} (want < {agree['agreement']:.3f})", flush=True)
    return row_out


def decide(rows, a):
    agree = np.array([r["agreement"] for r in rows])
    ablate = np.array([r["lesion_ablate_n_related"] for r in rows])
    shuf_below = np.array([r["shuffle_below_intact"] for r in rows])
    detail = {
        "agreement_mean": float(agree.mean()), "agreement_min": float(agree.min()),
        "ablate_related_total": int(ablate.sum()),
        "shuffle_below_intact_all_seeds": bool(np.all(shuf_below)),
        "agreement_bar": a.agreement_bar,
    }
    lesion_ok = bool(np.all(ablate == 0) and np.all(shuf_below))
    parity_ok = bool(np.all(agree >= a.agreement_bar))
    if parity_ok and lesion_ok:
        verdict = "GO"
    elif lesion_ok:
        verdict = "QUALIFIED_mechanism_load_bearing_below_host_parity"
    else:
        verdict = "HONEST_NEGATIVE_not_lesion_load_bearing"
    return verdict, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--pattern_size", type=int, default=8)
    ap.add_argument("--n_ensemble", type=int, default=4)
    ap.add_argument("--inter_density", type=float, default=0.08)
    ap.add_argument("--replay_cycles", type=int, default=100)
    ap.add_argument("--hebbian_learning_rate", type=float, default=0.4)
    ap.add_argument("--hebbian_max_weight", type=float, default=60.0)
    ap.add_argument("--agreement_bar", type=float, default=0.95)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    rows = [run_seed(s, a) for s in seeds]
    verdict, detail = decide(rows, a)
    result = {"status": verdict, "config": vars(a), "rows": rows, "detail": detail}
    print(f"\n=== VERDICT: {verdict} ===")
    print(json.dumps(detail, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
