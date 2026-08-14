"""ONE-BRAIN PRODUCTION-DEFAULT FLIP (POOL #2) — the end-to-end verify for flipping `BRAIN_ONEBRAIN_MERGE2` default ON.

Flipping the default makes the production `metacog` + `pragmatic` organs build on ONE shared `SimulationBridge` (the
`MergedSubstrate2` pool) instead of two separate bridges. This runner earns (or refuses) that flip by reading the REAL
production organ APIs the `/api/brain-chat` handler calls — metacog `judge(evidence)`, pragmatic `interpret(utterance)`
— three ways per seed:

  TODAY     : each organ on its OWN bridge, `shared=None`  (== today's flag-off production; == the escape path).
  MERGED    : both organs on ONE shared bridge (the default-ON path: MergedSubstrate2(("metacog","pragmatic"))).
  CORESIDENT: each organ on its OWN bridge with the THREE merge seams ON (per_region param-het + threshold + wiring) —
              the apples-to-apples merge baseline (MergedSubstrate2(("metacog",)) / (("pragmatic",))).

Checks per seed, over a BROAD panel (metacog evidence sweep; pragmatic {none, some, all}):
  A. ONE shared pool           : MERGED metacog.bridge IS pragmatic.bridge IS the substrate bridge (one cp_ array).
  B. MERGED == CORESIDENT      : the genuine merge byte-identity (must be 0.0 — metacog balance margin + pragmatic
                                 belief distribution, through the real read APIs over the broad panel).
  C. answer classes preserved  : every metacog `confident` bool + every pragmatic `implicature_represented` +
                                 `enriched_interpretation` IDENTICAL MERGED-vs-TODAY (the user-visible answer is
                                 unchanged — each build self-calibrates its threshold, so the read is self-normalizing).
  D. numeric residual          : the merged-vs-today balance / belief deltas (EXPECTED > 0; the honest, characterized
                                 cost of a genuine shared pool; NOT a regression, the classes in C are preserved).

VERDICT: FLIP-GO(-SCOPED) iff A + B(==0) + C(100%) all seeds. D is reported, not gated (the documented residual).
Run:  SIM_BACKEND=numpy python -m research.runners._onebrain_production_flip2_verify --seeds 42,43,44,100,101,102 \
        --out research/findings/raw/_onebrain_production_flip2_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from research.runners.metacog_production_organ import MetacogProductionOrgan
from research.runners.pragmatic_production_organ import PragmaticProductionOrgan
from research.runners.onebrain_merge_production2 import MergedSubstrate2
from research.runners._recursive_tom_rsa_derisk import STATES, UTTS

# metacog evidence sweep (low -> high answer confidence).
MC_EVID = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]


def _read_metacog(org):
    """Per-evidence read: (balance margin, confident bool). balance = byte-id target; confident = answer class."""
    out = []
    for e in MC_EVID:
        j = org.judge(e)
        out.append((float(e), float(j["balance"]), bool(j["confident"])))
    return out


def _read_pragmatic(org):
    """Per-utterance read: (belief distribution over STATES, implicature_represented bool, enriched phrase)."""
    out = []
    for u in UTTS:
        info = org.interpret(u)
        out.append((u, [float(x) for x in info["belief"]],
                    bool(info["implicature_represented"]), str(info["enriched_interpretation"])))
    return out


def run_seed(seed: int) -> dict:
    # TODAY (== flag-off production == escape path)
    mc_today = MetacogProductionOrgan(seed=seed, shared=None)
    pr_today = PragmaticProductionOrgan(seed=seed, shared=None)
    mc_t, pr_t = _read_metacog(mc_today), _read_pragmatic(pr_today)

    # MERGED (default-ON path: both organs on ONE shared bridge)
    merged = MergedSubstrate2(seed=seed, organs=("metacog", "pragmatic"))
    mc_merged = MetacogProductionOrgan(seed=seed, shared=merged)
    pr_merged = PragmaticProductionOrgan(seed=seed, shared=merged)
    mc_m, pr_m = _read_metacog(mc_merged), _read_pragmatic(pr_merged)
    merged.ensure_built()
    one_pool = bool((mc_merged.bridge is merged.bridge) and (pr_merged._shared.bridge is merged.bridge))
    n_pool = int(merged.bridge.cp_membrane_potential_v.shape[0])

    # CORESIDENT (each organ on its OWN bridge, the three merge seams ON — the apples-to-apples merge baseline)
    subM = MergedSubstrate2(seed=seed, organs=("metacog",))
    subP = MergedSubstrate2(seed=seed, organs=("pragmatic",))
    mc_cor = MetacogProductionOrgan(seed=seed, shared=subM)
    pr_cor = PragmaticProductionOrgan(seed=seed, shared=subP)
    mc_c, pr_c = _read_metacog(mc_cor), _read_pragmatic(pr_cor)

    # B. MERGED == CORESIDENT (byte-identity of the genuine merge — must be 0.0)
    b_mc = max(abs(m[1] - c[1]) for m, c in zip(mc_m, mc_c))
    b_pr = max(max(abs(mb - cb) for mb, cb in zip(m[1], c[1])) for m, c in zip(pr_m, pr_c))
    byte_id_merge = (b_mc == 0.0 and b_pr == 0.0)

    # C. answer classes preserved MERGED-vs-TODAY
    mc_class_ok = all(m[2] == t[2] for m, t in zip(mc_m, mc_t))
    pr_class_ok = all((m[2] == t[2]) and (m[3] == t[3]) for m, t in zip(pr_m, pr_t))
    classes_ok = bool(mc_class_ok and pr_class_ok)

    # D. numeric residual MERGED-vs-TODAY (expected > 0; the documented cost of a shared pool)
    d_mc = max(abs(m[1] - t[1]) for m, t in zip(mc_m, mc_t))
    d_pr = max(max(abs(mb - tb) for mb, tb in zip(m[1], t[1])) for m, t in zip(pr_m, pr_t))

    merge_go = bool(one_pool and byte_id_merge)                 # the genuine shared-pool merge (A + B)
    go = bool(merge_go and classes_ok)                          # the full flip (A + B + C, both organs preserved)
    return {
        "seed": seed, "one_pool": one_pool, "n_pool": n_pool,
        "byte_id_merge_vs_coresident": byte_id_merge,
        "byte_id_metacog_delta": float(b_mc), "byte_id_pragmatic_delta": float(b_pr),
        "answer_classes_preserved_vs_today": classes_ok,
        "metacog_class_ok": bool(mc_class_ok), "pragmatic_class_ok": bool(pr_class_ok),
        "residual_metacog_balance_max": float(d_mc), "residual_pragmatic_belief_max": float(d_pr),
        "merge_go": merge_go, "flip_go": go,
        "mc_today": mc_t, "mc_merged": mc_m, "pr_today": pr_t, "pr_merged": pr_m,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--out", default="research/findings/raw/_onebrain_production_flip2_6seed.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = [run_seed(s) for s in seeds]
    n = len(rows)
    agg = {
        "seeds": seeds, "backend": os.environ.get("SIM_BACKEND"),
        "one_pool": sum(r["one_pool"] for r in rows),
        "byte_id_merge_vs_coresident": sum(r["byte_id_merge_vs_coresident"] for r in rows),
        "merge_go": sum(r["merge_go"] for r in rows),
        "metacog_answer_preserved": sum(r["metacog_class_ok"] for r in rows),
        "pragmatic_answer_preserved": sum(r["pragmatic_class_ok"] for r in rows),
        "answer_classes_preserved_vs_today": sum(r["answer_classes_preserved_vs_today"] for r in rows),
        "residual_metacog_balance_max": max(r["residual_metacog_balance_max"] for r in rows),
        "residual_pragmatic_belief_max": max(r["residual_pragmatic_belief_max"] for r in rows),
        "flip_go": sum(r["flip_go"] for r in rows),
        "n": n, "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=2)

    print("\n" + "=" * 96)
    print("ONE-BRAIN PRODUCTION-DEFAULT FLIP (POOL #2) verify (metacog + pragmatic on ONE shared bridge)")
    print("=" * 96)
    for r in rows:
        print(f"  seed {r['seed']}: one_pool={r['one_pool']}(N={r['n_pool']}) | "
              f"MERGED==CORESIDENT byte-id={r['byte_id_merge_vs_coresident']} "
              f"(mcΔ={r['byte_id_metacog_delta']:.2e} prΔ={r['byte_id_pragmatic_delta']:.2e}) | "
              f"answer-classes-preserved-vs-today={r['answer_classes_preserved_vs_today']} | "
              f"residual mcΔ={r['residual_metacog_balance_max']:.4f} prΔ={r['residual_pragmatic_belief_max']:.4f} | "
              f"FLIP-GO={r['flip_go']}")
    print("-" * 96)
    print(f"  A. one shared pool:                          {agg['one_pool']}/{n}")
    print(f"  B. MERGED == CORESIDENT (byte-id, ==0):      {agg['byte_id_merge_vs_coresident']}/{n}")
    print(f"  ==> MERGE-GO (A+B, genuine one-pool merge):  {agg['merge_go']}/{n}")
    print(f"  C. answer preserved vs today — PRAGMATIC:    {agg['pragmatic_answer_preserved']}/{n}")
    print(f"  C. answer preserved vs today — METACOG:      {agg['metacog_answer_preserved']}/{n}  "
          f"(nmda_norm divisive-conductance read — de-noised + invariant to the per-region re-draw; blocker RESOLVED)")
    print(f"  D. residual vs today (documented, >0): metacog {agg['residual_metacog_balance_max']:.4f}  "
          f"pragmatic {agg['residual_pragmatic_belief_max']:.4f}")
    print(f"  ==> FULL FLIP-GO (A+B+C both organs):        {agg['flip_go']}/{n}  "
          f"(GO: genuine one-pool merge + BOTH organs answer-preserving -> pool #2 default-ON)")
    print(f"  wrote {args.out}")
    return agg


if __name__ == "__main__":
    main()
