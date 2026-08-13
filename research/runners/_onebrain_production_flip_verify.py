"""ONE-BRAIN PRODUCTION-DEFAULT FLIP — the end-to-end verify for flipping `BRAIN_ONEBRAIN_MERGE` default ON.

Flipping the default makes the production `surprise` + `world-model` organs build on ONE shared `SimulationBridge`
(the rung-1 `MergedSubstrate`) instead of two separate bridges. This runner earns (or refuses) that flip by reading
the REAL production organ APIs the `/api/brain-chat` handler calls — surprise `judge()`, world-model `expectation()`
+ `read_surprise()` — three ways per seed:

  TODAY     : each organ on its OWN bridge, `shared=None`  (== today's flag-off production; == the escape path).
  MERGED    : both organs on ONE shared bridge (the default-ON path: `MergedSubstrate(("surprise","worldmodel"))`).
  CORESIDENT: each organ on its OWN bridge with the TWO merge flags ON (per_region threshold + homeostasis) —
              the apples-to-apples merge baseline (`MergedSubstrate(("surprise",))` / `(("worldmodel",))`).

Checks per seed, over a BROAD query panel (each organ's turn classes):
  A. ONE shared pool           : MERGED surprise.bridge IS worldmodel.bridge IS the substrate bridge (one cp_ array).
  B. MERGED == CORESIDENT      : the genuine merge byte-identity (must be 0.0 — this is what rung-1 proved, here through
                                 the full read APIs over the broad panel).
  C. answer classes preserved  : every surprise `surprised` bool + every world-model `pred_sign` is IDENTICAL
                                 MERGED-vs-TODAY (the user-visible answer is unchanged — each build calibrates its own
                                 threshold, so the read is self-normalizing under the shared-pool threshold re-draw).
  D. numeric residual          : the merged-vs-today Hz/margin deltas (EXPECTED > 0; the honest, characterized cost of
                                 a genuine shared pool — the shared global RNG cannot reproduce BOTH organs' standalone
                                 threshold draws; NOT a regression, the classes in C are preserved).

VERDICT: FLIP-GO(-SCOPED) iff A + B(==0) + C(100%) all seeds. D is reported, not gated (it is the documented residual).
Run:  SIM_BACKEND=numpy python -m research.runners._onebrain_production_flip_verify --seeds 42,43,44,100,101,102 \
        --out research/findings/raw/_onebrain_production_flip_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from research.runners.surprise_production_organ import SurpriseProductionOrgan
from research.runners.worldmodel_production_organ import WorldModelProductionOrgan
from research.runners.onebrain_merge_production import MergedSubstrate

# Broad surprise panel: (agent, action, stored_patient, asserted_patient, class). confirm = asserted==stored (cancel,
# ~0 Hz), surprise = a different asserted block (un-inhibited, fires). Several distinct concepts broaden the read.
SURP_PANEL = [
    ("dog", "chase", "cat", "cat", "confirm"),
    ("dog", "chase", "cat", "bone", "surprise"),
    ("cat", "eat", "fish", "fish", "confirm"),
    ("cat", "eat", "fish", "milk", "surprise"),
    ("bird", "sing", "song", "song", "confirm"),
    ("bird", "sing", "song", "worm", "surprise"),
    ("man", "read", "book", "book", "confirm"),
    ("man", "read", "book", "rock", "surprise"),
]
WM_EXP = [1, -1]                                  # expectation() context signs
WM_SURP = [(1, 1), (1, -1), (-1, -1), (-1, 1)]    # read_surprise() (context, observed): expected / violated


def _read_surprise_organ(org):
    """The surprise organ's per-turn read: (surprise_hz, surprised-bool) for each panel row, in order."""
    out = []
    for (a, v, ps, pa, cls) in SURP_PANEL:
        j = org.judge(a, v, ps, pa)
        out.append((cls, float(j["surprise_hz"]), bool(j["surprised"])))
    return out


def _read_wm_organ(org):
    """The world-model organ's per-turn read: expectation pred_sign/margin + read_surprise hz/surprised, in order."""
    out = []
    for cs in WM_EXP:
        e = org.expectation(cs)
        out.append(("exp", int(e["pred_sign"]), float(e["pred_margin"]),
                    float(e["pred_pos_rate"]), float(e["pred_neg_rate"])))
    for (cs, obs) in WM_SURP:
        s = org.read_surprise(cs, obs)
        out.append(("surp", None if s["surprised"] is None else bool(s["surprised"]),
                    float(s["surprise_hz"]), None, None))
    return out


def run_seed(seed: int) -> dict:
    # TODAY (== flag-off production == escape path)
    s_today = SurpriseProductionOrgan(seed=seed, shared=None)
    w_today = WorldModelProductionOrgan(seed=seed, shared=None)
    surp_today, wm_today = _read_surprise_organ(s_today), _read_wm_organ(w_today)

    # MERGED (default-ON path: both organs on ONE shared bridge)
    merged = MergedSubstrate(seed=seed, organs=("surprise", "worldmodel"))
    s_merged = SurpriseProductionOrgan(seed=seed, shared=merged)
    w_merged = WorldModelProductionOrgan(seed=seed, shared=merged)
    surp_merged, wm_merged = _read_surprise_organ(s_merged), _read_wm_organ(w_merged)
    merged.ensure_built()
    one_pool = bool((s_merged.bridge is merged.bridge) and (w_merged._st["bridge"] is merged.bridge))
    n_pool = int(merged.bridge.cp_membrane_potential_v.shape[0])

    # CORESIDENT (each organ on its OWN bridge, both merge flags ON — the apples-to-apples merge baseline)
    subS = MergedSubstrate(seed=seed, organs=("surprise",))
    subW = MergedSubstrate(seed=seed, organs=("worldmodel",))
    s_cor = SurpriseProductionOrgan(seed=seed, shared=subS)
    w_cor = WorldModelProductionOrgan(seed=seed, shared=subW)
    surp_cor, wm_cor = _read_surprise_organ(s_cor), _read_wm_organ(w_cor)

    # B. MERGED == CORESIDENT (byte-identity of the genuine merge — must be 0.0)
    b_surp = max(abs(m[1] - c[1]) for m, c in zip(surp_merged, surp_cor))
    b_wm = 0.0
    for m, c in zip(wm_merged, wm_cor):
        if m[0] == "exp":
            b_wm = max(b_wm, abs(m[2] - c[2]), abs(m[3] - c[3]), abs(m[4] - c[4]))
        else:
            b_wm = max(b_wm, abs(m[2] - c[2]))
    byte_id_merge = (b_surp == 0.0 and b_wm == 0.0)

    # C. answer classes preserved MERGED-vs-TODAY (surprised bools + pred_signs identical)
    surp_class_ok = all(m[2] == t[2] for m, t in zip(surp_merged, surp_today))
    wm_sign_ok = all((m[1] == t[1]) for m, t in zip(wm_merged, wm_today) if m[0] == "exp")
    wm_surp_class_ok = all((m[1] == t[1]) for m, t in zip(wm_merged, wm_today) if m[0] == "surp")
    classes_ok = bool(surp_class_ok and wm_sign_ok and wm_surp_class_ok)

    # D. numeric residual MERGED-vs-TODAY (expected > 0; the documented cost of a shared pool)
    d_surp = max(abs(m[1] - t[1]) for m, t in zip(surp_merged, surp_today))
    d_wm = 0.0
    for m, t in zip(wm_merged, wm_today):
        if m[0] == "exp":
            d_wm = max(d_wm, abs(m[2] - t[2]))
        else:
            d_wm = max(d_wm, abs(m[2] - t[2]))

    go = bool(one_pool and byte_id_merge and classes_ok)
    return {
        "seed": seed, "one_pool": one_pool, "n_pool": n_pool,
        "byte_id_merge_vs_coresident": byte_id_merge,
        "byte_id_surp_delta": float(b_surp), "byte_id_wm_delta": float(b_wm),
        "answer_classes_preserved_vs_today": classes_ok,
        "surp_class_ok": bool(surp_class_ok), "wm_sign_ok": bool(wm_sign_ok),
        "wm_surp_class_ok": bool(wm_surp_class_ok),
        "residual_surp_hz_max": float(d_surp), "residual_wm_margin_max": float(d_wm),
        "flip_go": go,
        "surp_today": surp_today, "surp_merged": surp_merged,
        "wm_today": wm_today, "wm_merged": wm_merged,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--out", default="research/findings/raw/_onebrain_production_flip_6seed.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = [run_seed(s) for s in seeds]
    n = len(rows)
    agg = {
        "seeds": seeds, "backend": os.environ.get("SIM_BACKEND"),
        "one_pool": sum(r["one_pool"] for r in rows),
        "byte_id_merge_vs_coresident": sum(r["byte_id_merge_vs_coresident"] for r in rows),
        "answer_classes_preserved_vs_today": sum(r["answer_classes_preserved_vs_today"] for r in rows),
        "residual_surp_hz_max": max(r["residual_surp_hz_max"] for r in rows),
        "residual_wm_margin_max": max(r["residual_wm_margin_max"] for r in rows),
        "flip_go": sum(r["flip_go"] for r in rows),
        "n": n, "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=2)

    print("\n" + "=" * 96)
    print("ONE-BRAIN PRODUCTION-DEFAULT FLIP verify (surprise + world-model on ONE shared bridge)")
    print("=" * 96)
    for r in rows:
        print(f"  seed {r['seed']}: one_pool={r['one_pool']}(N={r['n_pool']}) | "
              f"MERGED==CORESIDENT byte-id={r['byte_id_merge_vs_coresident']} "
              f"(surpΔ={r['byte_id_surp_delta']:.2e} wmΔ={r['byte_id_wm_delta']:.2e}) | "
              f"answer-classes-preserved-vs-today={r['answer_classes_preserved_vs_today']} | "
              f"residual surpΔ={r['residual_surp_hz_max']:.3f}Hz wmΔ={r['residual_wm_margin_max']:.1f} | "
              f"FLIP-GO={r['flip_go']}")
    print("-" * 96)
    print(f"  A. one shared pool:                         {agg['one_pool']}/{n}")
    print(f"  B. MERGED == CORESIDENT (byte-id, ==0):     {agg['byte_id_merge_vs_coresident']}/{n}")
    print(f"  C. answer classes preserved vs today:       {agg['answer_classes_preserved_vs_today']}/{n}")
    print(f"  D. residual vs today (documented, >0): surp {agg['residual_surp_hz_max']:.3f}Hz  "
          f"wm {agg['residual_wm_margin_max']:.1f}")
    print(f"  ==> FLIP-GO (A+B+C):                        {agg['flip_go']}/{n}")
    print(f"  wrote {args.out}")
    return agg


if __name__ == "__main__":
    main()
