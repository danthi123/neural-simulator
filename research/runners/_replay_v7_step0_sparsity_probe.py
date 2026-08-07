"""STEP 0 closability gate for replay-consolidation self-calibration (v7).

Spec: ``research/findings/raw/_replay_selfcalibration_scoping.md``.

The v6 order-STDP consolidation is per-seed GO on calibration 412/413 but MULTISEED
NO-GO on development 414/415/410 (retest false-recall 0.46-0.50 vs 0.15 ceiling)
because its interference-control operating point is a VECTOR of absolute-unit gains
frozen on 2 seeds -- it cannot track the per-brain E/I / competition regime. The
proposed surpass is an emergent homeostatic integral controller to a LABEL-FREE
WTA-sparsity REGIME set-point ``S*``.

STEP 0 decides closability BEFORE building any controller: instrument the FROZEN v6
runner (byte-identical mechanism/config) to log, per development seed, a label-free
sparsity statistic ``S`` of the ``cortical_target`` population at retest, ALONGSIDE
the scored false-recall. If a monotone relation holds across the seeds (a
concentrated / one-winner regime => low false-recall), the set-point EXISTS and a
controller can target it -> proceed to Step 1. If false-recall stays high where
``S`` looks one-winner, a single label-free regime statistic is insufficient to set
the retest competition point -> STOP; the boundary is precisely named and no
controller is built.

``S`` is computed ONLY from the raw per-neuron spike-count vector over
``cortical_target`` and the STRUCTURAL assembly SIZE -- never the assembly identity,
the seed, the correct/wrong labels, or the false-recall metric (asserted below).

    SIM_BACKEND=numpy .venv/bin/python -m research.runners._replay_v7_step0_sparsity_probe \
        --out research/findings/raw/replay_v5_sfa_order/replay_v7_step0_dev.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners import _replay_cortical_consolidation_gate_v6_order_stdp as v6  # noqa: E402

# Candidate label-free statistics and the SIGN of their expected correlation with
# false-recall under the "one-winner regime => low false-recall" hypothesis.
#   pr_eff  : effective # active target neurons; LOW=>one-winner   => +corr with false
#   pr_frac : pr_eff / N                                            => +corr with false
#   gini    : concentration; HIGH=>one-winner                       => -corr with false
#   top_assembly_conc: top-assembly spike share; HIGH=>one-winner   => -corr with false
#   active_fraction: fraction firing; HIGH=>diffuse                 => +corr with false
CANDIDATE_SIGN = {
    "pr_eff": +1.0,
    "pr_frac": +1.0,
    "gini": -1.0,
    "top_assembly_conc": -1.0,
    "active_fraction": +1.0,
}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (NaN-safe, ties via average rank)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")

    def _rank(v):
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(v) + 1, dtype=np.float64)
        # average ties
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        return (sums / counts)[inv]

    rx, ry = _rank(x), _rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def run_step0(out_path: Path | None) -> dict:
    started = time.time()
    cfg = v6.GateConfig()  # FROZEN v6 config -- nothing tuned
    seeds = tuple(v6.DEVELOPMENT_SEEDS)  # 414, 415, 410

    # Anti-cheat assertion: the statistic must not have access to labels. We verify
    # structurally that _label_free_sparsity's signature takes only counts + size.
    import inspect

    from research.runners import _replay_cortical_consolidation_gate as v1

    sig = list(inspect.signature(v1._label_free_sparsity).parameters)
    assert sig == ["counts", "assembly_size"], (
        f"label-free S must depend ONLY on (counts, assembly_size); got {sig}"
    )

    per_probe: list[dict] = []
    per_seed: list[dict] = []
    for seed in seeds:
        # Frozen v6 mechanism, INTACT condition (the scored operating point).
        row = v6.run_condition(seed, "intact", cfg)
        recall = row["recall"]
        seed_false = []
        seed_S = {k: [] for k in CANDIDATE_SIGN}
        for mem in ("A", "B"):
            r = recall[mem]
            S = r["sparsity_S"]
            false = float(r["false_recall_fraction"])
            seed_false.append(false)
            for k in CANDIDATE_SIGN:
                seed_S[k].append(float(S[k]))
            per_probe.append({
                "seed": int(seed),
                "memory": mem,
                "false_recall": false,
                "correct_rate": float(r["correct_rate"]),
                "wrong_rate": float(r["wrong_rate"]),
                "total_target_spikes": int(r["total_target_spikes"]),
                **{f"S_{k}": float(S[k]) for k in CANDIDATE_SIGN},
            })
        per_seed.append({
            "seed": int(seed),
            "false_recall_mean": float(np.mean(seed_false)),
            **{f"S_{k}_mean": float(np.mean(seed_S[k])) for k in CANDIDATE_SIGN},
        })

    # Quantify the monotone relation over ALL probe points (6 = 3 seeds x 2 memories)
    # and over the seed-MEANS (3), which is the granularity a per-seed controller sets.
    false_probe = np.array([p["false_recall"] for p in per_probe])
    false_seed = np.array([s["false_recall_mean"] for s in per_seed])
    analysis = {}
    for k, sign in CANDIDATE_SIGN.items():
        s_probe = np.array([p[f"S_{k}"] for p in per_probe])
        s_seed = np.array([s[f"S_{k}_mean"] for s in per_seed])
        rho_probe = _spearman(s_probe, false_probe)
        rho_seed = _spearman(s_seed, false_seed)
        # "Aligned" == correlation has the hypothesised sign AND is strong.
        aligned_probe = (
            np.isfinite(rho_probe) and np.sign(rho_probe) == sign and abs(rho_probe) >= 0.8
        )
        # Seed-mean monotone ordering in the hypothesised direction (3 seeds).
        order_false = np.argsort(false_seed)
        s_in_false_order = s_seed[order_false]
        seed_monotone = (
            np.all(np.diff(s_in_false_order) > 0) if sign > 0
            else np.all(np.diff(s_in_false_order) < 0)
        )
        analysis[k] = {
            "expected_sign": sign,
            "rho_probe_points": None if not np.isfinite(rho_probe) else round(rho_probe, 4),
            "rho_seed_means": None if not np.isfinite(rho_seed) else round(rho_seed, 4),
            "aligned_probe_points": bool(aligned_probe),
            "seed_mean_monotone": bool(seed_monotone),
        }

    # Boundary detector: a seed whose regime looks ONE-WINNER (concentrated) yet has
    # HIGH false-recall falsifies "S sets the point". Use gini (concentration) and
    # top_assembly_conc against the frozen 0.15 ceiling.
    gini_seed = np.array([s["S_gini_mean"] for s in per_seed])
    top_seed = np.array([s["S_top_assembly_conc_mean"] for s in per_seed])
    contradictions = []
    for i, s in enumerate(per_seed):
        looks_one_winner = (gini_seed[i] >= np.median(gini_seed)) and (
            top_seed[i] >= np.median(top_seed)
        )
        if looks_one_winner and false_seed[i] > 0.15:
            contradictions.append({
                "seed": s["seed"],
                "false_recall_mean": s["false_recall_mean"],
                "gini_mean": float(gini_seed[i]),
                "top_assembly_conc_mean": float(top_seed[i]),
                "note": "concentrated/one-winner regime yet false-recall > 0.15 ceiling",
            })

    # STEP-0 VERDICT. The set-point exists iff at least one label-free candidate is
    # BOTH strongly correlated on the probe points (correct sign) AND monotone across
    # the seed means, with NO concentrated-but-high-false contradiction.
    passing = [
        k for k, a in analysis.items()
        if a["aligned_probe_points"] and a["seed_mean_monotone"]
    ]
    step0_pass = bool(passing) and not contradictions
    if step0_pass:
        verdict = "STEP0_PASS_SETPOINT_EXISTS"
        message = (
            "A label-free regime statistic predicts false-recall across the dev seeds "
            f"(candidates: {passing}); a set-point S* exists -> proceed to Step 1."
        )
    else:
        verdict = "STEP0_NOGO_SINGLE_STATISTIC_INSUFFICIENT"
        message = (
            "A single label-free regime statistic is insufficient to set the retest "
            "competition point across development seeds. No monotone candidate "
            f"survives (passing={passing}, contradictions={len(contradictions)}). "
            "STOP: do not build the controller; the boundary is named."
        )

    payload = {
        "gate": "replay_v7_step0_sparsity_setpoint",
        "spec": "research/findings/raw/_replay_selfcalibration_scoping.md",
        "backend": "numpy",
        "frozen_mechanism": "v6 order-STDP consolidation (byte-identical; INTACT condition)",
        "seeds": list(seeds),
        "false_recall_ceiling": 0.15,
        "candidate_sign": CANDIDATE_SIGN,
        "per_probe_points": per_probe,
        "per_seed": per_seed,
        "analysis": analysis,
        "concentrated_but_high_false_contradictions": contradictions,
        "passing_candidates": passing,
        "step0_pass": step0_pass,
        "verdict": verdict,
        "message": message,
        "elapsed_seconds": time.time() - started,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    payload = run_step0(args.out)
    print(f"[v7 STEP-0] {payload['verdict']}  pass={payload['step0_pass']}")
    print(f"  {payload['message']}")
    print("  per-seed (false-recall mean | S candidates):")
    for s in payload["per_seed"]:
        print(
            f"    seed {s['seed']}: false={s['false_recall_mean']:.4f} "
            f"pr_eff={s['S_pr_eff_mean']:.3f} pr_frac={s['S_pr_frac_mean']:.3f} "
            f"gini={s['S_gini_mean']:.3f} top_conc={s['S_top_assembly_conc_mean']:.3f} "
            f"active={s['S_active_fraction_mean']:.3f}"
        )
    print("  monotone analysis (rho on 6 probe pts | rho on 3 seed means | aligned | seed-monotone):")
    for k, a in payload["analysis"].items():
        print(
            f"    {k:>18}: rho_probe={a['rho_probe_points']} rho_seed={a['rho_seed_means']} "
            f"aligned={a['aligned_probe_points']} seed_monotone={a['seed_mean_monotone']}"
        )
    if payload["concentrated_but_high_false_contradictions"]:
        print("  CONTRADICTIONS (concentrated regime yet false>0.15):")
        for c in payload["concentrated_but_high_false_contradictions"]:
            print(f"    seed {c['seed']}: false={c['false_recall_mean']:.4f} "
                  f"gini={c['gini_mean']:.3f} top_conc={c['top_assembly_conc_mean']:.3f}")


if __name__ == "__main__":
    main()
