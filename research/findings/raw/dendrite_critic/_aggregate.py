"""Aggregate the dendrite-critic deploy arms.

Score = sum over phases of final_quarter_mean_distance (the canonical nav metric,
reused verbatim from nav_gate2a_aggregate.score_from_data). Lower = better.
"""
import json
import os


def score_from_data(data: dict) -> float:
    """Nav score = sum over phases of final_quarter_mean_distance (lower=better).

    Identical to research.runners.nav_gate2a_aggregate.score_from_data (inlined to
    avoid the package-import path dependency). Raises if phase_stats is missing so a
    bad file is never silently scored 0 (= a fake-perfect result).
    """
    phases = data.get("phase_stats")
    if not phases:
        raise ValueError("run data has no non-empty 'phase_stats'")
    total = 0.0
    for p in phases:
        val = p.get("final_quarter_mean_distance")
        if val is None:
            raise ValueError(f"phase {p.get('phase')} missing 'final_quarter_mean_distance'")
        total += float(val)
    return total


RAW = os.path.dirname(__file__)
SEEDS = [42, 43, 44]
ARMS = ["dendcritic", "baseline", "lesion"]


def load(arm, seed):
    p = os.path.join(RAW, f"{arm}_seed{seed}.json")
    if not os.path.exists(p):
        return None, "missing"
    try:
        d = json.load(open(p))
        return score_from_data(d), None
    except Exception as e:  # noqa: BLE001
        return None, f"error: {e}"


def main():
    table = {}
    for arm in ARMS:
        scores = []
        for s in SEEDS:
            sc, err = load(arm, s)
            table.setdefault(arm, {})[s] = sc if err is None else err
            if err is None:
                scores.append(sc)
        table[arm]["_mean"] = (sum(scores) / len(scores)) if scores else None
        table[arm]["_n"] = len(scores)

    print("=== DENDRITE-CRITIC DEPLOY: nav score (sum of phase final-quarter mean distances; lower=better) ===")
    hdr = f"{'seed':>6} | {'dendcritic':>12} | {'baseline':>12} | {'lesion(str=0)':>14}"
    print(hdr)
    print("-" * len(hdr))
    for s in SEEDS:
        def fmt(arm):
            v = table[arm][s]
            return f"{v:12.3f}" if isinstance(v, float) else f"{str(v):>12}"
        print(f"{s:>6} | {fmt('dendcritic')} | {fmt('baseline')} | {fmt('lesion'):>14}")
    print("-" * len(hdr))

    def m(arm):
        v = table[arm]["_mean"]
        return f"{v:.3f} (n={table[arm]['_n']})" if v is not None else "n/a"
    print(f"  mean: dendcritic={m('dendcritic')}  baseline={m('baseline')}  lesion={m('lesion')}")

    dc, bl, le = table["dendcritic"]["_mean"], table["baseline"]["_mean"], table["lesion"]["_mean"]
    if dc is not None and bl is not None:
        ratio = dc / bl if bl else float("inf")
        gap_pct = 100.0 * (dc - bl) / bl if bl else float("inf")
        print(f"\n  dendrite/baseline ratio = {ratio:.3f}  (gap {gap_pct:+.1f}%; deploy GATE: within ~25% i.e. ratio<=1.25)")
        print(f"  GATE: {'PASS' if ratio <= 1.25 else 'CHARACTERIZE-GAP'}")
    if dc is not None and le is not None:
        print(f"  lesion vs dendrite: lesion={le:.3f} vs dendrite={dc:.3f}  "
              f"(lesion {'WORSE (plateau load-bearing)' if le > dc * 1.10 else 'NOT clearly worse'})")

    json.dump(table, open(os.path.join(RAW, "_aggregate.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
