"""6-seed A/B verdict: place-code NEURAL value critic vs STAGE-A host scaffold.

Acceptance = NEURAL summed final-quarter distance <= STAGE-A (no regression;
lower = better). Reports per-seed final-quarter distance for both conditions +
the summed-across-seeds totals + the accept/regress verdict.
"""
import json, glob, os, re

OUT = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 43, 44, 100, 101, 102]


def final_quarter(path):
    """Return the final-quarter MEAN distance (the comparable nav scalar)."""
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    q = d.get("mean_distance_quarters") or []
    fq = q[-1] if q else None
    extra = dict(
        snc_mean=(sum(d.get("snc_rate_log") or [0]) / max(1, len(d.get("snc_rate_log") or [1]))),
        wi=d.get("critic_weight_initial"), wf=d.get("critic_weight_final"),
        n_at_goal=d.get("n_steps_at_goal"), overall=d.get("mean_distance_overall"),
        window=d.get("enable_critic_window"),
    )
    return fq, extra


def main():
    print(f"{'seed':>6} | {'STAGE-A finalQ':>15} | {'NEURAL finalQ':>14} | {'NEURAL snc':>10} | {'NEURAL w0->wf':>20}")
    print("-" * 80)
    sa_sum = 0.0
    nu_sum = 0.0
    n_ok = 0
    rows = []
    for s in SEEDS:
        sa_fq, _ = final_quarter(os.path.join(OUT, f"_placecritic_stagea_s{s}.json"))
        nu_fq, nx = final_quarter(os.path.join(OUT, f"_placecritic_neural_s{s}.json"))
        w_str = (f"{nx['wi']:.3f}->{nx['wf']:.3f}" if (nx and nx['wi'] is not None and nx['wf'] is not None) else "n/a")
        snc_str = (f"{nx['snc_mean']:.2f}" if nx else "n/a")
        print(f"{s:>6} | {str(round(sa_fq,3)) if sa_fq is not None else 'MISSING':>15} | "
              f"{str(round(nu_fq,3)) if nu_fq is not None else 'MISSING':>14} | {snc_str:>10} | {w_str:>20}")
        if sa_fq is not None and nu_fq is not None:
            sa_sum += sa_fq
            nu_sum += nu_fq
            n_ok += 1
            rows.append((s, sa_fq, nu_fq))
    print("-" * 80)
    print(f"  seeds compared: {n_ok}/{len(SEEDS)}")
    print(f"  SUMMED final-quarter distance:  STAGE-A={sa_sum:.3f}   NEURAL={nu_sum:.3f}")
    if n_ok:
        delta = nu_sum - sa_sum
        per_seed_neural_better = sum(1 for _, a, n in rows if n <= a)
        print(f"  delta (NEURAL - STAGE-A) = {delta:+.3f}  (<=0 => NEURAL no worse => ACCEPT)")
        print(f"  per-seed NEURAL<=STAGE-A: {per_seed_neural_better}/{n_ok}")
        verdict = "ACCEPT (no regression)" if nu_sum <= sa_sum else "REGRESS (NEURAL worse)"
        print(f"\n  ==> A/B VERDICT: {verdict}")


if __name__ == "__main__":
    main()
