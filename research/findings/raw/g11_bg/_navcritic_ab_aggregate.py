"""Aggregate the 6-seed A/B (NEURAL value critic vs STAGE-A host scaffold).

Acceptance = NEURAL summed final-quarter distance <= STAGE-A (no nav regression). Also reports,
per NEURAL seed, whether the critic FIRED (striov_rate_log sum > 0) and LEARNED
(critic_weight_final > initial) -- the Stage-1 critic gate.

Usage: python _navcritic_ab_aggregate.py
"""
import json, os, glob

D = os.path.dirname(os.path.abspath(__file__))


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def fq(d):
    """Final-quarter mean distance (last entry of mean_distance_quarters)."""
    q = (d or {}).get("mean_distance_quarters") or []
    return q[-1] if q else None


def main():
    seeds = [42, 43, 44, 100, 101, 102]
    rows = []
    sumA = 0.0; sumN = 0.0; nA = 0; nN = 0
    for s in seeds:
        dA = load(f"{D}/_navcritic_stagea_s{s}.json")
        dN = load(f"{D}/_navcritic_neural_s{s}.json")
        fqA = fq(dA); fqN = fq(dN)
        # critic facts (NEURAL only)
        striov = (dN or {}).get("striov_rate_log") or []
        wi = (dN or {}).get("critic_weight_initial"); wf = (dN or {}).get("critic_weight_final")
        crit_fires = any(v != 0 for v in striov)
        crit_learns = (wi is not None and wf is not None and wf > wi)
        rows.append(dict(seed=s, fqA=fqA, fqN=fqN, crit_fires=crit_fires,
                         crit_learns=crit_learns, wi=wi, wf=wf,
                         striov_sum=sum(striov) if striov else 0))
        if fqA is not None: sumA += fqA; nA += 1
        if fqN is not None: sumN += fqN; nN += 1

    print(f"{'seed':>5} | {'STAGE-A fq':>11} | {'NEURAL fq':>10} | {'crit_fires':>10} | "
          f"{'crit_learns':>11} | {'w0->wf':>16} | striov_sum")
    print("-" * 95)
    for r in rows:
        fqA_s = f"{r['fqA']:.3f}" if r['fqA'] is not None else "MISSING"
        fqN_s = f"{r['fqN']:.3f}" if r['fqN'] is not None else "MISSING"
        w_s = (f"{r['wi']:.3f}->{r['wf']:.3f}" if (r['wi'] is not None and r['wf'] is not None)
               else "n/a")
        print(f"{r['seed']:>5} | {fqA_s:>11} | {fqN_s:>10} | {str(r['crit_fires']):>10} | "
              f"{str(r['crit_learns']):>11} | {w_s:>16} | {r['striov_sum']}")
    print("-" * 95)
    print(f"SUMMED final-quarter distance:  STAGE-A = {sumA:.3f} ({nA}/6 seeds)  |  "
          f"NEURAL = {sumN:.3f} ({nN}/6 seeds)")
    if nA == 6 and nN == 6:
        verdict = "ACCEPT (no regression)" if sumN <= sumA else f"REGRESSION (NEURAL +{sumN - sumA:.3f})"
        print(f"ACCEPTANCE (NEURAL summed fq <= STAGE-A): {verdict}")
        n_fires = sum(1 for r in rows if r['crit_fires'])
        n_learns = sum(1 for r in rows if r['crit_learns'])
        print(f"CRITIC (Stage-1 gate): fires {n_fires}/6 seeds, learns {n_learns}/6 seeds")
    else:
        print("INCOMPLETE: not all 12 runs present yet.")


if __name__ == "__main__":
    main()
