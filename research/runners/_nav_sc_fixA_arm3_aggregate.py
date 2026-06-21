"""#6 FIX A — ARM 3 multi-seed aggregator. Reads the per-seed scpv_FIXA_arm3_seed*.json verdicts and
emits the final multi-seed table + the GO/NEGATIVE call.

  python -m research.runners._nav_sc_fixA_arm3_aggregate \
      --glob "research/findings/raw/nav_gate_2a/scpv_FIXA_arm3_seed*.json" \
      --out research/findings/raw/nav_gate_2a/scpv_FIXA_arm3_aggregate.json
"""
import argparse
import glob as globmod
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str,
                    default="research/findings/raw/nav_gate_2a/scpv_FIXA_arm3_seed*.json")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/nav_gate_2a/scpv_FIXA_arm3_aggregate.json")
    args = ap.parse_args()

    files = sorted(globmod.glob(args.glob))
    rows = []
    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception as e:
            print(f"skip {fp}: {e}")
            continue
        v = d.get("verdict", {})
        rows.append(v)

    def col(key):
        return [r.get(key) for r in rows]

    def fmean(key):
        xs = [r.get(key) for r in rows if isinstance(r.get(key), (int, float)) and r.get(key) == r.get(key)]
        return (float(np.mean(xs)) if xs else None)

    print("\n[arm3-agg] ===== MULTI-SEED TABLE (FIX1+A vs FIX1 vs HOST vs SCRAM) =====")
    hdr = f"  {'seed':>5} {'HOST_post':>10} {'FIX1_post':>10} {'FIX1A_post':>11} {'SCRAM_post':>11} " \
          f"{'FIX1_selNS%':>11} {'FIX1A_selNS%':>12} {'h/FIX1A':>9} {'FIX1A/SCR':>10} {'FIX1A_track':>11}"
    print(hdr)
    for r in rows:
        def g(k, f="{}"):
            x = r.get(k)
            if x is None:
                return "—"
            try:
                return f.format(x)
            except Exception:
                return str(x)
        trk = (r.get("FIX1A_tracking") or {}).get("tracks_goal")
        print(f"  {g('seed'):>5} {g('HOST_post_change_finalQ_sum','{:.2f}'):>10} "
              f"{g('FIX1_post_change_finalQ_sum','{:.2f}'):>10} "
              f"{g('FIX1A_post_change_finalQ_sum','{:.2f}'):>11} "
              f"{g('SCRAM_post_change_finalQ_sum','{:.2f}'):>11} "
              f"{g('FIX1_sel_NS_pct','{:.1f}'):>11} {g('FIX1A_sel_NS_pct','{:.1f}'):>12} "
              f"{g('host_over_FIX1A_post_ratio','{:.2f}'):>9} {g('FIX1A_over_SCRAM_post_ratio','{:.2f}'):>10} "
              f"{str(trk):>11}")

    summary = {
        "n_seeds": len(rows),
        "seeds": col("seed"),
        "mean_HOST_post": fmean("HOST_post_change_finalQ_sum"),
        "mean_FIX1_post": fmean("FIX1_post_change_finalQ_sum"),
        "mean_FIX1A_post": fmean("FIX1A_post_change_finalQ_sum"),
        "mean_SCRAM_post": fmean("SCRAM_post_change_finalQ_sum"),
        "mean_FIX1_sel_NS_pct": fmean("FIX1_sel_NS_pct"),
        "mean_FIX1A_sel_NS_pct": fmean("FIX1A_sel_NS_pct"),
        "mean_host_over_FIX1A_post_ratio": fmean("host_over_FIX1A_post_ratio"),
        "mean_FIX1A_over_SCRAM_post_ratio": fmean("FIX1A_over_SCRAM_post_ratio"),
        "n_FIX1A_tracks_goal": sum(1 for r in rows if (r.get("FIX1A_tracking") or {}).get("tracks_goal")),
        "n_SCRAM_tracks_goal": sum(1 for r in rows if (r.get("SCRAM_tracking") or {}).get("tracks_goal")),
    }
    # surplus-shrink check: FIX1A sel NS% materially < FIX1 sel NS%
    summary["surplus_shrinks"] = (
        summary["mean_FIX1A_sel_NS_pct"] is not None and summary["mean_FIX1_sel_NS_pct"] is not None
        and abs(summary["mean_FIX1A_sel_NS_pct"]) < abs(summary["mean_FIX1_sel_NS_pct"]))
    # ceiling: FIX1A reaches host (post-change ratio ~>=1 means FIX1A approaches host; >1 means FIX1A BETTER).
    summary["reaches_host_ceiling"] = (
        summary["mean_host_over_FIX1A_post_ratio"] is not None
        and summary["mean_host_over_FIX1A_post_ratio"] >= 0.9)
    # SCRAM collapses: FIX1A materially better than SCRAM (FIX1A/SCRAM < 1 means SCRAM worse).
    summary["scram_collapses"] = (
        summary["mean_FIX1A_over_SCRAM_post_ratio"] is not None
        and summary["mean_FIX1A_over_SCRAM_post_ratio"] < 0.9)

    print("\n[arm3-agg] ===== SUMMARY =====")
    for k, val in summary.items():
        print(f"  {k}: {val}")

    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=2)
    print(f"[arm3-agg] wrote {args.out}")


if __name__ == "__main__":
    main()
