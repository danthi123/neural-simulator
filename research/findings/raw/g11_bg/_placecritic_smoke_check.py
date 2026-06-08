"""Stage-1 smoke gate check for the place-code neural value critic.

Reads a g11_bg neural-critic result JSON and reports the four smoke gates:
  (i)   RUNS         -> snc_rate_log non-empty AND not all-zero (mask bug gone)
  (ii)  CRITIC LEARNS-> critic_weight_final > critic_weight_initial; striov non-zero
  (iii) SNc FIRES    -> mean(snc_rate_log) > 0 (Stage-A ~7 Hz reference)
  (iv)  SANE NAV     -> summed final-quarter distance not catastrophic (Stage-A ~2-4)

Usage: python _placecritic_smoke_check.py <result.json> [<result2.json> ...]
"""
import json, sys


def summarize(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    snc = d.get("snc_rate_log") or []
    striov = d.get("striov_rate_log") or []
    wi = d.get("critic_weight_initial")
    wf = d.get("critic_weight_final")
    quarters = d.get("mean_distance_quarters") or []
    # summed final-quarter distance: the gate metric the build instructions use.
    # mean_distance_quarters is per-quarter MEAN distance; the last entry is the
    # final quarter. (The A/B acceptance is "summed final-quarter distance"; for a
    # single run we report the final-quarter mean as the comparable scalar.)
    fq = quarters[-1] if quarters else None
    snc_mean = (sum(snc) / len(snc)) if snc else 0.0
    snc_nonzero = any(v != 0 for v in snc)
    striov_mean = (sum(striov) / len(striov)) if striov else 0.0
    striov_nonzero = any(v != 0 for v in striov)
    learns = (wi is not None and wf is not None and wf > wi)

    print(f"\n=== {path} ===")
    print(f"  afferent              : {d.get('critic_afferent')}")
    print(f"  gabab_prop_strength   : {d.get('critic_gabab_propagation_strength')}")
    print(f"  enable_critic_window  : {d.get('enable_critic_window')}  lead_steps={d.get('critic_lead_steps')}")
    print(f"  (i)   RUNS  (snc non-zero)        : {snc_nonzero}  "
          f"(n_windows={len(snc)}, mean_spikes/window={snc_mean:.2f})")
    print(f"  (ii)  CRITIC LEARNS (wf>wi)       : {learns}  "
          f"(w0={wi}, wf={wf}); striov non-zero={striov_nonzero} (mean={striov_mean:.2f})")
    print(f"  (iii) SNc FIRES (mean>0)          : {snc_mean > 0}  (mean spikes/window={snc_mean:.2f})")
    print(f"  (iv)  SANE NAV (final-Q mean dist): {fq}  "
          f"(n_steps_at_goal={d.get('n_steps_at_goal')}, overall_mean={d.get('mean_distance_overall')})")
    verdict = snc_nonzero and (snc_mean > 0) and learns
    print(f"  --> RUN+LEARN+FIRE gate           : {'PASS' if verdict else 'CHECK'}")
    return dict(path=path, snc_nonzero=snc_nonzero, snc_mean=snc_mean,
                learns=learns, striov_nonzero=striov_nonzero, fq=fq,
                n_at_goal=d.get("n_steps_at_goal"),
                overall=d.get("mean_distance_overall"))


if __name__ == "__main__":
    rows = [summarize(p) for p in sys.argv[1:]]
    print("\n--- SUMMARY ---")
    for r in rows:
        print(r)
