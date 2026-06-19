"""Roadmap #4 — CHEAP few-seed GPU nav comparison on the MERGED 'one brain':
fully-spiking commit-burst read-out (+Cisek urgency) vs host-argmax(motor) vs thal-argmax.

For each (seed, readout_source) it runs the merged nav+conv episode (parser+dlPFC appended +
the index-based conv-finalization hook) at a CHEAP grid-8 / short multi-goal config (NOT the
grid-32/1800 flagship — this is the fast read-out delta, not the flagship score), then scores it
with nav_gate2a_aggregate.score_from_data (sum of per-phase final-quarter mean distance; LOWER is
better). Reports the per-source mean and the spiking_wta-vs-motor / spiking_wta-vs-thal deltas.

Determinism (CUBLAS_WORKSPACE_CONFIG) is set at module top BEFORE any CuPy import.
Reuse-by-import; NO sim/ edit.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from statistics import mean


def _goal_schedule(gs, n_steps):
    """A 4-phase multi-goal schedule scaled to n_steps (the 4 corners; transitions at quarters)."""
    far = (max(0, gs - 2), max(0, gs - 2))
    far_west = (max(0, 1), max(0, gs - 2))
    sw = (max(0, 1), max(0, 1))
    far_se = (max(0, gs - 2), max(0, 1))
    q = max(1, n_steps // 4)
    return [(0, far), (q, far_west), (2 * q, sw), (3 * q, far_se)]


def run_one(seed, readout_source, urgency_max_pa, n_steps, grid_size, out_dir, lever_kwargs=None):
    from research.runners.g11_bg_runner import run_moving_goal_episode
    from research.runners.nav_conv_merged_bridge import (
        conv_extra_regions_pathways, finalize_conv_for_nav_gate,
    )

    lever_kwargs = dict(lever_kwargs or {})
    os.makedirs(out_dir, exist_ok=True)
    # Tag the spiking_wta arm by its #4 cost-reduction levers so each sweep point's JSON + row is distinct.
    lever_tag = "".join(f"_{k}{v}" for k, v in sorted(lever_kwargs.items()))
    tag = f"{readout_source}{('_u%g' % urgency_max_pa) if urgency_max_pa else ''}{lever_tag}"
    out = os.path.join(out_dir, f"navcmp_merged_{tag}_seed{seed}.json")

    extra_regions, extra_pathways = conv_extra_regions_pathways()

    def hook(bridge):
        finalize_conv_for_nav_gate(bridge, seed=seed)

    print(f"[navcmp] seed={seed} readout={readout_source} urgency={urgency_max_pa} levers={lever_kwargs} "
          f"grid={grid_size} n_steps={n_steps} -> {out}", flush=True)
    run_moving_goal_episode(
        out_path=out, seed=seed, n_steps=n_steps, grid_size=grid_size,
        goal_schedule=_goal_schedule(grid_size, n_steps),
        enable_d1_d2_asymmetry=True,
        enable_striatal_fsis=True,
        enable_cluster_a_closed_loop=True,
        enable_cluster_e_topography=True,
        enable_pfc_nmda=True,
        enable_visual_cortex=True,
        visual_cortex_action_warmup_steps=min(300, max(1, n_steps // 3)),
        stdp_w_max_override=400.0,
        readout_source=readout_source,
        urgency_max_pA=urgency_max_pa,
        extra_regions=extra_regions, extra_pathways=extra_pathways,
        build_with_ou=True, prebuilt_post_init_hook=hook,
        **lever_kwargs,
    )
    with open(out) as f:
        data = json.load(f)
    from research.runners.nav_gate2a_aggregate import score_from_data
    score = score_from_data(data)
    dpc = data.get("decision_path_counts") or data.get("_decision_path_counts")
    print(f"[navcmp] seed={seed} {tag}: score={score:.4f}  decision_path={dpc}", flush=True)
    return {"seed": seed, "readout_source": readout_source, "urgency_max_pA": urgency_max_pa,
            "lever_kwargs": lever_kwargs, "tag": tag, "score": score,
            "decision_path_counts": dpc, "out": out}


def _coerce(v):
    """Coerce a CLI lever value string to int / float / bool / str (for run_moving_goal_episode kwargs)."""
    s = str(v).strip()
    low = s.lower()
    if low in ("true", "1", "yes", "on") and not s.replace(".", "", 1).isdigit():
        return True
    if low in ("false", "0", "no", "off") and low in ("false", "no", "off"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _parse_levers(lever_list):
    """['reset_losers_only=1', 'thal_to_sel_weight=40'] -> {'reset_losers_only': True, 'thal_to_sel_weight': 40}."""
    out = {}
    for item in (lever_list or []):
        if "=" not in item:
            raise SystemExit(f"--lever must be KEY=VALUE (got {item!r})")
        k, v = item.split("=", 1)
        out[k.strip()] = _coerce(v)
    return out


def main():
    ap = argparse.ArgumentParser(description="roadmap #4 cheap few-seed merged nav read-out comparison + lever sweep")
    ap.add_argument("--seeds", default="42,43", help="comma-separated seeds")
    ap.add_argument("--n-steps", type=int, default=600)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--urgency-max-pa", type=float, default=180.0)
    ap.add_argument("--out-dir", default="research/findings/raw/nav_gate_2a")
    ap.add_argument("--summary-out", default="research/findings/raw/nav_gate_2a/_navcmp_summary.json")
    # #4 cost-reduction sweep (CYCLE 228). --sweep KEY=v1,v2,... runs one spiking_wta arm per value of the
    # given run_moving_goal_episode kwarg (e.g. sel_recurrent_weight=1.0,0.7,0.5,0.3 = the ROUND-1 leak sweep).
    # --lever KEY=VAL (repeatable) applies a FIXED lever to every spiking_wta arm (e.g. reset_losers_only=1).
    # --no-baselines skips the motor/thal baselines (re-use a prior run's baselines). All inert without spiking_wta.
    ap.add_argument("--sweep", default=None, help="KEY=v1,v2,... — one spiking_wta arm per value (the swept lever)")
    ap.add_argument("--lever", action="append", default=[], help="KEY=VAL fixed lever on all spiking_wta arms (repeatable)")
    ap.add_argument("--no-baselines", action="store_true", help="skip motor/thal baselines (spiking_wta arms only)")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    fixed_levers = _parse_levers(args.lever)

    # Build the spiking_wta arm list: either the swept lever's values, or a single base arm.
    spiking_arms = []  # list of lever_kwargs dicts
    if args.sweep:
        if "=" not in args.sweep:
            raise SystemExit("--sweep must be KEY[+KEY2...]=v1,v2,...")
        skey, svals = args.sweep.split("=", 1)
        # "+"-joined keys are swept TOGETHER to the same value (e.g. n_sel_per_action+n_commit_per_action=20,40,80
        # = the plan's paired N-scaling, the accumulator + commit pools grown in lockstep).
        skeys = [k.strip() for k in skey.split("+") if k.strip()]
        for sv in svals.split(","):
            if sv.strip():
                val = _coerce(sv)
                spiking_arms.append({**fixed_levers, **{k: val for k in skeys}})
    else:
        spiking_arms.append(dict(fixed_levers))

    # arms = (readout_source, urgency, lever_kwargs)
    arms = []
    if not args.no_baselines:
        arms.append(("motor", 0.0, {}))
        arms.append(("thal", 0.0, {}))
    for lk in spiking_arms:
        arms.append(("spiking_wta", args.urgency_max_pa, lk))

    rows = []
    for seed in seeds:
        for readout_source, urg, lk in arms:
            try:
                rows.append(run_one(seed, readout_source, urg, args.n_steps, args.grid_size, args.out_dir,
                                    lever_kwargs=lk))
            except Exception as e:
                print(f"[navcmp] seed={seed} {readout_source} {lk} FAILED: {type(e).__name__}: {e}", flush=True)
                rows.append({"seed": seed, "readout_source": readout_source, "urgency_max_pA": urg,
                             "lever_kwargs": lk, "tag": readout_source, "score": None,
                             "error": f"{type(e).__name__}: {e}"})

    # per-tag mean + ANTI-CHEAT decision-path fractions (the win must come from the commit burst = `primary`,
    # not the sel-lean argmax fallback). A config that lowers SUM by raising fallback% is rejected.
    by_tag, dpc_by_tag = {}, {}
    for r in rows:
        if r.get("score") is not None:
            by_tag.setdefault(r["tag"], []).append(r["score"])
            dpc = r.get("decision_path_counts") or {}
            tot = sum(dpc.values()) if dpc else 0
            if tot:
                acc = dpc_by_tag.setdefault(r["tag"], {"primary": 0, "fallback": 0, "random": 0, "tot": 0})
                for k in ("primary", "fallback", "random"):
                    acc[k] += dpc.get(k, 0)
                acc["tot"] += tot
    means = {t: mean(v) for t, v in by_tag.items()}
    motor_mean = means.get("motor")
    deltas = {t: (means[t] - motor_mean) for t in means if motor_mean is not None}
    summary = {
        "seeds": seeds, "n_steps": args.n_steps, "grid_size": args.grid_size,
        "urgency_max_pA": args.urgency_max_pa, "sweep": args.sweep, "fixed_levers": fixed_levers,
        "rows": rows, "means_by_tag": means, "deltas_vs_motor": deltas,
        "decision_path_frac_by_tag": {
            t: {k: acc[k] / acc["tot"] for k in ("primary", "fallback", "random")}
            for t, acc in dpc_by_tag.items()},
    }
    os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("roadmap #4 merged nav read-out sweep (LOWER score = better; primary% = anti-cheat)")
    print("=" * 78)
    for t in sorted(means, key=lambda t: means[t]):
        frac = summary["decision_path_frac_by_tag"].get(t, {})
        pf = f"  primary={frac.get('primary', 0):.2f} fallback={frac.get('fallback', 0):.2f}" if frac else ""
        dv = f"  d_vs_motor={deltas[t]:+.4f}" if t in deltas else ""
        print(f"  {t:>34}: mean {means[t]:.4f} (n={len(by_tag[t])}){dv}{pf}")
    print(f"\n[wrote {args.summary_out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
