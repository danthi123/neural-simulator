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


def run_one(seed, readout_source, urgency_max_pa, n_steps, grid_size, out_dir):
    from research.runners.g11_bg_runner import run_moving_goal_episode
    from research.runners.nav_conv_merged_bridge import (
        conv_extra_regions_pathways, finalize_conv_for_nav_gate,
    )

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{readout_source}{('_u%g' % urgency_max_pa) if urgency_max_pa else ''}"
    out = os.path.join(out_dir, f"navcmp_merged_{tag}_seed{seed}.json")

    extra_regions, extra_pathways = conv_extra_regions_pathways()

    def hook(bridge):
        finalize_conv_for_nav_gate(bridge, seed=seed)

    print(f"[navcmp] seed={seed} readout={readout_source} urgency={urgency_max_pa} grid={grid_size} "
          f"n_steps={n_steps} -> {out}", flush=True)
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
    )
    with open(out) as f:
        data = json.load(f)
    from research.runners.nav_gate2a_aggregate import score_from_data
    score = score_from_data(data)
    dpc = data.get("decision_path_counts") or data.get("_decision_path_counts")
    print(f"[navcmp] seed={seed} {tag}: score={score:.4f}  decision_path={dpc}", flush=True)
    return {"seed": seed, "readout_source": readout_source, "urgency_max_pA": urgency_max_pa,
            "tag": tag, "score": score, "decision_path_counts": dpc, "out": out}


def main():
    ap = argparse.ArgumentParser(description="roadmap #4 cheap few-seed merged nav read-out comparison")
    ap.add_argument("--seeds", default="42,43", help="comma-separated seeds")
    ap.add_argument("--n-steps", type=int, default=600)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--urgency-max-pa", type=float, default=180.0)
    ap.add_argument("--out-dir", default="research/findings/raw/nav_gate_2a")
    ap.add_argument("--summary-out", default="research/findings/raw/nav_gate_2a/_navcmp_summary.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    # the three arms: motor (host argmax), thal (biologized argmax SOURCE), spiking_wta (+urgency = fully-spiking).
    arms = [("motor", 0.0), ("thal", 0.0), ("spiking_wta", args.urgency_max_pa)]
    rows = []
    for seed in seeds:
        for readout_source, urg in arms:
            try:
                rows.append(run_one(seed, readout_source, urg, args.n_steps, args.grid_size, args.out_dir))
            except Exception as e:
                print(f"[navcmp] seed={seed} {readout_source} FAILED: {type(e).__name__}: {e}", flush=True)
                rows.append({"seed": seed, "readout_source": readout_source, "urgency_max_pA": urg,
                             "tag": readout_source, "score": None, "error": f"{type(e).__name__}: {e}"})

    # per-source mean + deltas
    by_tag = {}
    for r in rows:
        if r.get("score") is not None:
            by_tag.setdefault(r["tag"], []).append(r["score"])
    means = {t: mean(v) for t, v in by_tag.items()}
    spk_tag = f"spiking_wta_u{args.urgency_max_pa:g}" if args.urgency_max_pa else "spiking_wta"
    summary = {
        "seeds": seeds, "n_steps": args.n_steps, "grid_size": args.grid_size,
        "urgency_max_pA": args.urgency_max_pa, "rows": rows, "means_by_tag": means,
        "delta_spiking_minus_motor": (means.get(spk_tag) - means["motor"]) if (spk_tag in means and "motor" in means) else None,
        "delta_spiking_minus_thal": (means.get(spk_tag) - means["thal"]) if (spk_tag in means and "thal" in means) else None,
    }
    os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 64)
    print("roadmap #4 merged nav read-out comparison (LOWER score = better)")
    print("=" * 64)
    for t in ("motor", "thal", spk_tag):
        if t in means:
            print(f"  {t:>18}: mean score {means[t]:.4f}  (n={len(by_tag[t])})")
    print("-" * 64)
    print(f"  spiking_wta - motor : {summary['delta_spiking_minus_motor']}")
    print(f"  spiking_wta - thal  : {summary['delta_spiking_minus_thal']}")
    print(f"\n[wrote {args.summary_out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
