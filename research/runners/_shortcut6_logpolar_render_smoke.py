"""#6 SURPASS — CPU render smoke (the cheap-first gate, no GPU).

Replays render_egocentric_goal for the FOUR grid-32 moving-goal schedule corner goals, at the
documented agent positions (the top-edge pin (16,31) and the foveal-neutral centre (16,16)), for
BOTH the deployed LINEAR render and the new biology-faithful LOG-POLAR render, and reports for each:
  - sc_retina ON-channel mass (sum of the painted ON pixels),
  - the painted-pixel count,
  - the blob CENTROID, and the bearing-quadrant it lands in (NE/NW/SW/SE/centre),
  - whether the bearing-quadrant MATCHES the goal's true egocentric bearing.

The deep-research residual (research/findings/2026-06-22-shortcut6-upstream-orienting-residual-
surpass.md, MOVE 1c) is that the LINEAR render clips every eccentric (>~4-cell) goal off the
32-pixel sc_retina -> mass 0.0, the SC bump ABSENT, for all four schedule goals. The log-polar
render must restore mass > 0 with the CORRECT bearing for every goal (the off-image clipping gone),
WITHOUT smuggling coordinates (it consumes only the (agent,goal) egocentric bearing exactly as the
linear render -- the brain sees only the rendered retina image).

PASS gate (the cheap-first gate): for ALL FOUR schedule goals at BOTH agent positions, the log-polar
render gives mass > 0 AND the correct bearing-quadrant, vs the linear render's 0.0.

Pure NumPy, CPU, seconds. NO GPU, NO bridge build, NO sim/ edit.
"""
import argparse
import json
import os

import numpy as np

from research.runners.g11_bg_runner import render_egocentric_goal

ACTION_NAMES = ["N", "E", "S", "W"]


def _goal_schedule(gs):
    """The EXACT schedule the popvector harness uses (research/runners/_nav_sc_popvector_readout_derisk.py)."""
    return {
        "phase0_NE": (max(0, gs - 2), max(0, gs - 2)),     # NE corner
        "phase1_farW": (max(0, 1), max(0, gs - 2)),         # NW corner (pure-lateral west goal)
        "phase2_SW": (max(0, 1), max(0, 1)),                # SW corner
        "phase3_SE": (max(0, gs - 2), max(0, 1)),           # SE corner
    }


def _bearing_quadrant(ddx, ddy, tol=1e-6):
    """+x=East, +y=North. Returns the quadrant label of the (ddx,ddy) bearing."""
    ns = "N" if ddy > tol else ("S" if ddy < -tol else "")
    ew = "E" if ddx > tol else ("W" if ddx < -tol else "")
    return (ns + ew) if (ns or ew) else "centre"


def _analyze(img, image_size):
    """Return (mass, n_pixels, centroid (cx,cy) or None, bearing-quadrant of centroid)."""
    on = np.asarray(img)[0]
    mass = float(on.sum())
    ys, xs = np.where(on > 0)
    if xs.size == 0:
        return mass, 0, None, "absent"
    c = image_size // 2
    cx = float(xs.mean())
    cy = float(ys.mean())
    # centroid bearing relative to foveal centre (+x=East image-right, +y=North image as rendered:
    # render writes py = c + ddy*..., so +ddy -> larger py; bearing-quadrant read in the SAME frame).
    q = _bearing_quadrant(cx - c, cy - c)
    return mass, int(xs.size), (round(cx, 2), round(cy, 2)), q


def main():
    ap = argparse.ArgumentParser(description="#6 SURPASS log-polar render CPU smoke")
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=32)
    ap.add_argument("--log-polar-d0", type=float, default=1.0)
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/nav_gate_2a/logpolar_render_smoke.json")
    args = ap.parse_args()

    gs = args.grid_size
    imgw = args.image_size
    goals = _goal_schedule(gs)
    agents = {"top_edge_pin": (16, 31), "foveal_centre": (16, 16)}

    rows = []
    for agent_name, agent in agents.items():
        for goal_name, goal in goals.items():
            ddx = float(goal[0] - agent[0])
            ddy = float(goal[1] - agent[1])
            true_q = _bearing_quadrant(ddx, ddy)

            lin = render_egocentric_goal(agent, goal, image_size=imgw, log_polar=False)
            lp = render_egocentric_goal(agent, goal, image_size=imgw, log_polar=True,
                                        log_polar_d0=args.log_polar_d0, log_polar_grid_size=gs)
            lin_mass, lin_n, lin_cen, lin_q = _analyze(lin, imgw)
            lp_mass, lp_n, lp_cen, lp_q = _analyze(lp, imgw)

            row = {
                "agent": agent_name, "agent_pos": list(agent),
                "goal": goal_name, "goal_pos": list(goal),
                "delta_cell": [round(ddx, 1), round(ddy, 1)],
                "true_bearing_quadrant": true_q,
                "linear_mass": round(lin_mass, 4), "linear_npix": lin_n,
                "linear_centroid": lin_cen, "linear_quadrant": lin_q,
                "logpolar_mass": round(lp_mass, 4), "logpolar_npix": lp_n,
                "logpolar_centroid": lp_cen, "logpolar_quadrant": lp_q,
                "logpolar_mass_positive": bool(lp_mass > 0),
                "logpolar_bearing_correct": bool(lp_q == true_q and lp_mass > 0),
            }
            rows.append(row)
            print(f"[smoke] agent={agent_name:13s} goal={goal_name:11s} true={true_q:2s} "
                  f"| LINEAR mass={lin_mass:6.2f} npix={lin_n:2d} q={lin_q:6s} "
                  f"| LOG-POLAR mass={lp_mass:6.2f} npix={lp_n:2d} q={lp_q:6s} "
                  f"bearing_ok={row['logpolar_bearing_correct']}", flush=True)

    all_lp_mass_pos = all(r["logpolar_mass_positive"] for r in rows)
    all_lp_bearing_ok = all(r["logpolar_bearing_correct"] for r in rows)
    all_lin_clipped = all(r["linear_mass"] == 0.0 for r in rows)
    verdict = {
        "grid_size": gs, "image_size": imgw, "log_polar_d0": args.log_polar_d0,
        "n_cases": len(rows),
        "all_linear_clipped_to_zero": all_lin_clipped,
        "all_logpolar_mass_positive": all_lp_mass_pos,
        "all_logpolar_bearing_correct": all_lp_bearing_ok,
        "SMOKE_PASS": bool(all_lp_mass_pos and all_lp_bearing_ok),
        "NOTE": ("PASS = log-polar render gives mass>0 AND correct bearing for ALL 4 schedule goals "
                 "at BOTH agent positions (vs the linear render clipping to mass 0.0). This is the "
                 "cheap-first gate for the #6 log-polar SURPASS; the grid-32 GPU nav GO is the verdict."),
    }
    out = {"rows": rows, "verdict": verdict}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n[smoke] ===== LOG-POLAR RENDER SMOKE VERDICT =====", flush=True)
    for k, v in verdict.items():
        print(f"  {k}: {v}", flush=True)
    print(f"[smoke] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
