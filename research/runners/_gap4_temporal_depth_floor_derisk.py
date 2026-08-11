"""gap#4 deep-credit-on-spikes — ISOLATE THE TEMPORAL-DEPTH FLOOR (the un-quantified confound behind the mapped boundary).

THE BOUNDARY (2026-08-02, multiply stated). Transport-free deep credit works AT RATE; on spikes, DFA e-prop is
DEPTH-ROBUST (trains N=2,3,4, inherit 0.91-0.96, exceeds BPTT) where CHAINED multi-hop FA collapses. BUT every one of
those findings carries the SAME honest caveat and names it as the open edge:

    "THE FLOOR IS HIGH (~0.951) — the task is ~1-layer-solvable on the spiking net (the temporal-depth-floor: LIF
     membrane integration over T=24 adds effective depth). So DFA at N=3,4 shows depth-ROBUSTNESS, NOT proven depth-3
     credit ASSIGNMENT."  (2026-08-02-gap4-DFA-eprop-is-depth-robust..., honest scope #1)

Every depth finding ran at the FIXED CONSTANT T=24. No finding ever swept T. This is the CLAUDE.md wall reframe made
concrete: *what companion process did we replace with a constant?* The companion process is the LIF membrane's
temporal integration; the constant is T=24; the proxy (the temporal floor) may OWN the measurement, masking whether
the SPATIAL feedforward depth-2 credit is doing anything at all.

THE PROBE (additive, reuse-by-import, NO sim/ edit). Sweep T on the SAME compositional-inheritance task + SAME LIF SNN
+ SAME transport-free DFA e-prop credit (`run_seed`, credit_mode=eprop) used by the DFA-depth findings, and read, at
each T:
  * floor_inherit  = the 1-hidden-layer net (the temporal-depth floor: how much a SHALLOW net solves via T-integration)
  * snn_inherit    = the trained 2-hidden DFA net (spatial depth + DFA credit)
  * spatial_gap    = snn_inherit - floor_inherit  (how much the SPATIAL depth-2 credit adds ABOVE the temporal floor)
  * oracle_inherit = rate DendriticMLP ceiling  (task fittable? the ceiling must exist for a null to be interpretable)
  * permuted_inherit ~ chance  (no label leakage — anti-cheat, must hold at EVERY T)

ISOLATION VERDICT (GO = the temporal-depth floor is a REAL, LARGE, T-DRIVEN confound AND reducing T makes the spatial
credit load-bearing — i.e. a small-T regime is the clean instrument the depth-3 frontier needs):
  (1) floor(T_hi) HIGH (>= floor_hi_bar, default 0.80): reproduce the "~1-layer-solvable" cited confound;
  (2) floor(T_hi) - floor(T_lo) >= floor_drop_bar (default 0.15): the temporal integration WAS the effective depth;
  (3) spatial_gap(T_lo) >= gap_open_bar (default 0.05) while spatial_gap(T_hi) <= gap_closed_bar (default 0.03):
      at small T the SPATIAL depth-2 DFA credit becomes load-bearing (trained >> floor), where at T=24 it did not.
  Anti-cheats (all T): permuted within perm_tol (default 0.05) of chance; oracle >= oracle_bar (default 0.80).

DECISIVE: (1)&(2)&(3) hold -> reducing T DEFEATS the temporal-depth floor -> a small-T regime is a valid instrument
where spatial deep credit is obligatory, so the DFA-depth result can be re-posed there to demonstrate GENUINE (not
redundant-depth) credit assignment -- the named next rung past the boundary. If floor stays HIGH at T=1 (no drop),
the effective depth is NOT temporal (it is the static input expansion) -> honest negative that REFRAMES the confound
and names the next mechanism (attack the input-encoding expansion, not T).

Sources: 2026-08-02 DFA-eprop-depth-robust + crux-transport-free-rule-matched-capacity (both name the temporal-depth
floor as the open edge); Neftci-Mostafa-Zenke surrogate-gradient SNN; Bellec 2020 e-prop. NO sim/ edit.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from research.runners._snn_bptt_forward_vs_learning_isolation_derisk import run_seed  # noqa: E402
from tools.lab import lever  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_gap4_temporal_depth_floor.json"


def _one_T(seed, hidden, T, epochs, lr, in_gain, subsample, task_kwargs, n_hidden_layers, credit_mode):
    r = run_seed(seed, hidden, T, epochs, lr, in_gain, subsample, task_kwargs,
                 n_hidden_layers=n_hidden_layers, credit_mode=credit_mode)
    floor = r["floor_inherit_heldout"]
    snn = r["snn_inherit_heldout"]
    gap = (snn - floor) if (not np.isnan(snn) and not np.isnan(floor)) else float("nan")
    return {"T": T, "chance": r["chance"], "floor_inherit": floor, "snn_inherit": snn,
            "spatial_gap": gap, "oracle_inherit": r["oracle_inherit"], "permuted_inherit": r["permuted_inherit"],
            "stage0_depth_separating": r["stage0_depth_separating"], "trains_at_all": r["trains_at_all"]}


def run_sweep(seed, t_list, hidden, epochs, lr, in_gain, subsample, task_kwargs, n_hidden_layers, credit_mode,
              floor_hi_bar, floor_drop_bar, gap_open_bar, gap_closed_bar, perm_tol, oracle_bar):
    rows = []
    for T in t_list:
        rows.append(_one_T(seed, hidden, T, epochs, lr, in_gain, subsample, task_kwargs,
                           n_hidden_layers, credit_mode))
    rows.sort(key=lambda d: d["T"])
    T_lo, T_hi = rows[0], rows[-1]

    # the manipulation must actually move T, else both arms are identical and the isolation is void.
    lever("timesteps_T", T_hi["T"], T_lo["T"], continuous="floor %.3f->%.3f gap %.3f->%.3f"
          % (T_hi["floor_inherit"], T_lo["floor_inherit"], T_hi["spatial_gap"], T_lo["spatial_gap"]))

    floor_drop = T_hi["floor_inherit"] - T_lo["floor_inherit"]
    chance = T_hi["chance"]
    # Leakage = a PERMUTED-label net scoring ABOVE chance on the real held-out labels (it recovered the true
    # signal despite shuffled training). One-sided: BELOW chance is the cleanest possible no-leakage result
    # (the net learned the shuffled mapping, which anti-generalizes) and must NOT trip the anti-cheat.
    perm_ok = all(r["permuted_inherit"] <= r["chance"] + perm_tol for r in rows
                  if not np.isnan(r["permuted_inherit"]) and not np.isnan(r["chance"]))
    oracle_ok = all((not np.isnan(r["oracle_inherit"])) and r["oracle_inherit"] >= oracle_bar for r in rows)

    c1_floor_hi = T_hi["floor_inherit"] >= floor_hi_bar
    c2_floor_drops = floor_drop >= floor_drop_bar
    c3_gap_opens = (T_lo["spatial_gap"] >= gap_open_bar) and (T_hi["spatial_gap"] <= gap_closed_bar)

    go = bool(c1_floor_hi and c2_floor_drops and c3_gap_opens and perm_ok and oracle_ok)

    if not oracle_ok:
        verdict = ("UNDEFINED: oracle ceiling < %.2f at some T (task not fittable there) -> a floor/gap null is "
                   "uninterpretable at that T; not a score." % oracle_bar)
    elif not perm_ok:
        # Label leakage is a PRECONDITION (instrument-validity) failure, NOT a mechanism verdict: a run whose
        # permuted-label control rises above chance cannot yield a NEGATIVE, only UNDEFINED (per verdict-preconditions
        # gate — a failed precondition makes floor/gap uninterpretable, the affect-eviction lesson).
        verdict = ("UNDEFINED (instrument invalid): permuted-label rises > chance+%.2f at some T (label leakage) -> "
                   "the floor/gap is uninterpretable at that T; not a mechanism score." % perm_tol)
    elif go:
        verdict = ("GO: TEMPORAL-DEPTH FLOOR ISOLATED. floor(T=%d)=%.3f HIGH drops to floor(T=%d)=%.3f "
                   "(-%.3f) as T shrinks, AND the spatial depth-2 DFA credit becomes load-bearing at low T "
                   "(spatial_gap %.3f@T%d vs %.3f@T%d). Reducing T DEFEATS the temporal floor -> small-T is the "
                   "clean instrument where spatial deep credit is obligatory (re-pose the DFA-depth result there)."
                   % (T_hi["T"], T_hi["floor_inherit"], T_lo["T"], T_lo["floor_inherit"], floor_drop,
                      T_lo["spatial_gap"], T_lo["T"], T_hi["spatial_gap"], T_hi["T"]))
    elif c1_floor_hi and not c2_floor_drops:
        verdict = ("NEGATIVE (reframes the confound): floor stays HIGH at T=%d (%.3f, drop only %.3f) -> the "
                   "effective depth is NOT temporal integration; it is the static input expansion / rate code. "
                   "Next mechanism: attack the input-encoding expansion, not T." % (T_lo["T"], T_lo["floor_inherit"], floor_drop))
    else:
        verdict = ("PARTIAL: floor_hi=%s(%.3f>=%.2f) floor_drops=%s(%.3f>=%.2f) gap_opens=%s(lo %.3f>=%.2f & hi %.3f<=%.2f)."
                   % (c1_floor_hi, T_hi["floor_inherit"], floor_hi_bar, c2_floor_drops, floor_drop, floor_drop_bar,
                      c3_gap_opens, T_lo["spatial_gap"], gap_open_bar, T_hi["spatial_gap"], gap_closed_bar))

    # Instrument-validity PRECONDITIONS (hold for GO or NEGATIVE alike — they make ANY verdict interpretable;
    # the c1/c2/c3 checks are the RESULT, not preconditions). Emitted so a verdict travels with what earned it.
    depth_sep_all = all(bool(row["stage0_depth_separating"]) for row in rows)
    v = Verdict("gap4_temporal_depth_floor", chance=chance)
    v.require("oracle_ceiling_exists", oracle_ok, expect=True,
              note="rate DendriticMLP oracle >= %.2f at every T -> a null is interpretable" % oracle_bar)
    v.require("no_label_leakage", perm_ok, expect=True,
              note="permuted-label <= chance+%.2f at every T (one-sided; leakage would be ABOVE chance)" % perm_tol)
    v.reaches("timesteps_lever_moved", T_hi["T"], T_lo["T"],
              note="the T sweep actually changed the temporal-integration window")
    v.require("task_depth_separating", depth_sep_all, expect=True,
              note="stage0 depth-genuineness holds at every T (shallow oracle underfits, deep oracle fits)")
    decided = v.decide(go)

    return {"rows": rows, "T_lo": T_lo["T"], "T_hi": T_hi["T"], "chance": chance,
            "floor_hi": T_hi["floor_inherit"], "floor_lo": T_lo["floor_inherit"], "floor_drop": floor_drop,
            "spatial_gap_lo": T_lo["spatial_gap"], "spatial_gap_hi": T_hi["spatial_gap"],
            "permuted_ok": perm_ok, "oracle_ok": oracle_ok,
            "checks": {"c1_floor_hi": bool(c1_floor_hi), "c2_floor_drops": bool(c2_floor_drops),
                       "c3_gap_opens": bool(c3_gap_opens)},
            "go": go, "verdict": verdict,
            "preconditions": decided["preconditions"], "undefined_reasons": decided["undefined_reasons"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t-list", type=str, default="1,2,4,8,16,24",
                    help="comma-separated timestep values to sweep (the temporal-integration window).")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-hidden-layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--in-gain", type=float, default=1.0)
    ap.add_argument("--credit-mode", type=str, default="eprop", choices=["bptt", "spatial", "eprop", "eprop_shuffle"])
    ap.add_argument("--train-subsample", type=int, default=400)
    # task knobs (defaults match the DFA-depth findings' compositional-inheritance task)
    ap.add_argument("--n-super", type=int, default=12)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=2)
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--feature-seed", type=int, default=0)
    # GO-gate bars
    ap.add_argument("--floor-hi-bar", type=float, default=0.80)
    ap.add_argument("--floor-drop-bar", type=float, default=0.15)
    ap.add_argument("--gap-open-bar", type=float, default=0.05)
    ap.add_argument("--gap-closed-bar", type=float, default=0.03)
    ap.add_argument("--perm-tol", type=float, default=0.05)
    ap.add_argument("--oracle-bar", type=float, default=0.80)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    t_list = sorted(set(int(x) for x in args.t_list.split(",") if x.strip()))
    task_kwargs = {"n_super": args.n_super, "n_members": args.n_members,
                   "held_per_super": args.held_per_super, "n_prop": args.n_prop,
                   "member_id_dim": args.member_id_dim, "n_obs": args.n_obs,
                   "noise": args.noise, "feature_seed": args.feature_seed}

    t0 = time.time()
    try:
        r = run_sweep(args.seed, t_list, args.hidden, args.epochs, args.lr, args.in_gain,
                      args.train_subsample, task_kwargs, args.n_hidden_layers, args.credit_mode,
                      args.floor_hi_bar, args.floor_drop_bar, args.gap_open_bar, args.gap_closed_bar,
                      args.perm_tol, args.oracle_bar)
    except Exception as e:
        r = {"seed": args.seed, "error": repr(e), "traceback": traceback.format_exc()}

    out = {"probe": "gap4_temporal_depth_floor_isolation", "seed": args.seed,
           "config": {"t_list": t_list, "hidden": args.hidden, "n_hidden_layers": args.n_hidden_layers,
                      "epochs": args.epochs, "lr": args.lr, "in_gain": args.in_gain,
                      "credit_mode": args.credit_mode, "train_subsample": args.train_subsample, "task": task_kwargs,
                      "bars": {"floor_hi": args.floor_hi_bar, "floor_drop": args.floor_drop_bar,
                               "gap_open": args.gap_open_bar, "gap_closed": args.gap_closed_bar,
                               "perm_tol": args.perm_tol, "oracle": args.oracle_bar}},
           "elapsed_seconds": round(time.time() - t0, 1), "result": r}
    out["verdict"] = r.get("verdict", r.get("error", "no result"))
    out["preconditions"] = r.get("preconditions", [])  # top-level for tools/gates/verdict_preconditions.py
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    if "rows" in r:
        print("  T  chance floor  snn   gap   oracle perm  depth-sep")
        for row in r["rows"]:
            print("  %-3d %.3f %.3f %.3f %+0.3f %.3f %.3f %s"
                  % (row["T"], row["chance"], row["floor_inherit"], row["snn_inherit"], row["spatial_gap"],
                     row["oracle_inherit"], row["permuted_inherit"], row["stage0_depth_separating"]))
    print(out["verdict"])
    print("[temporal-depth-floor] wrote %s" % args.out)


if __name__ == "__main__":
    main()
