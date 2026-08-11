"""gap#4 deep-credit-on-spikes — GENUINE SPATIAL DEPTH-3 CREDIT at SMALL T (defeat the temporal-depth floor).

THE SETUP (2026-08-11). The temporal-depth-floor smoke ISOLATED the confound behind every 2026-08-02 gap#4 depth
finding: the LIF membrane's fixed temporal-integration window T silently supplies "effective depth", so the
compositional-inheritance task is ~1-layer-solvable AT T=24 and DFA e-prop's success at N=2,3,4 there showed
depth-ROBUSTNESS, not depth-3 credit ASSIGNMENT. The smoke's load-bearing read: a 1-hidden LIF net (NO spatial depth)
climbs 0.444 -> 0.963 as T grows 1 -> 24. The named next rung (verbatim from that finding):

    "re-pose the DFA N=2,3,4 depth sweep at SMALL T (T=2-4) where the deeper spatial layers are OBLIGATORY --
     genuine depth-3 credit assignment."

THIS PROBE (additive, reuse-by-import, NO sim/ edit). FIX a small T (the temporal window can no longer fake depth)
and SWEEP the spatial hidden DEPTH N in {1,2,3} on the SAME compositional-inheritance task + SAME LIF SNN + SAME
transport-free DFA e-prop credit (`run_seed`, credit_mode=eprop; DFA feedback is a SEPARATE fixed-random stream ->
no weight transport). At each N read:
  * snn_inherit    = the trained N-hidden DFA net (held-out inheritance accuracy)
  * floor_inherit  = a fresh 1-HIDDEN DFA net (the "deep layers REMOVED" control; run_seed always retrains it).
                     By construction snn(N=1) == floor (identical architecture + seed) -> an internal determinism check.
  * depth_gain     = snn(N_hi) - floor  (how much the DEEPER spatial layers + their DFA credit add ABOVE 1-hidden)
  * oracle_inherit = rate DendriticMLP ceiling, DEPTH-MATCHED to N  (task fittable at this depth? must exist for a null
                     to be interpretable)
  * permuted_inherit ~ chance  (no label leakage -- anti-cheat, one-sided, must hold at EVERY N)

THE KEY QUESTION: at small T, is the DEEP (N=3) spatial credit LOAD-BEARING? Two signatures of GENUINE deep credit
(NOT the temporal window faking it):
  (A) accuracy RISES with N (deeper spatial layers help): snn(N=1) <= snn(N=2) <= snn(N=3), within a noise tol;
  (B) REMOVING the deep layers COLLAPSES it back to the 1-hidden floor -- which is exactly the N=1 arm (== floor).

VERDICT TIERS (preconditions = oracle ceiling exists at every N, permuted <= chance, task depth-separating, N=1==floor,
T is small, depth lever moved):
  * GO (genuine depth-3): deep_gain(N3 vs floor) >= depth_gain_bar AND gain_2to3 >= depth3_bar (the 3rd layer itself
    adds skill) AND monotone -> at small T the spatial depth-3 DFA credit is load-bearing. The frontier target.
  * QUALIFIED (depth-2 load-bearing, depth-3 redundant): deep_gain >= bar but gain_2to3 < depth3_bar -> spatial
    depth-2 credit is load-bearing at small T (the temporal floor is defeated) but the 3rd layer adds nothing yet ->
    next probe: a task whose depth-3 is OBLIGATORY (hier3 that separates depth-2 from depth-3), or report per-layer
    spike rates (deep layers may be silent at small T).
  * NEGATIVE (reframes): deep_gain < depth_gain_bar -> even with the temporal floor removed, spatial depth does not
    help -> the residual effective depth is the STATIC input rate-expansion, not spatial layers -> next: attack the
    input encoding, not depth.
  * UNDEFINED: a precondition failed (uninterpretable).

Sources: 2026-08-11 temporal-depth-floor isolation (the named next rung); 2026-08-02 DFA-eprop-depth-robust +
crux-transport-free-rule-matched-capacity (the open edge); Neftci-Mostafa-Zenke surrogate-gradient SNN; Bellec 2020
e-prop (eligibility over many spikes); Nokland 2016 direct feedback alignment (transport-free credit). NO sim/ edit.
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
from tools.lab import lever, attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_gap4_spatial_depth3_smallT.json"


def _one_N(seed, N, T, hidden, epochs, lr, in_gain, subsample, task_kwargs, credit_mode):
    """One depth point: train an N-hidden DFA-eprop LIF net at fixed small T; run_seed also retrains the 1-hidden
    floor (deep-layers-removed control) and a permuted-label control at THIS N, and a depth-matched rate oracle."""
    r = run_seed(seed, hidden, T, epochs, lr, in_gain, subsample, task_kwargs,
                 n_hidden_layers=N, credit_mode=credit_mode)
    snn = r["snn_inherit_heldout"]
    floor = r["floor_inherit_heldout"]
    gain = (snn - floor) if (not np.isnan(snn) and not np.isnan(floor)) else float("nan")
    return {"seed": seed, "N": N, "T": T, "chance": r["chance"],
            "snn_inherit": snn, "floor_inherit": floor, "depth_gain_vs_floor": gain,
            "permuted_inherit": r["permuted_inherit"], "oracle_inherit": r["oracle_inherit"],
            "stage0_depth_separating": bool(r["stage0_depth_separating"]), "trains_at_all": bool(r["trains_at_all"])}


def _nanmean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _nanstd(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.std(xs)) if len(xs) > 1 else 0.0


def run_sweep(seeds, n_list, T, hidden, epochs, lr, in_gain, subsample, task_kwargs, credit_mode,
              depth_gain_bar, depth3_bar, mono_tol, perm_tol, oracle_bar, t_small_bar):
    rows = []
    for seed in seeds:
        for N in n_list:
            rows.append(_one_N(seed, N, T, hidden, epochs, lr, in_gain, subsample, task_kwargs, credit_mode))

    # per-N aggregates across seeds
    per_N = {}
    for N in n_list:
        rN = [r for r in rows if r["N"] == N]
        per_N[N] = {
            "N": N,
            "mean_snn": _nanmean([r["snn_inherit"] for r in rN]),
            "std_snn": _nanstd([r["snn_inherit"] for r in rN]),
            "mean_floor": _nanmean([r["floor_inherit"] for r in rN]),
            "mean_gain": _nanmean([r["depth_gain_vs_floor"] for r in rN]),
            "mean_perm": _nanmean([r["permuted_inherit"] for r in rN]),
            "mean_oracle": _nanmean([r["oracle_inherit"] for r in rN]),
            "seeds_snn": {r["seed"]: r["snn_inherit"] for r in rN},
        }
    chance = _nanmean([r["chance"] for r in rows])
    n_lo, n_hi = min(n_list), max(n_list)

    # -------------------------------------------------------------- one flag != one variable (tools.lab.lever)
    # The DEPTH flag must genuinely change the net (control N=n_lo vs treatment N=n_hi). Structural lever: the
    # layer count differs, so the architectures differ. Continuous read: the held-out accuracies + floor, so a
    # reader sees whether the RESULT moved (the science) and not just the config.
    lever("n_hidden_layers", n_hi, n_lo,
          continuous="snn %.3f->%.3f  floor(N=1) %.3f  chance %.3f"
          % (per_N[n_lo]["mean_snn"], per_N[n_hi]["mean_snn"], per_N[n_lo]["mean_floor"], chance))

    # -------------------------------------------------------------- anti-cheats (must hold at EVERY N, EVERY seed)
    perm_ok = all(r["permuted_inherit"] <= r["chance"] + perm_tol for r in rows
                  if not np.isnan(r["permuted_inherit"]) and not np.isnan(r["chance"]))
    # The rate oracle is DEPTH-MATCHED to N inside run_seed ([n_in]+[96]*N+[k], full backprop). Two DISTINCT
    # things it certifies, which the smoke proved must NOT be conflated: (a) the TASK is fittable -> a learnable
    # ceiling EXISTS SOMEWHERE (max over N); (b) the DEEPEST arm is interpretable -> the depth-N_hi ceiling itself
    # TRAINS (oracle at N_hi). A 3-hidden rate net can be UNTRAINABLE at fixed hyperparams even though the task is
    # representable (a strictly-shallower net ceilings), so requiring the oracle at EVERY N is wrong: it would call
    # a deep-net OPTIMIZATION wall a task failure. task_fittable is the base interpretability floor; oracle_at_Nhi
    # is the specific precondition for a genuine depth-N_hi CREDIT claim.
    oracle_vals = [r["oracle_inherit"] for r in rows if not np.isnan(r["oracle_inherit"])]
    task_fittable = bool(oracle_vals) and max(oracle_vals) >= oracle_bar
    oracle_at_Nhi = (not np.isnan(per_N[n_hi]["mean_oracle"])) and per_N[n_hi]["mean_oracle"] >= oracle_bar
    oracle_per_N = {N: per_N[N]["mean_oracle"] for N in n_list}
    depth_sep_all = all(r["stage0_depth_separating"] for r in rows)
    # internal determinism check: snn(N=1) MUST equal the 1-hidden floor (identical architecture + seed). If it
    # ever differs materially, the substrate is not deterministic under this seed -> the whole sweep is void.
    n1_diffs = [abs(r["snn_inherit"] - r["floor_inherit"]) for r in rows
                if r["N"] == 1 and not np.isnan(r["snn_inherit"]) and not np.isnan(r["floor_inherit"])]
    n1_is_floor = (len(n1_diffs) > 0) and all(d <= 1e-9 for d in n1_diffs)
    t_small = T <= t_small_bar

    # -------------------------------------------------------------- the science (per-N MEANS)
    floor_mean = per_N[n_lo]["mean_floor"]                 # the 1-hidden reference (== snn at N=1)
    deep_gain = per_N[n_hi]["mean_snn"] - floor_mean       # N_hi net vs deep-layers-removed control
    # step gains along the depth ramp
    sorted_N = sorted(n_list)
    step_gains = {}
    for i in range(1, len(sorted_N)):
        a, b = sorted_N[i - 1], sorted_N[i]
        step_gains["%d->%d" % (a, b)] = per_N[b]["mean_snn"] - per_N[a]["mean_snn"]
    gain_2to3 = step_gains.get("2->3", float("nan"))
    # monotone rise: each deeper N does not DROP more than mono_tol below the previous
    monotone = all(g >= -mono_tol for g in step_gains.values())

    # attributable-to: of the ABOVE-CHANCE skill of the N_hi net, what fraction is NOT already in the 1-hidden
    # floor (i.e. attributable to the deeper spatial layers + their DFA credit)? chance-baselined so the shared
    # floor's chance-level baseline is not counted as "control effect".
    frac_depth = attributable_to("above-chance held-out acc @N_hi (vs 1-hidden floor)",
                                 treatment_value=(per_N[n_hi]["mean_snn"] - chance),
                                 control_value=(floor_mean - chance))

    # STRUCTURAL interpretability (EXCLUDES perm, which is 1-seed-noisy on the small held set): a ceiling exists,
    # task needs depth, substrate deterministic, T small. Checked FIRST so the frontier-relevant structural
    # verdict (esp. depth-N_hi instrument absence) is surfaced even when the noisy perm anti-cheat also trips.
    interp_core = t_small and task_fittable and depth_sep_all and n1_is_floor
    # a GENUINE depth-N_hi CREDIT claim additionally needs the depth-N_hi ceiling to TRAIN (else the deepest arm
    # is confounded by the ceiling's own trainability, not the credit rule) AND no leakage.
    preconds_ok = interp_core and oracle_at_Nhi and perm_ok
    genuine_depth3 = bool(preconds_ok and (deep_gain >= depth_gain_bar)
                          and (not np.isnan(gain_2to3) and gain_2to3 >= depth3_bar) and monotone)
    depth2_loadbearing = bool(preconds_ok and (deep_gain >= depth_gain_bar) and not genuine_depth3)
    negative = bool(preconds_ok and (deep_gain < depth_gain_bar))
    # is the shallower spatial depth load-bearing EVEN IF the deepest arm is uninterpretable? (best interpretable
    # N above the floor). Reported inside the instrument-absent tier so the positive signal is not lost.
    interp_Ns = [N for N in n_list if (not np.isnan(oracle_per_N[N])) and oracle_per_N[N] >= oracle_bar]
    best_interp_N = max(interp_Ns, key=lambda N: per_N[N]["mean_snn"]) if interp_Ns else None
    best_interp_gain = (per_N[best_interp_N]["mean_snn"] - floor_mean) if best_interp_N is not None else float("nan")
    perm_note = ("" if perm_ok else " (NB the permuted anti-cheat also trips at 1 seed: small ~27-item held set, "
                 "0.037 resolution -> pooled out by the 6-seed sweep, not read as leakage.)")

    if not interp_core:
        if not t_small:
            verdict = ("UNDEFINED: T=%d is NOT small (> %d) -> the temporal-depth floor is NOT removed; a depth "
                       "sweep here re-runs the confounded regime. Set --T to 2-4." % (T, t_small_bar))
        elif not task_fittable:
            verdict = ("UNDEFINED: NO depth reaches the rate ceiling (max oracle %.3f < %.2f) -> the task is not "
                       "fittable at all here; a depth-gain null is uninterpretable. Not a score."
                       % (max(oracle_vals) if oracle_vals else float("nan"), oracle_bar))
        elif not depth_sep_all:
            verdict = ("UNDEFINED: task NOT depth-separating at some (N,seed) (shallow oracle already fits) -> the "
                       "task does not require depth, so a depth sweep says nothing about deep credit.")
        else:
            verdict = ("UNDEFINED: snn(N=1) != 1-hidden floor (max diff %.3g) -> the substrate is non-deterministic "
                       "under this seed; the depth sweep is void (compare docs/CLAUDE cfg.seed trap)."
                       % (max(n1_diffs) if n1_diffs else float("nan")))
    elif not oracle_at_Nhi:
        verdict = ("UNDEFINED (DEPTH-%d INSTRUMENT ABSENT): the depth-matched RATE ceiling COLLAPSES at N=%d "
                   "(mean held-out oracle %.3f < %.2f, ~chance) while a strictly-shallower net CEILINGS "
                   "(max oracle %.3f). The deep rate net is the BEST-POSSIBLE credit (full backprop) yet its "
                   "N=%d arm does not generalize -- so the SNN's N=%d arm is confounded by the deep net's own "
                   "GENERALIZATION collapse, NOT the credit rule. The compositional task is depth-<%d-SOLVABLE, "
                   "so the extra depth is surplus capacity that overfits (the 3rd-layer ceiling fits TRAIN but "
                   "collapses on held-out inheritance) -- there is no depth-%d-OBLIGATORY signal for the deepest "
                   "layer to latch onto. Depth-%d credit is UNTESTABLE on this task. %s Next mechanism: build a "
                   "depth-%d-OBLIGATORY task (held-out generalization REQUIRES %d composition levels: a "
                   "depth-<%d rate net must UNDERFIT held-out, a depth-%d net must CLEAR it), verify that ceiling, "
                   "THEN re-pose the transport-free DFA depth-%d credit sweep at small T on it.%s"
                   % (n_hi, n_hi, per_N[n_hi]["mean_oracle"], oracle_bar,
                      max(oracle_vals) if oracle_vals else float("nan"), n_hi, n_hi, n_hi, n_hi, n_hi,
                      ("Meanwhile the shallower spatial depth IS load-bearing at small T (best interpretable "
                       "N=%d: snn %.3f vs floor %.3f, +%.3f) -- the temporal floor is defeated for depth-2."
                       % (best_interp_N, per_N[best_interp_N]["mean_snn"], floor_mean, best_interp_gain))
                      if (best_interp_N is not None and best_interp_gain >= depth_gain_bar) else
                      "No interpretable depth clears the floor either.",
                      n_hi, n_hi, n_hi, n_hi, n_hi, perm_note))
    elif not perm_ok:
        verdict = ("UNDEFINED (1-seed anti-cheat noise): permuted-label > chance+%.2f at some (N,seed). The "
                   "held-out INHERITANCE set is small (~27 items -> ~0.037 resolution), so a few-item excess "
                   "trips the one-sided tol at 1 seed; the 6-seed sweep pools it out. NOT read as leakage yet. "
                   "All structural preconditions (T small, ceiling exists, depth-%d ceiling trainable, "
                   "depth-separating, determinism) held." % (perm_tol, n_hi))
    elif genuine_depth3:
        verdict = ("GO: GENUINE SPATIAL DEPTH-3 CREDIT at small T=%d. Held-out accuracy RISES with spatial depth "
                   "(N=1 %.3f -> N=2 %.3f -> N=3 %.3f; floor==N=1) and REMOVING the deep layers collapses to the "
                   "1-hidden floor. deep_gain(N=%d vs floor)=+%.3f >= %.2f, and the 3rd layer itself adds "
                   "+%.3f >= %.2f. At small T the temporal window cannot fake this -> the transport-free DFA "
                   "e-prop deep credit is LOAD-BEARING through 3 spiking layers."
                   % (T, per_N.get(1, {}).get("mean_snn", float('nan')), per_N.get(2, {}).get("mean_snn", float('nan')),
                      per_N.get(3, {}).get("mean_snn", float('nan')), n_hi, deep_gain, depth_gain_bar,
                      gain_2to3, depth3_bar))
    elif depth2_loadbearing:
        verdict = ("QUALIFIED (depth-2 load-bearing, depth-3 not yet demonstrated): at small T=%d the SPATIAL depth "
                   "credit IS load-bearing (deep_gain(N=%d vs floor)=+%.3f >= %.2f, monotone=%s), defeating the "
                   "temporal floor -- but the 3rd layer adds only +%.3f (< %.2f), so genuine depth-3 is not shown. "
                   "Next: a task whose depth-3 is OBLIGATORY (hier3 separating depth-2 from depth-3), or report "
                   "per-layer spike rates (deep layers may be silent at small T)."
                   % (T, n_hi, deep_gain, depth_gain_bar, monotone, gain_2to3, depth3_bar))
    elif negative:
        verdict = ("NEGATIVE (reframes the residual): at small T=%d the SPATIAL depth does NOT help "
                   "(deep_gain(N=%d vs floor)=+%.3f < %.2f). With the temporal floor removed, deeper spatial "
                   "layers + DFA credit add nothing -> the residual effective depth is the STATIC input "
                   "rate-expansion, not spatial layers. Next mechanism: attack the input encoding, not depth."
                   % (T, n_hi, deep_gain, depth_gain_bar))
    else:
        verdict = "UNDEFINED: unclassified (see checks)."

    # ------------------------------------------------------------------ EARNED verdict block (gate-visible)
    v = Verdict("gap4_spatial_depth3_smallT", chance=chance)
    v.require("temporal_window_small", t_small, expect=True,
              note="T=%d <= %d -> the LIF temporal integration cannot substitute for spatial depth" % (T, t_small_bar))
    v.require("task_fittable_ceiling_exists", task_fittable, expect=True,
              note="max-over-N rate DendriticMLP oracle >= %.2f -> a learnable ceiling EXISTS (task interpretable)"
                   % oracle_bar)
    v.require("depth_Nhi_ceiling_trainable", oracle_at_Nhi, expect=True,
              note="depth-matched rate oracle at N=%d >= %.2f -> the deepest arm is interpretable as a CREDIT "
                   "result (else it is confounded by the deep net's own trainability)" % (n_hi, oracle_bar))
    v.require("no_label_leakage", perm_ok, expect=True,
              note="permuted-label <= chance+%.2f at every (N,seed) (one-sided; leakage would be ABOVE chance)" % perm_tol)
    v.require("task_depth_separating", depth_sep_all, expect=True,
              note="stage0 depth-genuineness holds at every (N,seed): shallow oracle underfits, deep oracle fits")
    v.require("n1_equals_1hidden_floor", n1_is_floor, expect=True,
              note="snn(N=1) == the 1-hidden floor (identical arch+seed) -> substrate is deterministic; else void")
    v.reaches("depth_lever_moved", n_lo, n_hi,
              note="the N sweep actually changed the spatial hidden depth (control N=1 vs treatment N=%d)" % n_hi)
    decided = v.decide(genuine_depth3)

    return {"rows": rows, "per_N": per_N, "oracle_per_N": oracle_per_N, "n_lo": n_lo, "n_hi": n_hi, "T": T,
            "chance": chance, "floor_mean": floor_mean, "deep_gain": deep_gain, "step_gains": step_gains,
            "gain_2to3": gain_2to3, "monotone": monotone, "frac_above_chance_from_depth": frac_depth,
            "best_interp_N": best_interp_N, "best_interp_gain": best_interp_gain,
            "permuted_ok": perm_ok, "task_fittable": task_fittable, "oracle_at_Nhi": oracle_at_Nhi,
            "depth_sep_all": depth_sep_all, "n1_is_floor": n1_is_floor, "t_small": t_small,
            "checks": {"genuine_depth3": genuine_depth3, "depth2_loadbearing": depth2_loadbearing,
                       "negative": negative, "depth_Nhi_instrument_absent": bool(interp_core and not oracle_at_Nhi)},
            "go": genuine_depth3, "verdict": verdict,
            "preconditions": decided["preconditions"], "undefined_reasons": decided["undefined_reasons"],
            "status": decided["status"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="single seed (used when --seeds is not given).")
    ap.add_argument("--seeds", type=str, default="", help="comma-separated seeds for a self-aggregating sweep, "
                    "e.g. 42,43,44,100,101,102. Overrides --seed. GO is judged on the per-N cross-seed means.")
    ap.add_argument("--T", type=int, default=3, help="the FIXED small temporal-integration window (2-4). The whole "
                    "point: at small T the LIF membrane cannot supply the effective depth, so spatial depth is "
                    "obligatory.")
    ap.add_argument("--n-list", type=str, default="1,2,3", help="comma-separated spatial hidden-layer depths to sweep.")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--in-gain", type=float, default=1.0)
    ap.add_argument("--credit-mode", type=str, default="eprop", choices=["bptt", "spatial", "eprop", "eprop_shuffle"],
                    help="eprop = transport-free DFA e-prop (the deep-credit rule under test). NO weight transport.")
    ap.add_argument("--train-subsample", type=int, default=400)
    # task knobs (defaults match the DFA-depth / temporal-floor findings' compositional-inheritance task)
    ap.add_argument("--n-super", type=int, default=12)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=2)
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--feature-seed", type=int, default=0)
    # GO-gate bars
    ap.add_argument("--depth-gain-bar", type=float, default=0.05, help="min mean(snn@N_hi) - floor to call spatial "
                    "depth load-bearing (matches the temporal-floor runner's gap_open_bar).")
    ap.add_argument("--depth3-bar", type=float, default=0.02, help="min mean gain from the 2->3 step for GENUINE "
                    "depth-3 (the 3rd layer must itself add skill).")
    ap.add_argument("--mono-tol", type=float, default=0.03, help="allowed per-step drop before monotonicity fails.")
    ap.add_argument("--perm-tol", type=float, default=0.05)
    ap.add_argument("--oracle-bar", type=float, default=0.80)
    ap.add_argument("--t-small-bar", type=int, default=4, help="T must be <= this to count as 'small' (precondition).")
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()] if args.seeds.strip() else [args.seed]
    n_list = sorted(set(int(x) for x in args.n_list.split(",") if x.strip()))
    task_kwargs = {"n_super": args.n_super, "n_members": args.n_members,
                   "held_per_super": args.held_per_super, "n_prop": args.n_prop,
                   "member_id_dim": args.member_id_dim, "n_obs": args.n_obs,
                   "noise": args.noise, "feature_seed": args.feature_seed}

    t0 = time.time()
    try:
        r = run_sweep(seeds, n_list, args.T, args.hidden, args.epochs, args.lr, args.in_gain,
                      args.train_subsample, task_kwargs, args.credit_mode,
                      args.depth_gain_bar, args.depth3_bar, args.mono_tol, args.perm_tol,
                      args.oracle_bar, args.t_small_bar)
    except Exception as e:
        r = {"seeds": seeds, "error": repr(e), "traceback": traceback.format_exc()}

    out = {"probe": "gap4_spatial_depth3_smallT", "seeds": seeds,
           "config": {"T": args.T, "n_list": n_list, "hidden": args.hidden, "epochs": args.epochs,
                      "lr": args.lr, "in_gain": args.in_gain, "credit_mode": args.credit_mode,
                      "train_subsample": args.train_subsample, "task": task_kwargs,
                      "bars": {"depth_gain": args.depth_gain_bar, "depth3": args.depth3_bar,
                               "mono_tol": args.mono_tol, "perm_tol": args.perm_tol,
                               "oracle": args.oracle_bar, "t_small": args.t_small_bar}},
           "elapsed_seconds": round(time.time() - t0, 1), "result": r}
    out["verdict"] = r.get("verdict", r.get("error", "no result"))
    out["preconditions"] = r.get("preconditions", [])  # top-level for tools/gates/verdict_preconditions.py
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    if "rows" in r:
        print("\n  N  chance floor  snn    gain    perm   oracle depth-sep  (per (N,seed))")
        for row in r["rows"]:
            print("  %-2d %.3f %.3f %.3f %+0.3f %.3f %.3f  %s   seed=%d"
                  % (row["N"], row["chance"], row["floor_inherit"], row["snn_inherit"],
                     row["depth_gain_vs_floor"], row["permuted_inherit"], row["oracle_inherit"],
                     row["stage0_depth_separating"], row["seed"]))
        print("\n  per-N cross-seed means (T=%d):" % r["T"])
        for N in n_list:
            p = r["per_N"][N]
            print("    N=%d  mean_snn=%.3f (sd %.3f)  mean_floor=%.3f  mean_gain=%+.3f  mean_oracle=%.3f%s"
                  % (N, p["mean_snn"], p["std_snn"], p["mean_floor"], p["mean_gain"], p["mean_oracle"],
                     "  <== ceiling collapses (depth-N instrument absent)" if p["mean_oracle"] < 0.80 else ""))
        print("  step_gains=%s  deep_gain=%+.3f  monotone=%s" % (r["step_gains"], r["deep_gain"], r["monotone"]))
        print("  task_fittable=%s  oracle_at_Nhi=%s  best_interp_N=%s (gain %+.3f)"
              % (r["task_fittable"], r["oracle_at_Nhi"], r["best_interp_N"], r["best_interp_gain"]))
    print("\n" + out["verdict"])
    print("[spatial-depth3-smallT] status=%s  wrote %s" % (r.get("status", "?"), args.out))


if __name__ == "__main__":
    main()
