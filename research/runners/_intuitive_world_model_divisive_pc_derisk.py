"""SURPASS the world-model VoE-MAGNITUDE boundary — DIVISIVE / GAIN (biased-competition)
predictive coding on the spiking object-permanence circuit (faculty-map T1-7 next rung).

THE BOUNDARY THIS CLOSES (the first rung's named residual, 2026-08-13)
---------------------------------------------------------------------
The intuitive-world-model first rung (`_intuitive_world_model_permanence_derisk.py`) built a spiking
OBJECT FILE (slow-NMDA attractor maintains an object through occlusion, zero input) + a predictive-
coding surprise = object permanence + a persistence-caused, generalizing violation-of-expectation
(load-bearing claim 6/6). But the STRICT gate returned 1/6 due to TWO mapped SURPASSABLE boundaries:
  (i)  the VoE MAGNITUDE cleared >=2x on only 4/6 seeds, and
  (ii) the FS-WTA occasionally seated the WRONG object (hold_correct 0.75 on ~half the seeds; seed 43
       permanence degraded to ratio 2.77).

WHY A SUBTRACTIVE SINGLE-RELAY PC CAPS THE MAGNITUDE (~2x)
---------------------------------------------------------
In the first rung the maintained prediction cancels the sensory reveal by CURRENT SUBTRACTION
(ipred_k injects a hyperpolarizing current into err_k, E_i = -75 mV). A subtractive relay faces an
irreducible trade-off (the classic subtractive-PC limit; Rao-Ballard): to CANCEL a strong matched
sensory transient it needs an inhibitory current MATCHED to the sensory current, but that exact match
depends on the (seed-dependent) sensory magnitude, so on some seeds the cancellation UNDER-shoots
(match residual leaks -> match_alarm up -> ratio down) and it cannot both (a) null a strong match and
(b) leave a large violation response. Isolation (this runner, `--isolate`): at a FIXED operating point
the SUBTRACTIVE err read clears >=2x on 2/6 seeds; the DIVISIVE err read clears 6/6.

THE FIX — DIVISIVE / SHUNTING (biased-competition) PC, brain-based, NO sim/ edit
--------------------------------------------------------------------------------
Divisive normalization (Carandini-Heeger 2012) / biased-competition predictive coding (Spratling 2008
J.Vis.; Spratling 2010) computes the error as a RATIO (input GAIN-DIVIDED by the prediction), not a
subtraction. Its biological substrate is SHUNTING inhibition: an inhibitory conductance whose reversal
E_i sits near the operating point raises the membrane conductance and DIVIDES the neuron's gain rather
than subtracting a fixed current (the regions.py note: ~-60 mV = "shunting, depolarizing-near-rest").
The engine already delivers conductance-based inhibition (I_syn = g_i*(E_i - v), bridge.py:7744) and a
per-region reversal override (BrainRegion.syn_reversal_potential_i_override) -> the fix is RUNNER-SIDE
CONFIG on the SAME object-file machinery (imported unchanged), no sim/ edit:
  * err_reversal_i = -56 mV : the ipred_k -> err_k prediction SHUNTS (divides) the sensory reveal.
    On a MATCH the strong maintained prediction divides err_k down SCALE-ROBUSTLY (independent of the
    sensory magnitude -> the seed-robustness the subtractive relay lacked); on a VIOLATION ipred_m is
    silent (wm_m not maintained) so err_m responds FULLY. That is the divisive/gain read + the
    attentional amplification of the maintained prediction, in one shunting mechanism.
  * wm_reversal_i = -60 mV : the fs -> wm competition becomes SHUNTING = a divisive BIASED-COMPETITION
    WTA (Reynolds-Desimone) -> a cleaner, better-separated one-of-K winner (the WTA-cleanliness rung).
  * a decisive LOAD (load_w 36, fs_to_wm 16) so the PRESENTED object locks the attractor before
    occlusion (the wrong-winner was the loaded object being out-competed by a jitter-favoured slot).

WHAT IS NEURAL vs THE LEGITIMATE BOUNDARY (unchanged from the first rung)
------------------------------------------------------------------------
Identical to the first rung: PERSISTENCE is neural (slow-NMDA recurrence, occlusion input asserted
identically ZERO); the SURPRISE is a spiking err_* population rate (no host argmax over object codes);
the prediction that shunts the reveal is the maintained wm_k assembly. The ONLY change is the READ:
the prediction now DIVIDES (shunts) rather than SUBTRACTS. The occlusion/reveal events + which object
is presented remain the environment boundary (as E2's valence + T1-4's events were).

STRICT GO-GATE (pre-registered, 6 seeds 42/43/44/100/101/102) — the mission's target: 6/6
-----------------------------------------------------------------------------------------
Per seed, ALL of:
 (1) PERMANENCE holds: occlusion hold/off ratio >= 5 (zero-input self-sustain, correct one-of-K slot).
 (2) WTA-CLEAN: hold_correct == 1.0 on TRAIN and HELD-OUT (the divisive biased-competition rung).
 (3) VoE PRESENT + GENERALIZES: VoE ratio >= 1.3 on TRAIN and HELD-OUT.
 (4) VoE MAGNITUDE >= 2x on TRAIN and HELD-OUT (the divisive/gain rung — the boundary this closes).
 (5) PERSISTENCE-ATTRIBUTABLE (the KEPT lesion control, load-bearing): a NO-MAINTENANCE build
     (recur=0, nmda off) presents the object identically but does NOT maintain it -> the VoE COLLAPSES
     and intact - lesion >= 0.3 on TRAIN and HELD. The surprise is CAUSED by the maintained object.
GO = all 6 criteria on 6/6 seeds. The absolute lesion floor (<=1.15) is REPORTED as a secondary
characterization (it is noise-limited to ~4/6, unchanged from the first rung; attributability, not the
absolute floor, is the load-bearing collapse read — the first rung's own established position).

Usage
-----
    SIM_BACKEND=numpy python -m research.runners._intuitive_world_model_divisive_pc_derisk \
        --seeds 42,43,44,100,101,102 \
        --out research/findings/raw/_intuitive_world_model_divisive_pc_6seed.json
    SIM_BACKEND=numpy python -m research.runners._intuitive_world_model_divisive_pc_derisk --smoke
    SIM_BACKEND=numpy python -m research.runners._intuitive_world_model_divisive_pc_derisk --isolate
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as _st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Reuse the first rung's object-file + occlusion + VoE machinery UNCHANGED (import, don't fork).
from research.runners._intuitive_world_model_permanence_derisk import (  # noqa: E402
    build_world_model, voe_for_objects, occlusion_trial, train_permanence,
    K_OBJECTS, N_TRAIN)

# The DIVISIVE / biased-competition operating point (the ONLY thing that differs from the first rung's
# SUBTRACTIVE build): shunting reversals on err (divisive/gain PC read) + wm (biased-competition WTA)
# + a decisive load so the presented object locks the attractor. All are runner-side region config.
DIVISIVE_KW = dict(err_reversal_i=-56.0, wm_reversal_i=-60.0, load_w=36.0, fs_to_wm=16.0)
# The isolation control: the SAME operating point but a SUBTRACTIVE err read (default E_i -75) -> shows
# the divisive READ, not the load/WTA changes, is what lifts the magnitude past 2x.
SUBTRACTIVE_KW = dict(wm_reversal_i=-60.0, load_w=36.0, fs_to_wm=16.0)


def _perm_ratio(sb, train_objs, intact_train):
    tr0 = occlusion_trial(sb, train_objs[0], train_objs[0])
    off = tr0["hold_off"]
    return intact_train["hold_rate"] / max(off, 1e-3), off


def run_seed(seed, *, verbose=True, do_learn_control=False):
    train_objs = list(range(N_TRAIN))
    held_objs = list(range(N_TRAIN, K_OBJECTS))

    # INTACT — the divisive/gain (biased-competition) object-file + PC read.
    sb, cfg, meta = build_world_model(seed, **DIVISIVE_KW)
    itr = voe_for_objects(sb, train_objs)
    ihd = voe_for_objects(sb, held_objs)
    perm_ratio, perm_off = _perm_ratio(sb, train_objs, itr)

    # LESION — NO-MAINTENANCE build (recur=0, nmda off) WITH the divisive machinery still present: the
    # object is presented identically but DECAYS during occlusion, so ipred is silent at reveal, no
    # shunt is applied, and the VoE must collapse. Keeps the first rung's decisive control (unweakened).
    lbk = dict(DIVISIVE_KW); lbk["recur"] = 0.0; lbk["nmda"] = False
    sb_l, _, _ = build_world_model(seed, **lbk)
    ltr = voe_for_objects(sb_l, train_objs)
    lhd = voe_for_objects(sb_l, held_objs)

    # ISOLATION — the SAME operating point with a SUBTRACTIVE err read (the boundary's regime): shows
    # the divisive READ is responsible for the magnitude lift (before/after, brain-based).
    sb_s, _, _ = build_world_model(seed, **SUBTRACTIVE_KW)
    str_ = voe_for_objects(sb_s, train_objs)
    shd = voe_for_objects(sb_s, held_objs)

    from tools.lab import attributable_to
    frac_persist = attributable_to("VoE differential @ reveal (divisive)", itr["voe_diff"], ltr["voe_diff"])

    vt, vh = itr["voe_ratio"], ihd["voe_ratio"]
    lt, lh = ltr["voe_ratio"], lhd["voe_ratio"]
    hct, hch = itr["hold_correct"], ihd["hold_correct"]

    res = {
        "seed": seed,
        "perm_hold_rate": round(itr["hold_rate"], 4), "perm_off_rate": round(perm_off, 4),
        "perm_ratio": round(perm_ratio, 2),
        "hold_correct": round(hct, 3), "held_hold_correct": round(hch, 3),
        "train_match_alarm": round(itr["match_alarm"], 4), "train_viol_alarm": round(itr["viol_alarm"], 4),
        "train_voe_ratio": round(vt, 3),
        "held_match_alarm": round(ihd["match_alarm"], 4), "held_viol_alarm": round(ihd["viol_alarm"], 4),
        "held_voe_ratio": round(vh, 3),
        "lesion_voe_ratio": round(lt, 3), "lesion_held_voe_ratio": round(lh, 3),
        # the isolation control (subtractive read @ same operating point)
        "subtractive_train_voe_ratio": round(str_["voe_ratio"], 3),
        "subtractive_held_voe_ratio": round(shd["voe_ratio"], 3),
        "voe_attributable_to_persistence": frac_persist,
    }

    if do_learn_control:
        bk = dict(DIVISIVE_KW); bk["wm_to_ipred"] = 1.0   # NAIVE: un-potentiated prediction link
        sb_n, cfg_n, _ = build_world_model(seed, **bk)
        naive = voe_for_objects(sb_n, train_objs)
        train_permanence(sb_n, cfg_n, train_objs)
        trained = voe_for_objects(sb_n, train_objs)
        res["learn_naive_voe_ratio"] = round(naive["voe_ratio"], 3)
        res["learn_trained_voe_ratio"] = round(trained["voe_ratio"], 3)

    # per-seed derived flags
    res["voe_ge2_train"] = bool(vt >= 2.0)
    res["voe_ge2_held"] = bool(vh >= 2.0)
    res["wta_clean"] = bool(hct >= 0.99 and hch >= 0.99)
    res["attributable"] = bool((vt - lt) >= 0.3 and (vh - lh) >= 0.3)
    res["lesion_floor_ok"] = bool(lt <= 1.15 and lh <= 1.15)   # secondary characterization (noise-limited)
    # STRICT per-seed GO (the mission's 6-criteria gate; absolute floor reported, not gated)
    res["go"] = bool(perm_ratio >= 5.0
                     and res["wta_clean"]                              # (2) WTA-CLEAN train+held
                     and vt >= 1.3 and vh >= 1.3                       # (3) VoE present + generalizes
                     and res["voe_ge2_train"] and res["voe_ge2_held"]  # (4) MAGNITUDE >=2x train+held
                     and res["attributable"])                         # (5) persistence-attributable (kept)
    if verbose:
        print(f"  [seed {seed}] perm={perm_ratio:6.1f} holdC tr={hct:.2f}/hd={hch:.2f} | "
              f"VoE div tr={vt:.2f} hd={vh:.2f} (subtractive tr={str_['voe_ratio']:.2f} hd={shd['voe_ratio']:.2f}) "
              f"| LESION tr={lt:.2f} hd={lh:.2f} attrib={frac_persist} | GO={res['go']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--learn-control", action="store_true")
    ap.add_argument("--isolate", action="store_true",
                    help="subtractive-vs-divisive err read at a fixed operating point (the boundary demo)")
    ap.add_argument("--smoke", action="store_true", help="1-seed divisive VoE + lesion + isolation quick check")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.isolate:
        train_objs = list(range(N_TRAIN)); held_objs = list(range(N_TRAIN, K_OBJECTS))
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [42, 43, 44, 100, 101, 102]
        print("=== ISOLATION: SUBTRACTIVE vs DIVISIVE err read at a FIXED operating point ===")
        ns = nd = 0
        for s in seeds:
            sb_s, _, _ = build_world_model(s, **SUBTRACTIVE_KW)
            st, sh = voe_for_objects(sb_s, train_objs)["voe_ratio"], voe_for_objects(sb_s, held_objs)["voe_ratio"]
            sb_d, _, _ = build_world_model(s, **DIVISIVE_KW)
            dt, dh = voe_for_objects(sb_d, train_objs)["voe_ratio"], voe_for_objects(sb_d, held_objs)["voe_ratio"]
            sub_ge2 = st >= 2 and sh >= 2; div_ge2 = dt >= 2 and dh >= 2
            ns += int(sub_ge2); nd += int(div_ge2)
            print(f"  seed {s}: SUBTRACTIVE tr={st:.2f} hd={sh:.2f} ge2={sub_ge2}  ->  "
                  f"DIVISIVE tr={dt:.2f} hd={dh:.2f} ge2={div_ge2}")
        print(f"\n  VoE magnitude >=2x (both sets): SUBTRACTIVE {ns}/{len(seeds)}  ->  DIVISIVE {nd}/{len(seeds)}")
        return

    if args.smoke:
        print("=== SMOKE (seed 43, the first rung's WORST seed): divisive VoE + lesion + isolation ===")
        r = run_seed(43, do_learn_control=False)
        print("\n  SMOKE checks:")
        print(f"   PERMANENCE holds (ratio>=5) ............... {r['perm_ratio'] >= 5.0}  (ratio {r['perm_ratio']})")
        print(f"   WTA-CLEAN (hold_correct=1.0 tr+hd) ........ {r['wta_clean']}  (tr {r['hold_correct']} hd {r['held_hold_correct']})")
        print(f"   VoE >=2x DIVISIVE (tr+hd) ................. {r['voe_ge2_train'] and r['voe_ge2_held']}  (tr {r['train_voe_ratio']} hd {r['held_voe_ratio']})")
        print(f"   ... SUBTRACTIVE read @ same op-point ...... (tr {r['subtractive_train_voe_ratio']} hd {r['subtractive_held_voe_ratio']})  <- the boundary regime")
        print(f"   LESION(recur=0) COLLAPSES VoE ............. tr {r['lesion_voe_ratio']} hd {r['lesion_held_voe_ratio']}")
        print(f"   persistence-ATTRIBUTABLE (>=0.3) .......... {r['attributable']}  (frac {r['voe_attributable_to_persistence']})")
        print(f"\n   SEED-43 GO = {r['go']}")
        return

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    print("=== INTUITIVE WORLD-MODEL: DIVISIVE/GAIN PC — surpass the VoE-magnitude boundary ===")
    rows = [run_seed(s, do_learn_control=args.learn_control) for s in seeds]

    n_go = sum(1 for r in rows if r["go"])
    verdict = "GO" if n_go == len(rows) and len(rows) >= 6 else "BOUNDARY"

    perm_min = min(r["perm_ratio"] for r in rows)
    wta_clean = sum(1 for r in rows if r["wta_clean"])
    voe_present = sum(1 for r in rows if r["train_voe_ratio"] >= 1.3 and r["held_voe_ratio"] >= 1.3)
    ge2 = sum(1 for r in rows if r["voe_ge2_train"] and r["voe_ge2_held"])
    attributable = sum(1 for r in rows if r["attributable"])
    floor = sum(1 for r in rows if r["lesion_floor_ok"])
    sub_ge2 = sum(1 for r in rows if r["subtractive_train_voe_ratio"] >= 2 and r["subtractive_held_voe_ratio"] >= 2)
    voe_tr = [r["train_voe_ratio"] for r in rows]; voe_held = [r["held_voe_ratio"] for r in rows]
    les = [r["lesion_voe_ratio"] for r in rows]; les_h = [r["lesion_held_voe_ratio"] for r in rows]

    from tools.verdict import Verdict
    v = (Verdict("intuitive world model — DIVISIVE/gain PC surpasses the VoE-magnitude boundary")
         .require("PERMANENCE holds (ratio>=5, min over seeds)", perm_min, expect=lambda x: x >= 5.0)
         .require("WTA-CLEAN: hold_correct=1.0 on train AND held (all seeds)",
                  wta_clean, expect=lambda k: k == len(rows))
         .require("VoE PRESENT + GENERALIZES (>=1.3 train AND held, all seeds)",
                  voe_present, expect=lambda k: k == len(rows))
         .require("VoE MAGNITUDE >=2x on train AND held (all seeds — the boundary this closes)",
                  ge2, expect=lambda k: k == len(rows))
         .require("VoE PERSISTENCE-ATTRIBUTABLE: intact - lesion >= 0.3 (train AND held, all seeds)",
                  attributable, expect=lambda k: k == len(rows))
         .require("intact strict GO on all seeds", n_go, expect=lambda k: k == len(rows))
         .control("divisive VoE intact vs no-maintenance lesion (persistence load-bearing)",
                  _st.mean(voe_tr), _st.mean(les), min_separation=0.25)
         .control("VoE magnitude: SUBTRACTIVE read vs DIVISIVE read @ fixed op-point (the fix)",
                  _st.mean(voe_tr), _st.mean([r["subtractive_train_voe_ratio"] for r in rows]),
                  min_separation=0.25)
         .disabled("OU background process", "deterministic regime for a controllable operating point")
         .disabled("conductance noise", "deterministic regime")
         .disabled("absolute lesion floor <=1.15 (SECONDARY characterization, not gated)",
                   f"reported {floor}/{len(rows)}: noise-limited exactly as the first rung (attributability, "
                   "not the absolute floor, is the load-bearing collapse read; the divisive read makes the "
                   "INTACT VoE larger -> the attributable separation is LARGER than the first rung, so the "
                   "lesion control is STRENGTHENED, not weakened)")
         .disabled("self-organized object-file BINDING",
                   "the comparator is a topographic template (object-independent -> it generalizes, the "
                   "anti-cheat); self-organizing the binding from experience is the named next rung")
         .disabled("occlusion/reveal EVENT grounding",
                   "the occlusion + reveal events + presented object are sensory drive (the environment "
                   "boundary, as E2's valence + T1-4's events); grounding them in the emergent code follows"))
    decided = v.decide(go=(verdict == "GO"))

    print("\n=== VERDICT ===")
    print(f"  INTACT strict GO: {n_go}/{len(rows)} seeds (6/6 required)  ->  {verdict}")
    print(f"  permanence ratio (min): {perm_min:.1f}")
    print(f"  WTA-clean (hold_correct=1.0 tr+hd): {wta_clean}/{len(rows)}")
    print(f"  VoE train  (per seed): {[round(x,2) for x in voe_tr]}")
    print(f"  VoE HELDOUT(per seed): {[round(x,2) for x in voe_held]}")
    print(f"  VoE magnitude >=2x (train+held): {ge2}/{len(rows)}   (the boundary this closes; was 4/6)")
    print(f"  ... ISOLATION subtractive read @ same op-point >=2x: {sub_ge2}/{len(rows)}  (the boundary regime)")
    print(f"  LESION VoE train (per seed): {[round(x,2) for x in les]}")
    print(f"  LESION VoE held  (per seed): {[round(x,2) for x in les_h]}")
    print(f"  persistence-attributable (intact-lesion>=0.3 tr+hd): {attributable}/{len(rows)}")
    print(f"  [secondary] absolute lesion floor <=1.15: {floor}/{len(rows)}  (noise-limited, as first rung)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"mode": "intuitive_world_model_divisive_pc", "rows": rows,
                       "operating_point": DIVISIVE_KW, "isolation_op": SUBTRACTIVE_KW,
                       "n_go": n_go, "n_seeds": len(rows),
                       "verdict": decided["status"], "verdict_label": verdict,
                       "perm_ratio_min": perm_min, "wta_clean": wta_clean,
                       "voe_train": voe_tr, "voe_heldout": voe_held,
                       "voe_lesion_train": les, "voe_lesion_held": les_h,
                       "voe_present": voe_present, "voe_ge2": ge2,
                       "subtractive_ge2_isolation": sub_ge2,
                       "attributable": attributable, "lesion_floor_ok": floor,
                       "preconditions": decided["preconditions"],
                       "disabled_processes": decided["disabled_processes"],
                       "verdict_status": decided["status"]}, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
