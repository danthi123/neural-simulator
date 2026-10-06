"""Grader for the sleep transitive-inference family (prereg
research/findings/2026-09-29-sleep-transitive-inference-PREREGISTRATION.md). Pure functions over a seed record written
by `research/runners/sleep_transitive_inference.py`; no brain is built here. `selftest()` runs the registered GO,
NO-GO and UNDEFINED fixtures.

Reads (per arm, from the record):
  DI   direct inference: mean over the two chains of act(D_k | A_k + Z, B_k silenced) - act(D_other | same probe)
  FI   free inference:   the same with B_k not silenced (retrieval-time chaining through B allowed)
  PAB  premise A->B:     act(B_k | A_k + Z) - act(B_other | same probe)
  PBD  premise B->D:     act(D_k | B_k + Z) - act(D_other | same probe)
  PBP  minor detail B->P: act(P_k | B_k + Z) - act(P_other | same probe)
  WAD  direct weight:    mean w(A_k -> D_k) - mean w(A_k -> D_other), pA
act(.) is the trial-mean peak smoothed active fraction of an assembly (0..1).
"""
from __future__ import annotations

# ---- registered bars (fixed at the prereg commit; see the prereg's "Bars") ----
PREMISE_MIN = 0.30          # STI0: a learned premise drives its partner at least this far above the foil
DI_MIN = 0.15               # STI1: the direct inference after sleep
GAIN_MIN = 0.15             # STI1/STI2/STI4: sleep gain over the comparison arm
NULL_MAX = 0.05             # STI2/STI4: an arm without the mechanism adds at most this over wake
SCR_FRACTION = 0.5          # STI3: the scramble removes at least half of the sleep gain
FOIL_SLACK = 0.05           # STI5: sleep raises the foil / unencoded reads by at most this over wake
SPURIOUS_MAX = 0.10         # STI5: an unencoded assembly never reaches this under the direct probe
W_GAIN_MIN = 20.0           # STI6: the direct weight rises by at least this (pA) after sleep
W_LESION_FRACTION = 0.1     # STI6: under the replay lesion A->D moves by at most this fraction of the sleep gain
KEEP_FRACTION = 0.9         # PM1: sleep keeps each premise at >= this fraction of its waking read
YOK_TOL = 1e-3              # instrument: the replica reproduces the engine to this (pA)
NECESSARY_FRACTION = 0.5    # necessity: a lesion removes at least half of the sleep gain

REQUIRED_ARMS = ("wake", "sleep", "off", "dis", "yok", "wperm", "scr", "rev", "cross_pre", "cross", "nohub_pre",
                 "nohub", "nostdp", "nocomp", "noadapt")
DIS_INIT_MIN = 0.9          # U2: the disrupted night still starts its events (the initiating assembly took part)
DIS_PROPAGATED_MAX = 0.10   # U2: ... and at most this fraction of its events reached a second assembly


def _r(rec, arm, key):
    return float(rec["arms"][arm]["reads"][key])


def _w(rec, arm, key):
    return float(rec["arms"][arm]["weights"][key])


def scr_valid(rec, arm="wperm"):
    """STI3's control is valid only if it keeps the premises and does not flood the foils (else a DI drop would come
    from wrecked specificity, not from missing structure). Binding on `wperm`; REPORTED for the raster scramble `scr`."""
    a = rec["arms"]
    return (_r(rec, arm, "PAB") >= PREMISE_MIN and _r(rec, arm, "PBD") >= PREMISE_MIN
            and float(a[arm]["reads"]["foil_direct_max"]) <= float(a["wake"]["reads"]["foil_direct_max"]) + FOIL_SLACK
            and float(a[arm]["reads"]["spurious_C_max"]) < SPURIOUS_MAX)


def undefined_reasons(rec):
    """U-rules: input, instrument and integrity failures (never a pass, never a zero). Lesions are read off the
    engine's own behaviour (the night's raster and weights), never off the host's flag."""
    out = []
    if rec.get("error"):
        out.append(f"U4 run error: {rec['error']}")
        return out
    arms = rec.get("arms", {})
    for a in REQUIRED_ARMS:
        if a not in arms:
            out.append(f"U4 arm missing: {a}")
    if out:
        return out
    for a, v in arms.items():
        rd = v.get("reads")
        if rd is not None and not rd.get("silence_held", False):
            out.append(f"U1 hub silencing did not hold in arm {a}")
    off = arms["off"]
    if sum(off.get("night", {}).get("replay_counts", [1])) != 0 or not off.get("weights_unchanged", False):
        out.append("U2 `off` was not silent (the raster shows replay participation, or a weight changed)")
    reach = arms["dis"].get("night", {}).get("reach", {})
    if not ((reach.get("frac_init_fired") or 0.0) >= DIS_INIT_MIN
            and reach.get("frac_propagated", 1.0) <= DIS_PROPAGATED_MAX):
        out.append(f"U2 the replay disruption did not hold (reach {reach})")
    if not arms["nostdp"].get("weights_unchanged", False):
        out.append("U2 plasticity lesion did not hold (nostdp changed the store)")
    for a in ("nocomp", "noadapt"):
        if not arms[a].get("lesion_held", False):
            out.append(f"U2 lesion did not hold in {a}")
    y = arms["yok"].get("replica", {})
    if not (y.get("max_abs_diff", 1e9) <= YOK_TOL):
        out.append(f"U3 replica does not reproduce the engine: max|dw|={y.get('max_abs_diff')}")
    for a in ("scr", "rev"):
        if not arms[a].get("spike_count_equal", False):
            out.append(f"U3 surrogate {a} changed per-cell spike counts")
    if not arms["wperm"].get("dw_multiset_equal", False):
        out.append("U3 the weight permutation changed the multiset of weight changes")
    if arms["sleep"].get("night", {}).get("n_init", 0) <= 0:
        out.append("U5 no SWR initiation in the sleep arm")
    if not scr_valid(rec, "wperm"):
        out.append("U6 the weight-permutation control lost the premises or flooded the foils: STI3 has no valid control")
    return out


def gates(rec):
    """Per-seed gate values. Each is a bool; the family verdict reads them."""
    g = {}
    d = {a: _r(rec, a, "DI") for a in ("wake", "sleep", "off", "dis", "wperm", "scr", "rev", "cross", "cross_pre",
                                       "nohub", "nohub_pre", "nocomp", "noadapt")}
    gain = d["sleep"] - d["wake"]
    g["STI0_premises_learned"] = (_r(rec, "wake", "PAB") >= PREMISE_MIN and _r(rec, "wake", "PBD") >= PREMISE_MIN)
    g["STI1_sleep_infers"] = (d["sleep"] >= DI_MIN and gain >= GAIN_MIN)
    # the replay lesion (disruption) removes A->D and leaves A->B and B->D
    g["STI2_replay_necessary"] = (d["dis"] - d["wake"] <= NULL_MAX and d["sleep"] - d["dis"] >= GAIN_MIN
                                  and _r(rec, "dis", "PAB") >= PREMISE_MIN and _r(rec, "dis", "PBD") >= PREMISE_MIN)
    # co-reactivation structure, at the same total plasticity (U6 guards the control's validity)
    g["STI3_structure_necessary"] = (gain > 0 and d["sleep"] - d["wperm"] >= SCR_FRACTION * gain)
    # temporal order: the reversed night loses the A->D inference, and the link it writes runs D->A
    g["STI3b_order_necessary"] = (gain > 0 and d["sleep"] - d["rev"] >= SCR_FRACTION * gain
                                  and _w(rec, "rev", "WDA") > _w(rec, "rev", "WAD"))
    # the chain decides where the inference goes: crossed chains flip its sign (premises of the crossed store learned)
    g["STI4_chain_decides"] = (d["cross"] - d["cross_pre"] <= -SCR_FRACTION * gain
                               and _r(rec, "cross_pre", "PAB") >= PREMISE_MIN
                               and -_r(rec, "cross_pre", "PBD") >= PREMISE_MIN)
    fw = float(rec["arms"]["wake"]["reads"]["foil_direct_max"])
    fs = float(rec["arms"]["sleep"]["reads"]["foil_direct_max"])
    g["STI5_no_false_inference"] = (fs <= fw + FOIL_SLACK
                                    and float(rec["arms"]["sleep"]["reads"]["spurious_C_max"]) < SPURIOUS_MAX)
    wgain = _w(rec, "sleep", "WAD") - _w(rec, "wake", "WAD")
    g["STI6_representation"] = (wgain >= W_GAIN_MIN
                                and abs(_w(rec, "dis", "WAD") - _w(rec, "wake", "WAD")) <= W_LESION_FRACTION * wgain)
    # the premises survive the night (the plan's "leaving A->B and B->D intact", for the normal night too)
    g["STI7_premises_survive"] = (_r(rec, "sleep", "PAB") >= max(PREMISE_MIN, KEEP_FRACTION * _r(rec, "wake", "PAB"))
                                  and _r(rec, "sleep", "PBD") >= max(PREMISE_MIN,
                                                                     KEEP_FRACTION * _r(rec, "wake", "PBD")))
    g["PM1_what_matters_kept"] = g["STI7_premises_survive"]
    cnt = rec["arms"]["sleep"]["night"]["replay_counts"]
    names = rec["names"]
    core = [cnt[names.index(n)] for n in ("A1", "B1", "D1", "A2", "B2", "D2")]
    det = [cnt[names.index(n)] for n in ("P1", "P2")]
    g["PM2_replay_follows_strength"] = (sum(core) / 6.0 > sum(det) / 2.0)
    g["PM3_detail_fades"] = (_r(rec, "sleep", "PBP") < _r(rec, "wake", "PBP"))
    return g


def necessity(rec):
    """Per mechanism: its lesion arm keeps less than half of the sleep gain (REPORTED per seed; counted over seeds).
    nohub is read against its own pre-sleep store."""
    d = {a: _r(rec, a, "DI") for a in ("wake", "sleep", "off", "dis", "wperm", "scr", "rev", "nohub", "nohub_pre",
                                       "nocomp", "noadapt")}
    gain = d["sleep"] - d["wake"]
    from tools.lab import attributable_to
    out = {}
    for mech, arm, base in (("offline_activity", "off", "wake"), ("swr_replay_propagation", "dis", "wake"),
                            ("weight_change_structure", "wperm", "wake"), ("replay_coreactivation", "scr", "wake"),
                            ("temporal_order", "rev", "wake"),
                            ("shared_item", "nohub", "nohub_pre"), ("pattern_completion", "nocomp", "wake"),
                            ("adaptation", "noadapt", "wake")):
        kept = d[arm] - d[base]
        frac = attributable_to(f"sleep gain vs {arm}", gain, kept) if gain > 0 else None
        out[mech] = dict(arm=arm, gain_kept=kept, attributable_fraction=frac,
                         necessary=bool(gain > 0 and kept < NECESSARY_FRACTION * gain))
    out["plasticity_during_replay"] = dict(arm="nostdp", by_construction=True,
                                           necessary=bool(rec["arms"]["nostdp"].get("weights_unchanged", False)
                                                          and gain > 0))
    return out


STI_GATES = ("STI0_premises_learned", "STI1_sleep_infers", "STI2_replay_necessary", "STI3_structure_necessary",
             "STI3b_order_necessary", "STI4_chain_decides", "STI5_no_false_inference", "STI6_representation",
             "STI7_premises_survive")
PM_GATES = ("PM1_what_matters_kept", "PM2_replay_follows_strength", "PM3_detail_fades")


def grade_seed(rec):
    u = undefined_reasons(rec)
    if u:
        return dict(verdict="UNDEFINED", undefined=u, pm_verdict="UNDEFINED")
    g = gates(rec)
    sti = "GO" if all(g[k] for k in STI_GATES) else "NO-GO"
    pm = "GO" if all(g[k] for k in PM_GATES) else "NO-GO"
    return dict(verdict=sti, pm_verdict=pm, gates=g, necessity=necessity(rec), scr_valid_reported=scr_valid(rec, "scr"),
                reads={a: {k: rec["arms"][a]["reads"][k] for k in ("DI", "FI", "PAB", "PBD", "PBP")}
                       for a in rec["arms"] if "reads" in rec["arms"][a]})


def family(seed_grades, seeds=(42, 43, 44, 100, 101, 102)):
    """INCOMPLETE if a seed is missing; UNDEFINED if any seed is UNDEFINED after its re-run budget; GO iff all six
    read GO; NO-GO otherwise. Same order for the PM verdict. Necessity: counted per mechanism over the defined seeds."""
    out = {}
    for key in ("verdict", "pm_verdict"):
        vals = [seed_grades.get(s, {}).get(key) for s in seeds]
        if any(v is None for v in vals):
            out[key] = "INCOMPLETE"
        elif any(v == "UNDEFINED" for v in vals):
            out[key] = "UNDEFINED"
        elif all(v == "GO" for v in vals):
            out[key] = "GO"
        else:
            out[key] = "NO-GO"
    nec = {}
    for s in seeds:
        for mech, v in (seed_grades.get(s, {}).get("necessity") or {}).items():
            nec.setdefault(mech, []).append(bool(v["necessary"]))
    out["necessity"] = {m: dict(necessary_on=sum(v), of=len(v),
                                label=("NECESSARY" if len(v) == len(seeds) and all(v) else
                                       "contributing" if any(v) else "not necessary"))
                        for m, v in nec.items()}
    return out


# ------------------------------------------------------------------------------------------------ selftest fixtures
def _fixture(**over):
    def reads(DI, FI=0.5, PAB=0.6, PBD=0.6, PBP=0.1, foil=0.02, spur=0.0):
        return dict(DI=DI, FI=FI, PAB=PAB, PBD=PBD, PBP=PBP, foil_direct_max=foil, spurious_C_max=spur,
                    silence_held=True)
    names = ["A1", "B1", "D1", "P1", "C1", "X1", "Y1", "A2", "B2", "D2", "P2", "C2", "X2", "Y2", "Z"]
    cnt = [20, 30, 30, 3, 0, 10, 10, 20, 30, 30, 3, 0, 10, 10, 0]
    arms = {
        "wake": dict(reads=reads(0.0, PBP=0.12), weights=dict(WAD=0.0, WDA=0.0)),
        "sleep": dict(reads=reads(0.4, PBP=0.08, PAB=0.58, PBD=0.57), weights=dict(WAD=120.0, WDA=60.0),
                      night=dict(n_init=160, replay_counts=cnt)),
        "off": dict(reads=reads(0.0), weights=dict(WAD=0.0, WDA=0.0), night=dict(n_init=0, replay_counts=[0] * 15),
                    weights_unchanged=True),
        "dis": dict(reads=reads(0.01), weights=dict(WAD=2.0, WDA=1.0),
                    night=dict(n_init=160, reach=dict(frac_init_fired=0.98, frac_propagated=0.03))),
        "yok": dict(replica=dict(max_abs_diff=0.0)),
        "wperm": dict(reads=reads(0.03, PAB=0.5, PBD=0.5), weights=dict(WAD=5.0, WDA=5.0), dw_multiset_equal=True),
        "scr": dict(reads=reads(0.05, PAB=0.1, PBD=0.1), weights=dict(WAD=10.0, WDA=10.0), spike_count_equal=True),
        "rev": dict(reads=reads(0.02, PAB=0.4, PBD=0.4), weights=dict(WAD=-5.0, WDA=110.0), spike_count_equal=True),
        "cross_pre": dict(reads=reads(0.0, PBD=-0.55), weights=dict(WAD=0.0, WDA=0.0)),
        "cross": dict(reads=reads(-0.38, PBD=-0.5), weights=dict(WAD=-115.0, WDA=-60.0)),
        "nohub_pre": dict(reads=reads(0.0), weights=dict(WAD=0.0, WDA=0.0)),
        "nohub": dict(reads=reads(0.01), weights=dict(WAD=0.0, WDA=0.0)),
        "nostdp": dict(reads=reads(0.0), weights=dict(WAD=0.0, WDA=0.0), weights_unchanged=True),
        "nocomp": dict(reads=reads(0.3), weights=dict(WAD=80.0, WDA=40.0), lesion_held=True),
        "noadapt": dict(reads=reads(0.35), weights=dict(WAD=90.0, WDA=40.0), lesion_held=True),
    }
    rec = dict(error=None, names=names, arms=arms)
    for k, v in over.items():
        arm, key = k.split("__")
        if key in ("DI", "FI", "PAB", "PBD", "PBP", "foil_direct_max", "spurious_C_max", "silence_held"):
            rec["arms"][arm]["reads"][key] = v
        elif key in ("WAD", "WDA"):
            rec["arms"][arm]["weights"][key] = v
        elif key == "max_abs_diff":
            rec["arms"][arm]["replica"][key] = v
        else:
            rec["arms"][arm][key] = v
    return rec


def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + name)
        ok = ok and cond

    g = grade_seed(_fixture())
    check("designed GO reads GO", g["verdict"] == "GO")
    check("designed PM reads GO", g["pm_verdict"] == "GO")
    check("no sleep gain -> NO-GO", grade_seed(_fixture(sleep__DI=0.02))["verdict"] == "NO-GO")
    check("disrupted replay also infers -> NO-GO", grade_seed(_fixture(dis__DI=0.3))["verdict"] == "NO-GO")
    check("disruption loses a premise -> NO-GO", grade_seed(_fixture(dis__PAB=0.1))["verdict"] == "NO-GO")
    check("weight permutation keeps the gain -> NO-GO", grade_seed(_fixture(wperm__DI=0.35))["verdict"] == "NO-GO")
    check("weight permutation wrecks the premises -> UNDEFINED (no valid control)",
          grade_seed(_fixture(wperm__PAB=0.05))["verdict"] == "UNDEFINED")
    check("weight permutation floods the foils -> UNDEFINED",
          grade_seed(_fixture(wperm__foil_direct_max=0.4))["verdict"] == "UNDEFINED")
    check("the raster scramble wrecking the premises is REPORTED, not a U-rule",
          grade_seed(_fixture())["verdict"] == "GO" and grade_seed(_fixture())["scr_valid_reported"] is False)
    check("reversed night keeps the gain -> NO-GO", grade_seed(_fixture(rev__DI=0.35))["verdict"] == "NO-GO")
    check("reversed night writes A->D not D->A -> NO-GO",
          grade_seed(_fixture(rev__WAD=100.0, rev__WDA=5.0))["verdict"] == "NO-GO")
    check("crossed chains do not flip -> NO-GO", grade_seed(_fixture(cross__DI=0.3))["verdict"] == "NO-GO")
    check("premises never learned -> NO-GO (scored, not voided)",
          grade_seed(_fixture(wake__PAB=0.1))["verdict"] == "NO-GO")
    check("the night erodes a premise -> STI NO-GO and PM NO-GO",
          (lambda r: r["verdict"] == "NO-GO" and r["pm_verdict"] == "NO-GO")(grade_seed(_fixture(sleep__PBD=0.2))))
    check("sleep raises the foil -> NO-GO", grade_seed(_fixture(sleep__foil_direct_max=0.3))["verdict"] == "NO-GO")
    check("weights do not move -> NO-GO", grade_seed(_fixture(sleep__WAD=5.0))["verdict"] == "NO-GO")
    check("disruption moves A->D -> NO-GO", grade_seed(_fixture(dis__WAD=50.0))["verdict"] == "NO-GO")
    check("replica mismatch -> UNDEFINED", grade_seed(_fixture(yok__max_abs_diff=0.5))["verdict"] == "UNDEFINED")
    check("silencing failed -> UNDEFINED", grade_seed(_fixture(sleep__silence_held=False))["verdict"] == "UNDEFINED")
    check("`off` not silent (engine raster) -> UNDEFINED",
          grade_seed(_fixture(off__night=dict(n_init=0, replay_counts=[1] + [0] * 14)))["verdict"] == "UNDEFINED")
    check("disruption did not truncate -> UNDEFINED",
          grade_seed(_fixture(dis__night=dict(n_init=160, reach=dict(frac_init_fired=0.98, frac_propagated=0.6))))
          ["verdict"] == "UNDEFINED")
    check("completion lesion regrew -> UNDEFINED", grade_seed(_fixture(nocomp__lesion_held=False))["verdict"] == "UNDEFINED")
    check("run error -> UNDEFINED", grade_seed(dict(error="boom"))["verdict"] == "UNDEFINED")
    check("detail kept -> PM NO-GO, STI unaffected",
          (lambda r: r["pm_verdict"] == "NO-GO" and r["verdict"] == "GO")(grade_seed(_fixture(sleep__PBP=0.2))))
    fam = family({s: grade_seed(_fixture()) for s in (42, 43, 44, 100, 101, 102)})
    check("six GO seeds -> family GO", fam["verdict"] == "GO")
    check("mechanisms necessary on 6/6 labelled NECESSARY",
          fam["necessity"]["swr_replay_propagation"]["label"] == "NECESSARY")
    fam2 = family({s: grade_seed(_fixture()) for s in (42, 43, 44, 100, 101)})
    check("a missing seed -> INCOMPLETE", fam2["verdict"] == "INCOMPLETE")
    fam3 = {s: grade_seed(_fixture()) for s in (42, 43, 44, 100, 101)}
    fam3[102] = grade_seed(dict(error="x"))
    check("an UNDEFINED seed -> family UNDEFINED", family(fam3)["verdict"] == "UNDEFINED")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


OP_SATURATION_MAX = 0.05    # operating point: the normal night leaves the foil and never-encoded reads below this


def devcheck(rec):
    """The registered operating-point criteria (prereg section 3), on a DEV record; none of them reads DI.
    Returns {criterion: bool} and 'all'."""
    a = rec.get("arms", {})
    c = rec.get("cfg", {})
    out = {}
    try:
        out["premises_learned"] = (_r(rec, "wake", "PAB") >= PREMISE_MIN and _r(rec, "wake", "PBD") >= PREMISE_MIN)
        out["no_saturation"] = (float(a["sleep"]["reads"]["foil_direct_max"]) < OP_SATURATION_MAX
                                and float(a["sleep"]["reads"]["spurious_C_max"]) < OP_SATURATION_MAX)
        out["events_apart"] = (float(c["swr_period"]) * float(c["dt"]) > 5.0 * float(c["stdp_tau"]))
        reach = a["dis"]["night"]["reach"]
        out["disruption_holds"] = ((reach.get("frac_init_fired") or 0.0) >= DIS_INIT_MIN
                                   and reach.get("frac_propagated", 1.0) <= DIS_PROPAGATED_MAX)
        out["no_error"] = not rec.get("error")
    except (KeyError, TypeError) as e:
        out["record_complete"] = False
        out["missing"] = str(e)
    out["all"] = all(v for k, v in out.items() if isinstance(v, bool))
    return out


def merge_parts(parts):
    """Merge one seed's part records (the pool runs a seed as several jobs, each a subset of arms). U7: the parts must
    share the seed, the constants and, for every part that encoded the hub store, the post-waking weight hash."""
    errs = [p.get("error") for p in parts if p.get("error")]
    base = dict(parts[0])
    out = dict(seed=base.get("seed"), cfg=base.get("cfg"), names=base.get("names"), arms={}, parts=[], error=None)
    if errs:
        out["error"] = "; ".join(str(e) for e in errs)
        return out
    hashes = set()
    for p in parts:
        cfg_p = {k: v for k, v in (p.get("cfg") or {}).items()}
        if p.get("seed") != out["seed"] or cfg_p != out["cfg"]:
            out["error"] = "U7 parts disagree on the seed or the constants"
            return out
        if p.get("encode"):
            hashes.add(p["encode"]["w_hash"])
            out.setdefault("encode", p["encode"]); out.setdefault("pre_weights", p.get("pre_weights"))
        for a, v in p["arms"].items():
            if a in out["arms"]:
                out["error"] = f"U7 arm {a} appears in two parts"
                return out
            out["arms"][a] = v
        if p.get("sleep_trainctx_DI") is not None:
            out["sleep_trainctx_DI"] = p["sleep_trainctx_DI"]
        out["parts"].append(dict(arms=p.get("arms_requested"), backend=p.get("backend"),
                                 timing=p.get("timing"), encode_hash=(p.get("encode") or {}).get("w_hash")))
    if len(hashes) > 1:
        out["error"] = f"U7 the parts' waking stores differ: {sorted(hashes)}"
    return out


def main():
    import argparse
    import json
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--merge", nargs="+", metavar=("OUT", "PART"), help="OUT PART [PART ...]: merge and grade one seed")
    ap.add_argument("--family", metavar="DIR", help="grade DIR/seed<S>.json for the six gate seeds")
    ap.add_argument("--devcheck", nargs="+", metavar="DEVJSON", help="operating-point criteria on dev records; writes "
                    "opcheck.json beside the first one and exits 0 iff every record passes every criterion")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    if a.devcheck:
        res = {}
        for fn in a.devcheck:
            rec = json.load(open(fn))
            if int(rec.get("seed", -1)) in (42, 43, 44, 100, 101, 102):
                raise SystemExit(f"{fn}: a gate seed is not a dev record")
            res[fn] = devcheck(rec)
        ok = bool(res) and all(v["all"] for v in res.values())
        outp = os.path.join(os.path.dirname(a.devcheck[0]), "opcheck.json")
        backends = sorted({str(json.load(open(fn)).get("backend")) for fn in a.devcheck})
        json.dump(dict(records=res, all_pass=ok, backend=",".join(backends),
                       device="as recorded in each dev record (cupy = the local RTX 3090 via tools/gpu_queue.sh)"),
                  open(outp, "w"), indent=1)
        print(json.dumps(dict(records=res, all_pass=ok), indent=1))
        raise SystemExit(0 if ok else 1)
    if a.merge:
        outp, parts = a.merge[0], a.merge[1:]
        rec = merge_parts([json.load(open(p)) for p in parts])
        rec["grade"] = grade_seed(rec)
        json.dump(rec, open(outp, "w"), indent=1, default=float)
        print(outp, rec["grade"].get("verdict"), rec["grade"].get("pm_verdict"), rec["grade"].get("undefined"))
        return
    if a.family:
        seeds = (42, 43, 44, 100, 101, 102)
        grades = {}
        for sd in seeds:
            fn = os.path.join(a.family, f"seed{sd}.json")
            if os.path.exists(fn):
                grades[sd] = grade_seed(json.load(open(fn)))
        fam = family(grades, seeds)
        agg = dict(family=fam, per_seed={str(k): dict(verdict=v.get("verdict"), pm_verdict=v.get("pm_verdict"),
                                                      gates=v.get("gates"), undefined=v.get("undefined"))
                                         for k, v in grades.items()})
        json.dump(agg, open(os.path.join(a.family, "aggregate.json"), "w"), indent=1, default=float)
        print(json.dumps(fam, indent=1))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
