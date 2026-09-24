"""BRAIN_XEDGE_IN_WAVE3 verification (prereg: docs/plans/2026-09-24-xedge-in-wave3-PREREG.md).

Modes (each ARM runs in its OWN fresh process -- pools are memoized per process, so arms never share state):
  --g1-arm off|on   other-organ isolated reads (OFF = plain Wave-3 pool; ON = flag-ON pool, read at build AND after
                    exercising the cross-edge: credited live turns, per-session d6 loads, focused comprehension)
  --g1-compare OFF.json ON.json   -> the G1 verdict (exact, tolerance 0.0)
  --g2-run alone_A|alone_B|interleaved [--learn]   the two-session script (see prereg §3 G2)
  --g2-compare A.json B.json I.json               -> the G2 verdict
  --selftest        G3: _selftest_loadbearing on the fresh holder (W0) and after the in-pool PART-2 curriculum,
                    plus _selftest_livelearn after growing

Every run: SIM_BACKEND=numpy, cfg.seed set by the pool builders (merge_organs -> _base_config(seed)).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

# prereg §3 constants (copied, not derived)
N_EXERCISE_TURNS = 6
G1_TOL = 0.0
G3_INTACT_MIN = 1e-3
G3_LESION_MAX = 1e-9
G3_CURRICULUM_TURNS = 80
ENDPOINTS = ("d6_multiref_wm", "comprehension")
SESSIONS = {
    "A": {"msg": "the wolf chased the dog", "item": ("wolf", "chase", "dog"), "refs": ("wolf", "dog")},
    "B": {"msg": "the cat watched the bird", "item": ("cat", "watch", "bird"), "refs": ("cat", "bird")},
}
HOLD_QUERY = "who are we talking about?"
STEPS = ("load", "hold", "comp", "hold2")


def _peak_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 3)


def _jsonable(x):
    """Exact, order-stable serialisation (floats via repr so equality is bitwise)."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return repr(float(x))
    if x is None or isinstance(x, str):
        return x
    return repr(x)


def _set_flag(on: bool):
    if on:
        os.environ["BRAIN_XEDGE_IN_WAVE3"] = "1"
    else:
        os.environ.pop("BRAIN_XEDGE_IN_WAVE3", None)


def _isolated(pool, seed):
    from research.runners._onebrain_wave3_organread_verify import _wave3_descriptors
    from research.runners._onebrain_wave1_organread_verify import _isolated_reads
    descs = _wave3_descriptors()
    R, A, _ = _isolated_reads(pool, descs, seed)
    return {"reads": _jsonable(R), "answers": _jsonable(A)}


def _exercise(holder, pool, seed):
    """Drive every new code path on the flag-ON pool: credited live turns (both directions), two per-session d6
    loads, and focused comprehension judge + repair_target."""
    from research.runners.d6_multiref_wm_production_organ import MultiReferentWMOrgan
    pa, pp = holder.role["p_agent"], holder.role["p_patient"]
    traces = []
    for t in range(N_EXERCISE_TURNS):
        agent = (t % 2 == 0)
        tr = holder.credit_live_turn("agent" if agent else "patient", focus=(pa if agent else pp))
        traces.append({k: tr[k] for k in ("credited", "teach", "margin")} if tr else None)
    outs = []
    corg = holder.comp_organ
    for key in ("A", "B"):
        s = SESSIONS[key]
        org = MultiReferentWMOrgan(seed=seed, shared=holder.pool)
        dj = org.judge(s["msg"])
        text = " ".join(s["item"])
        cj = corg.judge(text, wm_focus=org.current_focus())
        rt = corg.repair_target(text, wm_focus=org.current_focus())
        outs.append({"session": key, "d6": _jsonable(dj), "comp": _jsonable(cj), "repair": _jsonable(rt)})
    return {"credit_traces": _jsonable(traces), "session_outputs": outs}


def g1_arm(arm: str, seed: int, out: str):
    t0 = time.time()
    _set_flag(arm == "on")
    from research.runners.onebrain_wave3_pool_production import get_merged_cortical_pool
    from research.runners import comprehension_production_organ as CO
    from research.runners import onebrain_xedge_production as XE
    pool = get_merged_cortical_pool(seed, min_wave=1)
    res = {"mode": "g1_arm", "arm": arm, "seed": seed, "backend": os.environ.get("SIM_BACKEND"),
           "n_neurons": int(pool.bridge.core_config.num_neurons),
           "pool_is_xedge_in_wave3": bool(getattr(pool, "xedge_in_wave3", False))}
    corg = CO.get_organ(seed)
    corg.ensure_built()
    if arm == "on":
        holder = XE.get_xedge_pool(seed)
        assert holder is not None and holder.pool is pool and holder.comp_organ is corg, "routing not reconciled"
        res["plasticity_values"] = holder._r3pool.plasticity_values()
        res["cross_weights_build"] = dict(holder.cross_weights)
    res["build"] = _isolated(pool, seed)
    res["t_build_reads_s"] = round(time.time() - t0, 1)
    if arm == "on":
        res["exercise"] = _exercise(holder, pool, seed)
        res["cross_weights_exercised"] = dict(holder.cross_weights)
        res["n_session_resets_exercise"] = int(holder._r3pool.n_session_resets)
        res["exercised"] = _isolated(pool, seed)
    res["peak_rss_gb"] = _peak_gb()
    res["wall_s"] = round(time.time() - t0, 1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print(f"wrote {out} wall={res['wall_s']}s rss={res['peak_rss_gb']}GB", flush=True)


def _maxdelta(a, b):
    """max |a-b| over the union of (flattened) numeric keys; strings compared for equality."""
    worst, wk, missing = 0.0, None, []
    ka, kb = set(a), set(b)
    for k in sorted(ka | kb):
        if k not in ka or k not in kb:
            missing.append(k)
            continue
        va, vb = a[k], b[k]
        try:
            d = abs(float(va) - float(vb))
        except (TypeError, ValueError):
            d = 0.0 if va == vb else float("inf")
        if d > worst:
            worst, wk = d, k
    return worst, wk, missing


def _flat(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(_flat(v, f"{prefix}{i}."))
    else:
        out[prefix.rstrip(".")] = d
    return out


def g1_compare(off_path, on_path, out):
    off = json.loads(Path(off_path).read_text())
    on = json.loads(Path(on_path).read_text())
    rows = {}
    for state in ("build", "exercised"):
        for organ, r_off in off["build"]["reads"].items():
            r_on = on[state]["reads"].get(organ, {})
            d, wk, miss = _maxdelta(_flat(r_off), _flat(r_on))
            ans_same = off["build"]["answers"].get(organ) == on[state]["answers"].get(organ)
            rows.setdefault(organ, {})[state] = {"maxdelta": d, "worst_key": wk, "missing": miss,
                                                 "answer_same": bool(ans_same),
                                                 "identical": bool(d <= G1_TOL and not miss and ans_same)}
    others = [k for k in rows if k not in ENDPOINTS]
    g1 = all(rows[k][s]["identical"] for k in others for s in ("build", "exercised"))
    endpoints_ok = all(rows[k][s]["identical"] for k in ENDPOINTS if k in rows for s in ("build", "exercised"))
    from tools.verdict import Verdict
    v = Verdict(f"G1 other-organ read identity, flag ON vs OFF, seed {on['seed']}")
    v.require("ON arm read the flag-ON (xedge-in-wave3) pool", on.get("pool_is_xedge_in_wave3"), expect=True)
    v.require("OFF arm read the plain Wave-3 pool", off.get("pool_is_xedge_in_wave3"), expect=False)
    v.require("9 non-endpoint organs read in both arms", len(others), expect=9)
    cwb, cwe = on.get("cross_weights_build") or {}, on.get("cross_weights_exercised") or {}
    v.reaches("the exercise moved the cross-edge (max candidate weight)",
              before=max(cwb.values()) if cwb else None, after=max(cwe.values()) if cwe else None)
    dec = v.decide(go=bool(g1), verbose=False)
    res = {"mode": "g1_compare", "seed": on["seed"], "off": off_path, "on": on_path, "rows": rows,
           "n_other_organs": len(others), "G1_other_organs_identical": bool(g1),
           "endpoints_identical": bool(endpoints_ok),
           "cross_weights_build": on.get("cross_weights_build"),
           "cross_weights_exercised": on.get("cross_weights_exercised"),
           "exercise_credit_traces": on.get("exercise", {}).get("credit_traces"),
           "status": dec["status"], "preconditions": dec["preconditions"],
           "undefined_reasons": dec["undefined_reasons"],
           "verdict": ("UNDEFINED" if dec["status"] == "UNDEFINED" else
                       "G1 PASS (no substrate change)" if g1 else
                       "G1 FAIL -> SUBSTRATE CHANGE (re-measure pooled rows)")}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("G1_other_organs_identical", "endpoints_identical", "verdict")}))
    for k, v in rows.items():
        print(f"  {k:20s} build d={v['build']['maxdelta']:.3g} ans={v['build']['answer_same']}  "
              f"exercised d={v['exercised']['maxdelta']:.3g} ans={v['exercised']['answer_same']}")


def g2_run(script: str, learn: bool, seed: int, out: str, pregrow: int = 0):
    t0 = time.time()
    _set_flag(True)
    from research.runners.onebrain_wave3_pool_production import get_merged_cortical_pool
    from research.runners import comprehension_production_organ as CO
    from research.runners import onebrain_xedge_production as XE
    from research.runners.d6_multiref_wm_production_organ import MultiReferentWMOrgan
    pool = get_merged_cortical_pool(seed, min_wave=1)
    holder = XE.get_xedge_pool(seed)
    corg = CO.get_organ(seed)
    assert holder is not None and holder.pool is pool and holder.comp_organ is corg, "routing not reconciled"
    # PREGROW (additional arm, not in the prereg): grow the edge with `pregrow` credited turns BEFORE any session
    # exists (deterministic per process, identical across the three runs), so the leak test also runs with a
    # non-trivial edge that the WM-resolved-role read actually engages.
    pre = []
    pa, pp = holder.role["p_agent"], holder.role["p_patient"]
    for t in range(int(pregrow)):
        agent = (t % 2 == 0)
        tr = holder.credit_live_turn("agent" if agent else "patient", focus=(pa if agent else pp))
        pre.append(bool(tr and tr.get("credited")))
    cw_pre = dict(holder._r3pool.cross_weights())
    # per-session organs exactly as webapp.server._get_multiref_organ builds them (shared = the holder's pool)
    organs = {}
    if script == "alone_A":
        order = [("A", s) for s in STEPS]
    elif script == "alone_B":
        order = [("B", s) for s in STEPS]
    else:
        order = [(k, s) for s in STEPS for k in ("A", "B")]
    outputs = {"A": [], "B": []}
    credits = []
    for key, step in order:
        if key not in organs:
            organs[key] = MultiReferentWMOrgan(seed=seed, shared=holder.pool)
        org, s = organs[key], SESSIONS[key]
        if step == "load":
            o = org.judge(s["msg"])
        elif step in ("hold", "hold2"):
            o = org.judge(HOLD_QUERY)
        else:
            text = " ".join(s["item"])
            foc = org.current_focus()
            cj = corg.judge(text, wm_focus=foc)
            rt = corg.repair_target(text, wm_focus=foc)
            o = {"focus": foc, "judge": cj, "repair": rt}
            if learn and cj is not None:
                tr = XE.credit_live_turn_from_comprehension(corg, cj["svo"], wm_focus=foc)
                credits.append({"session": key, "trace": _jsonable(tr)})
                o["credit"] = tr
        outputs[key].append({"step": step, "out": _jsonable(o)})
    res = {"mode": "g2_run", "script": script, "learn": bool(learn), "seed": seed,
           "pregrow": int(pregrow), "pregrow_credited": pre, "cross_weights_after_pregrow": cw_pre,
           "outputs": outputs, "credits": credits,
           "n_session_resets": int(holder._r3pool.n_session_resets),
           "cross_weights_end": dict(holder._r3pool.cross_weights()),
           "peak_rss_gb": _peak_gb(), "wall_s": round(time.time() - t0, 1)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print(f"wrote {out} resets={res['n_session_resets']} wall={res['wall_s']}s rss={res['peak_rss_gb']}GB",
          flush=True)


def _names_in(obj, names):
    blob = json.dumps(obj).lower()
    return sorted(n for n in names if f'"{n}"' in blob or f" {n} " in f" {blob} " or f"'{n}'" in blob)


def g2_compare(a_path, b_path, i_path, out):
    A = json.loads(Path(a_path).read_text())
    B = json.loads(Path(b_path).read_text())
    I = json.loads(Path(i_path).read_text())
    rows = {}
    ok_equal = True
    for key, alone in (("A", A), ("B", B)):
        per = []
        for sa, si in zip(alone["outputs"][key], I["outputs"][key]):
            fa, fi = _flat(sa["out"]), _flat(si["out"])
            diff = sorted(k for k in set(fa) | set(fi) if fa.get(k) != fi.get(k))
            per.append({"step": sa["step"], "equal": not diff, "diff_keys": diff[:20]})
            ok_equal &= not diff
        rows[key] = per
    other = {"A": SESSIONS["B"]["refs"], "B": SESSIONS["A"]["refs"]}
    cross = {k: _names_in(I["outputs"][k], other[k]) for k in ("A", "B")}
    no_cross = not any(cross.values())
    guard_fired = I["n_session_resets"] > 0 and A["n_session_resets"] == 0 and B["n_session_resets"] == 0
    learn = bool(I.get("learn"))
    # NON-VACUITY: every session's d6 load recovered exactly its own referents, and every comp step produced a judge
    from tools.verdict import Verdict
    v = Verdict(f"G2 two-session leak ({'learning' if learn else 'transient'}), seed {I['seed']}")

    def _own_recovered(run, key):
        o = run["outputs"][key][0]["out"] or {}
        return sorted((o.get("recovered") or {}).values()) == sorted(SESSIONS[key]["refs"])

    def _judged(run, key):
        return all((s["out"] or {}).get("judge") is not None for s in run["outputs"][key] if s["step"] == "comp")

    v.require("alone_A / alone_B / interleaved all ran the same learn setting",
              A.get("learn") == B.get("learn") == I.get("learn"), expect=True)
    v.require("each session's load recovered exactly its own referents (all 4 runs)",
              all(_own_recovered(r, k) for r, k in ((A, "A"), (B, "B"), (I, "A"), (I, "B"))), expect=True)
    v.require("every comp step produced a comprehension judge", all(_judged(r, k) for r, k in (
        (A, "A"), (B, "B"), (I, "A"), (I, "B"))), expect=True)
    v.require("session guard fired interleaved and never alone", bool(guard_fired), expect=True)
    # learning arm: the claim is that interleaving changes NOTHING but the shared learned weight (the credit trace)
    only_weight = all(k.startswith("credit.") for per in rows.values() for r in per for k in r["diff_keys"])
    dec = v.decide(go=bool(no_cross and (only_weight if learn else ok_equal)), verbose=False)
    res = {"mode": "g2_compare", "learn": learn, "seed": I["seed"], "rows": rows,
           "status": dec["status"], "preconditions": dec["preconditions"],
           "undefined_reasons": dec["undefined_reasons"],
           "all_outputs_equal_alone_vs_interleaved": bool(ok_equal),
           "cross_session_referent_names": cross, "no_cross_referent": bool(no_cross),
           "session_guard_fired": bool(guard_fired),
           "n_session_resets": {"alone_A": A["n_session_resets"], "alone_B": B["n_session_resets"],
                                "interleaved": I["n_session_resets"]},
           "cross_weights_end": {"alone_A": A["cross_weights_end"], "alone_B": B["cross_weights_end"],
                                 "interleaved": I["cross_weights_end"]}}
    res["diffs_confined_to_shared_weight"] = bool(only_weight)
    if learn:
        res["verdict"] = ("INFORMATIONAL (learning arm): interleaving changed only the shared learned weight"
                          if dec["status"] == "GO" else
                          "INFORMATIONAL (learning arm): interleaving changed more than the shared weight"
                          if dec["status"] == "NO-GO" else "UNDEFINED")
    else:
        g2 = bool(ok_equal and no_cross and guard_fired and dec["status"] == "GO")
        res["G2_transient_pass"] = g2
        res["verdict"] = ("UNDEFINED" if dec["status"] == "UNDEFINED" else
                          "G2 PASS (no transient cross-session leak)" if g2 else "G2 FAIL")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("verdict", "all_outputs_equal_alone_vs_interleaved", "no_cross_referent",
                                          "session_guard_fired", "n_session_resets")}))


def selftest(seed: int, out: str):
    t0 = time.time()
    _set_flag(True)
    from research.runners import onebrain_xedge_production as XE
    from research.runners._onebrain_integration_r2_threefactor_selforganized import GATE
    from sim.backend import to_host
    holder = XE.get_xedge_pool(seed)                       # production default: per-turn build (edge at W0)
    assert holder is not None and holder.in_wave3, "flag-ON holder did not build in-wave3"
    p = holder._r3pool
    b = holder.bridge

    def snap_w():
        return np.asarray(to_host(b.cp_connections.data)).copy()

    def restore_w(w):
        b.cp_connections.data = holder.pool.xp.asarray(w, dtype=b.cp_connections.data.dtype)

    res = {"mode": "selftest", "seed": seed, "role": holder.role, "in_wave3": True}
    w0 = snap_w()
    res["cross_weights_w0"] = p.cross_weights()
    lb0 = XE._selftest_loadbearing(holder, seed)
    restore_w(w0)
    res["loadbearing_w0"] = _jsonable(lb0)
    t1 = time.time()
    b.set_plasticity_gate(GATE, 1.0)                        # R3Pool leaves the gate OPEN for the build curriculum
    traj = XE.grow_live_selfsupervised(p, n_turns=G3_CURRICULUM_TURNS)   # re-freezes on return
    holder.grow_traj = traj
    holder.cross_weights = p.cross_weights()
    res["t_curriculum_s"] = round(time.time() - t1, 1)
    res["grow_traj"] = _jsonable(traj)
    res["cross_weights_grown"] = holder.cross_weights
    wg = snap_w()
    lbg = XE._selftest_loadbearing(holder, seed)
    restore_w(wg)
    res["loadbearing_grown"] = _jsonable(lbg)
    try:
        ll = XE._selftest_livelearn(holder, seed)
        res["livelearn_grown"] = _jsonable(ll)
    except Exception as e:                                  # informational; never masks G3
        res["livelearn_grown"] = {"error": f"{type(e).__name__}: {e}"}
    restore_w(wg)
    mi, ml = float(lbg["max_abs_dNet_intact"]), float(lbg["max_abs_dNet_lesioned"])
    res["G3_intact_max_abs_dNet"] = mi
    res["G3_lesioned_max_abs_dNet"] = ml
    # ATTRIBUTION: how much of the intact hold(p_agent)-vs-hold(p_patient) net-lean differential the cross-edge owns.
    from tools.lab import attributable_to
    frac = attributable_to(f"seed{seed} in-wave3 xedge net-lean drive vs cross-edge lesion", mi, ml)
    res["G3_frac_attributable_to_cross_edge"] = None if frac is None else float(frac)
    from tools.verdict import Verdict
    v = Verdict(f"G3 in-wave3 xedge load-bearing selftest, seed {seed}")
    v.require("holder built in-wave3 (flag ON)", bool(holder.in_wave3), expect=True)
    v.reaches("the curriculum grew the cross-edge (max candidate weight)",
              before=max(res["cross_weights_w0"].values()), after=max(res["cross_weights_grown"].values()))
    v.control("cross-edge lesion vs intact (max |dNet|)", treatment=mi, control=ml)
    dec = v.decide(go=bool(mi > G3_INTACT_MIN and ml < G3_LESION_MAX), verbose=False)
    res["status"], res["preconditions"] = dec["status"], dec["preconditions"]
    res["undefined_reasons"] = dec["undefined_reasons"]
    res["G3_pass"] = bool(dec["status"] == "GO")
    res["verdict"] = ("UNDEFINED" if dec["status"] == "UNDEFINED" else
                      "G3 PASS (load-bearing, lesion-attributable; de-risk seed %d)" % seed if res["G3_pass"]
                      else "G3 FAIL")
    res["peak_rss_gb"] = _peak_gb()
    res["wall_s"] = round(time.time() - t0, 1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("verdict", "G3_intact_max_abs_dNet", "G3_lesioned_max_abs_dNet",
                                          "cross_weights_grown", "peak_rss_gb", "wall_s")}, default=str), flush=True)


CHAT_SCRIPT = (("A", "the wolf chased the dog"), ("B", "the cat watched the bird"),
               ("A", "who are we talking about?"), ("B", "who are we talking about?"),
               ("A", "the wolf watched the dog"), ("B", "the cat chased the bird"))


def chat_smoke(flag: str, seed: int, out: str):
    """INTEGRATION SMOKE (not a prereg gate): the two-session interleaved script through the REAL
    `webapp.server.brain_chat` handler (tiny-demo, stub renderer, LLM disabled), flag on or off. Records each turn's
    answer + the multiref / comprehension / xedge_live_learn fields, and the live object identities afterwards."""
    t0 = time.time()
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    os.environ["BRAIN_CHAT_SEED"] = str(int(seed))
    _set_flag(flag == "on")
    import webapp.server as S
    turns = []
    for sess, msg in CHAT_SCRIPT:
        r = S.brain_chat(S.BrainChatRequest(session=f"a4-{sess}", message=msg, brain="tiny-demo",
                                            renderer="stub", rich=False))
        body = json.loads(bytes(r.body))
        keep = {k: body.get(k) for k in ("answer", "response", "abstained", "multiref", "comprehension",
                                         "inner_state_readout") if k in body}
        turns.append({"session": sess, "message": msg, "out": _jsonable(keep)})
    from research.runners import comprehension_production_organ as CO
    from research.runners import onebrain_xedge_production as XE
    from research.runners.onebrain_wave3_pool_production import get_merged_cortical_pool
    cseed = S._brain_chat_seed()
    xp = XE.get_xedge_pool(cseed)
    merged = get_merged_cortical_pool(cseed, min_wave=1)
    orgs = dict(S._SESSION_MULTIREF)
    ident = {"comp_is_xedge_comp_organ": bool(xp is not None and CO.get_organ(cseed) is xp.comp_organ),
             "multiref_shared_is_merged": {str(k): bool(o._shared is merged) for k, o in orgs.items()},
             "n_session_resets": (int(xp._r3pool.n_session_resets)
                                  if xp is not None and getattr(xp, "in_wave3", False) else None),
             "cross_weights": (dict(xp.cross_weights) if xp is not None else None)}
    other = {"A": SESSIONS["B"]["refs"], "B": SESSIONS["A"]["refs"]}
    cross = {k: _names_in([t["out"] for t in turns if t["session"] == k], other[k]) for k in ("A", "B")}
    res = {"mode": "chat_smoke", "flag": flag, "seed": seed, "turns": turns, "identity": ident,
           "cross_session_referent_names": cross, "peak_rss_gb": _peak_gb(), "wall_s": round(time.time() - t0, 1)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"identity": ident, "cross": cross, "rss": res["peak_rss_gb"], "wall": res["wall_s"]},
                     default=str), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--g1-arm", choices=("off", "on"))
    ap.add_argument("--g1-compare", nargs=2, metavar=("OFF", "ON"))
    ap.add_argument("--g2-run", choices=("alone_A", "alone_B", "interleaved"))
    ap.add_argument("--learn", action="store_true")
    ap.add_argument("--pregrow", type=int, default=0)
    ap.add_argument("--g2-compare", nargs=3, metavar=("ALONE_A", "ALONE_B", "INTERLEAVED"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--chat-smoke", choices=("on", "off"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    if a.g1_arm:
        g1_arm(a.g1_arm, a.seed, a.out)
    elif a.g1_compare:
        g1_compare(a.g1_compare[0], a.g1_compare[1], a.out)
    elif a.g2_run:
        g2_run(a.g2_run, a.learn, a.seed, a.out, pregrow=a.pregrow)
    elif a.g2_compare:
        g2_compare(*a.g2_compare, a.out)
    elif a.selftest:
        selftest(a.seed, a.out)
    elif a.chat_smoke:
        chat_smoke(a.chat_smoke, a.seed, a.out)
    else:
        ap.error("pick a mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
