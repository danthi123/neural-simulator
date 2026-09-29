"""Live-turn host share of the BG selection seam and the ignition-bus read, before and after the neural seam.

Plan step 3 (docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md). Preregistration:
research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md. Handoff: docs/drafts/selection-seam/HANDOFF.md.

Scoring reuses lane C's instrument (`research.runners.host_share`: `reply_diff`, `sham_diff`, `classify_seam`, `_get`);
this module adds only the BG turns, the arms and the in-process wrappers. If lane C's module is not on this branch
yet, set HOST_SHARE_INSTRUMENT_DIR to the directory holding its host_share.py (DEV only).

TURNS (one session, one fresh ChatBrain per arm, the live `webapp.server.brain_chat`; stub renderer, numpy):
  hsb_empty1 '...'   hsb_q 'what does the dog chase'   hsb_empty2 '?'   hsb_hi 'hi'      (all target turns)

ARMS
  prod_a                flag off, no wrapper                    (production)
  prod_b                flag off, observe wrappers              (sham: prod_a == prod_b proves observing is inert)
  prod_intervene        flag off; on target turns the host decision is flipped: a turn the host would not send to
                        the race is sent with (speak 0, silent 1); a turn it would send is sent with (speak 1, silent 0)
  prod_substitute       flag off; on target turns decide_action is replaced by decide_action_neural (the neural seam)
  seam_b                BRAIN_BG_NEURAL_SEAM=1 + BRAIN_GNW_UNIQUE_IGNITION_READ=1, observe wrappers
  seam_readout          as seam_b; on target turns the committed action is re-read with the production first-to-12
                        motor rule on the SAME trial (neural state fixed, only the host read changes)
  seam_lesion_value     as seam_b + BRAIN_BG_NEURAL_SEAM_LESION=value (neural intervention: heard/hold -> D1 closed)

Run one arm (~6 GB RSS; one ChatBrain build):
  bash tools/memcap.sh 12 -- env PYTHONHASHSEED=0 .venv/bin/python -m research.runners.host_share_bg_seam \
      --worker --seed 7 --arm prod_a --out research/findings/raw/_bg_live_seam/dev/s7/prod_a.json
Score:
  .venv/bin/python -m research.runners.host_share_bg_seam --score research/findings/raw/_bg_live_seam/dev --seeds 7 \
      --out research/findings/raw/_bg_live_seam/dev/score.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.lab import attributable_to  # noqa: E402

GATE_SEEDS = (42, 43, 44, 100, 101, 102)
SESSION = "hs_bg_0"
TURNS = [("hsb_empty1", "...", "empty"), ("hsb_q", "what does the dog chase", "content"),
         ("hsb_empty2", "?", "empty"), ("hsb_hi", "hi", "one_word")]
ARMS = {
    "prod_a": ({}, "none"), "prod_b": ({}, "observe"),
    "prod_intervene": ({}, "intervene"), "prod_substitute": ({}, "substitute"),
    "seam_b": ({"BRAIN_BG_NEURAL_SEAM": "1", "BRAIN_GNW_UNIQUE_IGNITION_READ": "1"}, "observe"),
    "seam_readout": ({"BRAIN_BG_NEURAL_SEAM": "1", "BRAIN_GNW_UNIQUE_IGNITION_READ": "1"}, "readout"),
    "seam_lesion_value": ({"BRAIN_BG_NEURAL_SEAM": "1", "BRAIN_GNW_UNIQUE_IGNITION_READ": "1",
                           "BRAIN_BG_NEURAL_SEAM_LESION": "value"}, "observe"),
}
BG_FIELDS_EXTRA = ("bg_select.action",)
_EV = {"label": None, "events": {}}


def lane_c():
    """Lane C's host-share instrument module (the shared scorer)."""
    try:
        return importlib.import_module("research.runners.host_share")
    except ImportError:
        d = os.environ.get("HOST_SHARE_INSTRUMENT_DIR")
        if not d:
            raise
        import research.runners as RR
        RR.__path__.append(d)
        return importlib.import_module("research.runners.host_share")


def _rec(**kw):
    _EV["events"].setdefault(_EV["label"], []).append(kw)


def install(mode):
    """Observe / intervene / substitute / readout wrappers on the BG seam, plus an observe-only record of the ignition
    bus read (every norgan_hop call re-read under the alternative host rules; the returned value is untouched)."""
    from research.runners import bg_action_selection_production_organ as BG
    import research.runners._gnw_norgan_bus_derisk as NB
    from research.runners._gnw_coincidence_integrator_derisk import _assign_slots
    orig_decide, orig_neural = BG.decide_action, BG.decide_action_neural

    def decide_action(msg, seed=42, lesion=None):
        sp, si = BG.salience(msg)
        consulted = si > sp
        if mode == "intervene" and _EV["label"]:
            les = BG.bg_select_lesion() if lesion is None else lesion
            flip = (1.0, 0.0) if consulted else (0.0, 1.0)
            r = BG.get_organ(seed=seed, lesion=les).select_once(*flip)
            _rec(kind="bg", mode=mode, applied=True, consulted=consulted, flip=list(flip), winner=r["winner"])
            if not r["committed"] or r["winner"] != BG.STAY_SILENT:
                return None
            return {"action": BG.ACTION_NAME[BG.STAY_SILENT], "winner": int(r["winner"]), "speak_salience": flip[0],
                    "silent_salience": flip[1], "decision_step": r["decision_step"], "loser_ratio": r["loser_ratio"],
                    "lesion": les}
        if mode == "substitute" and _EV["label"]:
            out = orig_neural(msg, seed=seed)
            _rec(kind="bg", mode=mode, applied=True, neural_winner=(BG.decide_action_neural.last or {}).get("winner"))
            return out
        out = orig_decide(msg, seed=seed, lesion=lesion)
        _rec(kind="bg", mode="observe", consulted=consulted, speak_salience=sp, silent_salience=si,
             host_action=(out or {}).get("action"))
        return out

    def decide_action_neural(msg, seed=42, lesion=None):
        out = orig_neural(msg, seed=seed, lesion=lesion)
        t = BG.decide_action_neural.last or {}
        rd = dict(t.get("readouts") or {})
        _rec(kind="bg_neural", winner=t.get("winner"), latch_rates=t.get("latch_rates"), readouts=rd,
             v_range=[t.get("v_min"), t.get("v_max")])
        if mode == "readout" and _EV["label"]:
            alt = rd.get("first_to_12_motor (production rule)")
            _rec(kind="bg_readout", applied=True, latch_winner=t.get("winner"), alt_winner=alt)
            if alt != BG.STAY_SILENT:
                return None
            return {"action": BG.ACTION_NAME[BG.STAY_SILENT], "winner": 1, "speak_salience": None,
                    "silent_salience": None, "decision_step": None, "loser_ratio": None, "lesion": None}
        return out

    BG.decide_action, BG.decide_action_neural = decide_action, decide_action_neural
    decide_action_neural.last = None
    BG.decide_action_neural.last = None

    orig_hop = NB.norgan_hop

    def norgan_hop(bridge, xp, slots_dev, snap, candidates, decoy, d_sub, **kw):
        out = orig_hop(bridge, xp, slots_dev, snap, candidates, decoy, d_sub, **kw)
        present = [c for c in candidates if c is not None]
        _slot_of, order = _assign_slots(present + ([decoy] if decoy is not None else []))
        _rec(kind="bus", committed=out[0], n_ignited=out[3], rates=[round(float(r), 5) for r in out[1]],
             variants=NB.readout_variants(out[1], order, len(slots_dev)))
        return out

    for modname in ("webapp.gnw_three_organ_bus", "webapp.gnw_bus_shadow"):
        m = importlib.import_module(modname)
        if getattr(m, "norgan_hop", None) is orig_hop:
            m.norgan_hop = norgan_hop


def run_worker(seed, arm, out_path, backend="numpy"):
    env_extra, mode = ARMS[arm]
    C = lane_c()
    env = dict(C.BASE_ENV)
    if backend != "numpy":                               # DEV-only replication on the GPU backend
        env["SIM_BACKEND"] = backend
        env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(env_extra)
    env["BRAIN_CHAT_SEED"] = str(int(seed))
    for k in ("BRAIN_BG_NEURAL_SEAM", "BRAIN_GNW_UNIQUE_IGNITION_READ", "BRAIN_BG_NEURAL_SEAM_LESION"):
        env.setdefault(k, "")                         # explicit empty = OFF in every arm that does not set it
    for k, v in env.items():
        os.environ[k] = v
    import research.runners.onebrain_regression_battery as B
    for i, (label, msg, _cls) in enumerate(TURNS):
        B._TURN_BY_LABEL[label] = (label, msg, SESSION, i == 0, None, True)
    import webapp.server as S
    if mode != "none":
        install(mode)
    orig_bc = S.brain_chat
    by_msg = {msg: label for label, msg, _ in TURNS}

    def brain_chat(req):
        _EV["label"] = by_msg.get(req.message)
        try:
            return orig_bc(req)
        finally:
            _EV["label"] = None

    S.brain_chat = brain_chat
    t0 = time.time()
    resp_path = out_path + ".responses.json"
    B._collect_worker(json.dumps(env), [t[0] for t in TURNS], resp_path)
    responses = json.load(open(resp_path))
    os.remove(resp_path)
    res = {"instrument": "host_share_bg_seam", "seed": int(seed), "arm": arm, "mode": mode, "env": env,
           "pythonhashseed_ok": os.environ.get("PYTHONHASHSEED") == "0", "responses": responses,
           "events": _EV["events"], "wall_seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=1, default=str)
    print("[bg seam worker] seed=%s arm=%s -> %s (%.0fs)" % (seed, arm, out_path, res["wall_seconds"]), flush=True)
    return 0


def run_all(seed, out_dir, backend="numpy", arms=None):
    """Every arm of one seed, each in its OWN fresh subprocess (PYTHONHASHSEED=0), sequentially; an existing arm file is
    kept (resumable). One brain-loading process at a time, so it is safe as ONE gpu_queue job."""
    import subprocess
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    ok = True
    for arm in (arms or list(ARMS)):
        out = os.path.join(out_dir, "s%d" % int(seed), arm + ".json")
        if os.path.exists(out):
            continue
        cmd = [sys.executable, "-u", "-m", "research.runners.host_share_bg_seam", "--worker", "--seed", str(seed),
               "--arm", arm, "--out", out, "--backend", backend]
        if int(seed) in GATE_SEEDS:
            cmd.append("--allow-gate-seeds")
        p = subprocess.run(cmd, env=env)
        ok = ok and p.returncode == 0 and os.path.exists(out)
    return 0 if ok else 1


def _held(resp):
    return bool(((resp or {}).get("bg_select") or {}).get("action") == "STAY_SILENT")


def score(shard_dir, seeds):
    """Per seed and target turn, lane C's F / S / D and outcome labels for the production seam; the capability,
    neural share and readout share for the neural seam; the bus-read variant disagreement. UNDEFINED reasons kept."""
    C = lane_c()
    fields = tuple(C.DECISION_FIELDS) + BG_FIELDS_EXTRA
    out = {"seeds": [int(s) for s in seeds], "per_seed": {}, "fields": list(fields)}
    pooled = {"F": [], "S": [], "D": [], "cap": [], "ns": [], "readout": [], "bus_disagree": [],
              "latch_thr_change": [], "first12_change": [], "membrane_ok": [], "content_held": [], "empty_held": [],
              "sham_ok": [], "bus_unique_agree": [], "seam_intact_held_for_ns": [], "seam_lesion_held": []}
    for s in seeds:
        sh = {}
        for arm in ARMS:
            p = os.path.join(shard_dir, "s%d" % int(s), arm + ".json")
            sh[arm] = json.load(open(p)) if os.path.exists(p) else None
        rows = []
        for label, _msg, cls in TURNS:
            ev = {"turn": label, "class": cls, "undefined": []}
            R = {a: ((sh[a] or {}).get("responses") or {}).get(label) for a in ARMS}
            if any(R[a] is None or "_error" in (R[a] or {}) for a in ("prod_a", "prod_b")):
                ev["undefined"].append("production intact arm missing/errored")
            elif C.sham_diff(R["prod_a"], R["prod_b"]) or _held(R["prod_a"]) != _held(R["prod_b"]):
                ev["undefined"].append("sham null failed (prod_a != prod_b)")
            else:
                obs = [e for e in ((sh["prod_b"] or {}).get("events") or {}).get(label, []) if e.get("kind") == "bg"]
                ev["host_consulted"] = bool(obs and obs[0].get("consulted"))
                ev["prod_held"] = _held(R["prod_a"])
                host_alt_held = not ev["prod_held"]
                if R["prod_intervene"] is not None:
                    ev["F"] = int(_held(R["prod_intervene"]) == host_alt_held)
                    changed = bool(C.reply_diff(R["prod_a"], R["prod_intervene"], fields=fields))
                    ev["intervene_outcome"] = ("follows_host" if ev["F"] else
                                               ("returned_to_intact" if not changed else "other_change"))
                else:
                    ev["undefined"].append("prod_intervene missing")
                if R["prod_substitute"] is not None:
                    ev["S"] = int(bool(C.reply_diff(R["prod_a"], R["prod_substitute"], fields=fields)))
                    ev["D"] = int(_held(R["prod_substitute"]) == ev["prod_held"])
                else:
                    ev["undefined"].append("prod_substitute missing")
            if R["seam_b"] is not None and "_error" not in R["seam_b"]:
                want_hold = cls == "empty"
                ev["seam_held"] = _held(R["seam_b"])
                ev["capability"] = int(ev["seam_held"] == want_hold) if cls != "one_word" else None
                if R["seam_lesion_value"] is not None:
                    ev["neural_share"] = int(_held(R["seam_lesion_value"]) != ev["seam_held"]
                                             or bool(C.reply_diff(R["seam_b"], R["seam_lesion_value"], fields=fields)))
                    ev["seam_lesion_held"] = _held(R["seam_lesion_value"])
                if R["seam_readout"] is not None:
                    rd = [e for e in ((sh["seam_readout"] or {}).get("events") or {}).get(label, [])
                          if e.get("kind") == "bg_readout"]
                    ev["readout_alt_differs"] = bool(rd and rd[0].get("alt_winner") != rd[0].get("latch_winner"))
                    ev["readout_share"] = int(ev["readout_alt_differs"]
                                              and _held(R["seam_readout"]) != ev["seam_held"])
            else:
                ev["undefined"].append("seam_b missing/errored")
            for arm in ("prod_b", "seam_b"):
                for e in ((sh[arm] or {}).get("events") or {}).get(label, []):
                    if e.get("kind") == "bus":
                        v = e["variants"]
                        base = v["argmax_thr (production)"]
                        pooled["bus_disagree"].append(int(any(x != base for x in v.values())))
                        pooled["bus_unique_agree"].append(int(v["unique"] == base))
            for arm in ("seam_b", "seam_readout", "seam_lesion_value"):
                for e in ((sh[arm] or {}).get("events") or {}).get(label, []):
                    if e.get("kind") == "bg_neural":
                        lo, hi = e.get("v_range") or [None, None]
                        pooled["membrane_ok"].append(int(lo is not None and lo >= -150.0 and hi <= 100.0))
                        if arm == "seam_b":
                            rd = e.get("readouts") or {}
                            pooled["latch_thr_change"].append(int(rd.get("latch_rate_x0.5") != e.get("winner")
                                                                  or rd.get("latch_rate_x2") != e.get("winner")))
                            pooled["first12_change"].append(
                                int(rd.get("first_to_12_motor (production rule)") != e.get("winner")))
            ev["sham_ok"] = int(not any("sham" in u for u in ev["undefined"]))
            pooled["sham_ok"].append(ev["sham_ok"])
            if ev.get("seam_held") is not None and cls in ("empty", "content"):
                pooled["%s_held" % cls].append(int(ev["seam_held"]))
            if ev.get("seam_lesion_held") is not None:                    # paired with seam_held for attribution
                pooled["seam_intact_held_for_ns"].append(int(ev["seam_held"]))
                pooled["seam_lesion_held"].append(int(ev["seam_lesion_held"]))
            rows.append(ev)
            for k, key in (("F", "F"), ("S", "S"), ("D", "D"), ("cap", "capability"), ("ns", "neural_share"),
                           ("readout", "readout_share")):
                if ev.get(key) is not None:
                    pooled[k].append(ev[key])
        out["per_seed"][int(s)] = rows
    m = {k: (round(sum(v) / len(v), 4) if v else None) for k, v in pooled.items()}
    out["pooled"] = m
    out["n"] = {k: len(v) for k, v in pooled.items()}
    out["production_seam_class"] = C.classify_seam(m["F"], m["S"], m["D"])
    # ATTRIBUTION (gates/attribution_required): G3 (ns) already reports the FRACTION of events whose outcome
    # changes when the value pathway is lesioned -- but a fraction alone still leaves "whose is the seam's held
    # rate" unasked. Ask it directly: treatment = the intact seam's held rate on the same paired events; control
    # = that rate with the value pathway lesioned. Informational only (out["diagnostics"]); does not feed BARS,
    # verdict() or any G1-G5 gate above.
    intact_held_rate, lesion_held_rate = m["seam_intact_held_for_ns"], m["seam_lesion_held"]
    seam_held_attribution = None
    if intact_held_rate is not None and lesion_held_rate is not None:
        seam_held_attribution = attributable_to(
            "seam held-rate (intact vs value-lesioned, paired events)", intact_held_rate, lesion_held_rate)
    out["diagnostics"] = {"seam_held_attribution": seam_held_attribution}
    return out


# ═══ the preregistered verdict (bars frozen in research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md)
BARS = {"F_before_min": 0.8, "empty_held_min": 0.75, "content_held_max": 0.0, "ns_after_min": 0.3,
        "latch_thr_change_max": 0.10, "bus_unique_agree_min": 0.95, "membrane_ok_min": 1.0, "sham_ok_min": 1.0,
        "min_events_per_measure": 6}


def verdict(sc, bars=BARS):
    """G1-G5 over the pooled six-seed score. Preconditions (UNDEFINED, never NO-GO, when they fail): the sham null
    holds on every target event, no membrane-bound violation in any neural-seam trial, and every gated measure has
    at least `min_events_per_measure` events."""
    from tools.verdict import Verdict
    m, n = sc["pooled"], sc["n"]
    v = Verdict("BG live seam retirement (plan step 3)")
    v.require("sham null holds on every target event", m["sham_ok"], expect=lambda x: x is not None
              and x >= bars["sham_ok_min"])
    v.require("membrane in [-150, +100] mV in every neural-seam trial", m["membrane_ok"],
              expect=lambda x: x is not None and x >= bars["membrane_ok_min"])
    for k in ("F", "empty_held", "content_held", "ns", "latch_thr_change", "bus_unique_agree"):
        v.require("enough events for %s" % k, n.get(k, 0), expect=lambda x: x >= bars["min_events_per_measure"])
    g = {
        "G1_production_host_owns_the_reply": m["F"] is not None and m["F"] >= bars["F_before_min"],
        "G2a_neural_seam_holds_on_empty_turns": m["empty_held"] is not None
        and m["empty_held"] >= bars["empty_held_min"],
        "G2b_neural_seam_never_holds_a_content_turn": m["content_held"] is not None
        and m["content_held"] <= bars["content_held_max"],
        "G3_value_lesion_moves_the_reply": m["ns"] is not None and m["ns"] >= bars["ns_after_min"],
        "G4_latch_read_threshold_invariant": m["latch_thr_change"] is not None
        and m["latch_thr_change"] <= bars["latch_thr_change_max"],
        "G5_bus_argmax_does_no_work": m["bus_unique_agree"] is not None
        and m["bus_unique_agree"] >= bars["bus_unique_agree_min"],
    }
    d = v.decide(go=all(g.values()), verbose=False)
    return {"verdict": d["status"], "gates": g, "status": d["status"], "preconditions": d["preconditions"],
            "production_seam_class_lane_c": sc.get("production_seam_class")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--arm", choices=sorted(ARMS), default="prod_a")
    ap.add_argument("--score", type=str, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7])
    ap.add_argument("--allow-gate-seeds", action="store_true")
    ap.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    ap.add_argument("--run-all", type=str, default=None, help="out dir: run every arm of --seed sequentially")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args(argv)
    if a.worker:
        if a.seed in GATE_SEEDS and not a.allow_gate_seeds:
            raise SystemExit("gate seed without --allow-gate-seeds (preregistration first)")
        return run_worker(a.seed, a.arm, a.out, backend=a.backend)
    if a.run_all:
        if a.seed in GATE_SEEDS and not a.allow_gate_seeds:
            raise SystemExit("gate seed without --allow-gate-seeds (preregistration first)")
        return run_all(a.seed, a.run_all, backend=a.backend)
    if a.score:
        res = score(a.score, a.seeds)
        res["label"] = ("GATE" if sorted(a.seeds) == sorted(GATE_SEEDS) else "DEV, not a verdict")
        vd = verdict(res)
        if res["label"] == "GATE":
            res.update(vd)                         # top-level verdict + preconditions (gates/verdict_preconditions)
        else:
            res["dev_readout"] = {k: vd[k] for k in ("gates", "status")}
        print(json.dumps({"pooled": res["pooled"], "n": res["n"], "production_seam_class": res["production_seam_class"],
                          "label": res["label"], "verdict": res.get("verdict", res.get("dev_readout"))}))
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            json.dump(res, open(a.out, "w"), indent=1)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
