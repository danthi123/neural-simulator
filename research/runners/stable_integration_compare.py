"""STABLE-INTEGRATION ON vs OFF -- compare two membrane-census record sets of the same seed (research/stable-integration).

Reads the per-group census JSONs written by research.runners.membrane_census with SIM_STABLE_INTEGRATION unset (OFF)
and "1" (ON) and reports, per group: (1) membrane validity per organ (out-of-range neuron count, low/high/non-finite,
lowest and highest v, the stable-integration read-back the census recorded: sub-step pairs seen, sub-step calls,
rule-(a) model resets, re-asserts, the organ's own bounded setting); (2) the speed cost (the stage-2 workload wall
time and the census worker wall time, ON / OFF); (3) the reply impact: every turn whose per-turn response sha256
differs, with both reply texts and whether abstained / recalled_svo / derived changed, plus the gate's own scorer
(research.runners.stage2_exit_gate.SCORERS) on both shards. Observe-only: reads JSON, runs nothing.

  .venv/bin/python -m research.runners.stable_integration_compare \
      --off research/findings/raw/_stable_integration/census/off/s7 --on research/findings/raw/_stable_integration/census/on/s7 \
      --out research/findings/raw/_stable_integration/compare_s7.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

ORGAN_FIELDS = ("n_bridges", "n_steps", "n_oor_neurons", "n_low_neurons", "n_high_neurons", "n_nonfinite_neurons",
                "v_min", "v_min_region", "v_max", "n_spikes_total", "substep_calls", "refractory_model_resets",
                "stable_integration_reasserts", "izh_substeps_seen", "stable_integration_applied_seen",
                "organ_bounded_configured", "bounded_conductance_integration", "max_gtot_dt_over_C")
REPLY_FIELDS = ("answer", "abstained", "recalled_svo", "derived", "source")


def _load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*__*.json"))):
        if p.endswith(".prov.json"):
            continue
        with open(p) as f:
            rec = json.load(f)
        out[rec["group"]] = (p, rec)
    return out


def _organs(rec):
    return {o["organ"]: {k: o.get(k) for k in ORGAN_FIELDS} for o in rec["census"]["organs"]}


def _score(group, rec):
    shard = (rec.get("workload") or {}).get("shard")
    if not shard:
        return None
    from research.runners import stage2_exit_gate as SG
    try:
        sc = SG.SCORERS[group](shard)
    except Exception as e:                                  # a scorer fault is reported, never hidden
        return {"error": "%s: %s" % (type(e).__name__, e)}
    return json.loads(json.dumps(sc, default=str))


def _signature(resp):
    """The per-turn internal read-outs the reply changes trace to: the GNW thought-swap workspace (`swap_drives`), its
    stop circuit (`gnw_stop`), the recall verification and the authorship label. Rates rounded to 3 decimals."""
    if not isinstance(resp, dict) or "_error" in resp:
        return None
    sd, gs, au = resp.get("swap_drives") or {}, resp.get("gnw_stop") or {}, resp.get("authorship") or {}
    r = sd.get("new_rate_post")
    return {"swap_reason": sd.get("reason"), "swapped": sd.get("swapped"), "n_ignited_post": sd.get("n_ignited_post"),
            "new_rate_post": (round(float(r), 3) if r is not None else None),
            "stop": gs.get("reason_lead"), "verified": resp.get("verified"), "authorship": au.get("label")}


def compare_group(group, off, on):
    (p_off, r_off), (p_on, r_on) = off, on
    w_off, w_on = r_off.get("workload") or {}, r_on.get("workload") or {}
    o_off, o_on = _organs(r_off), _organs(r_on)
    organs = []
    for name in list(o_off) + [n for n in o_on if n not in o_off]:
        a, b = o_off.get(name), o_on.get(name)
        organs.append({"organ": name, "off": a, "on": b})
    tr_off, tr_on = w_off.get("turn_replies") or {}, w_on.get("turn_replies") or {}
    labels = [t["label"] for t in ((w_off.get("shard") or {}).get("turns") or [])] or sorted(set(tr_off) | set(tr_on))
    sh_off = ((w_off.get("shard") or {}).get("responses")) or {}
    sh_on = ((w_on.get("shard") or {}).get("responses")) or {}
    signatures = {lab: {"off": _signature(sh_off.get(lab)), "on": _signature(sh_on.get(lab))} for lab in labels}
    changed = []
    for lab in labels:
        a, b = tr_off.get(lab) or {}, tr_on.get(lab) or {}
        if a.get("sha256") == b.get("sha256"):
            continue
        changed.append({"label": lab,
                        "fields_changed": [k for k in REPLY_FIELDS if a.get(k) != b.get(k)],
                        "off": {k: a.get(k) for k in REPLY_FIELDS + ("error",) if k in a},
                        "on": {k: b.get(k) for k in REPLY_FIELDS + ("error",) if k in b}})
    wall_off, wall_on = w_off.get("wall_s"), w_on.get("wall_s")
    return {"group": group, "off_path": os.path.relpath(p_off, _REPO), "on_path": os.path.relpath(p_on, _REPO),
            "env_off": r_off.get("stable_integration_env"), "env_on": r_on.get("stable_integration_env"),
            "git_sha_off": r_off.get("git_sha"), "git_sha_on": r_on.get("git_sha"),
            "valid": {"rc_off": w_off.get("rc"), "rc_on": w_on.get("rc"),
                      "turn_errors_off": w_off.get("turn_errors"), "turn_errors_on": w_on.get("turn_errors"),
                      "pythonhashseed_ok": [w_off.get("pythonhashseed_ok"), w_on.get("pythonhashseed_ok")],
                      "corpus_present": [w_off.get("corpus_present"), w_on.get("corpus_present")],
                      "observer_errors": [r_off["census"].get("observer_errors"), r_on["census"].get("observer_errors")]},
            "speed": {"workload_wall_s_off": wall_off, "workload_wall_s_on": wall_on,
                      "workload_ratio_on_over_off": (round(wall_on / wall_off, 3) if wall_off and wall_on else None),
                      "census_wall_s_off": r_off.get("wall_s"), "census_wall_s_on": r_on.get("wall_s"),
                      "observe_s_off": r_off["census"].get("observe_s"), "observe_s_on": r_on["census"].get("observe_s"),
                      "peak_rss_mb_off": (w_off.get("process") or {}).get("peak_rss_mb"),
                      "peak_rss_mb_on": (w_on.get("process") or {}).get("peak_rss_mb"),
                      "total_bridge_steps_off": r_off["census"].get("total_bridge_steps"),
                      "total_bridge_steps_on": r_on["census"].get("total_bridge_steps")},
            "membrane": {"n_offender_organs_off": sum(1 for o in o_off.values() if o["n_oor_neurons"]),
                         "n_offender_organs_on": sum(1 for o in o_on.values() if o["n_oor_neurons"]),
                         "n_oor_neurons_off": sum(o["n_oor_neurons"] for o in o_off.values()),
                         "n_oor_neurons_on": sum(o["n_oor_neurons"] for o in o_on.values()),
                         "organs": organs},
            "replies": {"n_turns": len(labels), "n_changed": len(changed), "changed": changed,
                        "responses_sha256_off": w_off.get("responses_sha256"),
                        "responses_sha256_on": w_on.get("responses_sha256"),
                        "signatures": signatures},
            "gate_score": {"off": _score(group, r_off), "on": _score(group, r_on)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--off", required=True)
    ap.add_argument("--on", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    off, on = _load(a.off), _load(a.on)
    groups = [g for g in ("conv", "infer", "ask", "emo", "inner") if g in off and g in on]
    res = {"runner": "research/runners/stable_integration_compare.py", "off_dir": a.off, "on_dir": a.on,
           "groups_compared": groups, "groups_missing": sorted((set(off) | set(on)) - set(groups)),
           "groups": {g: compare_group(g, off[g], on[g]) for g in groups}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    for g in groups:
        r = res["groups"][g]
        s, m, rp = r["speed"], r["membrane"], r["replies"]
        print("[stable-integration] %s: offender organs off %d on %d | oor neurons off %d on %d | workload wall off %s s "
              "on %s s (x%s) | replies changed %d/%d | gate all off=%s on=%s" % (
                  g, m["n_offender_organs_off"], m["n_offender_organs_on"], m["n_oor_neurons_off"],
                  m["n_oor_neurons_on"], s["workload_wall_s_off"], s["workload_wall_s_on"],
                  s["workload_ratio_on_over_off"], rp["n_changed"], rp["n_turns"],
                  (r["gate_score"]["off"] or {}).get("all"), (r["gate_score"]["on"] or {}).get("all")))
    print("[stable-integration] -> %s" % a.out)


if __name__ == "__main__":
    main()
