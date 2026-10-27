"""STABLE+RECAL vs OFF dev check (research/stable-recal): replies, answer rate, the Stage-2 scorers (C4 = emo, C5 =
inner), recall verification and membrane validity, per group and dev seed. Observe-only: reads the census JSONs that
research.runners.membrane_census wrote and the per-seed comparisons research.runners.stable_integration_compare wrote
from them; runs no brain.

Answer rate: tools/answer_rate.py `classify` / `rate` over every turn reply of the group's shard (and, separately,
over the non-teach turns), with `report` per group-seed: FLIP-HURTS-ANSWERS when the STABLE+RECAL answer rate is below
OFF's (reported, gates nothing; this branch flips no default).

  .venv/bin/python -m research.runners.stable_recal_dev_check --seeds 7,8 \
      --off research/findings/raw/_stable_integration/census/off/s{seed} \
      --on research/findings/raw/_stable_recal/dev78/stable_recal/s{seed} \
      --compare research/findings/raw/_stable_recal/dev78/compare_s{seed}.json \
      --out research/findings/raw/_stable_recal/dev78/dev_check.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

GROUPS = ("conv", "infer", "ask", "emo", "inner")
WORKSPACES = {"bus": ("research.runners._p1_2_workspace_deliberation_loop_derisk",
                      "research.runners._gnw_reentrant_metacog_gated_deliberation_derisk",
                      "research.runners._gnw_norgan_bus_derisk", "research.runners._gnw_two_distinct_organs_derisk",
                      "research.runners._gnw_coincidence_integrator_derisk"),
              "swap": ("research.runners._gnw_active_overwrite_derisk",
                       "research.runners._gnw_neural_swap_intention_derisk"),
              "stop": ("research.runners._gnw_distributed_overwrite_workspace_derisk",)}


def _load(path):
    with open(path) as f:
        return json.load(f)


def _bodies(rec, teach):
    sh = rec["workload"]["shard"]
    roles = {t["label"]: t.get("role") for t in sh.get("turns") or []}
    return [b for lab, b in (sh.get("responses") or {}).items() if teach or roles.get(lab) != "teach"]


def _recall(rec):
    resp = rec["workload"]["shard"].get("responses") or {}
    return {"n_recalled_svo": sum(1 for b in resp.values() if isinstance(b, dict) and b.get("recalled_svo")),
            "n_verified": sum(1 for b in resp.values() if isinstance(b, dict) and b.get("verified") is True),
            "recalled_svo_turns": sorted(lab for lab, b in resp.items() if isinstance(b, dict) and b.get("recalled_svo"))}


def _ws_oor(rec):
    out = {}
    for k, prefixes in WORKSPACES.items():
        hits = [o for o in rec["census"]["organs"] if o["organ"].split(":")[0] in prefixes]
        out[k] = {"n_oor_neurons": sum(o.get("n_oor_neurons") or 0 for o in hits), "organs": [o["organ"] for o in hits]}
    return out


def _all(score):
    return None if not isinstance(score, dict) else score.get("all")


def run(seeds, off_t, on_t, cmp_t):
    sys.path.insert(0, os.path.join(_REPO, "tools"))
    import answer_rate as AR
    rows, per_all, per_probe = {}, {}, {}
    for seed in seeds:
        cmp_path = cmp_t.format(seed=seed)
        cmp = _load(cmp_path)
        for g in GROUPS:
            key = "%s-s%d" % (g, seed)
            p_off = os.path.join(off_t.format(seed=seed), "%s__intact_a.json" % g)
            p_on = os.path.join(on_t.format(seed=seed), "%s__intact_a.json" % g)
            if not (os.path.exists(p_off) and os.path.exists(p_on)):
                rows[key] = {"status": "UNMEASURED", "off": p_off if os.path.exists(p_off) else None,
                             "on": p_on if os.path.exists(p_on) else None}
                continue
            r_off, r_on = _load(p_off), _load(p_on)
            per_all[key] = {"off": _bodies(r_off, True), "on": _bodies(r_on, True)}
            per_probe[key] = {"off": _bodies(r_off, False), "on": _bodies(r_on, False)}
            c = (cmp.get("groups") or {}).get(g) or {}
            gs = c.get("gate_score") or {}
            rows[key] = {
                "off": os.path.relpath(p_off, _REPO), "on": os.path.relpath(p_on, _REPO),
                "env": {"off": [r_off.get("stable_integration_env"), r_off.get("stable_recal_env")],
                        "on": [r_on.get("stable_integration_env"), r_on.get("stable_recal_env")]},
                "valid": {"rc": [r_off["workload"].get("rc"), r_on["workload"].get("rc")],
                          "turn_errors": [r_off["workload"].get("turn_errors"), r_on["workload"].get("turn_errors")]},
                "answer_rate_all_turns": {"off": AR.rate(per_all[key]["off"]), "on": AR.rate(per_all[key]["on"])},
                "answer_rate_non_teach": {"off": AR.rate(per_probe[key]["off"]), "on": AR.rate(per_probe[key]["on"])},
                "gate_all": {"off": _all(gs.get("off")), "on": _all(gs.get("on"))},
                "recall": {"off": _recall(r_off), "on": _recall(r_on)},
                "replies_changed": (c.get("replies") or {}).get("n_changed"),
                "replies_visible_changed": {x["label"]: x["fields_changed"] for x in (c.get("replies") or {}).get(
                    "changed") or [] if x.get("fields_changed")},
                "n_turns": (c.get("replies") or {}).get("n_turns"),
                "membrane": {k: v for k, v in (c.get("membrane") or {}).items() if k != "organs"},
                "workspace_oor": {"off": _ws_oor(r_off), "on": _ws_oor(r_on)},
                "compare": os.path.relpath(cmp_path, _REPO)}
    crit = {"C4": "emo", "C5": "inner", "C1": "conv", "C2": "infer", "C3": "ask"}
    return {"runner": "research/runners/stable_recal_dev_check.py", "backend": "numpy", "device": "cpu",
            "analysis_only": True, "seeds": list(seeds), "arms": {"off": off_t, "on": on_t},
            "answer_rate_report_all_turns": AR.report(per_all, label="STABLE+RECAL vs OFF, all turn replies"),
            "answer_rate_report_non_teach": AR.report(per_probe, label="STABLE+RECAL vs OFF, non-teach turn replies"),
            "criteria": {c: {"group": g, "per_seed": {str(s): rows.get("%s-s%d" % (g, s), {}).get("gate_all")
                                                    for s in seeds}} for c, g in crit.items()},
            "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", default="7,8")
    ap.add_argument("--off", required=True, help="OFF census dir template with {seed}")
    ap.add_argument("--on", required=True, help="STABLE+RECAL census dir template with {seed}")
    ap.add_argument("--compare", required=True, help="per-seed stable_integration_compare JSON template with {seed}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    res = run(seeds, a.off, a.on, a.compare)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    for key, r in res["rows"].items():
        if "answer_rate_all_turns" not in r:
            print("[dev check] %-9s %s" % (key, r.get("status")))
            continue
        ar = r["answer_rate_all_turns"]
        print("[dev check] %-9s answer %d/%d -> %d/%d | gate all %s -> %s | recalled_svo %d -> %d | visible changed "
              "%d/%s | oor %s -> %s" % (key, ar["off"]["answered"], ar["off"]["n"], ar["on"]["answered"], ar["on"]["n"],
                                        r["gate_all"]["off"], r["gate_all"]["on"], r["recall"]["off"]["n_recalled_svo"],
                                        r["recall"]["on"]["n_recalled_svo"], len(r["replies_visible_changed"]),
                                        r["n_turns"], r["membrane"].get("n_oor_neurons_off"),
                                        r["membrane"].get("n_oor_neurons_on")))
    print("[dev check] answer rate (all turns): %s; hurt %s" % (res["answer_rate_report_all_turns"]["status"],
                                                               res["answer_rate_report_all_turns"]["hurt_seeds"]))
    print("[dev check] -> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
