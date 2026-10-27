"""STOP-LEAD ATTRIBUTION (research/stable-recal): which organ decides the global-stop lead, by intervention.

Reads the census JSONs that research.runners.stable_integration_exempt_probe wrote for inner, dev seed 7, arm
intact_a, with one organ put on the other integrator inside a whole brain (both directions), and records per arm the
turns whose reply carries the stop lead ("Setting the held thread aside"), the turns with the swap lead only, and the
census out-of-range count and integrator read-back of the three candidate organs. Observe-only: runs nothing.

  .venv/bin/python -m research.runners.stable_recal_stop_attribution \
      research/findings/raw/_stable_recal/stop_lead_attribution_s7.json
"""
import json
import os
import sys

C = "research/findings/raw/_stable_integration/census"
STOP = "Setting the held thread aside"
ORG = {"stop_workspace": "research.runners._gnw_distributed_overwrite_workspace_derisk",
       "accbg_trigger": "research.runners._gnw_acc_bg_stop_trigger_derisk",
       "swap_workspace": "research.runners._gnw_active_overwrite_derisk"}
arms = {"off": "whole brain old integrator", "on": "whole brain stable integrator",
        "on_except_stop": "stable brain, global-stop workspace on the old integrator",
        "off_except_stop": "old brain, global-stop workspace on the stable integrator",
        "on_except_swap": "stable brain, thought-swap workspace on the old integrator",
        "off_except_swap": "old brain, thought-swap workspace on the stable integrator",
        "on_except_accbg": "stable brain, ACC-BG stop trigger on the old integrator",
        "off_except_accbg": "old brain, ACC-BG stop trigger on the stable integrator"}
out = {"runner": "research/runners/stable_recal_stop_attribution.py", "backend": "numpy", "seed": 7, "group": "inner", "arm": "intact_a",
       "stop_lead": STOP, "rows": {}}
for arm, desc in arms.items():
    p = f"{C}/{arm}/s7/inner__intact_a.json"
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    resp = d["workload"]["shard"]["responses"]
    lead = [t for t, r in resp.items() if STOP in str(r.get("response") or r.get("answer") or "")]
    if not lead:
        lead = [t for t, r in resp.items() if STOP in json.dumps(r)]
    organs = d["census"]["organs"]

    def _org(prefix):
        hits = [o for o in organs if o["organ"].startswith(prefix)]
        return {"organ": [o["organ"] for o in hits], "n_oor_neurons": sum(o.get("n_oor_neurons") or 0 for o in hits),
                "izh_substeps_seen": [o.get("izh_substeps_seen") for o in hits]}
    out["rows"][arm] = {"what": desc, "source": p, "stop_lead_turns": lead, "n_stop_lead": len(lead),
                        "swap_lead_only_turns": [t for t, r in resp.items()
                                                 if ", then" in json.dumps(r) and STOP not in json.dumps(r)],
                        "organs": {k: _org(v) for k, v in ORG.items()}}
json.dump(out, open(sys.argv[1], "w"), indent=1)
for a, r in out["rows"].items():
    print(a, r["n_stop_lead"], {k: v["n_oor_neurons"] for k, v in r["organs"].items()})
