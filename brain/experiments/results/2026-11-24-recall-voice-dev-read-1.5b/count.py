# Capability: Episodic recall
# Record: brain/research/2026-11-23-recall-voice-dev-prereg.md (section 2 count method)
# Expected-fail: no (a counter; reproduces the prereg K-P table exactly)
"""Recall-voice prereg §2 count method, exactly. Usage: count.py LIFE.json [...]"""
import json, sys
from brain.experiments.strict_scorer import is_echo

def count(path):
    r = json.load(open(path)); L = r["ledger"]
    c = dict(i_r=0, i_n=0, o_r=0, o_n=0, echo=0, echo_n=0, oecho=0, oecho_n=0, hurt=0, helped=0, best=0)
    paired = {}
    for p in r["probes"]:
        if not (5 <= p["night"] <= 15):
            continue
        trunk = p.get("line") == "trunk" and not p.get("tag")
        for i, row in enumerate(p["rows"]):
            it = row.get("item")
            e = L.get(it) if it else None
            if not e or not e.get("telling"):
                continue
            ech = any(is_echo(row["brain"], k, e["telling"]) for k in e["keys"])
            if p["arm"] == "hippocampus-off":
                c["oecho_n"] += 1; c["oecho"] += ech
            if not trunk:
                continue
            if p["arm"] == "intact":
                c["echo_n"] += 1; c["echo"] += ech
            if p["arm"] in ("intact", "hippocampus-off") and row.get("ok") is not None:
                k = "i" if p["arm"] == "intact" else "o"
                c[k + "_n"] += 1; c[k + "_r"] += bool(row["ok"])
                paired.setdefault((p["night"], i, it, row["you"]), {})[p["arm"]] = bool(row["ok"])
    for v in paired.values():
        if len(v) == 2:
            a, b = v["intact"], v["hippocampus-off"]
            c["hurt"] += (b and not a); c["helped"] += (a and not b); c["best"] += (a or b)
    return r, c

if __name__ == "__main__":
    tot = {}
    for f in sys.argv[1:]:
        r, c = count(f)
        print(r["seed"], r.get("brain_kw_extra"), f"told intact {c['i_r']}/{c['i_n']} off {c['o_r']}/{c['o_n']} | echo intact {c['echo']}/{c['echo_n']} off {c['oecho']}/{c['oecho_n']} | hurt {c['hurt']} helped {c['helped']} best {c['best']}")
        for k, v in c.items(): tot[k] = tot.get(k, 0) + v
    c = tot
    print("pooled", f"told intact {c['i_r']}/{c['i_n']} off {c['o_r']}/{c['o_n']} | echo intact {c['echo']}/{c['echo_n']} off {c['oecho']}/{c['oecho_n']} | hurt {c['hurt']} helped {c['helped']} best {c['best']}")
