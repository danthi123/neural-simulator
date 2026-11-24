"""Host-only: on K-P told rows, would the RAW-similarity winner have been the right episode? Maps episode index -> text
from rows where idx and recalled are both recorded (store is append-only within a life unless pruned; checked by consistency)."""
import json, sys
from brain.eval.life import episode_is_source
from brain.eval import life_bank as LB
tot = dict(rows=0, fetched=0, wrong=0, wo=0, raw_known=0, raw_right_on_wrong=0, raw_right_on_right=0, right=0, conflict=0)
for f in sys.argv[1:]:
    r = json.load(open(f)); m = {}
    for p in r["probes"]:
        for row in p["rows"]:
            fx = row.get("fetch") or {}
            if fx.get("idx") is not None and row.get("recalled"):
                k = fx["idx"]
                if k in m and m[k] != row["recalled"]: tot["conflict"] += 1
                m[k] = row["recalled"]
    for p in r["probes"]:
        if not (5 <= p["night"] <= 15) or p.get("line") != "trunk" or p.get("tag") or p["arm"] != "intact-frozen": continue
        for row in p["rows"]:
            if row.get("kind") not in ("H", "rotating") or row.get("ok") is None: continue
            it = LB.ALL[row["item"]]; fx = row.get("fetch") or {}
            tot["rows"] += 1
            if row.get("recalled") is None: continue
            tot["fetched"] += 1
            right = episode_is_source(it, row["recalled"]); tot["right"] += right
            rw = fx.get("raw_winner"); txt = m.get(rw)
            if not right: tot["wrong"] += 1; tot["wo"] += bool(fx.get("weighted_only_winner"))
            if txt is None: continue
            tot["raw_known"] += 1; tot["known_right" if right else "known_wrong"] = tot.get("known_right" if right else "known_wrong",0)+1
            rr = episode_is_source(it, txt)
            if right: tot["raw_right_on_right"] += rr
            else: tot["raw_right_on_wrong"] += rr
print(tot)
