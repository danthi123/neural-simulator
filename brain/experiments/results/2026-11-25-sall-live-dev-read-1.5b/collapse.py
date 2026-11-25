# Capability: Memory selection (pattern completion and separation)
# Record: brain/research/2026-11-25-sall-live-dev-prereg.md (collapse guard: top reply share >= 0.4, seeds-ever per arm)
# Expected-fail: no (a counter)
"""Collapse guard: per arm, the number of seeds with any trunk probe night whose most common reply covers >= 0.4 of rows."""
import collections, json, sys
ever = collections.Counter()
for f in sys.argv[1:]:
    r = json.load(open(f)); hit = set()
    for p in r["probes"]:
        if p.get("tag") or p.get("line") != "trunk":
            continue
        c = collections.Counter(x["brain"].strip() for x in p["rows"])
        if c.most_common(1)[0][1] / len(p["rows"]) >= 0.4:
            hit.add(p["arm"])
    ever.update(hit)
print(dict(ever))
