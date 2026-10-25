"""Number the unique (turn, svo, surface) sentences of corpus.json in a fixed order (turn, then first appearance) so a labels
file can refer to them, and print them for hand-labelling.   usage: unique_sentences.py <corpus.json> [--json out.json]"""
import collections, json, sys


def unique(corpus):
    """-> list of dict(num, turn, entity, svo, surface, methods(Counter-as-dict)) in order."""
    out, idx = [], {}
    for t in corpus["turns"]:
        for c in t.get("calls", []):
            k = (t["label"], tuple(c["svo"]), c["surface"])
            if k not in idx:
                idx[k] = len(out)
                out.append({"num": len(out) + 1, "turn": t["label"], "entity": t["entity"], "svo": list(c["svo"]),
                            "surface": c["surface"], "methods": collections.Counter()})
            out[idx[k]]["methods"][c["method"]] += 1
    for u in out:
        u["methods"] = dict(u["methods"])
    return out


if __name__ == "__main__":
    corpus = json.load(open(sys.argv[1]))
    u = unique(corpus)
    if "--json" in sys.argv:
        json.dump(u, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
    cur = None
    for t in corpus["turns"]:
        if not t.get("calls"):
            print("\n=== %s %s: NO FACTS GATHERED (%s)" % (t["label"], t["entity"], t.get("error") or "no calls"))
            continue
        print("\n=== %s %s  gathered set (%d facts):" % (t["label"], t["entity"], len(t["gated"])))
        for f in t["gated"]:
            print("      GATED", " | ".join(f))
        by = collections.OrderedDict()
        for x in u:
            if x["turn"] == t["label"]:
                by.setdefault(tuple(x["svo"]), []).append(x)
        for svo, xs in by.items():
            print("    FACT", " | ".join(svo))
            for x in xs:
                print("      [%3d] %s" % (x["num"], x["surface"]))
