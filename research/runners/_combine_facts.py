#!/usr/bin/env python
"""Combine multiple corpus-extracted SVO fact JSONs (from _corpus_svo_extract.py) into one base for the
first-chat brain: sum counts for triples attested in more than one corpus, keep the highest-count source
sentence, sort by total count. Merges TinyStories (narrative facts) + Simple-Wiki (encyclopedic facts).
Host-side curriculum prep; the brain still stores/recalls via spikes/binding."""
import argparse
import json
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="fact JSONs to merge")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    counts = defaultdict(int)
    attest, best = {}, {}
    for path in a.inputs:
        recs = json.load(open(path, encoding="utf-8"))
        for rec in recs:
            k = (rec["agent"], rec["action"], rec["patient"])
            counts[k] += int(rec["count"])
            if int(rec["count"]) > best.get(k, 0):
                best[k] = int(rec["count"]); attest[k] = rec.get("attest", "")
        print(f"[combine] {path}: {len(recs)} facts", flush=True)
    merged = sorted(counts.items(), key=lambda kv: -kv[1])
    out = [{"agent": k[0], "action": k[1], "patient": k[2], "count": c, "attest": attest[k]}
           for k, c in merged]
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"[combine] -> {len(out)} distinct facts -> {a.out}", flush=True)
    print("[combine] top 12: " + ", ".join(
        f"{r['count']}x({r['agent']},{r['action']},{r['patient']})" for r in out[:12]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
