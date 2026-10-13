"""Compare two reply-hash runs (docs/drafts/one-brain-m2/conversation_hashes.py output) turn by turn.

    python docs/drafts/one-brain-m2/compare_hashes.py <A.json> <B.json> [--md]

Per turn: whether the sha256 of the whole reply body matches, each side's seconds, and for a mismatch every JSON path
whose value differs (with both values, truncated). Exit 0 when every turn matches and both runs have the same turn
count, 1 otherwise. No brain; reads two files.
"""
import json
import sys


def _paths(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _paths(v, "%s.%s" % (p, k))
    elif isinstance(o, list):
        yield (p + "#len", len(o))
        for i, v in enumerate(o):
            yield from _paths(v, "%s[%d]" % (p, i))
    else:
        yield (p, o)


def body_diff(a, b, limit=12):
    da, db = dict(_paths(a)), dict(_paths(b))
    out = []
    for k in sorted(set(da) | set(db)):
        if da.get(k, "<absent>") != db.get(k, "<absent>"):
            out.append((k, str(da.get(k, "<absent>"))[:90], str(db.get(k, "<absent>"))[:90]))
    return out[:limit], len(out)


def compare(a, b):
    ra, rb = a["replies"], b["replies"]
    rows, all_match = [], len(ra) == len(rb)
    for i in range(max(len(ra), len(rb))):
        x = ra[i] if i < len(ra) else None
        y = rb[i] if i < len(rb) else None
        same = bool(x and y and x["sha256"] == y["sha256"])
        all_match &= same
        diff, n = ([], 0) if same or not (x and y) else body_diff(x["body"], y["body"])
        rows.append({"i": i, "session": (x or y)["session"], "message": (x or y)["message"], "match": same,
                     "sec_a": x and x["sec"], "sec_b": y and y["sec"], "diff": diff, "n_diff": n})
    return rows, all_match


def main(argv):
    a, b = json.load(open(argv[1])), json.load(open(argv[2]))
    rows, ok = compare(a, b)
    print("A=%s  B=%s" % (argv[1], argv[2]))
    print("| turn | conv | message | sha match | A s | B s |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print("| %d | %s | %s | %s | %s | %s |" % (r["i"], r["session"], r["message"][:44].replace("|", "/"),
                                                "yes" if r["match"] else "**NO**", r["sec_a"], r["sec_b"]))
    sa = sum(r["sec_a"] or 0 for r in rows)
    sb = sum(r["sec_b"] or 0 for r in rows)
    print("\nTotal seconds: A %.1f, B %.1f; excluding turn 0 (the build): A %.1f, B %.1f"
          % (sa, sb, sa - (rows[0]["sec_a"] or 0), sb - (rows[0]["sec_b"] or 0)))
    for r in rows:
        if r["diff"]:
            print("\nturn %d (%s) %r: %d differing paths" % (r["i"], r["session"], r["message"], r["n_diff"]))
            for k, x, y in r["diff"]:
                print("  %s\n    A: %s\n    B: %s" % (k, x, y))
    print("\nALL MATCH" if ok else "\nMISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
