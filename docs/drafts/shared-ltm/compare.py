"""Compare the two three_sessions.py arms: per-turn sha256 of the whole reply body, and per-session RSS.

    python docs/drafts/shared-ltm/compare.py docs/drafts/shared-ltm/runs/off.json docs/drafts/shared-ltm/runs/on.json
"""
import json
import sys

GIB = 1048576.0


def main(off_path, on_path):
    off, on = json.load(open(off_path)), json.load(open(on_path))
    by = lambda rec: {(r["session"], r["turn"]): r for r in rec["replies"]}
    a, b = by(off), by(on)
    assert set(a) == set(b), "different turn sets"
    same = 0
    for k in sorted(a, key=lambda k: (k[1], k[0])):
        eq = a[k]["sha256"] == b[k]["sha256"]
        same += eq
        print("%s t%d %s off=%s on=%s svo=%s" % (k[0], k[1], "SAME" if eq else "DIFF", a[k]["sha256"][:16],
                                              b[k]["sha256"][:16], b[k]["recalled_svo"]))
        if not eq:
            for f in sorted(set(a[k]["body"]) | set(b[k]["body"])):
                if a[k]["body"].get(f) != b[k]["body"].get(f):
                    print("   field %s differs" % f)
    print("identical turns: %d/%d" % (same, len(a)))
    for name, rec in (("off", off), ("on", on)):
        built = [rec["built"][s]["rss_kb"] for s in ("s1", "s2", "s3")]
        steps = [built[i] - built[i - 1] for i in (1, 2)]
        print("%s: start %d KiB; built s1/s2/s3 %s KiB (%s GiB); per extra session %s KiB (%s GiB); end %d hwm %d; "
              "distinct LTM store objects %d" % (
                  name, rec["rss_start_kb"], built, ["%.2f" % (x / GIB) for x in built], steps,
                  ["%.3f" % (x / GIB) for x in steps], rec["rss_end_kb"], rec["hwm_end_kb"],
                  rec["distinct_ltm_store_ids"]))
    return 0 if same == len(a) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
