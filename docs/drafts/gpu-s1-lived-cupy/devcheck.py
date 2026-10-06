"""devcheck.py -- read the T24 dev smoke (seed 7, intact_a + intact_b, cupy) and print ONE verdict line.

    python docs/drafts/gpu-s1-lived-cupy/devcheck.py <dir holding lived__intact_a.json and lived__intact_b.json>

Prints `GO <wall_s>` or `STOP <reasons>`; exit 0 / 1. GO needs, in both arms: backend cupy, the brain source the gate
expects, the corpus present, PYTHONHASHSEED 0, no turn errors; and the null: the two same-seed rebuilds give the same
reply-level primary fields (answer, abstained, recalled_svo, hypothesis_svo) on every turn, which the gate's aggregate
requires per seed (else that seed reads UNDEFINED).
"""
import json
import os
import sys


def _prim(r):
    r = r or {}
    return (str(r.get("answer")), bool(r.get("abstained")), r.get("recalled_svo"), r.get("hypothesis_svo"))


def check(a, b):
    why = []
    for n, d in (("intact_a", a), ("intact_b", b)):
        if d.get("backend") != "cupy":
            why.append("%s backend=%s" % (n, d.get("backend")))
        if d.get("sources") != [d.get("expected_source")]:
            why.append("%s sources=%s expected=%s" % (n, d.get("sources"), d.get("expected_source")))
        if d.get("corpus_present") is not True:
            why.append("%s corpus_present=%s" % (n, d.get("corpus_present")))
        if d.get("pythonhashseed_ok") is not True:
            why.append("%s pythonhashseed_ok=%s" % (n, d.get("pythonhashseed_ok")))
        err = [k for k, v in (d.get("responses") or {}).items() if isinstance(v, dict) and "_error" in v]
        if err or not d.get("responses"):
            why.append("%s errors=%s" % (n, err or "no responses"))
    ra, rb = a.get("responses") or {}, b.get("responses") or {}
    diffs = [k for k in ra if _prim(ra[k]) != _prim(rb.get(k))]
    if diffs:
        why.append("null diffs on %s" % diffs)
    wall = max(a.get("wall_s") or 0, b.get("wall_s") or 0)
    return why, wall


def main(argv):
    if len(argv) != 2:
        print("usage: devcheck.py <dir>", file=sys.stderr)
        return 2
    a, b = (json.load(open(os.path.join(argv[1], "lived__intact_%s.json" % x))) for x in "ab")
    why, wall = check(a, b)
    print(("STOP " + "; ".join(why)) if why else "GO %s" % wall)
    return 1 if why else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
