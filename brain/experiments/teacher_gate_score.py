# Capability: Sleep consolidation
# Record: brain/research/2026-11-15-teacher-night-gate-1.5b.md (T1-T3)
"""Score the teacher-night gate (brain/research/2026-11-15-teacher-night-gate-1.5b.md, T1-T3), plus the own cues' voice.
Usage: teacher_gate_score.py <dir with tg_s*.json>"""
import collections, glob, json, re, sys
rows, voice = [], collections.Counter()
for f in sorted(glob.glob(sys.argv[1] + "/tg_s*.json")):
    d = json.load(open(f)); s = d["summary"]; rows.append(s)
    print(f"s{d['seed']} {d['host']} own_ok {s['own_ok']}/{s['own_n']} never_false {s['never_false']}/{s['never_n']}"
          f" overlap {s['overlap_with_tests']}")
    for r in d["rows"]:
        if r["kind"] == "own":
            q = r["q"].lower()
            v = "your" if re.search(r"\byou(r)?\b", q) else ("my" if re.search(r"\b(my|i)\b", q) else "neither")
            voice[(v, r["key"] in r["reply"].lower(), r.get("recalled") == r["telling"])] += 1
n = len(rows)
own = sum(r["own_ok"] for r in rows) / n
nf, nn = sum(r["never_false"] for r in rows), sum(r["never_n"] for r in rows)
ov = [c for r in rows for c in r["overlap_with_tests"]]
print(f"T1 own_ok mean {own:.2f}/12 (>= 8: {own >= 8}); T2 never_false {nf}/{nn} = {100 * nf / nn:.1f}% (<= 10%: {nf / nn <= 0.1});"
      f" T3 overlaps {ov} (listed, dropped)")
print("own cues by (voice, key in reply, recall returned the telling):")
for k, v in sorted(voice.items()): print("  ", k, v)
