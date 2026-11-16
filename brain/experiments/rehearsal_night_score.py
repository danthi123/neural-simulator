# Capability: Sleep consolidation
# Record: brain/research/2026-11-16-rehearsal-night-1.5b.md (R1-R3; R4 is the blind reading)
"""R1-R3 for the rehearsal-night / habituation arms on the open-conversation eval.
usage: rehearsal_night_score.py <dir> <seeds...>   expects <dir>/<arm>_s<seed>.{json,md}.rows.json for arm in ocd ocr och ochr
(missing arms are skipped). Collapse = an intact day-2 reply repeated more than 3 times of 16."""
import collections, json, os, sys

ARMS = {"ocd": "default", "ocr": "rehearse", "och": "habituation", "ochr": "habituation+rehearse",
        "ocy": "reply-rehearse", "ochy": "habituation+reply-rehearse"}


def stats(path):
    d = json.load(open(path))
    rows = d["rows"]
    out = {}
    for arm in ("intact", "hippocampus off"):
        R = [r for r in rows if r["arm"] == arm]
        top = collections.Counter(r["brain"].strip() for r in R).most_common(1)[0][1]
        s = d["summary"][arm]
        out[arm] = dict(top=top, collapse=top > 3, echo=sum(r["brain"].strip() == r["you"].strip() for r in R),
                        recall=s["direct_ok"] + s["reworded_ok"], false=s["false"], intr=s["intrusions"],
                        needless=s["needless_recall"])
    return out


if __name__ == "__main__":
    d, seeds = sys.argv[1], sys.argv[2:]
    S = {}
    for a in ARMS:
        for s in seeds:
            for ext in ("json", "md"):     # AWS writes <name>.json, the pool <name>.md
                p = f"{d}/{a}_s{s}.{ext}.rows.json"
                if os.path.exists(p):
                    S[a, s] = stats(p)
    for s in seeds:
        print(f"s{s} " + " | ".join(f"{ARMS[a]}: rep x{S[a, s]['intact']['top']} echo {S[a, s]['intact']['echo']} "
                                     f"rec {S[a, s]['intact']['recall']} (off {S[a, s]['hippocampus off']['recall']}) "
                                     f"false {S[a, s]['intact']['false']} intr {S[a, s]['intact']['intr']} "
                                     f"needless {S[a, s]['intact']['needless']}" for a in ARMS if (a, s) in S))
    have = [a for a in ARMS if all((a, s) in S for s in seeds)]
    tot = {a: {k: sum(S[a, s]["intact"][k] for s in seeds) for k in ("collapse", "recall", "false", "intr", "needless", "echo")}
           for a in have}
    for a in have:
        print(f"{ARMS[a]}: collapsed {tot[a]['collapse']}/{len(seeds)}; recall {tot[a]['recall']}/{6 * len(seeds)}; "
              f"false {tot[a]['false']}; intrusions {tot[a]['intr']}; needless {tot[a]['needless']}; echoes {tot[a]['echo']}")
    if "ocd" in have:
        for a in have:
            if a == "ocd":
                continue
            r1 = (len(seeds) - tot[a]["collapse"]) >= len(seeds) - 1 and tot[a]["collapse"] < tot["ocd"]["collapse"]
            r2 = tot[a]["recall"] >= tot["ocd"]["recall"]
            r3 = tot[a]["false"] + tot[a]["intr"] <= tot["ocd"]["false"] + tot["ocd"]["intr"]
            print(f"{ARMS[a]} vs default: R1 {r1} R2 {r2} R3 {r3} (R4 = blind reading)")
