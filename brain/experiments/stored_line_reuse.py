# Capability: Sleep consolidation
# Record: brain/experiments/results/2026-11-16-rehearsal-night-1.5b.md (paraphrase-aware loop count)
"""Day-2 replies that reuse a stored day-1 small-talk line, paraphrase included: the reply shares >= 2 content words
(not in STOP, length > 2) with one day-1 user turn that is not a telling or a question (greet, chat), and those words
are not in the current user turn. Exact-repeat counts miss a loop that habituation turns into paraphrase.
usage: stored_line_reuse.py <dir> <seeds...>   (reads <arm>_s<seed>.{json,md}.rows.json, intact arm)"""
import glob, json, re, sys

STOP = set("i me my you your the a an and or of to in on for at by with is are was were be been it this that these "
           "those so very really lately last today week morning night lots have has had do did does just went made "
           "going trying get got about from what how who which can like not".split())


def cw(t):
    return {w for w in re.findall(r"[a-z']+", t.lower()) if w not in STOP and len(w) > 2}


def count(rows):
    lines = [cw(r["you"]) for r in rows if r["arm"] == "day1" and r["kind"] not in ("tell", "direct")]
    return sum(any(len((L & cw(r["brain"])) - cw(r["you"])) >= 2 for L in lines) for r in rows if r["arm"] == "intact")


if __name__ == "__main__":
    d, seeds = sys.argv[1], sys.argv[2:]
    for a in ("oc", "ocd", "ocr", "och", "ochr", "ocy", "ochy"):
        per = []
        for s in seeds:
            f = glob.glob(f"{d}/{a}_s{s}.*.rows.json")
            if f:
                per.append(count(json.load(open(f[0]))["rows"]))
        if per:
            print(f"{a}: {sum(per)} of {16 * len(per)} day-2 replies; per seed {per}")
