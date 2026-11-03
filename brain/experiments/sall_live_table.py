# Capability: Memory selection (pattern completion and separation)
# Record: brain/experiments/results/2026-11-03-sall-in-brain.md (pre-registration this table scores)
"""Table + pre-registered criteria for sall_live_probe JSONs (2026-11-03-sall-in-brain.md).

  python -m brain.experiments.sall_live_table <dir>
"""
import glob
import json
import sys

FIELDS = ["recall", "paraphrase", "false_recall", "recall_correct", "paraphrase_correct", "sel_told", "sel_para",
          "never_fired", "echo"]


def main(d):
    runs = sorted((json.load(open(p)) for p in glob.glob(f"{d}/seed*.json")), key=lambda r: r["seed"])
    seeds = [r["seed"] for r in runs]
    print("seeds", seeds)
    arms = list(runs[0]["arms"])
    v = {a: {f: [r["arms"][a][f] for r in runs] for f in FIELDS} for a in arms}
    m = lambda xs: sum(xs) / len(xs)
    print("| arm | " + " | ".join(FIELDS) + " |")
    print("|---" * (len(FIELDS) + 1) + "|")
    for a in arms:
        print(f"| {a} | " + " | ".join(f"{m(v[a][f]):.2f} {v[a][f]}" for f in FIELDS) + " |")
    for p in ("a", "b"):
        s, z = v[f"{p}_Sall"], v[f"{p}_S0"]
        dp = m(s["paraphrase"]) - m(z["paraphrase"])
        ge = sum(x >= y for x, y in zip(s["paraphrase"], z["paraphrase"]))
        P1 = dp >= 1.0 and ge >= 5
        dfr = m(s["false_recall"]) - m(z["false_recall"])
        gt = sum(x > y for x, y in zip(s["false_recall"], z["false_recall"]))
        P2 = dfr <= 0.5 and gt <= 2
        drc = m(s["recall"]) - m(z["recall"])
        P3 = drc >= -0.5
        tag = "PRIMARY" if p == "a" else "secondary (reported)"
        print(f"{p} {tag}: P1 paraphrase d={dp:+.2f} Sall>=S0 on {ge}/{len(runs)} -> {'PASS' if P1 else 'FAIL'}; "
              f"P2 false recall d={dfr:+.2f} Sall>S0 on {gt}/{len(runs)} -> {'PASS' if P2 else 'FAIL'}; "
              f"P3 recall d={drc:+.2f} -> {'PASS' if P3 else 'FAIL'}")
    diff = sum(ra["reply"] != rc["reply"] for r in runs
               for ra, rc in zip(r["arms"]["a_Sall"]["rows"], r["arms"]["c_hippo_off"]["rows"]))
    tot = sum(len(r["arms"]["a_Sall"]["rows"]) for r in runs)
    print(f"lesion: c_hippo_off reply differs from a_Sall on {diff}/{tot} questions")


if __name__ == "__main__":
    main(sys.argv[1])
