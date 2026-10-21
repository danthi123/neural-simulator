"""Build-6 REVIEW DIAGNOSTIC (non-blind; disclosed in prereg Amendment 6): the pure matcher's verdict on every case of hunt rounds 3-7, at
build 6 vs the committed build-5 allowlist-arm verdict (results_build5.json / r3_rerun_build5.json, production claim gate, seed 42; the
matcher is pure, so one seed suffices). Labels are the committed adjudicated labels of each round. Reports, per round and label, the cases
whose verdict changed, and the build-6 reject reason of each newly rejected case. Writes no case, reads no held-out stratum.

usage (cwd = worktree, as a module; refuses to overwrite):
  python -m research.findings.raw._moat_allowlist_build6.matcher_diff_vs_build5 <out.json>
"""
import research.runners  # noqa: F401  (provenance door)
import collections
import json
import os
import socket
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from research.runners._moat_allowlist import allowlist_verify  # noqa: E402

H4 = "research/findings/raw/_moat_allowlist_hunt4"
ROUNDS = {"r3": (H4 + "/r3_rerun_build5.json", "research/findings/raw/_moat_allowlist_hunt3/cases.json"),
          "r4": (H4 + "/r4/results_build5.json", H4 + "/r4/cases.json"),
          "r5": (H4 + "/r5/results_build5.json", H4 + "/r5/cases.json"),
          "r6": (H4 + "/r6/results_build5.json", H4 + "/r6/cases.json"),
          "r7": (H4 + "/r7/results_build5.json", H4 + "/r7/cases.json")}


def main(out_path):
    if os.path.exists(out_path):
        raise SystemExit("refusing to overwrite %s" % out_path)
    out = {"_doc": __doc__.split("\n\n")[0], "backend": os.environ.get("SIM_BACKEND"), "device": "cpu", "host": socket.gethostname(),
           "argv": sys.argv, "rounds": {}}
    for rnd, (res_path, cases_path) in ROUNDS.items():
        rows = json.load(open(res_path))["rows"]
        cases = {c["id"]: c for c in json.load(open(cases_path))}
        tally = collections.Counter()
        changed = []
        for r in rows:
            c = cases[r["id"]]
            b5 = r["gate"]["allowlist"][0]["v"]
            res = allowlist_verify(c["sentence"], c["gathered"])
            b6 = bool(res["accepted"])
            tally[(r["label"], b5, b6)] += 1
            if b5 != b6:
                changed.append({"id": r["id"], "label": r["label"], "class": r.get("class"), "sentence": c["sentence"],
                                "build5_kept": b5, "build6_kept": b6, "build6_reason": res["reject_reason"]})
        out["rounds"][rnd] = {"n": len(rows), "cells(label,build5_kept,build6_kept)": {str(k): v for k, v in sorted(tally.items())},
                              "kept_build5_by_label": dict(collections.Counter(r["label"] for r in rows if r["gate"]["allowlist"][0]["v"])),
                              "kept_build6_by_label": dict(collections.Counter(lab for (lab, _b5, b6), n in tally.items() if b6
                                                                              for _ in range(n))),
                              "changed": changed}
        print(rnd, len(rows), "changed", len(changed), dict(collections.Counter((x["label"], x["build6_kept"]) for x in changed)), flush=True)
    json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
