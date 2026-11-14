# Capability: Episodic recall
# Record: brain/research/2026-11-14-key-default-1.5b.md (pre-registration, K0-K5)
"""Table + registered criteria for the 1.5B key-default test: part A (unslept_day_probe) and part B (mixed_day_probe
--arms c,s,r --self-q), arms c / s (S0, gain 0.5) / r (S_all + size bar, gain 0.5).

  python -m brain.experiments.key_default_table --a <dir> [--a-glob 'unslept_s*.json'] --b <dir> [--b-glob 'keydef_s*.json']
"""
import argparse
import glob
import json
import os
import re
from fractions import Fraction

from brain.experiments.strict_scorer import has_key

ABSTAIN = re.compile(r"\b(i don't know|i do not know|i'm sorry|i am sorry|i don't have|not mentioned|no information)\b",
                     re.I)
ARMS = "csr"


def load(d, g):
    R = {}
    for p in sorted(glob.glob(os.path.join(d, g))):
        x = json.load(open(p))
        R[x["seed"]] = x["rows"]
    return R


def cnt(rows, arm, kind, st=None):
    return sum(r[arm]["ok"] for r in rows if r["kind"] == kind and (st is None or r.get("set") == st))


def mean(R, arm, kind, st=None):
    return Fraction(sum(cnt(R[s], arm, kind, st) for s in R), max(1, len(R)))


def k0(R):
    rows = [r for s in R for r in R[s]]
    return bool(rows) and all(r["s"]["gain"] == 0.5 and r["r"]["gain"] == 0.5 for r in rows) and \
        all(r["s"].get("key", "S0") == "S0" and r["r"].get("key", "S_all") == "S_all" for r in rows)


def told_pass(R):
    d = mean(R, "r", "told") - mean(R, "s", "told")
    return d >= 1 and sum(cnt(R[s], "r", "told") >= cnt(R[s], "s", "told") for s in R) >= 5


def verdict(A, B):
    out = {"K0": k0(A) and k0(B), "K1": told_pass(A), "K2": told_pass(B),
           "K3": mean(B, "r", "told", "slept") >= mean(B, "s", "told", "slept") - Fraction(1, 2),
           "K4": all(mean(X, "r", "never") <= mean(X, "s", "never") + Fraction(1, 2) for X in (A, B)),
           "K5": all(mean(X, "r", "self") >= mean(X, "s", "self") - Fraction(1, 2) for X in (A, B))}
    if not out["K0"]:
        out["verdict"] = "VOID"
    elif all(out[k] for k in ("K1", "K2", "K3", "K4", "K5")):
        out["verdict"] = "FLIP"
    elif all(out[k] for k in ("K1", "K2", "K3", "K4")):
        out["verdict"] = "NOT YET (K5: self questions)"
    else:
        out["verdict"] = "KEEP S0"
    return out


def show(name, R):
    print(f"== part {name}: seeds {sorted(R)}")
    sets = name == "B"
    print("seed   told c/s/r   " + ("slept c/s/r   new c/s/r   " if sets else "") +
          "para c/s/r   false c/s/r   self c/s/r   abstain c/s/r   fact-word c/s/r")
    for s in sorted(R):
        rs = R[s]
        f3 = lambda kind, st=None: "/".join(str(cnt(rs, x, kind, st)) for x in ARMS)
        ab = "/".join(str(sum(bool(ABSTAIN.search(r[x]["reply"] or "")) for r in rs if r["kind"] in ("told", "para")))
                      for x in ARMS)
        fw = "/".join(str(sum(has_key(r[x]["reply"] or "", r["key"]) for r in rs if r["kind"] == "told")) for x in ARMS)
        mid = f"{f3('told', 'slept'):<14}{f3('told', 'new'):<12}" if sets else ""
        print(f"{s:<6} {f3('told'):<13}{mid}{f3('para'):<13}{f3('never'):<14}{f3('self'):<13}{ab:<16}{fw}")
    for x in ARMS:
        extra = (f"  slept {float(mean(R, x, 'told', 'slept')):.2f}  new {float(mean(R, x, 'told', 'new')):.2f}"
                 if sets else "")
        print(f"mean {x}: told {float(mean(R, x, 'told')):.2f}{extra}  para {float(mean(R, x, 'para')):.2f}  "
              f"false {float(mean(R, x, 'never')):.2f}  self {float(mean(R, x, 'self')):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--a-glob", default="unslept_s*.json")
    ap.add_argument("--b", default=None)
    ap.add_argument("--b-glob", default="keydef_s*.json")
    a = ap.parse_args()
    A = load(a.a, a.a_glob)
    show("A", A)
    if not a.b:
        print(f"\nK1 (A told r - s >= 1, r >= s on >= 5/6): {'holds' if told_pass(A) else 'FAILS'}  (part B pending)")
        return
    B = load(a.b, a.b_glob)
    show("B", B)
    v = verdict(A, B)
    print()
    for k, t in (("K0", "gain 0.5 and keys recorded"), ("K1", "A told r - s >= 1, >= 5/6"),
                 ("K2", "B told r - s >= 1, >= 5/6"), ("K3", "B slept r >= s - 0.5"),
                 ("K4", "false r <= s + 0.5, A and B"), ("K5", "self r >= s - 0.5, A and B")):
        print(f"{k} ({t}): {'holds' if v[k] else 'FAILS'}")
    print(f"VERDICT: {v['verdict']}")


if __name__ == "__main__":
    main()
