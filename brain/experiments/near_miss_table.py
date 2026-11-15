# Capability: Sleep consolidation
# Record: brain/research/2026-11-14-near-miss-guard-1.5b.md (pre-registration, N0-N4)
"""Paired mixed-day table for the near-miss guard: mixed_day_probe --arms c,s,r --self-q without (nmoff_s*.json) and
with --near-miss 4 (nmon_s*.json). Primary arm s (the shipped 1.5B package); arms c and r reported. The optional ref dir
holds the 11-14 tellmix_s*.json (N0 reproduction check). Run on re-scored copies (brain.experiments.rescore).

  python -m brain.experiments.near_miss_table <dir> [<ref_dir>] [--on <prefix>]   (default nmon; M4b: nmft)
M4b (brain/research/2026-11-15-near-miss-told-filter-1.5b.md) adds N5 (inventions on never-told rows not up by more
than 1) and N6 (abstentions on told rows not up by more than 0.5); abstention = ABSTAIN below, invented = a never-told
reply that is neither a told fact nor an abstention.
"""
import re
import glob
import json
import os
import sys
from fractions import Fraction


def load(d, pre):
    R = {}
    for p in sorted(glob.glob(os.path.join(d, f"{pre}_s*.json"))):
        x = json.load(open(p))
        R[x["seed"]] = x
    return R


def cnt(x, arm, kind, st=None):
    return sum(r[arm]["ok"] for r in x["rows"] if r["kind"] == kind and (st is None or r.get("set") == st))


ABSTAIN = re.compile(r"i'm sorry|i am sorry|don't have (access|any|information|that)|i don't know|i do not know|"
                     r"not sure|no information|can't (know|recall|access)|cannot (know|recall|access)", re.I)


def abst(x, arm, kind):
    return sum(bool(ABSTAIN.search(r[arm]["reply"])) for r in x["rows"] if r["kind"] == kind)


def invented(x, arm):
    return sum(not r[arm]["ok"] and not ABSTAIN.search(r[arm]["reply"]) for r in x["rows"] if r["kind"] == "never")


def lure(x, arm):
    return sum(r["kind"] == "never" and "aunt" in r["q"] and "oslo" in r[arm]["reply"].lower() for r in x["rows"])


def verdict(OFF, ON, REF=None, arm="s", m4b=False):
    seeds = sorted(set(OFF) & set(ON))
    S = len(seeds)
    m = lambda R, kind, st=None: Fraction(sum(cnt(R[s], arm, kind, st) for s in seeds), S)
    flags = all(not OFF[s].get("near_miss") and ON[s].get("near_miss") for s in seeds)
    kept = all(sum(len(e["kept"]) for e in ON[s].get("near_miss_log", [])) >= 1 for s in seeds)
    repro = None
    if REF:
        repro = all(s in REF and all(cnt(OFF[s], a, k) == cnt(REF[s], a, k) for a in ("s", "r") for k in ("told", "never"))
                    for s in seeds)
    n0 = S == 6 and flags and kept and repro is not False
    n1 = m(ON, "never") <= Fraction(1, 2) * m(OFF, "never") and \
        sum(cnt(ON[s], arm, "never") < cnt(OFF[s], arm, "never") for s in seeds) >= 5
    lead_false = m(OFF, "never") - m(ON, "never") >= 2 and \
        sum(cnt(ON[s], arm, "never") < cnt(OFF[s], arm, "never") for s in seeds) >= 5
    n2 = m(ON, "told") >= Fraction(85, 100) * m(OFF, "told")
    n3 = m(ON, "para") >= m(OFF, "para") - 1
    n4 = m(ON, "self") >= m(OFF, "self") - Fraction(1, 2)
    mi = lambda R, f, *k: Fraction(sum(f(R[s], arm, *k) for s in seeds), S)
    n5 = mi(ON, invented) <= mi(OFF, invented) + 1
    n6 = mi(ON, abst, "told") <= mi(OFF, abst, "told") + Fraction(1, 2)
    extra = (n5 and n6) if m4b else True   # M4b's prereg adds N5-N6 to GO and LEAD
    if not n0:
        v = "VOID"
    elif n1 and n2 and n3 and n4 and extra:
        v = "GO"
    elif lead_false and n2 and n3 and n4 and extra:
        v = "LEAD"
    else:
        v = "NOT A LEAD"
    return {"N0": n0, "repro": repro, "N1": n1, "lead_false": lead_false, "N2": n2, "N3": n3, "N4": n4,
            "N5": n5, "N6": n6, "verdict": v, "m": m, "mi": mi, "seeds": seeds}


def main():
    args = sys.argv[1:]
    on = "nmon"
    if "--on" in args:
        i = args.index("--on")
        on = args[i + 1]
        del args[i:i + 2]
    d = args[0]
    REF = load(args[1], "tellmix") if len(args) > 1 else None
    OFF, ON = load(d, "nmoff"), load(d, on)
    for arm in ("s", "c", "r"):
        v = verdict(OFF, ON, REF, arm, m4b=on == "nmft")
        m = v["m"]
        print(f"== arm {arm} (off/on per seed)")
        for s in v["seeds"]:
            f = lambda kind, st=None: f"{cnt(OFF[s], arm, kind, st)}/{cnt(ON[s], arm, kind, st)}"
            print(f"  {s:<5} told {f('told'):<6} slept {f('told', 'slept'):<5} new {f('told', 'new'):<5} para "
                  f"{f('para'):<6} false {f('never'):<6} self {f('self'):<4} aunt->oslo {lure(OFF[s], arm)}/{lure(ON[s], arm)}")
        for name, R in (("off", OFF), ("on", ON)):
            print(f"  mean {name:<4}: told {float(m(R, 'told')):.2f}  slept {float(m(R, 'told', 'slept')):.2f}  new "
                  f"{float(m(R, 'told', 'new')):.2f}  para {float(m(R, 'para')):.2f}  false {float(m(R, 'never')):.2f}  "
                  f"self {float(m(R, 'self')):.2f}  invented {float(v['mi'](R, invented)):.2f}  abstain told "
                  f"{float(v['mi'](R, abst, 'told')):.2f} never {float(v['mi'](R, abst, 'never')):.2f}")
    for s in sorted(ON):
        log = ON[s].get("near_miss_log", [])
        print(f"  cues s{s}: generated {sum(len(e['cues']) for e in log)} kept {sum(len(e['kept']) for e in log)}")
    v = verdict(OFF, ON, REF, "s", m4b=on == "nmft")
    print(f"N0 (flags, cues kept, reproduces 11-14: {v['repro']}): {'holds' if v['N0'] else 'FAILS'}")
    for k, t in (("N1", "false <= 0.5 x off, lower on >= 5/6"), ("N2", "told >= 0.85 x off"), ("N3", "para >= off - 1"),
                 ("N4", "self >= off - 0.5")):
        print(f"{k} ({t}): {'holds' if v[k] else 'FAILS'}")
    print(f"LEAD false bar (-2, >= 5/6): {'holds' if v['lead_false'] else 'fails'}")
    for k, t in (("N5", "invented never-told <= off + 1 (M4b)"), ("N6", "told abstentions <= off + 0.5 (M4b)")):
        print(f"{k} ({t}): {'holds' if v[k] else 'FAILS'}")
    print(f"VERDICT (primary arm s): {v['verdict']}")


if __name__ == "__main__":
    main()
