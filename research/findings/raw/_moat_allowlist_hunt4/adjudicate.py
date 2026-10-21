"""Mechanical adjudication of a blind moat hunt round from two judges' labels and judge_prep.py's private key. No checker is run.

usage (cwd = worktree): python -m research.findings.raw._moat_allowlist_hunt4.adjudicate <cases.json> <key.json> <out adjudication.json>
                        --judge literal <files...> --judge ordinary <files...> [--f-rule rounds4-7|stays-F]
  a judge's files are its chunk outputs, each a JSON list of {"jid", "label": "U"|"F"|"B", "reason"}; every jid of the key must be
  labelled exactly once by each judge (a missing, repeated or unknown jid, or another label, stops the script: exit 2, nothing written).

THE RULE (rounds 4-7, recorded in their adjudication.json `_doc`, fixed before any run and applied mechanically):
  a case (its sentence as written): an expect-U case becomes B if the skeptic flagged it arguable ("arguable": true), or EITHER judge
  read it as supported (F), or BOTH judges read it as either-way (B); an expect-F case becomes B if either judge read it as unsupported
  (U) or both read it as either-way (B) (--f-rule stays-F: an expect-F case stays F, whatever the judges say); every other case keeps
  the skeptic's label (so a U case with one judge U and one judge B stays U).
  a PREFIX of a reply case (and, from judge_prep --subsequences, a non-prefix selection) carries no expect label of its own: U if both
  judges label it U, F if both label it F, else B.
OUTPUT keys (rounds 4-7's, so run_hunt.py and the verdict scripts read the round the same way): _doc, relabel_to_B ({id: "U->B" |
  "F->B"}), reasons, final_counts, label_crosstab, judge_labels ({id: {judge: [label, reason]}}), provenance, backend, device; new:
  final_labels ({id: label}), prefix_labels ({reply case id: [label of prefix 1, ..., label of prefix k]}, the last = the case's final
  label), prefix_judge_labels, prefix_counts, and subset_labels / subset_judge_labels when the key holds selections. Round-specific
  notes (bar3_count_note, split_rule, blindness_note) are Claude's to add.
"""
import research.runners  # noqa: F401  (the provenance door; stdlib-only, must be first)
import argparse
import hashlib
import json
import os
import sys
from collections import Counter

LABELS = ("U", "F", "B")
F_RULES = ("rounds4-7", "stays-F")
RULE_DOC = ("The rule, fixed before any run and applied mechanically: a U case becomes B if the skeptic flagged it arguable, or EITHER "
            "judge read it as supported (F), or BOTH judges read it as either-way (B); an F case becomes B if either judge read it as "
            "unsupported or both read it as either-way; every other case keeps the skeptic's label (so a U case with one judge U and "
            "one judge B stays U). A prefix of a reply case (or a non-prefix selection) has no expect label: U if both judges read it "
            "U, F if both read it F, else B.")
RULE_DOC_STAYS_F = RULE_DOC.replace("an F case becomes B if either judge read it as unsupported or both read it as either-way; ",
                                    "an F case stays F; ")


def case_label(expect, a, b, arguable=False, f_rule="rounds4-7"):
    """(final label, reason or None) for a case whose skeptic label is `expect`, judged `a` and `b`."""
    if f_rule not in F_RULES:
        raise ValueError("unknown f_rule %r" % (f_rule,))
    if expect == "U":
        if arguable:
            return "B", "skeptic flagged it arguable"
        if "F" in (a, b):
            return "B", "a blind judge read it as supported"
        if a == b == "B":
            return "B", "both blind judges: either way"
        return "U", None
    if expect == "F":
        if f_rule == "stays-F":
            return "F", None
        if "U" in (a, b):
            return "B", "a blind judge read it as unsupported"
        if a == b == "B":
            return "B", "both blind judges: either way"
        return "F", None
    if expect == "B":
        return "B", None
    raise ValueError("expect label %r is not U, F or B" % (expect,))


def part_label(a, b):
    """A prefix (or selection): U if both judges read U, F if both read F, else B."""
    return "U" if a == b == "U" else "F" if a == b == "F" else "B"


def collect(judge_lists, key):
    """{jid: (label, reason)} from one judge's item lists; ValueError on a missing / repeated / unknown jid or a bad label."""
    got, problems = {}, []
    for items in judge_lists:
        if not isinstance(items, list):
            raise ValueError("a judge file is not a JSON list")
        for it in items:
            jid, lab = (it or {}).get("jid"), (it or {}).get("label")
            if jid not in key:
                problems.append("unknown jid %r" % (jid,))
            elif jid in got:
                problems.append("jid %s labelled twice" % jid)
            elif lab not in LABELS:
                problems.append("jid %s label %r is not U, F or B" % (jid, lab))
            else:
                got[jid] = (lab, it.get("reason"))
    missing = sorted(set(key) - set(got))
    if missing:
        problems.append("%d jid(s) unlabelled, e.g. %s" % (len(missing), missing[:3]))
    if problems:
        raise ValueError("; ".join(problems[:10]))
    return got


def adjudicate(cases, key, judges, f_rule="rounds4-7"):
    """The adjudication dict. `key` = {jid: [case id, part]} (part "full", an int prefix j, or a "1+3" selection); `judges` = an ordered
    {name: {jid: (label, reason)}} of exactly two judges. ValueError when the key does not cover every case's full item exactly once and
    every reply case's prefixes 1..k-1 exactly once, or names an unknown case or part."""
    if len(judges) != 2:
        raise ValueError("exactly two judges are needed, got %d" % len(judges))
    names = list(judges)
    by_id = {c["id"]: c for c in cases}
    parts = {}
    for jid, (cid, part) in key.items():
        if cid not in by_id:
            raise ValueError("key names unknown case %r" % (cid,))
        if (cid, part) in parts:
            raise ValueError("key holds %s %r twice" % (cid, part))
        parts[(cid, part)] = jid
    for c in cases:
        if (c["id"], "full") not in parts:
            raise ValueError("key has no full item for %s" % c["id"])
        k = len(c.get("sentences") or []) if c.get("reply") is True else 0
        for j in range(1, k):
            if (c["id"], j) not in parts:
                raise ValueError("key has no prefix %d item for reply case %s" % (j, c["id"]))
    for (cid, part) in parts:
        c = by_id[cid]
        k = len(c.get("sentences") or []) if c.get("reply") is True else 0
        if part == "full":
            continue
        if isinstance(part, int) and not isinstance(part, bool):
            if not 1 <= part < k:
                raise ValueError("key prefix %r of %s is not 1..%d" % (part, cid, k - 1))
        elif isinstance(part, str) and k:
            idx = [int(x) for x in part.split("+")]
            if idx != sorted(set(idx)) or not all(1 <= i <= k for i in idx) or idx == list(range(1, len(idx) + 1)):
                raise ValueError("key selection %r of %s is not an in-order non-prefix selection" % (part, cid))
        else:
            raise ValueError("key part %r of %s is not 'full', a prefix or a selection of a reply case" % (part, cid))

    def two(jid):
        return [judges[n][jid] for n in names]

    final, relabel, reasons, jl, crosstab = {}, {}, {}, {}, Counter()
    prefix_labels, prefix_jl, subset_labels, subset_jl, pcounts = {}, {}, {}, {}, Counter()
    for c in cases:
        (a, ra), (b, rb) = two(parts[(c["id"], "full")])
        lab, why = case_label(c["expect"], a, b, arguable=bool(c.get("arguable")), f_rule=f_rule)
        final[c["id"]] = lab
        jl[c["id"]] = {names[0]: [a, ra], names[1]: [b, rb]}
        crosstab["expect %s, %s %s, %s %s" % (c["expect"], names[0], a, names[1], b)] += 1
        if lab != c["expect"]:
            relabel[c["id"]] = "%s->%s" % (c["expect"], lab)
            reasons[c["id"]] = why
        if c.get("reply") is True:
            k = len(c["sentences"])
            pl, pj = [], {}
            for j in range(1, k):
                (pa, rpa), (pb, rpb) = two(parts[(c["id"], j)])
                pl.append(part_label(pa, pb))
                pj[str(j)] = {names[0]: [pa, rpa], names[1]: [pb, rpb]}
                pcounts[pl[-1]] += 1
            prefix_labels[c["id"]] = pl + [lab]
            prefix_jl[c["id"]] = pj
            for (cid, part), jid in parts.items():
                if cid == c["id"] and isinstance(part, str) and part != "full":
                    (sa, rsa), (sb, rsb) = two(jid)
                    subset_labels.setdefault(cid, {})[part] = part_label(sa, sb)
                    subset_jl.setdefault(cid, {})[part] = {names[0]: [sa, rsa], names[1]: [sb, rsb]}
    out = {"_doc": "Mechanical adjudication (adjudicate.py) of a blind hunt round from two blind judges (%s, %s) who labelled every item "
                   "U/F/B from an anonymised, shuffled, label-free copy (judge_prep.py). %s" % (names[0], names[1],
                                                                                               RULE_DOC if f_rule == "rounds4-7" else RULE_DOC_STAYS_F),
           "f_rule": f_rule,
           "relabel_to_B": {i: relabel[i] for i in sorted(relabel) if relabel[i].endswith("->B")},
           "reasons": {i: reasons[i] for i in sorted(reasons)},
           "final_labels": final, "final_counts": dict(Counter(final.values())),
           "label_crosstab": dict(sorted(crosstab.items())), "judge_labels": jl,
           "prefix_labels": prefix_labels, "prefix_judge_labels": prefix_jl, "prefix_counts": dict(pcounts)}
    if subset_labels:
        out["subset_labels"] = subset_labels
        out["subset_judge_labels"] = subset_jl
    out["backend"] = out["device"] = "none (label adjudication, no simulation)"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cases")
    ap.add_argument("key")
    ap.add_argument("out")
    ap.add_argument("--judge", nargs="+", action="append", metavar="NAME_THEN_FILES", required=True,
                    help="a judge's name, then its output files (give --judge twice)")
    ap.add_argument("--f-rule", choices=F_RULES, default="rounds4-7")
    a = ap.parse_args(argv)
    cases = json.load(open(a.cases))
    key_doc = json.load(open(a.key))
    key = key_doc["key"]
    try:
        if len(a.judge) != 2 or any(len(j) < 2 for j in a.judge):
            raise ValueError("give --judge NAME FILE [FILE ...] exactly twice")
        if a.judge[0][0] == a.judge[1][0]:
            raise ValueError("the two judges need different names")
        judges = {j[0]: collect([json.load(open(f)) for f in j[1:]], key) for j in a.judge}
        out = adjudicate(cases, key, judges, a.f_rule)
    except ValueError as e:
        print("adjudicate: %s" % e, file=sys.stderr)
        return 2
    sha = hashlib.sha256(open(a.cases, "rb").read()).hexdigest()
    out["provenance"] = ("written by adjudicate.py from %s (sha256 %s), the judge key %s and the judge files %s; no checker was run"
                         % (a.cases, sha[:12], a.key, {j[0]: j[1:] for j in a.judge}))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    research.runners.declare_output(a.out)
    print("adjudicate: %s; prefixes %s -> %s" % (out["final_counts"], out["prefix_counts"], a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
