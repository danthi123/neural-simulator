"""Mechanical adjudication of a blind moat hunt round from two judges' labels and judge_prep.py's private key. No checker is run.

usage (cwd = worktree): python -m research.findings.raw._moat_allowlist_hunt4.adjudicate <cases.json> <key.json> <out adjudication.json>
                        --judge literal <files...> --judge ordinary <files...>
  a judge's files are its chunk outputs, each a JSON list of {"jid", "label": "U"|"F"|"B", "reason"}; every jid of the key must be
  labelled exactly once by each judge (a missing, repeated or unknown jid, or another label, stops the script: exit 2, nothing written).
  The key must hold every case's full item and every proper selection of every reply case (only the proper prefixes when judge_prep ran
  with --prefixes-only, which key.json records as "selections": "prefixes").

THE RULE (fixed before any run and applied mechanically):
  a CASE (its sentence as written), the rule recorded in rounds 4-7's adjudication.json `_doc`, the only behaviour: an expect-U case
  becomes B if the skeptic flagged it arguable ("arguable": true), or EITHER judge read it as supported (F), or BOTH judges read it as
  either-way (B); an expect-F case becomes B if either judge read it as unsupported (U) or both read it as either-way (B); every other
  case keeps the skeptic's label (so a U case with one judge U and one judge B stays U).
  a SELECTION of a reply case (a proper prefix or another in-order subsequence) carries no expect label of its own (Claude ruling
  2026-10-21, the conservative direction for a leak bar, so a U/B split counts as U): U if NEITHER judge reads it F and AT LEAST ONE
  reads it U; F if both read it F; B otherwise.
OUTPUT keys (rounds 4-7's, so run_hunt.py and the verdict scripts read the round the same way): _doc, relabel_to_B ({id: "U->B" |
  "F->B"}), reasons, final_counts, label_crosstab, judge_labels ({id: {judge: [label, reason]}}), provenance, backend, device; new:
  final_labels ({id: label}); subset_labels ({reply case id: {selection key: label}} for every judged selection AND the full reply,
  whose label is the case's final label: the one key family run_hunt reads); subset_judge_labels; subset_counts (over the selection
  items); and, for compatibility, prefix_labels ({reply case id: [label of prefix 1, ..., label of prefix k]}, derived from
  subset_labels). Round-specific notes (bar3_count_note, split_rule, blindness_note) are Claude's to add.
"""
import research.runners  # noqa: F401  (the provenance door; stdlib-only, must be first)
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from itertools import combinations

LABELS = ("U", "F", "B")
RULE_DOC = ("The rule, fixed before any run and applied mechanically: a U case becomes B if the skeptic flagged it arguable, or EITHER "
            "judge read it as supported (F), or BOTH judges read it as either-way (B); an F case becomes B if either judge read it as "
            "unsupported or both read it as either-way; every other case keeps the skeptic's label (so a U case with one judge U and "
            "one judge B stays U). A selection of a reply case (a proper prefix or another in-order subsequence of its sentences) has "
            "no expect label: U if neither judge read it F and at least one read it U (a U/B split counts as U, the conservative "
            "direction for a leak bar), F if both read it F, else B.")


def case_label(expect, a, b, arguable=False):
    """(final label, reason or None) for a case whose skeptic label is `expect`, judged `a` and `b` (the rounds 4-7 rule)."""
    if expect == "U":
        if arguable:
            return "B", "skeptic flagged it arguable"
        if "F" in (a, b):
            return "B", "a blind judge read it as supported"
        if a == b == "B":
            return "B", "both blind judges: either way"
        return "U", None
    if expect == "F":
        if "U" in (a, b):
            return "B", "a blind judge read it as unsupported"
        if a == b == "B":
            return "B", "both blind judges: either way"
        return "F", None
    if expect == "B":
        return "B", None
    raise ValueError("expect label %r is not U, F or B" % (expect,))


def part_label(a, b):
    """A selection: U if neither judge read it F and at least one read it U; F if both read F; else B."""
    if "F" not in (a, b) and "U" in (a, b):
        return "U"
    return "F" if a == b == "F" else "B"


def selection_key(idx):
    return "+".join(str(i) for i in idx)


def proper_selection_keys(k, prefixes_only=False):
    """Every proper in-order non-empty selection key of k sentences (only the proper prefixes when `prefixes_only`)."""
    keys = [selection_key(c) for n in range(1, k) for c in combinations(range(1, k + 1), n)]
    if prefixes_only:
        keys = [selection_key(range(1, j + 1)) for j in range(1, k)]
    return keys


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


def adjudicate(cases, key, judges, prefixes_only=False):
    """The adjudication dict. `key` = {jid: [case id, part]} (part "full" or a proper selection key "1", "1+3", ...); `judges` = an
    ordered {name: {jid: (label, reason)}} of exactly two judges. ValueError when the key does not hold every case's full item and every
    reply case's proper selections (proper prefixes only when `prefixes_only`) exactly once, or names an unknown case or part."""
    if len(judges) != 2:
        raise ValueError("exactly two judges are needed, got %d" % len(judges))
    names = list(judges)
    by_id = {c["id"]: c for c in cases}
    k_of = {c["id"]: (len(c.get("sentences") or []) if c.get("reply") is True else 0) for c in cases}
    parts = {}
    for jid, (cid, part) in key.items():
        if cid not in by_id:
            raise ValueError("key names unknown case %r" % (cid,))
        if (cid, part) in parts:
            raise ValueError("key holds %s %r twice" % (cid, part))
        if part != "full" and part not in proper_selection_keys(k_of[cid]):
            raise ValueError("key part %r of %s is not 'full' or a proper in-order selection of a reply case" % (part, cid))
        parts[(cid, part)] = jid
    for c in cases:
        for part in ["full"] + proper_selection_keys(k_of[c["id"]], prefixes_only):
            if (c["id"], part) not in parts:
                raise ValueError("key has no %r item for %s" % (part, c["id"]))

    def two(jid):
        return [judges[n][jid] for n in names]

    final, relabel, reasons, jl, crosstab = {}, {}, {}, {}, Counter()
    subset_labels, subset_jl, prefix_labels, scounts = {}, {}, {}, Counter()
    for c in cases:
        (a, ra), (b, rb) = two(parts[(c["id"], "full")])
        lab, why = case_label(c["expect"], a, b, arguable=bool(c.get("arguable")))
        final[c["id"]] = lab
        jl[c["id"]] = {names[0]: [a, ra], names[1]: [b, rb]}
        crosstab["expect %s, %s %s, %s %s" % (c["expect"], names[0], a, names[1], b)] += 1
        if lab != c["expect"]:
            relabel[c["id"]] = "%s->%s" % (c["expect"], lab)
            reasons[c["id"]] = why
        k = k_of[c["id"]]
        if k:
            sl, sj = {}, {}
            for part in proper_selection_keys(k):
                if (c["id"], part) in parts:
                    (sa, rsa), (sb, rsb) = two(parts[(c["id"], part)])
                    sl[part] = part_label(sa, sb)
                    sj[part] = {names[0]: [sa, rsa], names[1]: [sb, rsb]}
                    scounts[sl[part]] += 1
            sl[selection_key(range(1, k + 1))] = lab                    # the full reply: the case's own final label
            subset_labels[c["id"]], subset_jl[c["id"]] = sl, sj
            prefix_labels[c["id"]] = [sl[selection_key(range(1, j + 1))] for j in range(1, k + 1)]
    out = {"_doc": "Mechanical adjudication (adjudicate.py) of a blind hunt round from two blind judges (%s, %s) who labelled every item "
                   "U/F/B from an anonymised, shuffled, label-free copy (judge_prep.py). %s" % (names[0], names[1], RULE_DOC),
           "relabel_to_B": {i: relabel[i] for i in sorted(relabel) if relabel[i].endswith("->B")},
           "reasons": {i: reasons[i] for i in sorted(reasons)},
           "final_labels": final, "final_counts": dict(Counter(final.values())),
           "label_crosstab": dict(sorted(crosstab.items())), "judge_labels": jl,
           "subset_labels": subset_labels, "subset_judge_labels": subset_jl, "subset_counts": dict(scounts),
           "prefix_labels": prefix_labels,
           "selections": "prefixes" if prefixes_only else "all"}
    out["backend"] = out["device"] = "none (label adjudication, no simulation)"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cases")
    ap.add_argument("key")
    ap.add_argument("out")
    ap.add_argument("--judge", nargs="+", action="append", metavar="NAME_THEN_FILES", required=True,
                    help="a judge's name, then its output files (give --judge twice)")
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
        out = adjudicate(cases, key, judges, prefixes_only=(key_doc.get("selections") == "prefixes"))
    except ValueError as e:
        print("adjudicate: %s" % e, file=sys.stderr)
        return 2
    sha = hashlib.sha256(open(a.cases, "rb").read()).hexdigest()
    out["provenance"] = ("written by adjudicate.py from %s (sha256 %s), the judge key %s and the judge files %s; no checker was run"
                         % (a.cases, sha[:12], a.key, {j[0]: j[1:] for j in a.judge}))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    research.runners.declare_output(a.out)
    print("adjudicate: %s; selections %s -> %s" % (out["final_counts"], out["subset_counts"], a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
