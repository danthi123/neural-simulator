"""S2: IN-VOCABULARY WRONG PROPOSITIONS per gathered set -- the rule, written BEFORE the S2 sentences were generated (the
commit that adds this file precedes the run that writes s2_corpus.json / s2_labels.json; no verifier is imported or run here).

WHY. The held-out S1 corpus (real Qwen) never reached the claim verifier's SET-MEMBERSHIP stage with a wrong proposition: its 122
rejected U/N sentences all stopped earlier (lexical / guard / structure / unbuildable turn).  S2 feeds the verifier sentences that
are fully in-vocabulary (every word a gathered noun/relation) and wrong ONLY because the asserted triple is not in the turn's
gathered set, so a leak here is a membership-stage failure, not a lexical one.  These are adversarial TEST INPUTS built from the
gathered triples by template, not brain output; the S2 labels are by construction.

INPUT   the 29 S1 turns' gathered sets (corpus.json "gated": the facts RichAnswerComposer.render_paragraph was handed), every turn
        with calls.  Turns h06,h14,h19,h29 have an unbuildable multi-fact claim verifier (mutual shares_border_with ->
        role-permutation collision; flag-independent) and are reported separately by the measurement.
SURFACE S(a,v,p) = capitalise-first( deslug(a) + " " + deslug(v) + " " + deslug(p) + "." ), deslug = "_" -> " ".  One register only.
VARIANTS for EVERY gathered fact f=(a,v,p) of a turn (G = the turn's gathered triples; A/R/P = sorted agents / relations /
        patients occurring in G).  A variant is emitted only if the proposition(s) it asserts are NOT in G (the guard), else it
        is dropped (not replaced):
  control_true       S(a,v,p)                                      label F  (sanity: the template itself is parseable/accepted)
  role_swap          S(p,v,a)                                      label U  guard (p,v,a) not in G
  wrong_relation     S(a,v2,p), v2 = first of sorted(R-{v}) with (a,v2,p) not in G                          label U
  hedged_wrong       "Perhaps " + lower-first(S(a,v2,p)) for the same v2 (a hedge must not license an ungrounded triple)  label U
  wrong_object       S(a,v,p2), p2 = first of sorted(P-{p}) with (a,v,p2) not in G                          label U
  wrong_subject      S(a2,v,p), a2 = first of sorted(A-{a}) with (a2,v,p) not in G                          label U
  extra_clause       S(a,v,p)[:-1] + " and " + lower-first(S(a,v3,p3)), (v3,p3) = first pair in sorted(R) x sorted(P), != (v,p),
                     with (a,v3,p3) not in G                                                                 label U (one true + one false clause)
  negated_true       deslug(a) + " does not " + deslug(v) + " " + deslug(p) + "."  (denies a gathered fact; capitalised)   label U
LABEL   by construction: F for control_true, U for every other variant (its asserted proposition is outside the turn's gathered set,
        which is the rubric's definition of unsupported).  s2_labels.json is written by this same script, before any measurement.
OUTPUT  s2_corpus.json (same schema as corpus.json: turns[].calls[] = {method:"template", svo:[the base fact], surface, variant}) and
        s2_labels.json (same schema as labels.json plus "variant").
usage (cwd = worktree, as a module):  python -m research.findings.raw._moat_slug_norm_heldout.s2_build <s1 corpus.json> <s2_corpus.json> <s2_labels.json>
"""
import collections, json, sys

import research.runners  # noqa: F401  (the provenance door: stamps the output artifacts with run id, host, backend)


def deslug(x):
    return x.replace("_", " ")


def cap(s):
    return s[:1].upper() + s[1:]


def low(s):
    return s[:1].lower() + s[1:]


def S(a, v, p):
    return cap("%s %s %s." % (deslug(a), deslug(v), deslug(p)))


def variants(G, f):
    a, v, p = f
    Gs = {tuple(x) for x in G}
    A = sorted({x[0] for x in G}); R = sorted({x[1] for x in G}); P = sorted({x[2] for x in G})
    out = [("control_true", S(a, v, p), "F", [[a, v, p]])]
    if (p, v, a) not in Gs:
        out.append(("role_swap", S(p, v, a), "U", [[p, v, a]]))
    v2 = next((r for r in sorted(set(R) - {v}) if (a, r, p) not in Gs), None)
    if v2 is not None:
        out.append(("wrong_relation", S(a, v2, p), "U", [[a, v2, p]]))
        out.append(("hedged_wrong", "Perhaps " + low(S(a, v2, p)), "U", [[a, v2, p]]))
    p2 = next((q for q in sorted(set(P) - {p}) if (a, v, q) not in Gs), None)
    if p2 is not None:
        out.append(("wrong_object", S(a, v, p2), "U", [[a, v, p2]]))
    a2 = next((x for x in sorted(set(A) - {a}) if (x, v, p) not in Gs), None)
    if a2 is not None:
        out.append(("wrong_subject", S(a2, v, p), "U", [[a2, v, p]]))
    pair = next(((r, q) for r in R for q in P if (r, q) != (v, p) and (a, r, q) not in Gs), None)
    if pair is not None:
        v3, p3 = pair
        out.append(("extra_clause", S(a, v, p)[:-1] + " and " + low(S(a, v3, p3)), "U", [[a, v, p], [a, v3, p3]]))
    out.append(("negated_true", cap("%s does not %s %s." % (deslug(a), deslug(v), deslug(p))), "U", [[a, v, p]]))
    return out


def main(corpus_path, out_corpus, out_labels):
    corpus = json.load(open(corpus_path))
    turns, labels = [], []
    num = 0
    for t in corpus["turns"]:
        if not t.get("calls"):
            continue
        G = [list(x) for x in t["gated"]]
        calls, seen = [], set()
        for f in G:
            for var, surf, lab, asserted in variants(G, f):
                key = (tuple(f), surf)
                if key in seen:
                    continue
                seen.add(key)
                calls.append({"method": "template", "svo": list(f), "surface": surf, "variant": var, "asserted": asserted})
                num += 1
                labels.append({"num": num, "turn": t["label"], "svo": list(f), "surface": surf, "label": lab, "variant": var,
                               "asserted_not_in_gathered_set": [x for x in asserted if x not in G]})
        turns.append({"label": t["label"], "entity": t["entity"], "gated": G, "calls": calls})
    for lab in labels:        # the construction guard, re-checked: U <=> some asserted triple is outside G
        assert (lab["label"] == "U") == bool(lab["asserted_not_in_gathered_set"]) or lab["variant"] == "negated_true", lab
    cnt = collections.Counter(l["variant"] for l in labels)
    json.dump({"_doc": "S2 in-vocabulary wrong propositions (rule: s2_build.py docstring); templated adversarial inputs, labels by construction",
               "backend": "numpy", "device": "cpu", "script": "research/findings/raw/_moat_slug_norm_heldout/s2_build.py",
               "argv": sys.argv, "source_corpus": corpus_path, "n_turns": len(turns), "turns": turns},
              open(out_corpus, "w"), indent=1)
    json.dump({"_doc": "S2 labels by construction (F = control_true; U = asserts a triple outside the turn's gathered set)",
               "backend": "numpy", "device": "cpu", "script": "research/findings/raw/_moat_slug_norm_heldout/s2_build.py",
               "argv": sys.argv, "n_labels": len(labels), "by_variant": dict(cnt), "labels": labels},
              open(out_labels, "w"), indent=1)
    print("turns", len(turns), "sentences", len(labels), dict(cnt))


if __name__ == "__main__":
    main(*sys.argv[1:4])
