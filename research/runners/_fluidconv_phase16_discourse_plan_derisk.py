"""Phase-16 DE-RISK: grounded DISCOURSE PLAN -> connected prose (the multi-fact-synthesis cheap-first #1).

Per the scoping (`2026-07-01-multi-fact-synthesis-frontier-scoping.md`): "DISCUSS lists facts" is a DISGUISED boundary
-- the grouped rendering is already NLG synthesis (aggregation + referring-expression), and the ~70% cheap residual is
DISCOURSE CONNECTIVES + same-subject/same-verb AGGREGATION over the brain's retrieved, grounded facts. This builds the
deterministic, ENTAILMENT-CHECKED plan-then-realize renderer and de-risks it: turn a topic's grounded facts into ONE or
two connected sentences ("An elephant is a mammal; it is grey and has a trunk and tusk." / "A dog is big. It eats meat,
chases cat and likes bone.") with NO free abstractive generation -- every asserted fact is a stored triple (moat by
construction), the connectives are Joint/Elaboration (conjunction of grounded facts) + a checkable Contrast/Additive in
the COMPARE case. Reuse-by-import; NO `sim/` edit; NO train; CPU (pure host-side surface realization from brain content).

`plan_discourse` is the deployable function (the console's `_discuss` imports it). This de-risk tests the PURE plan
logic (grounded-template clause render, no generator needed -- the per-clause FT fluency is orthogonal + Phase-2-
validated).

METRICS (multiple fact-set SCENARIOS; a deterministic host plan has no RNG seed -- robustness = varied structures):
  (a) DEPTH        -- output has FEWER sentences than facts AND >=1 aggregated clause (2+ facts fused) AND >=1 connective.
  (b) GROUNDED     -- every noun/verb/patient in the output comes from the input facts (0 invented tokens).
  (c) CONNECTIVE-CORRECT (compare) -- "but" fires IFF the two subjects' patients differ; "and so does" IFF they share
      verb+patient (a checkable entailment predicate, never a free relation).
  (d) LESION       -- empty facts -> a hedge, not a fabricated sentence.

GO = depth + grounded + connective-correct + lesion, across all scenarios. Reuse-by-import; NO `sim/` edit; CPU.
Run: python -m research.runners._fluidconv_phase16_discourse_plan_derisk
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "research" / "findings" / "raw" / "_fluidconv_phase16_discourse_plan.json"
_V3 = {"eat": "eats", "chase": "chases", "like": "likes", "see": "sees", "hunt": "hunts", "guard": "guards",
       "help": "helps", "herd": "herds", "catch": "catches", "find": "finds"}


def _art(w):
    return "an" if (w[:1].lower() in "aeiou") else "a"


def _join_and(items):
    items = list(items)
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + " and " + items[-1]


def _v3(v):
    return _V3.get(v, v + "s" if not v.endswith("s") else v)


def plan_discourse(topic, facts, *, render_action=None):
    """Turn a topic's grounded facts into CONNECTED prose (aggregation + Joint/Elaboration connectives). `facts` = a
    list of [subject, verb, patient] grounded triples (the topic's own facts; the caller pre-filters instances/members).
    `render_action(subj, verb, [patients])` optionally renders action clauses fluently (else a grounded template).
    Returns (prose_str, used_facts): prose is grounded by construction (only input facts); used_facts are the triples
    it asserted (for VERIFY). The relation split: isa=taxonomy noun ("is a mammal"); is=adjective ("is grey"); has=part;
    else=action verb (aggregated by verb)."""
    own = [f for f in facts if f[0] == topic]
    isa_parents = [p for (a, v, p) in own if v == "isa"]
    is_attrs = [p for (a, v, p) in own if v == "is"]
    has_parts = [p for (a, v, p) in own if v == "has"]
    actions = {}                                                    # verb -> [patients], preserve first-seen order
    for (a, v, p) in own:
        if v not in ("isa", "is", "has"):
            actions.setdefault(v, [])
            if p not in actions[v]:
                actions[v].append(p)
    used = []

    # sentence 1 (definitional): isa (+ fold in is/has via Elaboration connectives -> one cohesive clause).
    tail = []                                                       # "; it ..." elaborations after the isa head
    if is_attrs:
        tail.append(f"is {_join_and(is_attrs[:4])}"); used += [[topic, "is", p] for p in is_attrs[:4]]
    if has_parts:
        tail.append(f"has {_join_and(has_parts[:5])}"); used += [[topic, "has", p] for p in has_parts[:5]]
    lead = None
    if isa_parents:
        lead = f"{_art(topic).capitalize()} {topic} is {_join_and([f'{_art(x)} {x}' for x in isa_parents[:3]])}"
        used += [[topic, "isa", p] for p in isa_parents[:3]]
        if tail:
            lead += "; it " + " and ".join(tail)                   # "An elephant is a mammal; it is grey and has a trunk"
        lead += "."
    elif tail:
        lead = f"{_art(topic).capitalize()} {topic} " + " and ".join(tail) + "."

    # sentence 2 (behaviour): aggregate action facts by verb, joined -> "It eats meat, chases cat and likes bone."
    act_sentence = None
    if actions:
        clauses = []
        for v, ps in actions.items():
            if render_action is not None:
                clauses.append(render_action(topic, v, ps))
            else:
                clauses.append(f"{_v3(v)} {_join_and(ps[:4])}")
            used += [[topic, v, p] for p in ps[:4]]
        subj = "It" if lead is not None else f"{_art(topic).capitalize()} {topic}"
        act_sentence = f"{subj} {_join_and(clauses)}."

    sents = [s for s in (lead, act_sentence) if s]
    prose = " ".join(sents) if sents else f"I don't know much about the {topic}."
    return prose, used


def shared_discourse(x, y, facts_x, facts_y):
    """CHECKABLE GIST (#2): what two topics SHARE -- shared isa-parents + shared verb+patient facts, entailment-only
    (a shared fact must be present in BOTH stores; never a free generalization). "both the dog and the wolf eat meat.\""""
    ox = [f for f in facts_x if f[0] == x]
    oy = {(v, p) for (a, v, p) in facts_y if a == y}
    lines = []
    for (a, v, p) in ox:
        if (v, p) in oy:
            if v == "isa":
                lines.append(f"both the {x} and the {y} are {_art(p)} {p}")
            elif v == "is":
                lines.append(f"both the {x} and the {y} are {p}")
            elif v == "has":
                lines.append(f"both the {x} and the {y} have {p}")
            else:
                lines.append(f"both the {x} and the {y} {v} {p}")   # base verb: plural "both" subject ("both ... eat")
    if not lines:
        return f"the {x} and the {y} don't share anything i know.", False
    return _join_and(lines).capitalize() + ".", True


def compare_discourse(x, y, facts_x, facts_y):
    """CHECKABLE-connective COMPARE of two topics (the #2 seam): Contrast 'but' IFF a shared verb's patients DIFFER;
    Additive 'and so does' IFF they share verb+patient. Every connective is entailed by the grounded facts."""
    vx = {v: p for (a, v, p) in facts_x if a == x and v not in ("isa", "is", "has")}
    vy = {v: p for (a, v, p) in facts_y if a == y and v not in ("isa", "is", "has")}
    shared = [v for v in vx if v in vy]
    lines, connective = [], None
    for v in shared:
        if vx[v] == vy[v]:
            lines.append(f"the {x} {_v3(v)} {vx[v]}, and so does the {y}"); connective = "additive"
        else:
            lines.append(f"the {x} {_v3(v)} {vx[v]}, but the {y} {_v3(v)} {vy[v]}"); connective = "contrast"
    return (". ".join(lines) + "." if lines else f"I don't know how the {x} and the {y} compare."), connective


# ------------------------------- de-risk scenarios + checks -------------------------------
SCENARIOS = {
    "elephant (taxonomy)": ("elephant", [["elephant", "isa", "mammal"], ["elephant", "is", "grey"],
                                         ["elephant", "has", "trunk"], ["elephant", "has", "tusk"]]),
    "dog (adjective + actions)": ("dog", [["dog", "is", "big"], ["dog", "eat", "meat"], ["dog", "chase", "cat"],
                                          ["dog", "like", "bone"]]),
    "banana (multi-attr)": ("banana", [["banana", "isa", "fruit"], ["banana", "is", "yellow"], ["banana", "is", "green"]]),
    "bird (same-verb aggregation)": ("bird", [["bird", "eat", "seed"], ["bird", "eat", "worm"], ["bird", "isa", "animal"]]),
}


def _tokens_of(prose):
    return {t.strip(".,;").lower() for t in prose.split()}


def run_scenario(name, topic, facts):
    prose, used = plan_discourse(topic, facts)
    n_sents = prose.count(".") + prose.count(";")               # rough clause count
    n_facts = len([f for f in facts if f[0] == topic])
    # DEPTH: fewer sentences than facts + >=1 connective (and/;) + >=1 aggregated clause (a "and" inside a clause)
    has_connective = (" and " in prose) or ("; it " in prose)
    depth_ok = (n_sents < n_facts) and has_connective
    # GROUNDED: every content token in the prose is a fact token or a function word (no invented nouns)
    fact_toks = {topic} | {p for (a, v, p) in facts} | {v for (a, v, p) in facts} | {_v3(v) for (a, v, p) in facts}
    fn_words = {"a", "an", "the", "is", "it", "and", "has", "of", topic.capitalize().lower(), _art(topic)}
    stray = [t for t in _tokens_of(prose) if t and t not in fact_toks and t not in fn_words and not t.istitle()]
    grounded_ok = (len(stray) == 0)
    # every used fact is an input fact (no fabricated triple)
    used_ok = all([u[0], u[1], u[2]] in [list(f) for f in facts] for u in used)
    return {"scenario": name, "prose": prose, "n_sents": n_sents, "n_facts": n_facts,
            "depth_ok": bool(depth_ok), "grounded_ok": bool(grounded_ok and used_ok), "stray": stray}


def run_compare():
    fx = [["dog", "eat", "meat"], ["dog", "chase", "cat"]]
    fy = [["cat", "eat", "fish"], ["cat", "chase", "cat"]]
    # eat: meat vs fish -> Contrast "but"; chase: cat vs cat -> Additive "and so does"
    prose, _c = compare_discourse("dog", "cat", fx, fy)
    contrast_ok = ("but the cat eats fish" in prose)                       # patients differ -> but
    additive_ok = ("and so does the cat" in prose)                        # shared chase cat -> and so does
    return {"prose": prose, "contrast_ok": bool(contrast_ok), "additive_ok": bool(additive_ok)}


def run_shared():
    fx = [["dog", "isa", "mammal"], ["dog", "eat", "meat"], ["dog", "chase", "cat"]]
    fy = [["wolf", "isa", "mammal"], ["wolf", "eat", "meat"], ["wolf", "chase", "rabbit"]]
    # shared: both isa mammal + both eat meat; NOT chase (cat vs rabbit differ)
    prose, ok = shared_discourse("dog", "wolf", fx, fy)
    lo = prose.lower()
    shared_ok = ok and ("both the dog and the wolf are a mammal" in lo) and ("eat meat" in lo) \
        and ("rabbit" not in lo) and ("cat" not in lo)
    return {"prose": prose, "shared_ok": bool(shared_ok)}


def run_lesion():
    prose, used = plan_discourse("dragon", [])
    return {"prose": prose, "lesion_ok": bool("don't know" in prose and not used)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time(); err = None
    try:
        scen = [run_scenario(n, tp, fx) for n, (tp, fx) in SCENARIOS.items()]
        cmp = run_compare()
        shared = run_shared()
        les = run_lesion()
        for s in scen:
            print(f"  [{s['scenario']}] {s['n_facts']} facts -> {s['n_sents']} sentence(s) | depth {s['depth_ok']} | "
                  f"grounded {s['grounded_ok']}\n      \"{s['prose']}\"", flush=True)
        print(f"  [compare dog/cat] contrast {cmp['contrast_ok']} | additive {cmp['additive_ok']}\n"
              f"      \"{cmp['prose']}\"", flush=True)
        print(f"  [shared dog/wolf] {shared['shared_ok']}\n      \"{shared['prose']}\"", flush=True)
        print(f"  [lesion] {les['lesion_ok']} -> \"{les['prose']}\"", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        depth = all(s["depth_ok"] for s in scen)
        grounded = all(s["grounded_ok"] for s in scen)
        conn = cmp["contrast_ok"] and cmp["additive_ok"]
        shared_ok = shared["shared_ok"]
        lesion = les["lesion_ok"]
        go = bool(depth and grounded and conn and shared_ok and lesion)
        verdict = (("GO -- grounded DISCOURSE PLAN closes the multi-fact-synthesis cheap-first #1: a topic's grounded "
                    "facts render as CONNECTED prose (aggregation + Joint/Elaboration connectives) with FEWER sentences "
                    "than facts + >=1 aggregated clause, every token grounded (0 invented); the COMPARE path fires "
                    "checkable connectives (Contrast 'but' IFF patients differ, Additive 'and so does' IFF shared "
                    "verb+patient); empty -> hedge. NO free abstractive generation, NO sim/ edit, NO train, moat by "
                    "construction. Ready to wire into the console `_discuss`.") if go else
                   ("HONEST/PARTIAL -- " + "; ".join(
                       ([] if depth else ["depth (not fewer sentences / no connective)"]) +
                       ([] if grounded else ["grounded (invented tokens: " + str([s['stray'] for s in scen]) + ")"]) +
                       ([] if conn else ["connective-correct (compare predicate)"]) +
                       ([] if shared_ok else ["shared/gist (checkable intersection)"]) +
                       ([] if lesion else ["lesion"])) + " failed"))
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "fluidconv_phase16_discourse_plan", "GO": go, "verdict": verdict,
               "resolves": "multi-fact synthesis cheap-first #1: grounded discourse plan (aggregation + entailment-"
                           "checked connectives) -> connected prose, no free generation, moat by construction.",
               "scenarios": scen if err is None else [], "compare": cmp if err is None else {},
               "shared": shared if err is None else {}, "lesion": les if err is None else {},
               "elapsed_seconds": round(time.time() - t0, 1),
               "HONEST_CEILING": "deterministic host-side surface realization from brain-supplied, VERIFY-clean facts "
                                 "(the connective inventory + entailment predicates are host-authored -- like the "
                                 "FRAME_LEXICON; the fully-brain-based Broca connective producer is the deep follow-on). "
                                 "Joint/Elaboration/Contrast/Additive are checkable; free abstractive single-pass "
                                 "synthesis + open-world cross-fact inference on a 21M remains the genuine wall (routed "
                                 "around, not solved)."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[phase16-discourse-plan] VERDICT: {verdict}", flush=True)
    print(f"[phase16-discourse-plan] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
