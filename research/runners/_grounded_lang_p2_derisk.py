"""Cheap-first DE-RISK P2 for the grounded-language faculty arc (scoping
`research/findings/2026-06-22-grounded-language-faculty-scoping.md` §4 Rank-1).

THE DE-RISK: a structured curriculum (authored OFFLINE by the controller at
`research/findings/raw/_grounded_lang_curriculum_p2.json`) is ingested by the BRAIN's validated
pipeline (parser -> composer store) through a normal `BrainConversationalAgent`, then tested for
structured RECALL + the no-confab MOAT. This validates the WHOLE knowledge+grounding half (P2 + P3's
gate) with ZERO model download / ZERO convert / ZERO new GPU mechanism. NO `sim/` edit.

Metrics per seed:
  (1) structured RECALL -- every `queries_recall` returns its `expect` (patient/agent/yes-no);
      recall = correct/total (~1.0 bar on the taught set).
  (2) the no-confab MOAT -- every `queries_moat` (untaught cue) returns None/"unknown"; the BAR is
      0 FALSE-ACCEPTS (an untaught cue returning a confident wrong answer = a breach).
  (3) the 2-hop CHAIN via reason_chain -> the chained answer ('mouse').

GO = recall ~= 1.0 on taught facts AND moat 0-false-accept on untaught cues, >=3 seeds (chain a bonus).

CURRICULUM-FORMAT decisions (the brain's validated machinery dictates these; documented in the JSON
output so the controller can refine the curriculum):
  - SVO `facts` -> agent.hear("a v p")  (the parser -> composer store path).
  - `attribute_facts` [noun, adj] -> stored as the SVO triple (noun, "is", adj), i.e. agent=noun,
    action="is", patient=adj. This makes the yes-no attribute moat ("apple is blue" -> unknown,
    since only "apple is red" was stored) work on the SAME store/query path. (The composer's native
    (adjs, noun) attribute-tuple binding is the alternative; the curriculum's only attribute query is
    the yes-no moat, which the (noun,"is",adj) SVO answers most directly.)
  - `clause_facts` [agent, action, [s,v,o]] -> agent.hear_clause_fact(agent, action, Clause(s,v,o))
    (the validated recursive-clause path; patient is an embedded clause).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

# the rf composer's Clause is what the agent imports + the composer duck-types on (agent,action,patient)
from research.runners.core_sim_composition import Clause
from research.runners.brain_conversational_agent import BrainConversationalAgent

CURRICULUM = os.path.join(os.path.dirname(__file__), "..", "findings", "raw",
                          "_grounded_lang_curriculum_p2.json")
OUT = os.path.join(os.path.dirname(__file__), "..", "findings", "raw", "_grounded_lang_p2_derisk.json")


def _collect_vocab(cur):
    """Every word that appears in any fact OR any query cue/answer -- the composer needs each in its
    vocab so a moat cue (lion/whale/drink/fly/plane/is/blue) can be ENCODED (else self.concepts[w]
    KeyErrors). The moat is then STRUCTURAL: an encodable-but-never-stored cue matches no fact -> abstain."""
    words = set()
    for f in cur.get("facts", []):
        words.update(f)
    for noun, adj in cur.get("attribute_facts", []):
        words.update([noun, "is", adj])
    for cf in cur.get("clause_facts", []):
        agent, action, clause = cf
        words.update([agent, action])
        words.update(clause)
    for ch in cur.get("chains", []):
        for f in ch.get("facts", []):
            words.update(f)
        words.update(ch.get("query", []))
        if isinstance(ch.get("expect_2hop"), str):
            words.add(ch["expect_2hop"])
    for q in cur.get("queries_recall", []) + cur.get("queries_moat", []):
        words.update(w for w in q.get("cue", []) if isinstance(w, str))
        if isinstance(q.get("expect"), str):
            words.add(q["expect"])
    # drop the non-word sentinels that appear as `expect` values
    words.discard("yes"); words.discard("no"); words.discard("no_or_unknown")
    return sorted(words)


def _teach(agent, cur):
    """Ingest the curriculum through the brain's validated comprehend+store pipeline.

    Every curriculum fact is an ASSERTION, so it is stored with the bound AFFIRM polarity tag
    (polarity='AFFIRM'). This is the VALIDATED yes-no pattern (test_brain_conversational_agent.py:40,
    test_rf_phasor_composer.py:35) -- a fact stored with polarity=None binds NO polarity tag, so
    ask_yes_no's unbind reads an unbound role -> a seed-fragile AFFIRM/NEGATE coin-flip (the seed-43
    'dog eat meat' -> 'no' miss). Binding AFFIRM makes 'is it true?' deterministically 'yes', and the
    moat is unaffected (an untaught SVO still matches no fact -> 'unknown')."""
    taught = {"facts": 0, "attribute_facts": 0, "clause_facts": 0, "chain_facts": 0}
    for a, v, p in cur.get("facts", []):
        agent.hear(f"{a} {v} {p}", polarity="AFFIRM")
        taught["facts"] += 1
    for noun, adj in cur.get("attribute_facts", []):
        agent.hear(f"{noun} is {adj}", polarity="AFFIRM")   # (noun, "is", adj) SVO -- see module docstring
        taught["attribute_facts"] += 1
    for ag, ac, clause in cur.get("clause_facts", []):
        agent.hear_clause_fact(ag, ac, Clause(*clause), polarity="AFFIRM")
        taught["clause_facts"] += 1
    # the chain facts may overlap with `facts`; store them so reason_chain has its trace either way
    for ch in cur.get("chains", []):
        for a, v, p in ch.get("facts", []):
            agent.hear(f"{a} {v} {p}", polarity="AFFIRM")
            taught["chain_facts"] += 1
    return taught


def _answer(agent, q):
    """Route one query through the agent's recall API and return its raw answer."""
    t = q["type"]; cue = q["cue"]
    if t == "patient":
        return agent.what_does(cue[0], cue[1])             # -> composer.query_patient
    if t == "agent":
        return agent.who_does(cue[0], cue[1])              # -> composer.query_agent
    if t == "yesno":
        return agent.is_it_true(cue[0], cue[1], cue[2])    # -> composer.ask_yes_no ('yes'/'no'/'unknown')
    raise ValueError(f"unknown query type {t!r}")


def _recall_ok(q, got):
    """Did the recall query return its expected answer?"""
    exp = q["expect"]
    if q["type"] == "yesno":
        if exp == "yes":
            return got == "yes"
        if exp in ("no", "no_or_unknown"):
            return got in ("no", "unknown")
        return got == exp
    return got == exp


def _moat_breach(q, got):
    """A MOAT breach = an untaught cue returns a CONFIDENT WRONG answer instead of abstaining.
    Abstention is None (patient/agent) or 'unknown' (yes-no). For a yes-no whose expect is
    'no_or_unknown', a 'no' is ALSO a correct (non-confabulating) answer -- only a confident 'yes'
    breaches. Any non-abstaining patient/agent string is a breach (the cue was never taught)."""
    exp = q.get("expect")
    if q["type"] == "yesno":
        # never-stored SVO -> must be 'unknown' OR 'no' (both decline to assert the false fact); 'yes' breaches
        return got == "yes"
    # patient/agent: expect is None -> any non-None word is a fabricated answer
    return got is not None


def run_seed(cur, seed, vocab):
    agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab}, composer_kind="rf")
    # NOTE: concepts={w: None} only sets the VOCAB (the agent extracts sorted(concepts.keys())); the rf
    # composer generates the actual phasor codes per seed. (grounded_codes would override; not used here.)
    taught = _teach(agent, cur)

    recall = []
    n_recall_ok = 0
    for q in cur.get("queries_recall", []):
        got = _answer(agent, q)
        ok = _recall_ok(q, got)
        n_recall_ok += int(ok)
        recall.append({"cue": q["cue"], "type": q["type"], "expect": q["expect"],
                       "got": got, "ok": ok})

    moat = []
    n_breach = 0
    for q in cur.get("queries_moat", []):
        got = _answer(agent, q)
        breach = _moat_breach(q, got)
        n_breach += int(breach)
        moat.append({"cue": q["cue"], "type": q["type"], "got": got,
                     "abstained": not breach, "note": q.get("note", "")})

    # 2-hop chain (bonus)
    chain_results = []
    for ch in cur.get("chains", []):
        cue = ch["query"][0]
        actions = [ch["query"][1]] * 2   # 2-hop: repeat the relation (dog -chase-> cat -chase-> mouse)
        got = agent.reason_chain(cue, actions)
        passed = (got == ch.get("expect_2hop"))
        chain_results.append({"desc": ch.get("desc", ""), "cue": cue, "actions": actions,
                              "expect": ch.get("expect_2hop"), "got": got, "ok": passed})

    n_recall = len(recall)
    return {
        "seed": seed,
        "taught": taught,
        "recall_correct": n_recall_ok,
        "recall_total": n_recall,
        "recall_rate": (n_recall_ok / n_recall) if n_recall else None,
        "moat_false_accepts": n_breach,
        "moat_total": len(moat),
        "chain_pass": all(c["ok"] for c in chain_results) if chain_results else None,
        "recall_detail": recall,
        "moat_detail": moat,
        "chain_detail": chain_results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    with open(os.path.abspath(CURRICULUM), "r", encoding="utf-8") as fh:
        cur = json.load(fh)
    vocab = _collect_vocab(cur)

    print(f"[p2-derisk] backend={os.environ.get('SIM_BACKEND', 'auto')} vocab={len(vocab)} words")
    print(f"[p2-derisk] vocab: {vocab}")

    per_seed = []
    t0 = time.time()
    for seed in args.seeds:
        ts = time.time()
        try:
            r = run_seed(cur, seed, vocab)
        except Exception as e:
            r = {"seed": seed, "error": repr(e), "traceback": traceback.format_exc()}
            print(f"[p2-derisk] seed {seed} ERROR: {e!r}")
            traceback.print_exc()
        per_seed.append(r)
        dt = time.time() - ts
        if "error" not in r:
            print(f"[p2-derisk] seed {seed}: recall {r['recall_correct']}/{r['recall_total']} "
                  f"(={r['recall_rate']:.3f})  moat false-accepts {r['moat_false_accepts']}/{r['moat_total']}  "
                  f"chain {'PASS' if r['chain_pass'] else 'FAIL'}  [{dt:.1f}s]")

    ok_seeds = [r for r in per_seed if "error" not in r]
    all_recall_perfect = bool(ok_seeds) and all(r["recall_rate"] == 1.0 for r in ok_seeds)
    all_moat_clean = bool(ok_seeds) and all(r["moat_false_accepts"] == 0 for r in ok_seeds)
    all_chain_pass = bool(ok_seeds) and all(r["chain_pass"] for r in ok_seeds)
    n_ge3 = len(ok_seeds) >= 3
    go = all_recall_perfect and all_moat_clean and n_ge3

    # min recall rate across seeds (honest partial-map number if not perfect)
    min_recall = min((r["recall_rate"] for r in ok_seeds), default=None)
    max_breach = max((r["moat_false_accepts"] for r in ok_seeds), default=None)

    verdict = (
        "GO -- recall ~=1.0 on taught facts AND moat 0-false-accept on untaught cues, >=3 seeds; "
        "the knowledge-teacher->brain->grounded-recall loop works end-to-end (P2 + P3's gate de-risked, NO LLM touched)."
        if go else
        "PARTIAL/NO-GO -- see per-seed recall+moat table (recall_rate / moat_false_accepts) for which queries miss + why."
    )

    summary = {
        "curriculum": os.path.relpath(os.path.abspath(CURRICULUM),
                                      os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))),
        "backend": os.environ.get("SIM_BACKEND", "auto"),
        "vocab_size": len(vocab),
        "vocab": vocab,
        "seeds": args.seeds,
        "n_seeds_ok": len(ok_seeds),
        "all_recall_perfect": all_recall_perfect,
        "min_recall_rate": min_recall,
        "all_moat_clean": all_moat_clean,
        "max_moat_false_accepts": max_breach,
        "all_chain_pass": all_chain_pass,
        "GO": go,
        "verdict": verdict,
        "elapsed_seconds": round(time.time() - t0, 1),
        "format_decisions": {
            "facts": "agent.hear('a v p') -- parser -> composer store",
            "attribute_facts": "[noun, adj] -> (noun, 'is', adj) SVO triple; yes-no moat ('apple is blue') tests it",
            "clause_facts": "[ag, ac, [s,v,o]] -> hear_clause_fact(ag, ac, Clause(s,v,o)) -- recursive-clause path",
        },
        "per_seed": per_seed,
    }

    out_path = os.path.abspath(args.out)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[p2-derisk] VERDICT: {verdict}")
    print(f"[p2-derisk] wrote {out_path}")
    return 0 if go else 0   # always 0: a partial map is a real result, not a runner failure


if __name__ == "__main__":
    sys.exit(main())
