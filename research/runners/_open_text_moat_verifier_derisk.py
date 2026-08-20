"""LANE 4 DE-RISK (#99): can the no-confab moat cover FREE, multi-clause Qwen
text, not just the bounded single-SVO answer path?

THE PROBLEM. Today's production moat (webapp/server.py, reused across every
composer in research/runners/{unified_brain_bridge,routed_composer,
one_brain_composer}.py) is an ENTAILMENT CHECK over a single pre-selected
(agent, action[, patient]) triple: `composer.query_patient(agent, action)` ->
patient-or-None, `composer.ask_yes_no(agent, action, patient)` ->
yes/no/unknown. The caller already knows WHICH triple to check because the
question was itself SVO-shaped. Free Qwen prose is not: one paragraph mixes
several factual claims, none pre-selected, plus hedged opinion clauses that
are not factual assertions at all and must NOT be flagged. Before letting
Qwen generate open text in production, we need (a) a claim EXTRACTOR that
turns a free paragraph into a set of checkable (agent, action, patient)
triples plus an opinion/hedge flag per clause, and (b) an entailment check
per extracted claim reusing the SAME abstain-on-unknown semantics as
`query_patient` / `ask_yes_no` (see research/runners/routed_composer.py:
218-224, unified_brain_bridge.py:800-810) but run over EVERY extracted claim
instead of one caller-selected triple.

THIS FILE. A numpy/host-only de-risk of the CLAIM-LEVEL layer (extraction +
classification), reusing the entailment SEMANTICS of the existing moat
verbatim (unknown SVO -> abstain/ungrounded; store hit with matching
polarity -> grounded; store hit with a DIFFERENT patient or opposite
polarity -> ungrounded/contradicted -- the same "same-SVO opposite-polarity
-> REJECT" rule server.py already applies at :4488-4495). The store itself
is a plain dict standing in for the spiking composer, exactly as
`abstention_gate.gate` stands in for the spiking threshold check that
`grounded_decode` calls (sim/grounded_decode.py) -- the CLAIM here is about
whether extraction+classification is tractable at all, not about re-proving
entailment (already GO, 6-seed, production).

Run: python -m research.runners._open_text_moat_verifier_derisk
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# --------------------------------------------------------------------------
# 1. The brain fact store -- a plain dict standing in for the spiking
#    composer's SVO memory (same abstraction level as routed_composer's
#    per-shard dict-backed test doubles). Keys are (agent, action) so
#    lookup mirrors `query_patient(agent, action) -> patient`.
# --------------------------------------------------------------------------

AFFIRM, NEGATE = "AFFIRM", "NEGATE"


@dataclass
class FactStore:
    # (agent, action) -> (patient, polarity)
    facts: dict = field(default_factory=dict)

    def store(self, agent, action, patient, polarity=AFFIRM):
        self.facts[(agent, action)] = (patient, polarity)

    def query_patient(self, agent, action):
        """Mirrors composer.query_patient: patient-or-None, AFFIRM only
        (a stored NEGATE fact does not hand back a patient to assert)."""
        hit = self.facts.get((agent, action))
        if hit is None or hit[1] != AFFIRM:
            return None
        return hit[0]

    def ask_yes_no(self, agent, action, patient):
        """Mirrors composer.ask_yes_no: 'yes'/'no'/'unknown' via the bound
        polarity tag, patient-sensitive (same-SVO opposite-polarity is a
        distinct case from an unrelated unknown patient)."""
        hit = self.facts.get((agent, action))
        if hit is None:
            return "unknown"
        stored_patient, polarity = hit
        if stored_patient != patient:
            return "unknown"
        return "yes" if polarity == AFFIRM else "no"


# --------------------------------------------------------------------------
# 2. Claim extraction -- lightweight, no dependency, clause-splits a free
#    paragraph and pulls an (agent, action, patient) triple per clause when
#    a verb from the store's lexicon is present. This is the part the task
#    calls out as "hard" for open text; the de-risk's job is to measure how
#    far a cheap heuristic gets, not to ship a production parser.
# --------------------------------------------------------------------------

HEDGES = (
    "i think", "i believe", "i feel", "i guess", "i suspect", "i wonder",
    "in my opinion", "it seems", "seems like", "maybe", "perhaps",
    "possibly", "probably", "might", "could be", "may be", "i'm not sure",
    "im not sure", "not sure", "supposedly", "allegedly", "i'd guess",
    "id guess",
)

CONNECTIVES = re.compile(r"\b(and|but|because|so|although|while|which)\b|,")
STOPWORDS = {"the", "a", "an", "to", "that", "this", "these", "those"}


@dataclass
class Claim:
    text: str
    kind: str                 # "opinion" | "assertion" | "unparsed"
    agent: str = None
    action: str = None
    patient: str = None
    negated: bool = False     # clause carries "not" / "doesn't" etc.


def split_clauses(paragraph):
    sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    clauses = []
    for sent in sentences:
        sent = sent.strip().rstrip(".!?")
        if not sent:
            continue
        parts = [p.strip() for p in CONNECTIVES.split(sent) if p and p.strip()]
        # CONNECTIVES.split keeps the matched connective as its own token
        # (e.g. 'and') interleaved with the clause text; drop pure
        # connective tokens, keep the clause text pieces.
        conn_words = {"and", "but", "because", "so", "although", "while",
                      "which"}
        clauses.extend([p for p in parts if p.lower() not in conn_words])
    return clauses


def is_opinion(clause_lower):
    return any(h in clause_lower for h in HEDGES)


def _verb_forms(verb_lexicon):
    """Maps both the canonical (3rd-person) form and its bare stem to the
    canonical form, so 'bites' and 'bite' (as in 'does not bite') both
    resolve to the store's canonical action key. A real system would use a
    lemmatizer; this is the cheapest thing that covers the de-risk set."""
    forms = {}
    for v in verb_lexicon:
        forms[v] = v
        # candidate bare stems for a 3rd-person '-s'/'-es' form: strip 's',
        # and (for '...es') also try stripping 'es' and re-adding 'e'
        # ('bites' -> 'bite', not 'bit'). Cheap coverage, not a lemmatizer.
        if v.endswith("s"):
            forms.setdefault(v[:-1], v)
        if v.endswith("es"):
            forms.setdefault(v[:-2] + "e", v)
            forms.setdefault(v[:-2], v)
    return forms


def extract_svo(clause, verb_lexicon):
    """Very small heuristic extractor: find the first verb-lexicon match,
    treat the tokens before it (skipping determiners/pronouns-as-articles)
    as the agent, and the tokens after it (skipping determiners) as the
    patient. Detects a preceding 'not'/negator on the verb. Returns
    (agent, action, patient, negated) or None if no lexicon verb is found."""
    verb_forms = _verb_forms(verb_lexicon)
    words = re.findall(r"[a-zA-Z']+", clause.lower())
    words = [w for w in words if w not in STOPWORDS]
    if not words:
        return None
    negators = {"not", "doesn't", "does", "n't", "never"}
    for i, w in enumerate(words):
        if w in verb_forms:
            canonical_verb = verb_forms[w]
            agent_toks = [t for t in words[:i] if t not in negators]
            if not agent_toks:
                continue
            negated = any(t in negators for t in words[:i]) or (
                i + 1 < len(words) and words[i + 1] in negators)
            patient_start = i + 1
            if negated and patient_start < len(words) and \
                    words[patient_start] in negators:
                patient_start += 1
            patient_toks = [t for t in words[patient_start:]
                             if t not in negators]
            if not patient_toks:
                continue
            return agent_toks[-1], canonical_verb, patient_toks[0], negated
    return None


def extract_claims(paragraph, verb_lexicon):
    claims = []
    for clause in split_clauses(paragraph):
        lower = clause.lower()
        if is_opinion(lower):
            claims.append(Claim(text=clause, kind="opinion"))
            continue
        parsed = extract_svo(clause, verb_lexicon)
        if parsed is None:
            claims.append(Claim(text=clause, kind="unparsed"))
            continue
        agent, action, patient, negated = parsed
        claims.append(Claim(text=clause, kind="assertion", agent=agent,
                             action=action, patient=patient,
                             negated=negated))
    return claims


# --------------------------------------------------------------------------
# 3. Entailment classification -- reuses the STORE'S query_patient /
#    ask_yes_no semantics claim-by-claim (the moat's actual decision rule,
#    just applied N times instead of once).
# --------------------------------------------------------------------------

def classify_claim(claim: Claim, store: FactStore):
    """Returns one of 'grounded' | 'ungrounded' | 'opinion' | 'unparsed'."""
    if claim.kind in ("opinion", "unparsed"):
        return claim.kind
    verdict = store.ask_yes_no(claim.agent, claim.action, claim.patient)
    if claim.negated:
        # "the cat does not eat fish" asserts NEGATE(cat,eat,fish) as fact.
        return "grounded" if verdict == "no" else "ungrounded"
    return "grounded" if verdict == "yes" else "ungrounded"


# --------------------------------------------------------------------------
# 4. De-risk harness -- a synthetic paragraph set with hand-labeled ground
#    truth per clause (grounded / ungrounded / opinion), run through
#    extract+classify, scored for precision/recall of catching the
#    UNGROUNDED clauses (the confabulation-catch rate). 'unparsed' clauses
#    count as MISSED coverage, tracked separately (a real ungrounded claim
#    the extractor failed to even parse is a recall loss, not a free pass).
# --------------------------------------------------------------------------

def build_store():
    s = FactStore()
    s.store("dog", "chases", "cat")
    s.store("cat", "eats", "fish")
    s.store("bird", "flies", "sky")
    s.store("dog", "sees", "bone")
    s.store("cat", "sleeps", "mat")
    s.store("mouse", "hides", "hole")
    s.store("cat", "bites", "mouse", polarity=NEGATE)   # explicit stored NO
    return s


VERB_LEXICON = {"chases", "eats", "flies", "sees", "sleeps", "hides",
                "bites", "climbs", "barks", "swims"}

# Each case: (paragraph, [(clause_substring, gold_label), ...])
# gold_label in {grounded, ungrounded, opinion}. clause_substring is matched
# by simple `in` containment against the extractor's clause text so the
# harness doesn't depend on exact clause-splitting boundaries.
CASES = [
    (
        "The dog chases the cat, and the cat eats the fish.",
        [("dog chases the cat", "grounded"),
         ("cat eats the fish", "grounded")],
    ),
    (
        "I think the dog chases the mailman, but the bird flies the sky.",
        [("dog chases the mailman", "opinion"),
         ("bird flies the sky", "grounded")],
    ),
    (
        "The bird flies the ocean.",
        [("bird flies the ocean", "ungrounded")],
    ),
    (
        "The mouse eats the cheese, because it was hungry.",
        [("mouse eats the cheese", "ungrounded")],
    ),
    (
        "Maybe the cat chases the dog, and the dog sees the bone.",
        [("cat chases the dog", "opinion"),
         ("dog sees the bone", "grounded")],
    ),
    (
        "The dog sees the bone, but the bird eats the cat.",
        [("dog sees the bone", "grounded"),
         ("bird eats the cat", "ungrounded")],
    ),
    (
        "The cat sleeps the mat, and perhaps the mouse hides the barn.",
        [("cat sleeps the mat", "grounded"),
         ("mouse hides the barn", "opinion")],
    ),
    (
        "The mouse hides the hole, and the cat does not bite the mouse.",
        [("mouse hides the hole", "grounded"),
         ("cat does not bite the mouse", "grounded")],  # matches stored NEGATE
    ),
    (
        "The cat bites the mouse.",
        [("cat bites the mouse", "ungrounded")],   # store says NEGATE -> stated as fact is wrong
    ),
    (
        "I believe the mouse hides the hole, probably in the evening.",
        [("mouse hides the hole", "opinion")],
    ),
]


def run_case(paragraph, gold, store):
    claims = extract_claims(paragraph, VERB_LEXICON)
    predicted = [classify_claim(c, store) for c in claims]
    rows = []
    for text, label in gold:
        # find the claim whose text contains this gold substring
        match_idx = None
        for i, c in enumerate(claims):
            if text.lower() in c.text.lower():
                match_idx = i
                break
        if match_idx is None:
            rows.append({"gold_text": text, "gold": label,
                         "predicted": "NOT_EXTRACTED", "claim_text": None})
        else:
            rows.append({"gold_text": text, "gold": label,
                         "predicted": predicted[match_idx],
                         "claim_text": claims[match_idx].text})
    return rows, claims, predicted


def score(all_rows):
    """Precision/recall of catching UNGROUNDED claims. Positive class =
    predicted 'ungrounded'. NOT_EXTRACTED on a gold-ungrounded clause is a
    recall miss (extraction failure = moat failure, counted honestly)."""
    tp = fp = fn = tn = 0
    opinion_false_flags = 0   # opinion mis-flagged as ungrounded (bad: over-flagging honest hedges)
    grounded_false_flags = 0  # grounded mis-flagged as ungrounded (bad: blocks true statements)
    for r in all_rows:
        gold, pred = r["gold"], r["predicted"]
        pred_ungrounded = (pred == "ungrounded" or
                            (pred == "NOT_EXTRACTED" and gold == "ungrounded"))
        # NOT_EXTRACTED is scored as a MISS (not a catch) for recall purposes below;
        # we do not count it as a TP even for gold=ungrounded (see loop after).
        if gold == "ungrounded":
            if pred == "ungrounded":
                tp += 1
            else:
                fn += 1
        else:
            if pred == "ungrounded":
                fp += 1
                if gold == "opinion":
                    opinion_false_flags += 1
                elif gold == "grounded":
                    grounded_false_flags += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "opinion_false_flags": opinion_false_flags,
        "grounded_false_flags": grounded_false_flags,
    }


def main():
    store = build_store()
    all_rows = []
    per_case = []
    for paragraph, gold in CASES:
        rows, claims, predicted = run_case(paragraph, gold, store)
        all_rows.extend(rows)
        per_case.append({
            "paragraph": paragraph,
            "extracted_claims": [
                {"text": c.text, "kind": c.kind, "agent": c.agent,
                 "action": c.action, "patient": c.patient,
                 "negated": c.negated}
                for c in claims
            ],
            "predictions_vs_gold": rows,
        })
    result = score(all_rows)

    print("=== Open-text moat verifier de-risk (#99) ===")
    for case in per_case:
        print("\nParagraph:", case["paragraph"])
        for c in case["extracted_claims"]:
            print("  extracted:", c)
        for r in case["predictions_vs_gold"]:
            flag = "OK" if (
                (r["gold"] == "ungrounded") == (r["predicted"] == "ungrounded")
            ) else "MISS"
            print(f"  [{flag}] gold={r['gold']:<10} pred={r['predicted']:<14} "
                  f"claim={r['claim_text']!r}")

    print("\n=== SCORE (positive class = predicted 'ungrounded') ===")
    print(json.dumps(result, indent=2))
    print(f"\nconfabulation-catch precision={result['precision']:.3f} "
          f"recall={result['recall']:.3f}")
    print(f"opinion clauses wrongly flagged as ungrounded: "
          f"{result['opinion_false_flags']}")
    print(f"grounded clauses wrongly flagged as ungrounded: "
          f"{result['grounded_false_flags']}")

    out = {
        "cases": per_case,
        "score": result,
    }
    out_path = os.path.join(_REPO, "research", "findings", "raw",
                             "_open_text_moat_verifier_derisk.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")
    return result


if __name__ == "__main__":
    main()
