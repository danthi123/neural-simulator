"""EMERGE-88 -- FUNCTIONAL INTEGRATION: the form->role RESERVOIR COMPREHENDS -> the COMPOSER stores + ANSWERS.

The whole EMERGE-78..87 arc de-risked the reservoir as a comprehension MECHANISM (learned form->role, uncontingent
non-local, spiking, on-substrate, recursion-capable, one-brain co-resident) -- but always scored it in isolation
(role-labeling accuracy vs baselines). This closes the loop: the reservoir's role output DRIVES the production
`RFPhasorComposer`. The reservoir parses a sentence's thematic roles from the closed-class configuration (content
ABSTRACTED, so it cannot memorize lexemes); those roles map to the composer's (agent, action, patient); the composer
binds/stores them; and the who/what turn + the no-confab moat run on the reservoir's OWN comprehension -- replacing
the hand-labeler / BridgeParser for role assignment.

This is a genuine CAPABILITY COMPOSITION (comprehension -> composition), not another isolated score: the two validated
mechanisms interact -- the reservoir understands, the composer stores + answers.

Cheap-first rung: the RATE reservoir (EMERGE-78) comprehends -> the spiking `RFPhasorComposer` stores/answers. The
spiking-reservoir swap (EMERGE-82's `OnBridgeLSM.final_state` has the identical signature) is the mechanical follow-on.
Reuse-by-import; NO `sim/` edit.

Anti-cheats: (1) held-out CONTENT (fresh transitive draws the read-out never saw); (2) the no-confab MOAT (an
(agent, action) never stored -> the composer abstains); (3) a COMPREHENSION-LESION (the reservoir's closed-class
identity replaced by a single generic token -> role-labeling collapses -> the stored facts are wrong -> recall
collapses) = the reservoir is load-bearing for the whole turn.

Run:  SIM_BACKEND=numpy python -m research.runners._emerge88_reservoir_comprehends_composer_answers_derisk \
          --seeds 42 43 44 100 101 102 --json research/findings/raw/_emerge88_reservoir_comprehends.json
"""
import argparse
import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

import research.runners._emerge62_discover_function_words_derisk as m62  # noqa: E402
from research.runners._emerge78_reservoir_form_to_role_derisk import (  # noqa: E402
    Encoder, Reservoir, _fit_slots, _content_pools, _ROLES, _gen, _TRAIN_KINDS, _N_TRAIN_PER_CONSTRUCTION,
)
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402

# the reservoir's thematic roles -> the composer's fact fields (a transitive SVO fills all three)
_ROLE2FIELD = {"AGENT": "agent", "PREDICATE": "action", "THEME": "patient"}

_D = 256
_N_TEST = 24


class ReservoirComprehender:
    """The EMERGE-78 reservoir + slot read-out as a COMPREHENSION front-end: comprehend(tokens) -> {agent, action,
    patient} by labeling each content (OPEN) position's thematic role from the whole-sentence final state (Dominey-
    Hinaut). Content is abstracted to the OPEN marker in the encoder, so the roles come from STRUCTURE, not lexemes;
    the surface content word at each labeled position is read back out to fill the fact."""

    def __init__(self, seed, discovered, res=None, enc=None):
        self.enc = enc or Encoder(discovered)
        self.res = res or Reservoir(self.enc.dim, seed=seed)
        self.Ws = None
        self.closed = set(self.enc.idx)

    def fit(self, train_sentences):
        self.Ws = _fit_slots(self.res, self.enc, train_sentences)
        return self

    def comprehend(self, tokens, lesion=False):
        """Parse a sentence into a fact dict via the reservoir's learned form->role map. `lesion=True` collapses the
        closed-class identity (the necessity control)."""
        f = np.concatenate([self.res.final_state(self.enc.encode(tokens, lesion=lesion)), [1.0]])
        content = [t for t, w in enumerate(tokens) if w not in self.closed]   # OPEN positions, left-to-right -> slots
        fact = {}
        for k, t in enumerate(content):
            if self.Ws is None or k not in self.Ws:
                continue
            role = _ROLES[int(np.argmax(f @ self.Ws[k]))]
            field = _ROLE2FIELD.get(role)
            if field is not None and field not in fact:
                fact[field] = tokens[t]
        return fact


def _build_test_facts(seed, subj, verb, obj, n=_N_TEST):
    """Distinct-(agent, action) transitive sentences with fresh CONTENT draws (held out from the read-out fit)."""
    trng = np.random.default_rng(seed * 733 + 11)
    facts, seen = [], set()
    guard = 0
    while len(facts) < n and guard < 5000:
        guard += 1
        s = str(trng.choice(subj)); vv = str(trng.choice(verb)); o = str(trng.choice(obj))
        v3 = vv + "s"
        if (s, v3) in seen:                       # distinct (agent, action) -> query is unambiguous
            continue
        seen.add((s, v3))
        facts.append((["the", s, v3, "the", o], s, v3, o))
    return facts, seen, trng


def _recall_over(composer, comprehender, test, lesion=False):
    """COMPREHEND each sentence -> STORE the parsed fact -> query_patient over all -> fraction recalling the true
    patient. A fresh composer so the lesion condition is isolated."""
    for toks, s, v3, o in test:
        fact = comprehender.comprehend(toks, lesion=lesion)
        if {"agent", "action", "patient"} <= set(fact):
            composer.store(fact["agent"], fact["action"], fact["patient"])
    hit = 0
    for toks, s, v3, o in test:
        hit += int(composer.query_patient(s, v3) == o)
    return hit / max(1, len(test))


def _derisk_one(seed):
    # setup: the SELF-DISCOVERED closed class + content pools (EMERGE-62/78), the reservoir + slot read-out fit
    stream = m62.build_stream(seed, n_sentences=6000)
    words, freq, cover, _c = m62.compute_stats(stream)
    discovered, *_ = m62.discover_closed_class(words, freq, cover)
    subj, verb, obj = _content_pools(discovered)
    rng = np.random.default_rng(seed * 101 + 5)

    comp = ReservoirComprehender(seed, discovered)
    comp.fit(_gen(_TRAIN_KINDS, _N_TRAIN_PER_CONSTRUCTION, rng, subj, verb, obj))

    v3 = [v + "s" for v in verb]
    vocab = sorted(set(subj) | set(v3) | set(obj))
    test, seen, trng = _build_test_facts(seed, subj, verb, obj)

    # PARSE accuracy: did the reservoir map each sentence to the right (agent, action, patient)?
    parse_hit = 0
    for toks, s, v3s, o in test:
        fact = comp.comprehend(toks)
        parse_hit += int(fact.get("agent") == s and fact.get("action") == v3s and fact.get("patient") == o)
    parse_acc = parse_hit / len(test)

    # THE INTEGRATION: reservoir comprehends -> composer stores -> who/what recall
    composer = RFPhasorComposer(seed=seed, D=_D, vocab=vocab)
    recall = _recall_over(composer, comp, test, lesion=False)

    # MOAT: (agent, action) pairs NEVER stored -> the composer abstains (None). A non-None = a false-accept (confab).
    stored_keys = {(s, v3s) for _t, s, v3s, _o in test}
    fa = tot = 0
    mguard = 0
    while tot < 40 and mguard < 4000:
        mguard += 1
        s = str(trng.choice(subj)); v3q = str(trng.choice(verb)) + "s"
        if (s, v3q) in stored_keys:
            continue
        tot += 1
        fa += int(composer.query_patient(s, v3q) is not None)
    moat_fa = fa / max(1, tot)

    # NECESSITY: lesion the reservoir's comprehension -> roles collapse -> wrong facts -> recall collapses.
    composer_l = RFPhasorComposer(seed=seed, D=_D, vocab=vocab)
    lesion_recall = _recall_over(composer_l, comp, test, lesion=True)

    return {
        "seed": seed, "n_discovered_closed": len(discovered), "n_test_facts": len(test),
        "parse_acc": parse_acc, "recall": recall, "moat_false_accept": moat_fa, "lesion_recall": lesion_recall,
    }


def _go(rows):
    def mean(k):
        return float(np.mean([r[k] for r in rows]))
    return {
        "n_seeds": len(rows),
        "parse_acc": mean("parse_acc"), "recall": mean("recall"),
        "moat_false_accept": mean("moat_false_accept"), "lesion_recall": mean("lesion_recall"),
        # GO: the reservoir's comprehension drives correct who/what answers (recall), it parses transitive sentences
        # (parse), the no-confab moat holds (0 false-accepts), and comprehension is load-bearing (lesion collapses).
        "go": (mean("parse_acc") >= 0.90 and mean("recall") >= 0.90
               and mean("moat_false_accept") <= 0.05 and mean("lesion_recall") <= 0.55),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    rows = []
    for s in args.seeds:
        d = _derisk_one(s)
        rows.append(d)
        print(f"[seed {s}] parse {d['parse_acc']:.3f} | recall {d['recall']:.3f} | "
              f"moat-FA {d['moat_false_accept']:.3f} | lesion-recall {d['lesion_recall']:.3f}", flush=True)

    agg = _go(rows)
    verdict = "GO" if agg["go"] else "NO-GO"
    print(f"\n[emerge88] VERDICT: {verdict} -- the form->role RESERVOIR COMPREHENDS and the COMPOSER ANSWERS "
          f"(parse {agg['parse_acc']:.3f}; who/what recall {agg['recall']:.3f}; no-confab moat "
          f"{agg['moat_false_accept']:.3f} false-accept; comprehension-lesion collapses recall to "
          f"{agg['lesion_recall']:.3f} -- the reservoir is load-bearing for the whole turn).", flush=True)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "agg": agg}, fh, indent=2)
        print(f"[emerge88] wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
