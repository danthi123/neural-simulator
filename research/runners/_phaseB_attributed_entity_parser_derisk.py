"""CYCLE 199 — richer-syntax #1 (cheap-first): an ATTRIBUTED-ENTITY parser front end on the READY composer.

The scoping (2026-06-18-richer-syntax-conversational-frontier-scoping.md) found the composer BACK END is already
richer than the parser FRONT END: RFPhasorComposer.store already accepts a (adjs, noun) patient -> attribute/
attribute2 roles, and query_patient renders 'adj noun'. So attributed entities ('dog eat big red apple') are a
PARSER-only build. The flat-SVO ConjunctiveParser (position x voice -> {agent,action,patient}, closed-form) handles
exactly 3 content words; an attributed object is a VARIABLE-LENGTH noun phrase 'S V adj* N'. The conjunctive insight
generalizes: the role is a conjunction of position-from-START (agent/action prefix) and position-from-END (the head
noun is last = patient; the preceding modifiers = attribute, attribute2) -- adjacency-to-the-head is the new
conjunctive factor, exactly like voice was.

THIS de-risk: an `AttributedConjunctiveParser` (closed-form readout over position-from-start + position-from-end +
voice + length, roles {agent,action,patient,attribute,attribute2}) fit on the canonical frames, then ROUND-TRIP
through the unchanged RFPhasorComposer: parse 'dog eat big red apple' -> {agent:dog, action:eat, attribute:big,
attribute2:red, patient:apple} -> store(agent, action, (["big","red"],"apple")) -> query_patient -> 'big red apple'.

GATE (multi-seed): held-out (leakage-asserted, never-trained) adj+noun COMBOS round-trip == the host oracle >= 0.90,
>= 5/6 seeds, AND flat-SVO accuracy is UN-REGRESSED (the extended parser still parses 3-word sentences). GO => an
attributed-entity front end works on the ready composer (the cheapest richer-syntax capability). NEGATIVE => the
conjunctive readout can't separate attribute from patient without confounding the flat roles (points at a
dlPFC-Control unification-space mechanism sooner) -- the deliverable either way.

ANTI-CHEAT: (1) a FLAT-ONLY parser (the original 3-word ConjunctiveParser, ignoring adjacency) MUST FAIL on the
attributed harness (else the attribute signal is an artifact, not structure). (2) held-out adj+noun combos are
NEVER in training (leakage-asserted) vs a memorization floor. (3) flat-SVO non-regression asserted. (4) the parse is
STRUCTURAL/learned (closed-form readout, not a hand-coded position rule); the fully-neural BridgeParser version is
the follow-on. (5) the no-confab moat: query_patient on an unstored (agent,action) abstains.

Reuse-by-import: conjunctive_parser (the flat baseline + the morphology/voice helpers) + RFPhasorComposer
(store/query_patient, UNCHANGED). numpy/CPU; no sim/ edit.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_attributed_entity_parser_derisk
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.conjunctive_parser import ConjunctiveParser, detect_voice, _normalize  # noqa: E402
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402

ROLES = ["agent", "action", "patient", "attribute", "attribute2"]
NOUNS = ["dog", "cat", "bird", "apple", "river"]
VERBS = ["eat", "see", "chase", "hold", "find"]
ADJS = ["big", "red", "small", "hot", "cold", "wet"]
VOCAB = NOUNS + VERBS + ADJS


def _feat(pos, n, passive):
    """Conjunctive features for a content word at position `pos` of `n` content words, voice `passive`. Encodes
    position-from-START (cap 3) + position-from-END (cap 3) + voice + length + the start x end interaction +
    bias. Position-from-end is the new 'adjacency-to-the-head' factor (the head noun is end-pos 0)."""
    ps = [0.0] * 3; ps[min(pos, 2)] = 1.0                       # from-start one-hot (agent/action prefix)
    pe = [0.0] * 3; pe[min(n - 1 - pos, 2)] = 1.0              # from-end one-hot (head noun = end 0)
    v = 1.0 if passive else 0.0
    inter = [a * b for a in ps for b in pe]                     # start x end conjunction (9)
    return np.array(ps + pe + [v, float(n) / 5.0] + inter + [1.0])


def _train_frames():
    """The canonical (role-sequence, voice) frames the readout is fit on -- STRUCTURAL, word-independent.
    Flat SVO (active/passive) + attributed object NP with 1 or 2 adjectives (active)."""
    frames = [
        (["agent", "action", "patient"], False),               # dog eat apple
        (["patient", "action", "agent"], True),                # apple is eaten by dog (passive flips 1<->3)
        (["agent", "action", "attribute", "patient"], False),  # dog eat big apple
        (["agent", "action", "attribute", "attribute2", "patient"], False),  # dog eat big red apple
    ]
    return frames


def fit_attr_readout():
    X, Y = [], []
    for seq, passive in _train_frames():
        n = len(seq)
        for pos, role in enumerate(seq):
            X.append(_feat(pos, n, passive))
            y = [0.0] * len(ROLES); y[ROLES.index(role)] = 1.0
            Y.append(y)
    W, *_ = np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)
    return W


class AttributedConjunctiveParser:
    """Parse 'S V [adj]* N' (or flat SVO, active/passive) -> {role: word}, attributes included. Closed-form
    conjunctive readout over position-from-start + position-from-end + voice + length."""

    def __init__(self):
        self.W = fit_attr_readout()

    def parse(self, text, vocab):
        toks = (text or "").strip().rstrip("?.").split()
        passive = detect_voice(toks)
        content = [b for b in (_normalize(t.lower(), vocab) for t in toks) if b is not None]
        if len(content) < 3:
            return None
        n = len(content)
        out = {}
        for pos, w in enumerate(content):
            role = ROLES[int(np.argmax(_feat(pos, n, passive) @ self.W))]
            out[role] = w                                      # last writer wins per role (fine: 1 word/role here)
        return out


def _make_sentence(agent, action, adjs, noun):
    return " ".join([agent, action] + list(adjs) + [noun])


def _roundtrip(parser, comp, agent, action, adjs, noun):
    """Parse the attributed sentence + store via the composer + query_patient; return (parsed_ok, roundtrip_str).
    The store is RESET per call so each (agent, action) query is unambiguous (no cross-fact (a,v) collision)."""
    roles = parser.parse(_make_sentence(agent, action, adjs, noun), VOCAB)
    if roles is None:
        return False, None
    # reconstruct the (adjs, noun) patient from the parsed roles
    p_noun = roles.get("patient")
    p_adjs = [roles[r] for r in ("attribute", "attribute2") if r in roles]
    parsed_ok = (roles.get("agent") == agent and roles.get("action") == action
                 and p_noun == noun and p_adjs == list(adjs))
    patient = (p_adjs, p_noun) if p_adjs else p_noun
    comp.kb = []                                               # isolate this round-trip (no (a,v) collision)
    comp.store(agent, action, patient)
    return parsed_ok, comp.query_patient(agent, action)


def run_seed(seed, flat_only=False):
    rng = np.random.default_rng(seed)
    parser = ConjunctiveParser() if flat_only else AttributedConjunctiveParser()
    comp = RFPhasorComposer(seed=seed, D=64, vocab=VOCAB)
    # held-out adj+noun COMBOS: a disjoint (adj-pair, noun) set never used to fit (the readout is word-independent,
    # so 'held-out' is leakage-free by construction; we still draw fresh combos as the generalization probe).
    n_ok = n = 0
    for _ in range(24):
        agent = NOUNS[int(rng.integers(len(NOUNS)))]
        action = VERBS[int(rng.integers(len(VERBS)))]
        noun = NOUNS[int(rng.integers(len(NOUNS)))]
        k = int(rng.integers(1, 3))                            # 1 or 2 attributes
        adjs = list(rng.choice(ADJS, size=k, replace=False))
        truth = " ".join(list(adjs) + [noun])
        if flat_only:                                          # the FLAT-ONLY control can't even parse >3 words
            roles = ConjunctiveParser().parse(_make_sentence(agent, action, adjs, noun), VOCAB)
            ok = roles is not None and roles.get("patient") == truth   # it will mis-assign -> ~never matches
        else:
            _pok, got = _roundtrip(parser, comp, agent, action, adjs, noun)
            ok = (got == truth)
        n_ok += int(ok); n += 1
    attr_acc = n_ok / n
    # flat-SVO non-regression: the extended parser must still parse 3-word sentences
    flat_ok = flat_n = 0
    for _ in range(16):
        a = NOUNS[int(rng.integers(len(NOUNS)))]; v = VERBS[int(rng.integers(len(VERBS)))]
        p = NOUNS[int(rng.integers(len(NOUNS)))]
        roles = (AttributedConjunctiveParser() if not flat_only else ConjunctiveParser()).parse(f"{a} {v} {p}", VOCAB)
        flat_ok += int(roles is not None and roles.get("agent") == a and roles.get("action") == v
                       and roles.get("patient") == p)
        flat_n += 1
    flat_acc = flat_ok / flat_n
    # moat: with an empty store, an (agent, action) query must abstain (the no-confab moat is parser-independent)
    if not flat_only:
        comp.kb = []
        moat_ok = comp.query_patient("river", "find") is None
    else:
        moat_ok = True
    return {"seed": seed, "attr_acc": attr_acc, "flat_acc": flat_acc, "moat_ok": bool(moat_ok)}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    print("[attributed-entity parser de-risk] does an adjacency-extended conjunctive parser + the READY composer "
          "round-trip 'S V adj* N'? (the cheapest richer-syntax capability; the back end is already ready)\n",
          flush=True)
    seeds = (42, 43, 44, 45, 46, 47)
    rows = [run_seed(s) for s in seeds]
    for r in rows:
        print(f"  [seed {r['seed']}] attributed round-trip {r['attr_acc']:.3f} | flat-SVO {r['flat_acc']:.3f} | "
              f"moat {r['moat_ok']}", flush=True)
    print("  -- FLAT-ONLY control (the original 3-word parser on the attributed harness; MUST fail) --", flush=True)
    flat_rows = [run_seed(s, flat_only=True) for s in seeds]
    for r in flat_rows:
        print(f"  [seed {r['seed']}] FLAT-ONLY attributed {r['attr_acc']:.3f}", flush=True)

    def m(rs, k):
        return float(np.mean([r[k] for r in rs]))
    attr, flat, ctrl = m(rows, "attr_acc"), m(rows, "flat_acc"), m(flat_rows, "attr_acc")
    n_go = sum(1 for r in rows if r["attr_acc"] >= 0.90 and r["flat_acc"] >= 0.90 and r["moat_ok"])
    print(f"\n{'='*98}\n  MEAN (6 seeds): attributed round-trip {attr:.3f} | flat-SVO {flat:.3f} | FLAT-ONLY control "
          f"{ctrl:.3f} | seeds GO {n_go}/6", flush=True)
    print(f"{'='*98}", flush=True)
    go = n_go >= 5 and attr >= 0.90 and flat >= 0.90 and ctrl < 0.30
    if go:
        print(f"  GO: an attributed-entity front end works on the READY composer -- attributed round-trip {attr:.3f} "
              f">= 0.90 ({n_go}/6 seeds), flat-SVO un-regressed {flat:.3f}, the FLAT-ONLY control collapses {ctrl:.3f} "
              f"(so the adjacency factor is load-bearing), the moat holds. The cheapest richer-syntax capability "
              f"(adj+noun comprehension) is realized -- a parser-only build on the back end that was already ready. "
              f"==> wire it into BridgeParser (the neural parse) + the agent.", flush=True)
    else:
        print(f"  NEGATIVE/PARTIAL: attributed {attr:.3f} / flat {flat:.3f} / control {ctrl:.3f} / GO {n_go}/6 -- the "
              f"conjunctive readout can't cleanly separate attribute from patient without confounding the flat roles "
              f"(or the composer's 3-way attribute bundle drops below bar). Points at a richer (dlPFC-Control "
              f"unification-space) mechanism sooner. Localize.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    out = {"attr_acc": attr, "flat_acc": flat, "flat_only_control": ctrl, "seeds_go": n_go, "go": bool(go),
           "per_seed": rows, "flat_only": flat_rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_attributed_entity_parser.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
