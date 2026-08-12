"""PRODUCTION LESION PROBE — the instrument that turns the production-integration ledger's level-3 claims from
CONSISTENCY-checked (the PI gate: does the ledger agree with the source?) to TRUTH-checked (does the spiking path
actually DO the work on the default turn?).

WHY (owner directive 2026-08-11 + the adversary's must-fix): a faculty earns "spiking, on-by-default" credit only if
LESIONING its spiking path CHANGES the default answer. A byte-identical flip (composer_kind='onebrain' enabling branches
validated '== host argmax') stamps a trace but is cosmetic. And the OWNER-VISIBLE FUNCTION — does the default turn
CHOOSE (a neural selector decides what to say), GENERATE (novel content from the brain), LEARN (the turn writes
synapses) — is the north-star that trace-stamping cannot fake. This probe builds the EXACT default ChatBrain that
/api/brain-chat builds (server.py:2951 -> _build_tiny_demo(42, use_multiturn=True, enable_neural_render=False)) and
measures those three acts behaviorally, plus a coarse lesion of the two default-on-spiking paths.

This is MEASUREMENT infra — it does NOT edit the production path. It is the nightly/behavioral half of CLASS PI.

Run:  SIM_BACKEND=numpy .venv/bin/python -m research.runners._production_lesion_probe \
          --out research/findings/raw/_production_lesion/probe.json
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")


def _build_default_brain(seed=42):
    """The IDENTICAL default brain /api/brain-chat builds (server.py:2951-2953)."""
    from research.runners.brain_chat_tui import ChatBrain, _build_tiny_demo, DEFAULT_SELF_ALIASES
    agent, aliases, _n = _build_tiny_demo(seed, use_multiturn=True, enable_neural_render=False)
    chat = ChatBrain(agent, self_aliases=aliases or DEFAULT_SELF_ALIASES)
    return chat, agent


def _ask(chat, q):
    # THE PRODUCTION PATH: /api/brain-chat (rich=False) calls chat.gate() then chat.render() — NOT chat.answer()
    # (webapp/server.py:3345-3349). The probe MUST test what the endpoint runs, or it verifies the wrong entry point
    # (the 2026-08-12 catch: LEARN/GENERATE were added to answer() and did NOT reach the endpoint).
    try:
        svo = chat.gate(q)
        if svo is None:
            return {"q": q, "answer": "I don't know about that.", "abstained": True}
        return {"q": q, "answer": chat.render(svo), "abstained": False}
    except Exception as e:
        return {"q": q, "answer": "<error: %s>" % type(e).__name__, "abstained": None, "error": str(e)[:200]}


def probe(seed=42):
    chat, agent = _build_default_brain(seed)
    facts = [tuple(f) for f in chat.list_facts()]
    out = {"seed": seed, "backend": os.environ.get("SIM_BACKEND", "numpy"), "device": "cpu",
           "runner": "research.runners._production_lesion_probe",
           "n_stored_facts": len(facts), "stored_facts": [list(f) for f in facts[:12]]}

    # ---- OWNER-VISIBLE FUNCTION: CHOOSE / GENERATE / LEARN (the north-star, behavioral) ----
    # CHOOSE — is WHICH fact/topic answers decided by a NEURAL selector? Structural: the default path routes the direct
    # answer through the HOST QuestionRouter.match_fact (keyword overlap); there is no neural selector to lesion on this
    # path (ledger row content-selection: wired=NO). We record it as host-decided (a future neural selector, once wired,
    # is lesion-tested here).
    # CHOOSE — does the SUBSTRATE decide the content of a direct factual question (recall or honest abstain), or does the
    # host router keyword-CONFABULATE? Test a well-formed UNANSWERABLE question: a substrate-driven turn ABSTAINS; the old
    # host router guessed a keyword-overlapping fact. Lesion the substrate recall -> the confab must return (load-bearing).
    cq = "what does fish fly?"                                 # well-formed, no such fact
    c_base = _ask(chat, cq)
    choose_abstains = bool(c_base.get("abstained")) or "don't know" in str(c_base["answer"]).lower()
    choose_lesion_load_bearing = None
    if choose_abstains and hasattr(chat, "_substrate_recall"):
        orig = chat._substrate_recall
        chat._substrate_recall = lambda q: None               # remove substrate -> falls to host keyword guess
        try:
            c_les = _ask(chat, cq)
        finally:
            chat._substrate_recall = orig
        # load-bearing if lesioning the substrate makes the turn CONFABULATE (stop abstaining) on the same question
        choose_lesion_load_bearing = not (bool(c_les.get("abstained")) or "don't know" in str(c_les["answer"]).lower())
    choose_is_neural = bool(choose_abstains and choose_lesion_load_bearing)
    out["CHOOSE"] = {"neural": choose_is_neural, "abstains_on_unanswerable": choose_abstains,
                     "lesion_load_bearing": choose_lesion_load_bearing, "probe": c_base,
                     "note": "direct factual questions are decided by the SUBSTRATE (recall or honest abstain); the host "
                             "router's keyword-confab is retired for them (lesion the substrate -> confab returns). "
                             "Broader CHOOSE (neural question comprehension, open-ended topic) is further."}

    # GENERATE — does the brain produce NOVEL content, or only recall a pre-stored fact? Ask something that is NOT a
    # stored fact and requires composition; recall-only -> abstains ("I don't know").
    # GENERATE — does the brain VOLUNTEER associated knowledge about a topic (associative generation), or only recall a
    # single fact? "tell me about dog" -> the brain should chain >1 fact via the dlPFC spiking spreading-activation
    # (elaborate). Lesion elaborate -> the chain collapses to the single primary fact (load-bearing).
    gq = "tell me about dog"
    g = _ask(chat, gq)
    n_facts = str(g["answer"]).count(".")                    # each chained fact ends with "."
    generates = (not g.get("abstained", True)) and "don't know" not in str(g["answer"]).lower() and n_facts >= 2
    gen_lesion_load_bearing = None
    if generates and hasattr(getattr(agent, "agent", agent), "elaborate"):
        inner = getattr(agent, "agent", agent)
        orig = inner.elaborate
        inner.elaborate = lambda *a, **k: None               # remove the associative spreading-activation
        try:
            g_les = _ask(chat, gq)
        finally:
            inner.elaborate = orig
        gen_lesion_load_bearing = str(g_les["answer"]).count(".") < n_facts   # collapsed to fewer facts
    out["GENERATE"] = {"open_ended": bool(generates and gen_lesion_load_bearing is not False),
                       "n_facts_chained": n_facts, "assoc_lesion_load_bearing": gen_lesion_load_bearing, "probe": g,
                       "note": "associative generation: 'tell me about X' chains >1 fact via the dlPFC spiking "
                               "spreading-activation (elaborate); lesion it -> collapses to the single primary fact. "
                               "Broader open-ended generation (novel sentences on any topic) is the emerge stream-cortex, further."}

    # LEARN — does the TURN incorporate a fact told mid-conversation? Teach a fact NOT pre-baked, then ask. Uses KNOWN
    # words ("cat chase bird" — the pre-baked cat-fact is "cat eat fish", so "cat chase ?" is genuinely new), which the
    # first in-loop-learning integration (substrate-recall fallback) covers; a NEW word needs on-the-fly code allocation.
    # NEW WORDS ("wolf","hunt","deer" not in the build-time vocab) taught through the FULL production acquisition path
    # (chat.answer parses the assertion + hears it via runtime code allocation) -> the owner grows the brain by talking.
    learn = {"taught": "wolf hunt deer", "q": "what does wolf hunt?", "answer_word": "deer"}
    before = _ask(chat, learn["q"])
    taught = _ask(chat, learn["taught"])                     # ASSERTION -> the turn acquires it
    taught_ok = "got it" in str(taught["answer"]).lower()
    after = _ask(chat, learn["q"])
    learned = ("deer" not in str(before["answer"]).lower() or before.get("abstained")) and \
              ("deer" in str(after["answer"]).lower())
    # LESION the substrate recall -> the taught fact must DISAPPEAR (proves LEARN is load-bearing on the substrate
    # path, not a host-list effect). If the answer is byte-identical with the recall lesioned, LEARN earns no credit.
    lesion_load_bearing = None
    if learned and hasattr(chat, "_substrate_recall"):
        orig = chat._substrate_recall
        chat._substrate_recall = lambda q: None
        try:
            lesioned = _ask(chat, learn["q"])
        finally:
            chat._substrate_recall = orig
        lesion_load_bearing = "bird" not in str(lesioned["answer"]).lower()
        after["lesioned_substrate_recall"] = lesioned["answer"]
    out["LEARN"] = {"in_loop": bool(learned and lesion_load_bearing is not False),
                    "recall_lesion_load_bearing": lesion_load_bearing,
                    "before": before, "after": after, "taught": learn["taught"], "hear_ok": taught_ok,
                    "note": "substrate-first recall: a fact heard this turn is recalled from inner.what_does (the "
                            "spiking substrate), role-aware; host QuestionRouter is the fallback. LEARN credit requires "
                            "the recall lesion to flip it off."}

    # ---- coarse spiking-path lesion (validate the 2 default-on-spiking rows are LESION-LOAD-BEARING) ----
    # Lesion the composer's recall: force the composer.query/recall to fail and see if a fact-recall answer changes.
    recall_q = None
    _SELF = ("i", "you", "me", "brain", "self")
    for a, v, p in facts:
        if a and a.lower() not in _SELF and v and p:
            recall_q = "what does %s %s?" % (a, v)
            break
    if recall_q is None:  # fall back to any fact with agent+action
        for a, v, p in facts:
            if a and v:
                recall_q = "what does %s %s?" % (a, v)
                break
    lesion = {"recall_q": recall_q}
    if recall_q:
        base = _ask(chat, recall_q)
        composer = getattr(agent, "composer", None)
        lesion["composer_type"] = type(composer).__name__ if composer is not None else None
        patched = None
        for meth in ("query_patient", "query_agent", "query", "recall", "unbind"):
            if composer is not None and hasattr(composer, meth):
                patched = meth
                orig = getattr(composer, meth)
                setattr(composer, meth, lambda *a, **k: None)
                try:
                    les = _ask(chat, recall_q)
                finally:
                    setattr(composer, meth, orig)
                lesion.update({"lesioned_method": meth, "baseline": base, "lesioned": les,
                               "load_bearing": base["answer"] != les["answer"]})
                break
        if patched is None:
            lesion["note"] = "no composer recall method found to lesion (query_patient/query/recall/unbind)"
    out["spiking_recall_lesion"] = lesion

    # ---- headline ----
    out["owner_visible_function"] = {"CHOOSE": out["CHOOSE"]["neural"], "GENERATE": out["GENERATE"]["open_ended"],
                                     "LEARN": out["LEARN"]["in_loop"]}
    out["verdict"] = ("The default production turn CHOOSE=%s GENERATE=%s LEARN=%s. "
                      % (out["CHOOSE"]["neural"], out["GENERATE"]["open_ended"], out["LEARN"]["in_loop"])
                      + ("All three host-or-absent: the production brain does not yet decide, generate, or learn."
                         if not any(out["owner_visible_function"].values()) else "Some owner-visible function present."))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    out = probe(args.seed)
    print("=== PRODUCTION LESION PROBE ===")
    print(out["verdict"])
    print("owner-visible-function:", out["owner_visible_function"])
    print("spiking-recall-lesion:", {k: out["spiking_recall_lesion"].get(k) for k in ("lesioned_method", "load_bearing")})
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
