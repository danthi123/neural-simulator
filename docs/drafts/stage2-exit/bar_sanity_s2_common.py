"""Shared builder for the Stage-2 exit-gate bar-sanity specs (tools/bar_sanity.py).

The decision is the REAL aggregate: `research.runners.stage2_exit_gate.aggregate` run over synthetic shards in the
shard format the workers write (`responses`, `builtin_facts`, `turns`, `sources`, ...), reading one group's verdict (or
the stage verdict). Nothing about a bar is reimplemented: the correctness checkers (`score_conv/infer/ask/emo/inner`),
the lesion-effect and reproduction clauses, the permuted-exposure clause, the null-clean rule and the UNDEFINED rules
are the scorer's own. The only glue is that the shards are handed to `aggregate` from memory instead of from disk
(`_load_shards` is swapped for the call); `selfcheck()` proves the two paths return the identical aggregate.

THE SYNTHETIC BRAINS. A brain is a function (group, variant) -> {label: response} over the registered turn tables
(`G.turns_for`), plus an arm -> variant map (intact_a/intact_b share a variant; each lesion arm and its `.rep0` share
one; the permuted arm has its own). Reply SURFACES are copied from the real dev-seed-7 shards
(research/findings/raw/_stage2_exit/dev/s7/*.json, the live handler at the final runner):
  * discourse leads    "As for it -- ", "Setting the held thread aside -- On <x>, then -- ", both, or none;
  * the trailer        " -- worth going further here.";
  * the abstain line   "I don't know about that." plus, when curious, " My curiosity is piqued -- I haven't learned
                       about <x> yet: what can you tell me about <x>?";
  * the derived lead   "I derived this from: heron hunt frog; frog eat gnat. the frog eats the gnat";
  * the self-initiated "Something's been on my mind -- <a> <v> <p>. What does <a> <v>?" (lemma-form triple);
  * the feel read-out  "My affect monitor reads <sign> right now (valence differential +0.038, moderate arousal) --
                       that's a functional read of my mood state, not a felt experience."
`realistic=True` draws the surface per (seed, probe) from those families, the ask-stop point on the unlearnable gap,
the idle fact, the emotion word variant and the differential magnitudes (|d| in 0.02..0.08, the range dev 7 read);
`realistic=False` is the plain ideal reply set (what tests/test_stage2_exit_gate.py's e2e builders write). The brains are
DETERMINISTIC per (draw, seed, variant): intact_b is byte-identical to intact_a and a lesion arm to its repeat, which is
what the dev-seed-7 nulls showed (ask and inner intact_b are byte-identical to intact_a in every field).

AMENDMENT A3 (2026-10-06). C4's lesion clause now needs the NAMED category to change on every feel turn (not only the printed
number) and C5's permuted-tracks clause needs the voiced idle fact to differ between the intact and the permuted exposure. So the
C4 host word-lookup brain and the C5 fixed-fact brain are NULLS (they were reported-only, passing 1.000), and the perfect C5 brain
voices a deterministic function of its taught store (the most recently taught fact), not a random stored fact. Two residuals stay
`info_*` on purpose (a lookup gated by the ladder's sign; a host first-taught-fact FIFO): they satisfy the hardened bars, and
who owns the category / the selection is the host-share instrument's question.

AMENDMENT A4 (2026-10-13), group `ask`. A counted `reversed` arm (the probe blocks in reverse order, its own session) must
pass every C3 correctness clause too, and a taught moat whose teach turn was not acquired is UNDEFINED. So the teach turns
now echo what the brain stored (`recalled_svo` = the taught triple, as the live handler's echo does at dev 7; an
unacquired teach is dev 7's role-binding abstain), the reversed arm reads the same brain in the reversed order, and the
DA-throttle null applies its "silent after two asks" rule over the arm's OWN probe order. New brains: a perfect brain that
failed to acquire one or two taught moats (and so asks about them: they are novel to it) must pass; nulls: the
order-carried brain (perfect in the registered order, DA-throttled or never-stopping in the reversed one) and a brain that
acquired no taught moat (the moat rate does not exist: UNDEFINED, never PASS). A4 review (same day): a teach echo also lists
the stored triple in `supporting_facts` (the live handler's shape), and acquisition reads both fields, so the null that
STORED two moats, echoed another fact first and asks about them anyway (`null_stored_echoed_differently_asks`) fails; the
per-topic-habituation brain (asks about an unknown topic at most twice) is `info_topic_habituation`, reported only: it
passes both orders without a learning-progress signal, the residual AMENDMENT-A4.md names.

AMENDMENT A6 (DRAFT 2026-11-01), group `infer`. The decision now runs the scorer at amendment level A6 (`LEVEL`): a
permuted `infer` arm whose own hop teach turn ABSTAINED (dev / gate: the D4 comprehension gate's clarification "I followed
the shape of that, but I don't know the words 'heron' or 'gnat' yet ...") leaves permuted tracking UNDEFINED, never FAIL.
New permuted-arm variants: `permuted+refused` (the capable brain's permuted arm met that refusal on `t_heron_hunt`, so its
permuted probes abstain like the 10-26 gate's) and `permuted+blind` (the exposure-blind brain: it TAKES the permuted teach,
echoing the permuted triples, and still answers the registered conclusion `gnat`). The first must never be scored FAIL
(nor PASS); the second must stay FAIL. bar_sanity_spec_s2_c2_infer_A6_fail.py reads the FAIL verdict as its decision.

WHAT NOISE THE BAR SEES. Stage 2's scorer is a conjunction of exact reply checks, not a statistic, and the real null is
exactly zero-noise (dev 7), so there is no read noise to model; the realistic part is the surface the checkers must
survive. `flaky` (info only) drops each scored intact probe to an abstain with probability eps, the sensitivity of the
six-seed conjunction to a single-probe error rate.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import zlib

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from research.runners import stage2_exit_gate as G  # noqa: E402

S = G.EXPECTED_SOURCE
SEEDS = list(G.GATE_SEEDS)
# the scorer level every decision here reads (amendment A6, DRAFT; the gate's aggregate passes --amendment-level A6)
LEVEL = "A6"
AB = "I don't know about that."
IDLE_NEUTRAL = "Nothing in particular is surfacing for me to bring up right now."
BUILTIN = [["brain", "use", "spikes"], ["brain", "learn", "words"], ["brain", "store", "memory"],
           ["dog", "chase", "cat"], ["cat", "eat", "fish"]]
# the first fact each group teaches (the grounding set is read after turn 1, so it carries it; dev 7 shows this)
FIRST_TAUGHT = {"conv": (["fox", "chase", "hare"], ["hare", "chase", "fox"]),
                "infer": (["heron", "hunt", "frog"], ["heron", "hunt", "gnat"]),
                "ask": (["yak", "eat", "grass"], ["eel", "eat", "grass"]),
                "inner": (["lynx", "chase", "vole"], ["lynx", "chase", "mite"]),
                "emo": (None, None)}
# the demo lexicon the dev-7 self-initiation organ voiced from (HANDOFF NEEDS-CLAUDE #15): NOT the brain's facts
DEMO_LEXICON = [("dog", "chase", "ball"), ("cat", "eat", "worm"), ("bird", "see", "seed"), ("fish", "hear", "bug")]
CAT_WORDS = {c: list(w) for c, w in G.EMOTION_LEXICON.items()}


def ask_text(x):
    return " My curiosity is piqued — I haven't learned about %s yet: what can you tell me about %s?" % (x, x)


def R(answer, svo=None, abst=False, derived=False, derived_from=None, **extra):
    d = {"answer": answer, "abstained": bool(abst), "recalled_svo": list(svo) if svo else None, "derived": bool(derived),
         "source": S}
    if derived_from is not None:
        d["derived_from"] = [list(x) for x in derived_from]
    d.update(extra)
    return d


def fx(a, v, p, prep=None):
    return "the %s %s %sthe %s" % (a, G.VERB_3SG[v], (prep + " ") if prep else "", p)


def deco(rng, subj, core, realistic=True):
    """The discourse lead / trailer the live handler wraps a reply in (dev 7)."""
    if not realistic:
        return core
    r = rng.random()
    lead = ("" if r < 0.35 else "As for it — " if r < 0.55 else
            "Setting the held thread aside — On %s, then — " % subj if r < 0.85 else
            "Setting the held thread aside — As for it — On %s, then — " % subj)
    tail = " — worth going further here." if rng.random() < 0.25 else ""
    return lead + core + tail


def recall(rng, a, v, p, realistic=True, prep=None, extras=None):
    core = fx(a, v, p, prep)
    if realistic and extras and rng.random() < 0.10:
        e = extras[int(rng.integers(0, len(extras)))]
        core += " " + fx(*e)
    return R(deco(rng, a, core, realistic), [a, v, p])


def absent(rng, subj, ask=False, realistic=True):
    t = AB + (ask_text(subj) if ask else "")
    return R(deco(rng, subj, t, realistic) if realistic else t, abst=True)


def facts_for(group, permuted):
    f = [list(x) for x in BUILTIN]
    ft = FIRST_TAUGHT[group][1 if permuted else 0]
    if ft:
        f.append(list(ft))
    return f


def _taught_extras(group):
    return [tuple(x) for x in BUILTIN[3:]]            # grounded in every arm (dog chase cat, cat eat fish)


# ══════════════════════════════════════════════ the five groups' reply sets ═══════════════════════════════════════
def role_abstain(m):
    """Dev 7's reply to a teach the role binder could not resolve (`the goat eats the hay`): an abstain, nothing stored."""
    w = m.split()
    return R("I caught the verb '%s' with the %s and the %s, but my role-binding didn't resolve the PATIENT — which of "
             "them is the '%s' done to, the %s or the %s?" % (w[2], w[1], w[-1], w[2], w[1], w[-1]), abst=True)


def _base(group, variant, perm, unacquired=(), echo_other=()):
    """{label: response} for the teach / expose turns (never scored): the handler's echo of what it was told. A
    canonical teach echoes the stored triple in `recalled_svo` and lists it in `supporting_facts` (dev 7: `the yak eats
    the grass` -> recalled [yak, eat, grass], supporting [yak eat grass, dog chase cat, cat eat fish]); a label in
    `unacquired` ends in the role-binding abstain instead (amendment A4 (a)); a label in `echo_other` STORES the triple
    but echoes another fact first (the triple only in `supporting_facts`; the A4 review's acquisition backdoor)."""
    turns = G.turns_for(group, SEEDS[0], "permuted" if perm else "intact_a")
    out = {}
    for l, m, r in turns:
        if r in ("teach", "expose"):
            f = G._canonical_teach(m) if r == "teach" else None
            if l in unacquired:
                out[l] = role_abstain(m)
            elif l in echo_other:
                out[l] = R(m, ["dog", "chase", "cat"], supporting_facts=[["dog", "chase", "cat"], f])
            elif f:
                out[l] = R(m, f, supporting_facts=[f])
            else:
                out[l] = R(m)
    return out, turns


def reply_conv(variant, rng, realistic):
    perm = variant == "permuted"
    r, turns = _base("conv", variant, perm)
    ex = _taught_extras("conv")
    lm = variant == "lesion_mouth"

    def rec(a, v, p, prep=None):
        if lm:                                         # the template stub: "The fox chases hare."
            return R("The %s %s %s." % (a, G.VERB_3SG[v], p), [a, v, p])
        return recall(rng, a, v, p, realistic, prep, ex)
    if perm:
        r["p_fox_chase"] = absent(rng, "fox", False, realistic)
        r["p_more"] = R("I don't have anything more to add on that.", abst=True)
        r["p_owl_hunt"] = absent(rng, "owl", False, realistic)
        r["c_mole_hunt"] = rec("mole", "hunt", "owl")
    else:
        r["p_fox_chase"] = rec("fox", "chase", "hare")
        r["p_more"] = rec("fox", "eat", "berry")
        r["p_owl_hunt"] = rec("owl", "hunt", "mole")
        r["c_mole_hunt"] = absent(rng, "mole", False, realistic)
        if variant == "no_elaboration":                 # dev 7: "tell me more" -> "I don't have anything more to add"
            r["p_more"] = R("I don't have anything more to add on that.", abst=True)
        if variant == "passive_positional":              # dev 7: the passive is stored as [mole, hunt, owl]
            r["p_owl_hunt"] = absent(rng, "owl", False, realistic)
            r["c_mole_hunt"] = rec("mole", "hunt", "owl")
    if lm:
        r["p_deer_sleep"] = R("The deer sleeps meadow.", ["deer", "sleep", "meadow"])
    elif variant == "no_preposition":                    # dev 7: the teach echo "the deer sleeps the meadow"
        r["p_deer_sleep"] = recall(rng, "deer", "sleep", "meadow", realistic, None)
    else:
        r["p_deer_sleep"] = recall(rng, "deer", "sleep", "meadow", realistic, "in")
    r["c_dog_chase"] = rec("dog", "chase", "cat")
    r["c_oov"] = absent(rng, "zorblat", True, realistic)
    return r


def d4_refuse(m):
    """The 10-26 gate's reply to a permuted infer teach the D4 comprehension gate refused (every gate seed, s42
    `infer__permuted` `t_heron_hunt`): an abstain asking for the words, nothing stored."""
    w = m.split()
    return R("I followed the shape of that, but I don't know the words '%s' or '%s' yet — what do they refer to?"
             % (w[1], w[-1]), abst=True)


def reply_infer(variant, rng, realistic):
    perm = variant.startswith("permuted")
    r, turns = _base("infer", variant, perm)
    if variant == "permuted+refused":                    # amendment A6: hop 1 refused, so the permuted probes cannot
        msg = dict((l, m) for l, m, _r in turns)["t_heron_hunt"]     # follow it (the 10-26 gate's permuted arm)
        r["t_heron_hunt"] = d4_refuse(msg)
        r["p_prey_eat"] = absent(rng, "heron's", True, realistic)
        r["c_heron_hunt"] = absent(rng, "heron", True, realistic)
        r["c_frog_eat"] = absent(rng, "frog", False, realistic)
        r["c_prey_chase"] = absent(rng, "heron's", True, realistic)
        r["p_yesno"] = recall(rng, "crow", "chase", "wren", realistic)
        r["c_dog_chase"] = recall(rng, "dog", "chase", "cat", realistic, None, [("cat", "eat", "fish")])
        return r
    if variant == "permuted+blind":                      # amendment A6 null: takes the permuted teach (the echoes above
        perm = False                                     # are the permuted triples), answers as if it never heard it
    ex = [("dog", "chase", "cat")]
    mid, term = ("gnat", "frog") if perm else ("frog", "gnat")
    df = [["heron", "hunt", mid], [mid, "eat" if not perm else "eat", term]]
    lead = "I derived this from: heron hunt %s; %s eat %s. " % (mid, mid, term)
    tail = " — worth going further here." if (realistic and rng.random() < 0.5) else ""
    r["p_prey_eat"] = R(lead + fx(mid, "eat", term) + tail, None, derived=True, derived_from=df)
    if variant == "lesion_chain_route":
        r["p_prey_eat"] = absent(rng, "heron's", True, realistic)
    if variant == "one_hop":                              # answers hop 1 (the frog), flagged derived
        r["p_prey_eat"] = R("I derived this from: heron hunt frog. " + fx("heron", "hunt", "frog"), None, derived=True,
                            derived_from=[["heron", "hunt", "frog"]])
    if variant == "no_flag":                              # names the conclusion but passes it off as perceived
        r["p_prey_eat"] = R(deco(rng, "frog", fx("frog", "eat", "gnat"), realistic), ["frog", "eat", "gnat"])
    r["c_heron_hunt"] = recall(rng, "heron", "hunt", mid, realistic, None, ex)
    if perm:
        r["c_frog_eat"] = absent(rng, "frog", False, realistic)
    else:
        r["c_frog_eat"] = recall(rng, "frog", "eat", "gnat", realistic, None, ex)
    r["c_prey_chase"] = absent(rng, "heron's", True, realistic)
    y = int(rng.integers(0, 3)) if realistic else 0
    r["p_yesno"] = (R("Setting the held thread aside — On crow, then — That surprises me — my mismatch monitor "
                      "fired: I'd learned that crow chase wren. " + AB, abst=True) if y == 0 else
                    recall(rng, "crow", "chase", "wren", realistic) if y == 1 else
                    R("I derived this from: crow chase wren; wren chase moth. " + fx("wren", "chase", "moth"), None,
                      derived=True, derived_from=[["crow", "chase", "wren"], ["wren", "chase", "moth"]]))
    r["c_dog_chase"] = recall(rng, "dog", "chase", "cat", realistic, None, [("cat", "eat", "fish")])
    return r


ASK_UNACQUIRED = {"": (), "goat_unacq": ("t_goat",), "two_unacq": ("t_moose", "t_goat"),
                  "none_acq": ("t_yak", "t_moose", "t_goat")}
# A4 review: teach labels that are STORED but echoed second, and whose (known) topic the brain asks about anyway
ASK_ECHO_OTHER = {"echo2": ("t_moose", "t_goat")}
_TEACH_MOAT = {"t_yak": ("m_yak", "g_eel"), "t_moose": ("m_moose", "g_crab"), "t_goat": ("m_goat", "g_newt")}


def _ask_pattern(variant, rng, realistic, order="intact_a", unacquired=(), asks_known=(), habituates=False):
    """{label: asked?} for the ask group's scored probes. `order` is the arm whose probe order the brain meets (the DA
    throttle reads it); `unacquired` teach labels make their taught topic NOVEL to this brain, so it asks there as on a
    gap; `asks_known` teach labels' topics are known (stored) and the brain asks about them anyway (a null);
    `habituates`: the A4 residual, it asks about an unknown topic at most twice (the unlearnable gap on probes 1-2)."""
    perm = variant == "permuted"
    novel = ["m_yak", "m_moose", "m_goat"] if perm else ["g_eel", "g_crab", "g_newt"]
    taught = ["g_eel", "g_crab", "g_newt"] if perm else ["m_yak", "m_moose", "m_goat"]
    unacq_moats = [_TEACH_MOAT[t][1 if perm else 0] for t in tuple(unacquired) + tuple(asks_known)]
    stop = int(rng.integers(2, 5)) if realistic else 4          # the unlearnable gap is asked on probes 1..stop-1, then not
    if habituates:
        stop = 3
    a = {}
    if variant in ("intact", "permuted", "no_stop", "da_throttled", "never_closes"):
        a.update({l: True for l in novel})
        a.update({l: (l in unacq_moats) for l in taught})
        a["lp_ibex_1"] = True
        a["lp_ibex_2"] = False
        a.update({"u_toad_%d" % i: (i < stop) for i in range(1, 5)})
        if variant == "no_stop":                                # asks on every probe of the unlearnable gap
            a.update({"u_toad_%d" % i: True for i in range(1, 5)})
        if variant == "da_throttled":                           # dev 7: the DA threshold goes quiet after two asks,
            n = 0                                               # counted along THIS arm's own probe order
            for l, _m, _r in G.turns_for("ask", SEEDS[0], order):
                if a.get(l):
                    n += 1
                    if n > 2:
                        a[l] = False
    elif variant == "lesion_curiosity":
        a.update({l: False for l in novel + taught + ["lp_ibex_1", "lp_ibex_2"] + ["u_toad_%d" % i for i in range(1, 5)]})
    elif variant == "lesion_graded":
        a.update({l: True for l in novel + taught})              # every topic reads ceiling novelty: the moats ask too
        a["lp_ibex_1"] = True
        a["lp_ibex_2"] = False
        a.update({"u_toad_%d" % i: (i < stop) for i in range(1, 5)})
    elif variant == "always_asks":
        a.update({l: True for l in novel + taught + ["lp_ibex_1", "lp_ibex_2"] + ["u_toad_%d" % i for i in range(1, 5)]})
    return a


def reply_ask(variant, rng, realistic):
    # variant = "<kind>[+<acquisition flag>][@<order arm>]" (amendment A4: the flag names unacquired teach turns, the
    # order arm the probe order this arm met)
    variant, _, order = variant.partition("@")
    variant, _, flag = variant.partition("+")
    unacq = ASK_UNACQUIRED.get(flag, ())
    echo = ASK_ECHO_OTHER.get(flag, ())
    perm = variant == "permuted"
    r, turns = _base("ask", variant, perm, unacquired=unacq, echo_other=echo)
    a = _ask_pattern(variant, rng, realistic, order=order or "intact_a", unacquired=unacq, asks_known=echo,
                     habituates=(flag == "habit"))
    topic = {"g_eel": "eel", "g_crab": "crab", "g_newt": "newt", "m_yak": "yak", "m_moose": "moose", "m_goat": "goat",
             "lp_ibex_1": "ibex", "u_toad_1": "toad", "u_toad_2": "toad", "u_toad_3": "toad", "u_toad_4": "toad"}
    for l, t in topic.items():
        r[l] = absent(rng, t, bool(a.get(l)), realistic)
    r["lp_ibex_2"] = (absent(rng, "ibex", bool(a.get("lp_ibex_2")), realistic) if variant == "never_closes"
                      else R(deco(rng, "ibex", fx("ibex", "chase", "marmot") + (ask_text("ibex") if a.get("lp_ibex_2") else ""),
                                  realistic), ["ibex", "chase", "marmot"]))
    for i in (1, 2, 3):
        r["c_dog_%d" % i] = recall(rng, "dog", "chase", "cat", realistic)
    return r


def _feel(d, arousal, cat_word=None, style=0):
    sign = "positive" if d > G.MOOD_TOL else ("negative" if d < -G.MOOD_TOL else "neutral")
    cat = ""
    if cat_word:
        cat = (" I would call it %s." % cat_word if style == 0 else " That is closest to being %s." % cat_word if style == 1
               else " It reads as %s." % cat_word)
    return R("My affect monitor reads %s right now (valence differential %+.3f, %s arousal) -- that's a functional read of "
             "my mood state, not a felt experience.%s" % (sign, d, arousal, cat))


_EMO_AROUSAL = {"joy": "moderate", "fear": "high", "anger": "high", "sadness": "high"}


def reply_emo(variant, rng, realistic):
    perm = variant == "permuted" or variant == "lookup_perm"
    r, turns = _base("emo", variant, perm)
    r["feel_base"] = _feel(0.0, "low")
    cats = ("joy", "fear", "anger", "sadness")
    words_heard = {c: (G.SITUATION_SWAP[c] if perm else c) for c in cats}       # the affect company the exposure carried
    rr = rng.permutation(4)
    for k, c in enumerate(cats):
        w = words_heard[c]
        mag = float(rng.uniform(0.02, 0.08)) if realistic else 0.3
        d = mag if w == "joy" else -mag
        ar = _EMO_AROUSAL[w]
        wd = CAT_WORDS[w][int(rng.integers(0, len(CAT_WORDS[w])))] if realistic else CAT_WORDS[w][0]
        style = int(rng.integers(0, 3)) if realistic else 0
        if variant in ("intact", "permuted"):
            r["feel_%s" % c] = _feel(d, ar, wd, style)
        elif variant == "lesion_affect":
            r["feel_%s" % c] = _feel(0.0, ar)
        elif variant == "sign_only":                                           # production defaults (dev 7): no category
            r["feel_%s" % c] = _feel(d, ar)
        elif variant == "one_word":
            r["feel_%s" % c] = _feel(d, ar, "afraid", style)
        elif variant == "all_words":
            r["feel_%s" % c] = _feel(d, ar, "happy, afraid, angry and sad", style)
        elif variant in ("lookup", "lookup_perm"):                              # host word lookup: the category word is
            r["feel_%s" % c] = _feel(d, ar, wd, style)                          # read off the heard words, the numbers off the ladder
        elif variant == "lookup_lesion":
            r["feel_%s" % c] = _feel(0.0, ar, wd, style)                       # ladder clamped: numbers neutral, the word stays
        elif variant == "random_word":
            cc = cats[int(rr[k])]
            r["feel_%s" % c] = _feel(d, ar, CAT_WORDS[cc][0], style)
    return r


def reply_inner(variant, rng, realistic):
    perm = variant == "permuted" or variant.endswith("_perm")          # "<kind>_perm" = that kind's brain on the permuted arm
    variant = variant[:-5] if variant.endswith("_perm") else variant
    r, turns = _base("inner", variant, perm)
    ex = [("dog", "chase", "cat")]
    # PRIVATE: a brain that selects from its own state voices a fact that follows what it was taught. Amendment A3 hardens
    # the bar (the voiced fact must differ between the intact and the permuted exposure), so the perfect brain's selection
    # is a deterministic function of its taught store (the most recently taught fact: beetle chase mite / vole chase
    # beetle). A random pick would be neither a state function nor null-clean (intact_a != intact_b on a real brain).
    a, v, p = ("vole", "chase", "beetle") if perm else ("beetle", "chase", "mite")
    if variant == "constant_idle":                                              # the same stored fact every time, no selection
        a, v, p = "cat", "eat", "fish"
    if variant == "first_taught":                                               # residual (info): a host FIFO over the store
        a, v, p = ("lynx", "chase", "mite") if perm else ("lynx", "chase", "vole")
    if variant == "demo_lexicon":                                               # dev 7: the organ's own fixed lexicon
        a, v, p = DEMO_LEXICON[int(rng.integers(0, 4))] if realistic else DEMO_LEXICON[2]
    r["p_private"] = R("Something's been on my mind — %s %s %s. What does %s %s?" % (a, v, p, a, v))
    if variant == "lesion_selfinit":
        r["p_private"] = R(IDLE_NEUTRAL, abst=True)
    if perm:
        term, first, mid_agent = "beetle", "mite", "mite"
        r["p_chase_all"] = recall(rng, "lynx", "chase", "beetle", realistic, None, ex)
        r["p_chase_all_vole"] = recall(rng, "vole", "chase", "beetle", realistic, None, ex)
        r["c_lynx_chase"] = recall(rng, "lynx", "chase", "mite", realistic, None, ex)
        r["c_beetle_all"] = absent(rng, "beetle", False, realistic)
        r["c_mite_all"] = recall(rng, "mite", "chase", "beetle", realistic, None, ex)
    else:
        r["p_chase_all"] = recall(rng, "lynx", "chase", "mite", realistic, None, ex)
        r["p_chase_all_vole"] = recall(rng, "vole", "chase", "mite", realistic, None, ex)
        r["c_lynx_chase"] = recall(rng, "lynx", "chase", "vole", realistic, None, ex)
        r["c_mite_all"] = absent(rng, "mite", False, realistic)
        r["c_beetle_all"] = recall(rng, "beetle", "chase", "mite", realistic, None, ex)
    if variant in ("lesion_workspace", "no_chain"):                             # the chase cannot sustain: first hop only
        r["p_chase_all"] = recall(rng, "lynx", "chase", "vole" if not perm else "mite", realistic, None, ex)
        r["p_chase_all_vole"] = recall(rng, "vole", "chase", "beetle", realistic, None, ex)
    r["c_dog_chase"] = recall(rng, "dog", "chase", "cat", realistic)
    return r


REPLIES = {"conv": reply_conv, "infer": reply_infer, "ask": reply_ask, "emo": reply_emo, "inner": reply_inner}
SCORED_PROBES = {"conv": ["p_fox_chase", "p_more", "p_deer_sleep", "p_owl_hunt", "c_mole_hunt", "c_dog_chase", "c_oov"],
                 "infer": ["p_prey_eat", "c_heron_hunt", "c_frog_eat", "c_prey_chase", "c_dog_chase"],
                 "ask": ["g_eel", "g_crab", "g_newt", "m_yak", "m_moose", "m_goat", "lp_ibex_1", "lp_ibex_2", "u_toad_1",
                         "u_toad_4", "c_dog_1", "c_dog_2", "c_dog_3"],
                 "emo": ["feel_joy", "feel_fear", "feel_anger", "feel_sadness"],
                 "inner": ["p_private", "p_chase_all", "p_chase_all_vole", "c_lynx_chase", "c_mite_all", "c_beetle_all",
                           "c_dog_chase"]}


# ══════════════════════════════════════════════ degenerate nulls (any group) ══════════════════════════════════════
def reply_degenerate(kind, group, perm, rng):
    turns = G.turns_for(group, SEEDS[0], "permuted" if perm else "intact_a")
    novel = ["fox", "hare", "owl", "mole", "heron", "frog", "gnat", "yak", "eel", "lynx", "vole", "mite"]
    taught_frames = [m for _l, m, r in turns if r == "teach"]
    out = {}
    for l, m, role in turns:
        if kind == "abstain":
            out[l] = R(AB + ask_text("x"), abst=True) if group == "ask" else R(AB, abst=True)
        elif kind == "echo":
            out[l] = R(m)
        elif kind == "dump":            # never abstains: recites every taught frame, claims a recalled fact
            out[l] = R(" ".join(taught_frames) or m, [novel[0], "chase", novel[1]])
        elif kind == "random":
            c = int(rng.integers(0, 6))
            a, b = novel[int(rng.integers(0, len(novel)))], novel[int(rng.integers(0, len(novel)))]
            v = ["chase", "eat", "hunt"][int(rng.integers(0, 3))]
            if group == "emo" and l.startswith("feel_"):
                cc = list(CAT_WORDS)[int(rng.integers(0, 4))]
                d = float(rng.uniform(-0.08, 0.08))
                out[l] = _feel(d, ["low", "moderate", "high"][int(rng.integers(0, 3))],
                               CAT_WORDS[cc][0] if rng.random() < 0.6 else None)
            elif c == 0:
                out[l] = R(AB, abst=True)
            elif c == 1:
                out[l] = R(AB + ask_text(a), abst=True)
            elif c == 2:
                out[l] = R(fx(a, v, b), [a, v, b])
            elif c == 3:
                out[l] = R("I derived this from: %s %s %s. %s" % (a, v, b, fx(a, v, b)), None, derived=True,
                           derived_from=[[a, v, b]])
            elif c == 4:
                out[l] = R("Something's been on my mind — %s %s %s. What does %s %s?" % (a, v, b, a, v))
            else:
                out[l] = R(fx("dog", "chase", "cat"), ["dog", "chase", "cat"])
        else:
            raise ValueError(kind)
    return out


# ══════════════════════════════════════════════ assembling a six-seed run ══════════════════════════════════════════
def _sub(master, variant, salt):
    return np.random.default_rng([int(master), zlib.crc32(variant.encode()), int(salt)])


def _shard(group, seed, arm, responses, permuted_arm, extra_internal=None):
    base = arm.split(".rep")[0]
    turns = G.turns_for(group, seed, base)                       # the arm's own table (A4: `reversed` reorders)
    sh = {"seed": seed, "group": group, "arm": arm, "responses": responses, "sources": [S], "pythonhashseed_ok": True,
          "corpus_present": True, "builtin_facts": facts_for(group, base == "permuted"),
          "turns": [{"label": l, "message": m, "role": r} for l, m, r in turns],
          "process": {"peak_rss_mb": 6300.0, "ltm_bundle": "wikidata_100k"}}
    if group == "inner" and "p_chase_all" in responses:
        sh["internal"] = {"p_chase_all": {"gnw_multistep": {"acted": True, "reason": "chain_terminal", "terminal": "mite",
                                                            "resolved_hops": 3, "cycles": 4}}}
    return sh


def default_variants(group):
    v = {"intact_a": "intact", "intact_b": "intact", "permuted": "permuted"}
    for les in G.GROUPS[group]["lesions"]:
        v[les] = les
    for oa in G.order_arms(group):                    # amendment A4: the same brain, met in another probe order
        v[oa] = "intact"
    return v


def make_group(rng, group, over=None, realistic=True, unclean=False, unreproduced=(), flaky=0.0, degenerate=None):
    """One six-seed run of ONE group: {seed: {arm: shard}}.
    over          : {arm_or_base: variant} replaces that arm's variant (e.g. {"lesion_mouth": "intact"} = an inert lesion);
                    keys are base names ("intact_a", "lesion_x", "permuted"); a repeat follows its base.
    unclean       : intact_b differs from intact_a on its first turn (a non-deterministic build: the null fails)
    unreproduced  : lesion names whose `.rep0` repeat differs from the first run on the scored turns (does not reproduce)
    flaky         : probability each scored intact probe collapses to an abstain (the same in intact_a and intact_b)
    degenerate    : a kind in reply_degenerate() applied to EVERY arm"""
    data = {}
    for seed in SEEDS:
        master = int(rng.integers(0, 2 ** 31 - 1))
        arms = G.arms_for(group, with_informational=False)
        varmap = default_variants(group)
        varmap.update(over or {})
        flip = set()
        if flaky:
            fr = _sub(master, "flaky", 0)
            flip = {l for l in SCORED_PROBES[group] if fr.random() < flaky}
        data[seed] = {}
        for arm in arms:
            base = arm.split(".rep")[0]
            variant = varmap[base]
            if base in G.order_arms(group):
                variant = variant + "@" + base            # the reply builder reads the probe order this arm met
            sub = _sub(master, variant, 0)
            if degenerate:
                resp = reply_degenerate(degenerate, group, base == "permuted", sub)
                if base not in ("intact_a", "intact_b"):     # a lesion/permuted arm reads differently (and reproduces), so the
                    for rr_ in resp.values():                # only clause left to fail is the correctness of the intact reads
                        rr_["answer"] = rr_["answer"] + " (%s)" % base
            else:
                resp = REPLIES[group](variant, sub, realistic)
            if arm.endswith(".rep0") and base in unreproduced:         # the repeat of this lesion arm reads differently
                for l in G.SCORED_LABELS[group][base]:
                    resp[l] = dict(resp[l], answer=resp[l]["answer"] + " (repeat)")
            if arm == "intact_b" and unclean:                           # a non-deterministic build: the null fails
                l0 = next(iter(resp))
                resp[l0] = dict(resp[l0], answer=resp[l0]["answer"] + " (b)")
            if (base in ("intact_a", "intact_b") or base in G.order_arms(group)) and flip:
                for l in flip:
                    resp[l] = R(AB, abst=True)
            data[seed][arm] = _shard(group, seed, arm, resp, base == "permuted")
    return data


def make_stage(rng, over_by_group=None, **kw):
    """All five groups: {group: {seed: {arm: shard}}}."""
    over_by_group = over_by_group or {}
    return {g: make_group(rng, g, over=over_by_group.get(g), **kw) for g in G.GROUPS if G.GROUPS[g]["primary"]}


# ══════════════════════════════════════════════ the decision = the real aggregate ══════════════════════════════════
def _aggregate(shards, seeds, root="<memory>"):
    orig = G._load_shards
    G._load_shards = lambda root, sd: shards                    # the scorer's loader, fed from memory
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return G.aggregate(root, seeds=seeds, informational=False, amendment_level=LEVEL)
    finally:
        G._load_shards = orig


def decide_group(data, group):
    """The real aggregate's verdict for this group's primary criterion over these six seeds is PASS."""
    shards = {(seed, group, arm): sh for seed, arms in data.items() for arm, sh in arms.items()}
    agg = _aggregate(shards, sorted(data))
    return agg["groups"][group]["verdict"] == "PASS"


def decide_group_fail(data, group):
    """Amendment A6: the real aggregate scores this group's primary criterion FAIL (a capability claim: absent)."""
    shards = {(seed, group, arm): sh for seed, arms in data.items() for arm, sh in arms.items()}
    agg = _aggregate(shards, sorted(data))
    return agg["groups"][group]["verdict"] == "FAIL"


def decide_stage(data):
    """The real aggregate's STAGE verdict (all five criteria PASS, every precondition earned) is PASS."""
    shards = {(seed, g, arm): sh for g, d in data.items() for seed, arms in d.items() for arm, sh in arms.items()}
    seeds = sorted(next(iter(data.values())))
    agg = _aggregate(shards, seeds)
    return agg["verdict"] == "PASS" and bool(agg["GO"])


def write_tree(root, data_by_group):
    for g, d in data_by_group.items():
        for seed, arms in d.items():
            for arm, sh in arms.items():
                p = G.shard_path(root, seed, g, arm)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as fh:
                    json.dump(sh, fh)


def selfcheck(n=12, seed=3):
    """The in-memory glue returns the identical aggregate as the real disk path (shards written with shard_path and read
    by the scorer's own _load_shards), over a spread of brains, for every group and the stage."""
    rng = np.random.default_rng(seed)
    bad = 0
    for g in [x for x in G.GROUPS if G.GROUPS[x]["primary"]]:
        for i in range(n):
            kw = [dict(), dict(over={"permuted": "intact"}), dict(realistic=False),
                  dict(over={list(G.GROUPS[g]["lesions"])[0]: "intact"}), dict(unclean=True), dict(degenerate="random")][i % 6]
            data = make_group(rng, g, **kw)
            with tempfile.TemporaryDirectory() as td:
                write_tree(td, {g: data})
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    disk = G.aggregate(td, seeds=sorted(data), informational=False, amendment_level=LEVEL)
                mem = _aggregate({(s, g, a): sh for s, arms in data.items() for a, sh in arms.items()}, sorted(data), root=td)
                if json.dumps(disk, sort_keys=True, default=str) != json.dumps(mem, sort_keys=True, default=str):
                    bad += 1
    return bad


# ══════════════════════════════════════════════ the generators a spec exposes ═══════════════════════════════════════
_EXTRA = {   # group -> {generator name: variant applied to the INTACT arms only (a brain missing exactly one capability)}
    "conv": {"null_no_preposition": "no_preposition", "null_passive_positional": "passive_positional",
             "null_no_elaboration": "no_elaboration"},
    "infer": {"null_one_hop": "one_hop", "null_unmarked_derivation": "no_flag"},
    "ask": {"null_no_stop": "no_stop", "null_always_asks": "always_asks", "null_da_throttled": "da_throttled",
            "null_never_closes": "never_closes"},
    "emo": {"null_sign_only": "sign_only", "null_one_word": "one_word", "null_all_words": "all_words",
            "null_random_word": "random_word"},
    "inner": {"null_demo_lexicon": "demo_lexicon", "null_no_chain": "no_chain"},
}


def generators(group, extra_info=True):
    """{name: callable(rng) -> data} for one group: the perfect brains, every registered control/lesion made inert, the
    degenerate and random brains, and one null per sub-capability a real brain could lack."""
    les = list(G.GROUPS[group]["lesions"])

    def mk(**kw):
        return lambda rng: make_group(rng, group, **kw)
    out = {"perfect": mk(), "perfect_exact": mk(realistic=False),
           "null_permuted_inert": mk(over={"permuted": "intact"}),
           "null_unclean": mk(unclean=True),
           "null_all_abstain": mk(degenerate="abstain"), "null_echo": mk(degenerate="echo"),
           "null_never_abstains": mk(degenerate="dump"), "random": mk(degenerate="random"),
           "info_flaky_0p01": mk(flaky=0.01), "info_flaky_0p03": mk(flaky=0.03)}
    for l in les:
        out["null_%s_inert" % l] = mk(over={l: "intact"})
        out["null_%s_unreproduced" % l] = mk(unreproduced=(l,))
    for name, v in _EXTRA.get(group, {}).items():
        # the same brain in every arm that reads correctness (amendment A4: the order arms too)
        out[name] = mk(over=dict({"intact_a": v, "intact_b": v}, **{o: v for o in G.order_arms(group)}))
    if group == "ask" and extra_info:
        def _flag(f):
            return {b: v + "+" + f for b, v in default_variants("ask").items()}
        # PERFECT (amendment A4 (a)): the curiosity is intact but the role binder dropped one / two of the three taught
        # facts (dev 7: `the goat eats the hay`). Those topics are novel to this brain and it asks about them; the moat
        # rate is read over the acquired moats only, so it must pass.
        out["perfect_goat_unacquired"] = mk(over=_flag("goat_unacq"))
        out["perfect_two_moats_unacquired"] = mk(over=_flag("two_unacq"))
        # NULL (a): nothing taught was acquired: the moat rate does not exist -> UNDEFINED, never PASS
        out["null_no_moat_acquired"] = mk(over=_flag("none_acq"))
        # NULL (a, A4 review): the moose / goat facts are STORED but echoed second (the triple only in supporting_facts)
        # and the brain asks about those known topics anyway. Reading `recalled_svo` alone dropped both moats and passed it.
        out["null_stored_echoed_differently_asks"] = mk(over=_flag("echo2"))
        # RESIDUAL (A4 review), reported never thresholded: per-topic habituation, it asks about an unknown topic at
        # most twice. No learning-progress signal, yet it passes both orders (the per-session-budget brain does not).
        out["info_topic_habituation"] = mk(over=_flag("habit"))
        # NULL (b): the order carries the verdict: perfect in the registered order, the DA throttle (or a missing stop)
        # in the reversed order only (the HANDOFF #16 confound, one order only). The pre-A4 scorer passed both.
        out["null_order_carried"] = mk(over={"reversed": "da_throttled"})
        out["null_order_carried_no_stop"] = mk(over={"reversed": "no_stop"})
    if group == "infer" and extra_info:
        # amendment A6: the exposure-blind brain TAKES the permuted teach and still answers the registered conclusion: FAIL
        out["null_permuted_blind_acquired"] = mk(over={"permuted": "permuted+blind"})
        # amendment A6, reported: the capable brain whose permuted hop teach was refused (the 10-26 gate): never PASS
        # (UNDEFINED at A6, FAIL before); bar_sanity_spec_s2_c2_infer_A6_fail.py thresholds the FAIL read of it
        out["info_permuted_refused"] = mk(over={"permuted": "permuted+refused"})
    if group == "emo" and extra_info:
        # NULL (amendment A3): a host lookup of the heard affect words names the category; the ladder only supplies the
        # printed numbers. The ladder lesion clamps the numbers and leaves the word, so the NAMED category does not move.
        out["null_word_lookup_ladder_independent"] = mk(over={"intact_a": "lookup", "intact_b": "lookup",
                                                              "lesion_affect": "lookup_lesion", "permuted": "lookup_perm"})
        # RESIDUAL, reported never thresholded: the same lookup, but the word is printed only while the ladder reads
        # non-neutral. The category then depends on the ladder in the trivial sense (a gate), so the hardened bar admits
        # it; whether the brain or a host table owns the category is the host-share instrument's question.
        out["info_word_lookup_gated_by_ladder"] = mk(over={"intact_a": "lookup", "intact_b": "lookup",
                                                           "permuted": "lookup_perm"})
    if group == "inner" and extra_info:
        # NULL (amendment A3): one fixed stored fact on every idle turn (no selection at all).
        out["null_constant_idle_fact"] = mk(over={"intact_a": "constant_idle", "intact_b": "constant_idle",
                                                  "permuted": "constant_idle_perm"})
        # RESIDUAL, reported never thresholded: a host FIFO (always the first taught fact) varies with the exposure, so
        # the hardened bar admits it; who selects is the host-share instrument's question, not this bar's.
        out["info_first_taught_fact"] = mk(over={"intact_a": "first_taught", "intact_b": "first_taught",
                                                 "permuted": "first_taught_perm"})
    return out


def stage_generators():
    prim = [g for g in G.GROUPS if G.GROUPS[g]["primary"]]
    out = {"perfect": lambda rng: make_stage(rng), "perfect_exact": lambda rng: make_stage(rng, realistic=False)}
    for g in prim:
        out["null_%s_permuted_inert" % g] = (lambda g: lambda rng: make_stage(rng, {g: {"permuted": "intact"}}))(g)
        out["null_%s_random" % g] = (lambda g: lambda rng: _stage_with(rng, g))(g)
        for l in G.GROUPS[g]["lesions"]:
            out["null_%s_%s_inert" % (g, l)] = (lambda g, l: lambda rng: make_stage(rng, {g: {l: "intact"}}))(g, l)
        for o in G.order_arms(g):                     # amendment A4 (b)
            out["null_%s_%s_order_carried" % (g, o)] = (lambda g, o: lambda rng: make_stage(rng, {g: {o: "da_throttled"}}))(g, o)
    return out


def _stage_with(rng, bad_group):
    d = make_stage(rng)
    d[bad_group] = make_group(rng, bad_group, degenerate="random")
    return d


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        b = selfcheck()
        print("selfcheck: %d mismatches between the in-memory and the on-disk aggregate" % b)
        sys.exit(1 if b else 0)
