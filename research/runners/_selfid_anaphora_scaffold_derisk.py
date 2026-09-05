"""SELF/IDENTITY + ANAPHORA-MISS scaffold-retirement DE-RISK (scaffold_retirement_backlog.md rank 13).

THE RESIDUAL. The 2026-08-12 CHOOSE-1 integration made a factual-SVO question's (agent, action) COMPREHENSION
NEURAL (`ChatBrain._neural_question_parse`, the on-brain `BridgeParser.role_of`) and AUTHORITATIVE: a comprehended
parse feeds the substrate recall (+, since 2026-08-13, the GNW N-organ ignition-bus combiner,
`webapp/gnw_bus_shadow.py`, installed UNCONDITIONALLY on the production turn by `webapp/server.py::brain_reply`);
a DECLINED parse honestly ABSTAINS instead of falling to `QuestionRouter.match_fact`'s role-blind keyword
bag-of-words. That finding's own "Honest scope" and `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s content-selection
row both name the SAME residual verbatim: "the router... still owns self/identity + the anaphora-fallback".

THIS DE-RISK extends the SAME already-proven recipe to that residual (two flags, both default OFF, in
`research/runners/brain_chat_tui.py` — see the module comment above `_neural_selfid_enabled`):

  BRAIN_NEURAL_SELFID:
    (a) self-referential FACTUAL SVO ("what do you use?") -- resolves the self-alias to 'brain' BEFORE
        `_extract_route`'s has_self_alias gate, so it reaches the SAME on-brain `_neural_question_parse` (+ the
        SAME GNW bus) any other factual-SVO question uses. Genuinely neural (BridgeParser.role_of).
    (b) bare identity ("what are you?" / "who are you?") -- extends `_definitional_copula_route`'s EXISTING
        'what is X?' -> [X, 'isa'] comprehension-helper to a self-alias subject, plus a MISS-ONLY
        candidate-relation retry (has/have/is/uses/use, the HOST router's OWN preference order) mirrored at BOTH
        call sites that can author a covered-class answer: `ChatBrain._substrate_recall` (the plain/non-bus gate)
        and `gnw_bus_shadow.gate_via_bus` (the actual production combiner). NOT a neural-parser claim for this
        shape -- a host regex/preference-list comprehension helper, same honesty class as the pre-existing
        copula/relation-fronted/kb-relation routes.
  BRAIN_NEURAL_ANAPHORA_ABSTAIN:
    an anaphora-resolved query the substrate/bus can't confirm ABSTAINS instead of falling to the host router's
    keyword "rescue" of a possibly-wrong WM referent (the SAME honesty as the pre-existing direct-query abstain).

THIS RUNNER CHECKS, per the task's own (a)/(b)/(c):
  (a) CORRECT comprehension: self-factual + self-identity questions route to the RIGHT recall, flag ON.
  (b) HONEST abstain: an anaphora-miss abstains (flag ON) instead of the host router's keyword-confab (flag OFF
      reproduces the confab, on THIS tiny-demo fixture -- "what does it fly?" after "dog chase cat" wrongly
      re-answers "dog chase cat").
  (c) LOAD-BEARING + regression: (c1) a genuine BridgeParser LESION (role_of -> a junk role, the EXACT 2026-08-12
      recipe) collapses the self-factual answer to abstain (class (a) only -- the ONLY class here that reaches a
      genuinely spiking mechanism; onebrain composer required, --onebrain, slow); (c2) a RETIREMENT proof (call
      counters on `_gate_router_combine` / `QuestionRouter.match_fact`) that flag-ON classes are answered WITHOUT
      the host router ever running, on BOTH the plain gate() and the installed GNW-bus gate() (the actual
      production combiner); (c3) NO REGRESSION -- every pre-existing class (stored/unstored/anaphora-HIT) stays
      byte-identical flag-on vs flag-off.
  SEED SCOPE: the fast (rf composer) battery runs the full project 6-seed panel (42/43/44/100/101/102). The slow
  (onebrain composer) lesion battery is --onebrain-gated (~180s/build); it also runs all 6 seeds when passed,
  documented honestly if truncated for time.

Run (numpy-CPU, fast rf recall path -- classes a/b/(c2)/(c3)):
  SIM_BACKEND=numpy python -u -m research.runners._selfid_anaphora_scaffold_derisk
Add the slow onebrain lesion battery (class c1, genuinely-neural BridgeParser lesion):
  SIM_BACKEND=numpy python -u -m research.runners._selfid_anaphora_scaffold_derisk --onebrain
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")

from tools.lab import attributable_to, void_if   # noqa: E402
from tools.verdict import Verdict   # noqa: E402

SEEDS = [42, 43, 44, 100, 101, 102]

# ── panels ──────────────────────────────────────────────────────────────────────────────────────────────────
# (a) self-referential FACTUAL SVO -- tiny-demo's OWN self-facts (`_build_tiny_demo`), asked via self-alias phrasing.
SELF_FACTUAL = [
    ("what do you use?", ["brain", "use", "spikes"]),
    ("what do you learn?", ["brain", "learn", "words"]),
    ("what do you store?", ["brain", "store", "memory"]),
]
# (b) bare self/identity -- tiny-demo has NO 'brain isa X' fact, so this exercises the candidate-relation retry.
# Expected answer matches the HOST router's OWN preference order (has/have/is/uses/use, first hit wins) on the
# SAME facts -- 'use' is the only stored action among those five, giving ['brain','use','spikes'] either way
# (verified against `_gnw_bus_scaffold_retire_verify.py`'s OWN SELF fixture: self_svo == ["brain","use","spikes"]).
SELF_IDENTITY = [
    ("what are you", ["brain", "use", "spikes"]),
    ("who are you", ["brain", "use", "spikes"]),
]
# no-regression controls (byte-identical flag-on vs flag-off expected)
STORED = [
    ("what does dog chase?", ["dog", "chase", "cat"]),
    ("what does cat eat?", ["cat", "eat", "fish"]),
    ("what does brain learn?", ["brain", "learn", "words"]),
]
UNSTORED = ["what does fish fly?", "what does ball roll?"]
ANAPHORA_HIT_SEQ = [("what does dog chase?", ["dog", "chase", "cat"]), ("what does it eat?", ["cat", "eat", "fish"])]
# (b)-anaphora-miss: 'it' resolves to 'cat' (the just-established referent); cats have no 'fly' fact, so this is a
# well-formed-but-unanswerable anaphora-resolved query -- FLAG OFF reproduces the exact host-router keyword-confab
# shape the direct-query fix already retired (verified below to be a REAL confab on this fixture, not assumed).
ANAPHORA_MISS_SEQ = [("what does dog chase?", ["dog", "chase", "cat"]), ("what does it fly?", None)]


def _svo_eq(x, y) -> bool:
    if x is None and y is None:
        return True
    if x is None or y is None:
        return False
    return list(x) == list(y)


def _set_flags(selfid: bool, anaphora_abstain: bool):
    os.environ["BRAIN_NEURAL_SELFID"] = "1" if selfid else "0"
    os.environ["BRAIN_NEURAL_ANAPHORA_ABSTAIN"] = "1" if anaphora_abstain else "0"


def _build(seed, flags_on: bool, with_bus: bool):
    """A FRESH production ChatBrain (rf composer, numpy-CPU). `with_bus=True` installs the GNW ignition bus
    (`gnw_bus_shadow.install_bus_gate`) -- the ACTUAL unconditional production combiner
    (`webapp/server.py::brain_reply`) -- so `chat.gate` is the real `/api/brain-chat` entry point, not just the
    plain host `ChatBrain.gate`. BRAIN_LTM_SHIP_DEFAULT=off skips the (unrelated) bulk-knowledge LTM bundle
    `_build_chat_brain` otherwise attaches by default -- irrelevant to this de-risk's tiny-demo fixture facts and
    VERY expensive to rebuild per probe (each attach builds its own tens-of-thousands-of-neuron bridge); disabling
    it is a pure speed win with no effect on the mechanism under test (`_extract_route` / `_definitional_copula_route`
    / `_substrate_recall` / `gate_via_bus` never consult the LTM for these fixture facts either way)."""
    os.environ["BRAIN_COMPOSER_KIND"] = "rf"          # the numpy fast-path recall (a real production path)
    os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "off"
    _set_flags(flags_on, flags_on)
    from webapp.server import _build_chat_brain
    from webapp import gnw_bus_shadow as gbs
    chat, _src = _build_chat_brain("tiny-demo", "stub")
    if with_bus:
        gbs.install_bus_gate(chat)
    return chat, gbs


def _call_counts(chat):
    """Wrap `_gate_router_combine` + `QuestionRouter.match_fact` with counters (restored by the returned
    closure), mirroring `_gnw_bus_scaffold_retire_verify.py`'s OWN retirement-proof instrumentation -- proof that
    an answer was authored WITHOUT the host router running, not merely that it agrees with what the router
    would have said."""
    counts = {"_gate_router_combine": 0, "match_fact": 0}
    orig_rc = chat._gate_router_combine
    orig_mf = chat.router.match_fact

    def _rc(q, *a, **k):
        counts["_gate_router_combine"] += 1
        return orig_rc(q, *a, **k)

    def _mf(q, *a, **k):
        counts["match_fact"] += 1
        return orig_mf(q, *a, **k)

    chat._gate_router_combine = _rc
    chat.router.match_fact = _mf

    def _restore():
        chat._gate_router_combine = orig_rc
        chat.router.match_fact = orig_mf

    return counts, _restore


def _build_and_batch(seed, flags_on, with_bus, questions):
    """Build ONE fresh chat for (seed, flags_on, with_bus) and run EVERY stateless question in `questions`
    against it, returning {question: (svo, per_question_call_counts)}. Safe to batch (a single build instead of
    one per question): none of these are pronoun-led, so an earlier answer's discourse-WM referent write
    (`_note_referent`) cannot perturb a LATER stateless answer in this same list -- only a genuinely stateful
    pronoun SEQUENCE (handled separately by `_stateful_probe`, always its OWN fresh build) depends on WM state."""
    chat, _gbs = _build(seed, flags_on, with_bus)
    counts, restore = _call_counts(chat)
    out = {}
    try:
        for q in questions:
            counts["_gate_router_combine"] = 0
            counts["match_fact"] = 0
            svo = chat.gate(q)
            out[q] = (svo, dict(counts))
    finally:
        restore()
    return out


def _stateful_probe(seed, flags_on, with_bus, seq):
    chat, _gbs = _build(seed, flags_on, with_bus)
    counts, restore = _call_counts(chat)
    try:
        answers = [chat.gate(utt) for utt, _want in seq]
        return answers, dict(counts)
    finally:
        restore()


def _eval_combiner(seed, with_bus):
    """Run the full battery for ONE combiner ('plain' host gate() or the installed GNW 'bus' gate()), both flag
    arms, at ONE seed. 6 builds total (2 stateless batches + 4 stateful sequences), not one build per question."""
    rows = {"combiner": ("bus" if with_bus else "plain")}

    stateless_qs = ([q for q, _w in SELF_FACTUAL] + [q for q, _w in SELF_IDENTITY]
                    + [q for q, _w in STORED] + list(UNSTORED))
    off_batch = _build_and_batch(seed, False, with_bus, stateless_qs)
    on_batch = _build_and_batch(seed, True, with_bus, stateless_qs)

    self_factual = []
    for q, want in SELF_FACTUAL:
        off_svo, _off_c = off_batch[q]
        on_svo, on_c = on_batch[q]
        self_factual.append({
            "q": q, "want": want, "off_svo": off_svo, "on_svo": on_svo,
            "off_correct": _svo_eq(off_svo, want),         # sanity: today's host answer (via router) is ALSO right
            "on_correct": _svo_eq(on_svo, want),
            "retired": bool(_svo_eq(on_svo, want) and on_c["_gate_router_combine"] == 0 and on_c["match_fact"] == 0),
        })
    rows["self_factual"] = self_factual

    self_identity = []
    for q, want in SELF_IDENTITY:
        off_svo, off_c = off_batch[q]
        on_svo, on_c = on_batch[q]
        self_identity.append({
            "q": q, "want": want, "off_svo": off_svo, "on_svo": on_svo,
            "off_via_router": off_c["match_fact"] >= 1,     # today: the host router IS how this gets answered
            "on_correct": _svo_eq(on_svo, want),
            "retired": bool(_svo_eq(on_svo, want) and on_c["_gate_router_combine"] == 0 and on_c["match_fact"] == 0),
        })
    rows["self_identity"] = self_identity

    off_ans, off_c = _stateful_probe(seed, False, with_bus, ANAPHORA_MISS_SEQ)
    on_ans, on_c = _stateful_probe(seed, True, with_bus, ANAPHORA_MISS_SEQ)
    rows["anaphora_miss"] = {
        "seq": [u for u, _ in ANAPHORA_MISS_SEQ],
        "off_first_turn": off_ans[0], "off_confab": off_ans[-1],
        "on_first_turn": on_ans[0], "on_answer": on_ans[-1],
        "off_confabulated": off_ans[-1] is not None,        # the shape being retired: an unanswerable query got a
                                                            # WRONG answer (no ('cat','fly',*) fact was ever taught)
        "on_abstains": on_ans[-1] is None,
        "first_turn_unaffected": _svo_eq(off_ans[0], on_ans[0]) and _svo_eq(off_ans[0], ANAPHORA_MISS_SEQ[0][1]),
        "retired": bool(on_ans[-1] is None and on_c["_gate_router_combine"] == 0),
    }

    regression = []
    for q, want in STORED:
        off_svo, _ = off_batch[q]
        on_svo, _ = on_batch[q]
        regression.append({"cls": "stored", "q": q, "off": off_svo, "on": on_svo,
                           "identical": _svo_eq(off_svo, on_svo), "correct": _svo_eq(off_svo, want)})
    for q in UNSTORED:
        off_svo, _ = off_batch[q]
        on_svo, _ = on_batch[q]
        regression.append({"cls": "unstored", "q": q, "off": off_svo, "on": on_svo,
                           "identical": _svo_eq(off_svo, on_svo), "moat_ok": (off_svo is None and on_svo is None)})
    off_hit, _ = _stateful_probe(seed, False, with_bus, ANAPHORA_HIT_SEQ)
    on_hit, _ = _stateful_probe(seed, True, with_bus, ANAPHORA_HIT_SEQ)
    regression.append({"cls": "anaphora_hit_seq", "q": "dog chase cat / it eat",
                       "off": off_hit, "on": on_hit,
                       "identical": all(_svo_eq(a, b) for a, b in zip(off_hit, on_hit)),
                       "correct": all(_svo_eq(a, w) for a, (_u, w) in zip(off_hit, ANAPHORA_HIT_SEQ))})
    rows["regression"] = regression
    return rows


def _lesion_selfid(seed):
    """LOAD-BEARING (class (a) ONLY -- the sole genuinely-neural mechanism here): lesion the on-brain
    `BridgeParser` (`role_of` -> a fixed junk role, the EXACT 2026-08-12 CHOOSE-1 recipe) and confirm the
    self-factual answer COLLAPSES to abstain. Requires the SLOW onebrain composer (the only composer carrying a
    `.parser`) -- not run in the default sweep."""
    os.environ["BRAIN_COMPOSER_KIND"] = "onebrain"
    os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "off"      # see `_build`'s docstring -- irrelevant + expensive here
    _set_flags(True, False)
    from webapp.server import _build_chat_brain
    chat, _src = _build_chat_brain("tiny-demo", "stub")
    q, want = "what do you use?", ["brain", "use", "spikes"]
    intact = chat.gate(q)
    parser = chat.inner.composer.parser
    orig_role_of = parser.role_of
    parser.role_of = lambda *a, **k: "junk_role"          # the 2026-08-12 lesion recipe verbatim
    try:
        lesioned = chat.gate(q)
    finally:
        parser.role_of = orig_role_of
    reflex = chat.inner.composer.query_patient("brain", "use")   # the workspace/parser-INDEPENDENT recall reflex
    intact_correct = _svo_eq(intact, want)
    lesion_collapses = (lesioned is None)
    attribution = attributable_to("self-factual answer owed to the on-brain BridgeParser (not a host fallback)",
                                  1.0 if intact_correct else 0.0, 0.0 if lesion_collapses else 1.0)
    return {"seed": seed, "q": q, "want": want,
            "intact": (list(intact) if intact is not None else None),
            "lesioned": (list(lesioned) if lesioned is not None else None),
            "reflex": reflex, "intact_correct": intact_correct, "lesion_collapses": lesion_collapses,
            "reflex_survives": (reflex == "spikes"), "attribution_to_parser": attribution}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onebrain", action="store_true",
                    help="ALSO run the slow onebrain BridgeParser-lesion battery (class c1, ~180s/build/seed)")
    ap.add_argument("--onebrain-seeds", type=int, default=6,
                    help="how many of the 6 canonical seeds to run the onebrain lesion battery on (default: all 6)")
    args = ap.parse_args()

    fast_rows = {"plain": [], "bus": []}
    for seed in SEEDS:
        fast_rows["plain"].append(_eval_combiner(seed, with_bus=False))
        fast_rows["bus"].append(_eval_combiner(seed, with_bus=True))

    def _agg(combiner_rows, key, subkey="retired"):
        vals = []
        for row in combiner_rows:
            item = row[key]
            if isinstance(item, list):
                vals.extend(bool(x[subkey]) for x in item)
            else:
                vals.append(bool(item[subkey]))
        return vals

    def _all_true(vals):
        return bool(vals) and all(vals)

    summary = {}
    for combiner in ("plain", "bus"):
        rows = fast_rows[combiner]
        self_factual_retired = _agg(rows, "self_factual")
        self_identity_retired = _agg(rows, "self_identity")
        anaphora_miss_retired = _agg(rows, "anaphora_miss")
        self_factual_correct = [x["on_correct"] for row in rows for x in row["self_factual"]]
        self_identity_correct = [x["on_correct"] for row in rows for x in row["self_identity"]]
        off_confab_seen = [row["anaphora_miss"]["off_confabulated"] for row in rows]
        on_abstains = [row["anaphora_miss"]["on_abstains"] for row in rows]
        first_turn_unaffected = [row["anaphora_miss"]["first_turn_unaffected"] for row in rows]
        regression_ok = [r["identical"] for row in rows for r in row["regression"]]
        summary[combiner] = {
            "n_seeds": len(rows),
            "self_factual_correct": _all_true(self_factual_correct),
            "self_factual_retired": _all_true(self_factual_retired),
            "self_identity_correct": _all_true(self_identity_correct),
            "self_identity_retired": _all_true(self_identity_retired),
            "anaphora_off_confabulated_on_this_fixture": _all_true(off_confab_seen),
            "anaphora_on_abstains": _all_true(on_abstains),
            "anaphora_first_turn_unaffected": _all_true(first_turn_unaffected),
            "anaphora_miss_retired": _all_true(anaphora_miss_retired),
            "no_regression": _all_true(regression_ok),
            "n_regression_checks": len(regression_ok),
        }

    lesion = None
    if args.onebrain:
        lesion_rows = [_lesion_selfid(s) for s in SEEDS[: max(1, min(6, args.onebrain_seeds))]]
        os.environ["BRAIN_COMPOSER_KIND"] = "rf"      # restore for any subsequent in-process rf-path call
        lesion = {
            "rows": lesion_rows, "n_seeds": len(lesion_rows),
            "all_intact_correct": _all_true([r["intact_correct"] for r in lesion_rows]),
            "all_lesion_collapses": _all_true([r["lesion_collapses"] for r in lesion_rows]),
            "all_reflex_survives": _all_true([r["reflex_survives"] for r in lesion_rows]),
        }

    panel_void = void_if(not fast_rows["plain"] or not fast_rows["bus"],
                         "the fast panel produced ZERO rows -- the de-risk verdict is UNDEFINED, not a GO")

    plain, bus = summary["plain"], summary["bus"]
    go_core = bool(
        not panel_void
        and plain["self_factual_correct"] and plain["self_factual_retired"]
        and bus["self_factual_correct"] and bus["self_factual_retired"]
        and plain["self_identity_correct"] and plain["self_identity_retired"]
        and bus["self_identity_correct"] and bus["self_identity_retired"]
        and plain["anaphora_off_confabulated_on_this_fixture"] and plain["anaphora_on_abstains"]
        and bus["anaphora_off_confabulated_on_this_fixture"] and bus["anaphora_on_abstains"]
        and plain["anaphora_first_turn_unaffected"] and bus["anaphora_first_turn_unaffected"]
        and plain["no_regression"] and bus["no_regression"]
    )
    go_lesion = True if lesion is None else bool(
        lesion["all_intact_correct"] and lesion["all_lesion_collapses"] and lesion["all_reflex_survives"])
    go = bool(go_core and go_lesion)

    v = Verdict("rank-13 de-risk: on-brain comprehension + GNW bus extended to self/identity + the anaphora-miss")
    v.require("self-factual SVO ('what do you use?' etc): flag-ON answers CORRECTLY, plain gate()",
              plain["self_factual_correct"], expect=True)
    v.require("self-factual SVO: flag-ON RETIRES the host router (0 calls), plain gate()",
              plain["self_factual_retired"], expect=True)
    v.require("self-factual SVO: flag-ON answers CORRECTLY, installed GNW-bus gate() (the production combiner)",
              bus["self_factual_correct"], expect=True)
    v.require("self-factual SVO: flag-ON RETIRES the host router, installed GNW-bus gate()",
              bus["self_factual_retired"], expect=True)
    v.require("bare self/identity ('what are you?' etc): flag-ON answers CORRECTLY, plain gate()",
              plain["self_identity_correct"], expect=True)
    v.require("bare self/identity: flag-ON RETIRES the host router, plain gate()",
              plain["self_identity_retired"], expect=True)
    v.require("bare self/identity: flag-ON answers CORRECTLY, installed GNW-bus gate()",
              bus["self_identity_correct"], expect=True)
    v.require("bare self/identity: flag-ON RETIRES the host router, installed GNW-bus gate()",
              bus["self_identity_retired"], expect=True)
    v.require("anaphora-miss: flag-OFF genuinely CONFABULATES on this fixture (the defect being retired is real)",
              plain["anaphora_off_confabulated_on_this_fixture"] and bus["anaphora_off_confabulated_on_this_fixture"],
              expect=True)
    v.require("anaphora-miss: flag-ON ABSTAINS honestly instead, plain gate() + installed GNW-bus gate()",
              plain["anaphora_on_abstains"] and bus["anaphora_on_abstains"], expect=True)
    v.require("anaphora-HIT (legitimate anaphora recall) is UNAFFECTED by the flag",
              plain["anaphora_first_turn_unaffected"] and bus["anaphora_first_turn_unaffected"], expect=True)
    v.require("NO REGRESSION on stored/unstored/anaphora-hit, flag-on vs flag-off, both combiners",
              plain["no_regression"] and bus["no_regression"], expect=True,
              note=f"{plain['n_regression_checks']}+{bus['n_regression_checks']} checks")
    if lesion is not None:
        v.require("LOAD-BEARING: lesioning BridgeParser.role_of collapses the self-factual answer to abstain",
                  lesion["all_lesion_collapses"], expect=True, note=f"n_seeds={lesion['n_seeds']}")
        v.require("intact (non-lesioned) onebrain self-factual answer is correct",
                  lesion["all_intact_correct"], expect=True)
        v.require("the parser-INDEPENDENT recall reflex (query_patient) survives the lesion (dissociation)",
                  lesion["all_reflex_survives"], expect=True)
        v.control("lesion dissociation (self-factual answer needs the on-brain parser, not a host fallback)",
                  treatment=(1.0 if lesion["all_intact_correct"] else 0.0),
                  control=(0.0 if lesion["all_lesion_collapses"] else 1.0), min_separation=0.0)
    else:
        v.disabled("BridgeParser lesion battery (class c1, genuinely-neural load-bearing proof)",
                  why="--onebrain not passed this run; class (a) is the ONLY class here with a spiking mechanism "
                      "to lesion -- (b)/(anaphora-miss) are host comprehension-helper / control-flow extensions "
                      "(see the module docstring), so their load-bearing evidence is the flag-toggle + "
                      "retirement-proof counters above, not a spiking lesion.")
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])

    print("\n" + "=" * 112, flush=True)
    print("  RANK-13 DE-RISK — self/identity + anaphora-miss extension of the on-brain parser + GNW bus", flush=True)
    print("=" * 112, flush=True)
    for combiner in ("plain", "bus"):
        s = summary[combiner]
        print(f"  [{combiner:5s}] self_factual correct={s['self_factual_correct']} retired={s['self_factual_retired']}"
              f" | self_identity correct={s['self_identity_correct']} retired={s['self_identity_retired']}"
              f" | anaphora off_confab={s['anaphora_off_confabulated_on_this_fixture']} "
              f"on_abstains={s['anaphora_on_abstains']} first_turn_ok={s['anaphora_first_turn_unaffected']}"
              f" | no_regression={s['no_regression']} (n={s['n_regression_checks']})", flush=True)
    for row in fast_rows["plain"] + fast_rows["bus"]:
        seed_tag = f"{row['combiner']}"
        for x in row["self_factual"]:
            print(f"      [{seed_tag}] self_factual  {x['q']:20s} off={x['off_svo']} on={x['on_svo']} "
                  f"on_correct={x['on_correct']} retired={x['retired']}", flush=True)
        for x in row["self_identity"]:
            print(f"      [{seed_tag}] self_identity {x['q']:20s} off={x['off_svo']} on={x['on_svo']} "
                  f"on_correct={x['on_correct']} retired={x['retired']}", flush=True)
        am = row["anaphora_miss"]
        print(f"      [{seed_tag}] anaphora_miss off_confab={am['off_confab']} on_answer={am['on_answer']} "
              f"retired={am['retired']}", flush=True)
    if lesion is not None:
        print(f"  LESION (onebrain, n_seeds={lesion['n_seeds']}): "
              f"intact_correct={lesion['all_intact_correct']} lesion_collapses={lesion['all_lesion_collapses']} "
              f"reflex_survives={lesion['all_reflex_survives']}", flush=True)
        for r in lesion["rows"]:
            print(f"      seed={r['seed']:4d} intact={r['intact']} lesioned={r['lesioned']} reflex={r['reflex']!r}",
                  flush=True)
    verdict = "GO (de-risk earned; both flags default OFF -- see the finding for scope)" if go else "NO-GO / CHARACTERIZED"
    print(f"\n  VERDICT: {verdict}\n" + "=" * 112, flush=True)

    out = {"runner": "_selfid_anaphora_scaffold_derisk", "go": go, "status": decided["status"],
           "seeds": SEEDS, "summary": summary, "fast_rows": fast_rows, "lesion": lesion,
           "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
           "undefined_reasons": decided["undefined_reasons"]}
    op = "research/findings/raw/_selfid_anaphora_scaffold_derisk/result.json"
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  [saved] {op}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
