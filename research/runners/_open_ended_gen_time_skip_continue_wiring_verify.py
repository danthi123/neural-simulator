"""WIRING SANITY for the 2026-08-28 SKIP-AND-CONTINUE extension of the token-id continuation added to
`_open_ended_gen_time_consensus_veto_derisk.py` (`_generate_tokenid_continuation_skip`, `skip_continue=` on
`generate_with_generation_time_veto`, `BRAIN_HONESTY_SKIP_CONTINUE` in `webapp/open_ended_chat.py`). This is the
NEXT rung BOTH the 2026-08-27 PARTIAL finding and the 2026-08-28 token-id-continuation finding named ("skip-and-
continue past a dropped sentence to reach later same-reply residuals") and neither attempted -- see those two
findings' own "Honest scope"/NEXT sections.

NO GPU, NO real Qwen load, NO organs/GNW build -- reuses the SAME fake tokenizer/fake model technique
`_open_ended_gen_time_tokenid_continuation_wiring_verify.py` established (imported verbatim, not re-implemented)
to exercise the REAL, unmodified `_generate_tokenid_continuation` / `_generate_tokenid_continuation_skip` /
`generate_with_generation_time_veto` against the REAL, unmodified `clause_filter_sentence` -- plus a webapp-level
harness (mirroring `_open_ended_gen_time_consensus_veto_wiring_verify.py`'s technique) that exercises the REAL
`webapp.open_ended_chat.answer_turn` / `skip_continue_enabled()` with `generate_with_generation_time_veto`
stubbed, so the env-var -> answer_turn -> mechanism-call routing is checked in isolation from the mechanism
itself.

THE ADVERSARIAL SCRIPT (deliberately UNREPAIRABLE, unlike the border-list sentence the other wiring verify
uses -- this is the case that actually distinguishes skip-and-continue from the existing conservative-truncate
path). Two fixed "sentences" a fake model emits across two `model.generate()` calls:
  S1 = "The capital of Canada is Toronto." -- with `facts=[("capital", "ottawa")]`, `sentence_contradicts`
       flags "wrong capital: toronto"; `clause_filter_sentence` has NO removable span for a wrong-capital
       reason (`_bad_relation_tokens` only locates border/continent spans), so `candidate == original` and it
       returns None -- a FULL DROP, not a repair. This is the honest reason the existing (non-skip) path's own
       controlled unit battery never demonstrates a "later sentence survives a drop" property: its adversarial
       sentences are all REPAIRABLE (a wrong border list-item removed, the correct one kept) -- this file's
       script is the first place that residual is actually tested.
  S2 = "It has ten provinces and three territories." -- no relation clause in scope, `clause_filter_sentence`
       returns it unchanged (kept) -- the LATER, SUPPORTED content the mission's "skip-and-continue" rung names.
  LESIONED (`facts=[]`): S1 has nothing to check against -> kept too (matches the EXISTING coupling-lesion
  property already proven by `_open_ended_gen_time_tokenid_continuation_wiring_verify.py`'s checks; not
  re-claimed as this file's own contribution -- see check (E) below, reported as a sanity cross-check).

THE LOAD-BEARING LEVER THIS FILE PROVES (checks A/B/C): holding the CONSENSUS VETO FIRING constant
(`facts=FACTS_ON`, i.e. S1 IS flagged, on every arm), does `skip_continue=True` vs `skip_continue=False` MOVE
the final accepted text -- specifically, does the LATER sentence (S2, "ten provinces") survive skip_continue=
True and VANISH when skip_continue is lesioned back to False (the pre-existing, still-default truncating
behavior)? `tools.lab.lever` makes the attribution explicit: `skip_continue` is the ONLY term that varies
between the two arms.

    SIM_BACKEND=numpy /home/dant123/Projects/sim/.venv/bin/python -m \
        research.runners._open_ended_gen_time_skip_continue_wiring_verify
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
import logging
logging.disable(logging.INFO)

from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.verdict import Verdict  # noqa: E402
from tools.lab import lever  # noqa: E402
from research.runners._open_ended_clause_contradiction_filter_derisk import clause_filter_sentence  # noqa: E402
import research.runners._open_ended_gen_time_consensus_veto_derisk as GT  # noqa: E402
# reuse-by-import: the EXACT fake tokenizer/model/faculty harness the 2026-08-28 token-id-continuation wiring
# verify established -- not re-implemented here.
from research.runners._open_ended_gen_time_tokenid_continuation_wiring_verify import (  # noqa: E402
    _FakeTok, _FakeModel, _FakeFac, _FakeGen,
)

OUT = _REPO / "research" / "findings" / "raw" / "_open_ended_gen_time_skip_continue_wiring_verify.json"

# The UNREPAIRABLE adversarial sentence (see module docstring for why this, not a border-list sentence, is the
# case that actually exercises skip-and-continue -- a repairable sentence never reaches the `repaired is None`
# branch either implementation is defined by).
S1 = "The capital of Canada is Toronto."
S2 = "It has ten provinces and three territories."
S1_WORDS = S1.split()      # 7 words
S2_WORDS = S2.split()      # 7 words
FACTS_ON = [("capital", "ottawa")]
FACTS_LESIONED: list = []

# sanity: confirm the fixture is genuinely unrepairable/kept as designed BEFORE building any check on top of it
# (a fixture that silently repairs instead of dropping would make every check below vacuously trivial).
assert clause_filter_sentence(S1, "canada", FACTS_ON) is None, "fixture S1 must be a FULL DROP, not a repair"
assert clause_filter_sentence(S1, "canada", FACTS_LESIONED) == S1, "fixture S1 must survive when lesioned"
assert clause_filter_sentence(S2, "canada", FACTS_ON) == S2, "fixture S2 must be kept unchanged (ON)"


def _run_mechanism(*, facts, skip_continue, eos_after_last=True, max_new_tokens=60):
    """Build a FRESH fake tok/model/fac/gen (model call-count is call-scoped) and drive the REAL
    `generate_with_generation_time_veto` dispatcher through the REAL `consensus_facts_for_topic` signature --
    but with `GT.consensus_facts_for_topic` monkeypatched to hand back `facts` directly (never touching the
    live GNW organ buses this file must stay CPU-only/no-organs). Everything AFTER that hand-off --
    `continuation="token_id"` dispatch, `_generate_tokenid_continuation` vs `_generate_tokenid_continuation_skip`
    selection, `clause_filter_sentence`, the token-id plumbing -- runs the REAL, unmodified code."""
    tok = _FakeTok()
    model = _FakeModel(tok, [S1_WORDS + ["It", "has"], S2_WORDS], eos_after_last=eos_after_last)
    fac = _FakeFac(tok, model)
    gen = _FakeGen(fac)

    orig = GT.consensus_facts_for_topic
    GT.consensus_facts_for_topic = lambda chat, topic, seed, **kw: (facts, {})
    try:
        text, trace, info = GT.generate_with_generation_time_veto(
            gen, None, "canada", 42, "system prompt", "user prompt",
            max_new_tokens=max_new_tokens, sentence_budget=30, max_sentences=4,
            continuation="token_id", skip_continue=skip_continue)
    finally:
        GT.consensus_facts_for_topic = orig
    return text, trace, model


def main():
    checks = {}

    # ---- (A) skip_continue=True, coupling ON: S1 dropped-and-skipped, S2 (the LATER, supported content) is
    # STILL reached and kept -- the load-bearing "reaches the residuals AFTER the dropped sentence" property. ----
    text_skip, trace_skip, model_skip = _run_mechanism(facts=FACTS_ON, skip_continue=True)
    checks["A_two_attempts_traced"] = len(trace_skip) == 2
    checks["A_s1_dropped_skip_not_stop"] = trace_skip[0]["action"] == "dropped_skip" and trace_skip[0]["kept"] is None
    checks["A_s2_kept_unchanged"] = trace_skip[1]["action"] == "kept" and trace_skip[1]["kept"] == S2
    checks["A_final_text_is_s2_only"] = text_skip.strip() == S2
    checks["A_toronto_excluded"] = "toronto" not in text_skip.lower()
    checks["A_provinces_included"] = "provinces" in text_skip.lower()
    checks["A_model_called_twice"] = model_skip.calls == 2

    # ---- (B) skip_continue=False (the pre-existing, still-default path), SAME coupling ON: S1 dropped ->
    # generation STOPS there -- S2 never reached, never even generated (model called ONCE, not twice). ----
    text_stop, trace_stop, model_stop = _run_mechanism(facts=FACTS_ON, skip_continue=False)
    checks["B_one_attempt_traced"] = len(trace_stop) == 1
    checks["B_s1_dropped_stop"] = trace_stop[0]["action"] == "dropped_stop" and trace_stop[0]["kept"] is None
    checks["B_final_text_empty"] = text_stop.strip() == ""
    checks["B_model_called_once_only"] = model_stop.calls == 1

    # ---- (C) THE LEVER: holding the veto firing constant (facts=FACTS_ON on both arms), skip_continue is the
    # ONLY thing that varies -- does it MOVE the final text (S2 present vs absent)? tools.lab.lever makes the
    # attribution explicit rather than merely observing "both arms differ". This is vary -> differ; check (B)
    # above (skip_continue lesioned back to False) is the lesion -> vanish half of vary/differ/lesion/vanish. ----
    try:
        lever("skip_continue: True accepted-text -> False (lesioned) accepted-text", text_skip, text_stop,
              required=True)
        checks["C_lever_moved"] = True
    except Exception:  # noqa: BLE001 -- tools.lab.LeverError: both arms identical, the lever is void
        checks["C_lever_moved"] = False
    checks["C_diverge"] = text_skip != text_stop
    checks["C_skip_has_what_stop_lacks"] = ("provinces" in text_skip.lower()) and ("provinces" not in text_stop.lower())

    # ---- (D) BYTE-IDENTICAL WHEN OFF, proven by EXECUTION, not by comment: skip_continue=False through the
    # dispatcher must reproduce the pre-existing, untouched `_generate_tokenid_continuation` EXACTLY -- same
    # text, same trace -- since that is literally the function the dispatcher calls when skip_continue=False. ----
    tok_d = _FakeTok()
    model_d = _FakeModel(tok_d, [S1_WORDS + ["It", "has"], S2_WORDS], eos_after_last=True)
    fac_d = _FakeFac(tok_d, model_d)
    gen_d = _FakeGen(fac_d)
    text_direct, trace_direct = GT._generate_tokenid_continuation(
        gen_d, "canada", 42, "system prompt", "user prompt", FACTS_ON,
        max_new_tokens=60, sentence_budget=30, max_sentences=4)
    checks["D_text_matches_pre_existing_fn"] = text_direct == text_stop
    checks["D_trace_matches_pre_existing_fn"] = trace_direct == trace_stop

    # also confirm the DEFAULT (not passing skip_continue at all) equals skip_continue=False explicitly --
    # proves the parameter's own default preserves the exact pre-existing call signature's behavior.
    tok_e = _FakeTok()
    model_e = _FakeModel(tok_e, [S1_WORDS + ["It", "has"], S2_WORDS], eos_after_last=True)
    fac_e = _FakeFac(tok_e, model_e)
    gen_e = _FakeGen(fac_e)
    orig_cft = GT.consensus_facts_for_topic
    GT.consensus_facts_for_topic = lambda chat, topic, seed, **kw: (FACTS_ON, {})
    try:
        text_default, trace_default, _info = GT.generate_with_generation_time_veto(
            gen_e, None, "canada", 42, "system prompt", "user prompt",
            max_new_tokens=60, sentence_budget=30, max_sentences=4, continuation="token_id")
    finally:
        GT.consensus_facts_for_topic = orig_cft
    checks["D_default_param_matches_explicit_false"] = (text_default == text_stop and trace_default == trace_stop)

    # ---- (E) sanity cross-check (NOT this file's own lever -- the coupling lesion is already proven by
    # `_open_ended_gen_time_tokenid_continuation_wiring_verify.py`): with facts=[] (nothing to suppress at all),
    # skip_continue has ZERO effect -- S1 survives either way, confirming this extension only changes behavior
    # WHEN the veto actually fires. ----
    text_les_skip, trace_les_skip, _m1 = _run_mechanism(facts=FACTS_LESIONED, skip_continue=True)
    text_les_stop, trace_les_stop, _m2 = _run_mechanism(facts=FACTS_LESIONED, skip_continue=False)
    checks["E_lesioned_facts_skip_and_stop_identical"] = text_les_skip == text_les_stop == f"{S1} {S2}"
    checks["E_lesioned_toronto_survives_both"] = ("toronto" in text_les_skip.lower()
                                                  and "toronto" in text_les_stop.lower())

    # ---- (F) budget exhaustion under skip_continue=True: a token budget smaller than the full script
    # truncates cleanly, never raises, never loops unboundedly. ----
    try:
        text_short, trace_short, _m = _run_mechanism(facts=FACTS_ON, skip_continue=True, max_new_tokens=8)
        checks["F_budget_exhaustion_no_crash"] = True
        checks["F_budget_exhaustion_bounded"] = len(trace_short) <= 4          # never exceeds max_sentences
    except Exception as exc:  # noqa: BLE001
        checks["F_budget_exhaustion_no_crash"] = False
        checks["F_budget_exhaustion_error"] = repr(exc)

    # =====================================================================================================
    # WEBAPP-LEVEL: env var -> answer_turn -> generate_with_generation_time_veto kwarg routing (mirrors
    # `_open_ended_gen_time_consensus_veto_wiring_verify.py`'s technique -- a stubbed mechanism function, no
    # GPU/organs, isolating the WIRING from the mechanism this file already proved above).
    # =====================================================================================================
    import webapp.open_ended_chat as OE

    checks["G_flag_reader_default_false"] = OE.skip_continue_enabled() is False
    saved_skip_flag = os.environ.pop("BRAIN_HONESTY_SKIP_CONTINUE", None)
    saved_gt_flag = os.environ.pop("BRAIN_OPEN_ENDED_GEN_TIME_HONESTY", None)
    try:
        os.environ["BRAIN_HONESTY_SKIP_CONTINUE"] = "1"
        checks["G_flag_reader_true_on_1"] = OE.skip_continue_enabled() is True
        os.environ["BRAIN_HONESTY_SKIP_CONTINUE"] = "true"
        checks["G_flag_reader_true_on_word"] = OE.skip_continue_enabled() is True
        os.environ["BRAIN_HONESTY_SKIP_CONTINUE"] = "0"
        checks["G_flag_reader_false_on_0"] = OE.skip_continue_enabled() is False
    finally:
        os.environ.pop("BRAIN_HONESTY_SKIP_CONTINUE", None)

    class _FakeGenObj:
        def generate(self, system, user, seed=42, max_new_tokens=110):
            return "ONE-SHOT: stub", 0.01

    class _FakeChat:
        marker = "fake-chat-for-skip-continue-wiring-verify"

    fake_gen_obj = _FakeGenObj()
    orig_get_generator = OE.get_generator
    orig_gt_fn = GT.generate_with_generation_time_veto
    calls = []

    def _stub_get_generator(_warm_faculty):
        return fake_gen_obj

    def _stub_gt_veto(gen, chat, topic, seed, system, user, **kw):
        calls.append({"skip_continue": kw.get("skip_continue")})
        return ("GEN-TIME: stub reply", [{"raw": "x", "kept": "x", "action": "kept"}], {})

    OE.get_generator = _stub_get_generator
    GT.generate_with_generation_time_veto = _stub_gt_veto

    def _turn(*, skip_flag):
        if skip_flag is None:
            os.environ.pop("BRAIN_HONESTY_SKIP_CONTINUE", None)
        else:
            os.environ["BRAIN_HONESTY_SKIP_CONTINUE"] = "1" if skip_flag else "0"
        os.environ["BRAIN_OPEN_ENDED_GEN_TIME_HONESTY"] = "1"
        return OE.answer_turn("tell me about canada", None, 0.1, 0.4, ltm_bundle=None,
                              brain_bundle=None, seed=42, max_new_tokens=110, chat=_FakeChat())

    try:
        # need `known=True`: answer_turn's retrieve() needs a nonempty by_agent index. Patch build_index/retrieve
        # via a tiny stub instead of a real bundle on disk (keeps this file CPU/memory-trivial).
        orig_build_index = OE.build_index
        OE.build_index = lambda *_a, **_k: {"canada": [("canada", "capital", "ottawa")]}
        try:
            calls.clear()
            r_default = _turn(skip_flag=None)
            calls_default = list(calls)
            calls.clear()
            r_false = _turn(skip_flag=False)
            calls_false = list(calls)
            calls.clear()
            r_true = _turn(skip_flag=True)
            calls_true = list(calls)
        finally:
            OE.build_index = orig_build_index
    finally:
        OE.get_generator = orig_get_generator
        GT.generate_with_generation_time_veto = orig_gt_fn
        os.environ.pop("BRAIN_HONESTY_SKIP_CONTINUE", None)
        os.environ.pop("BRAIN_OPEN_ENDED_GEN_TIME_HONESTY", None)
        if saved_skip_flag is not None:
            os.environ["BRAIN_HONESTY_SKIP_CONTINUE"] = saved_skip_flag
        if saved_gt_flag is not None:
            os.environ["BRAIN_OPEN_ENDED_GEN_TIME_HONESTY"] = saved_gt_flag

    checks["H_known_true_all_turns"] = (r_default["known"] is True and r_false["known"] is True
                                        and r_true["known"] is True)
    checks["H_gen_time_used_all_turns"] = (r_default["gen_time_honesty_used"] is True
                                           and r_false["gen_time_honesty_used"] is True
                                           and r_true["gen_time_honesty_used"] is True)
    # THE routing check: the env var, read fresh each turn by `skip_continue_enabled()` inside `answer_turn`,
    # must land as the EXACT `skip_continue=` kwarg `generate_with_generation_time_veto` is called with --
    # unset (env var absent) and explicit "0" both -> False; "1" -> True.
    checks["H_default_env_kwarg_false"] = calls_default == [{"skip_continue": False}]
    checks["H_flag_false_kwarg_false"] = calls_false == [{"skip_continue": False}]
    checks["H_flag_true_kwarg_true"] = calls_true == [{"skip_continue": True}]
    checks["H_default_matches_explicit_false"] = calls_default == calls_false

    v = Verdict("SKIP-AND-CONTINUE (_generate_tokenid_continuation_skip / skip_continue= / "
               "BRAIN_HONESTY_SKIP_CONTINUE) is wired correctly against the REAL, unmodified "
               "clause_filter_sentence and generate_with_generation_time_veto dispatcher, and routes correctly "
               "from the env var through webapp.open_ended_chat.answer_turn -- fake tokenizer/model, no GPU, "
               "no Qwen, no organs")
    v.require("(A) skip_continue=True + veto firing: the dropped sentence is skipped (not stopped) and the "
              "LATER, supported sentence is still reached and kept",
              all([checks["A_two_attempts_traced"], checks["A_s1_dropped_skip_not_stop"],
                   checks["A_s2_kept_unchanged"], checks["A_final_text_is_s2_only"],
                   checks["A_toronto_excluded"], checks["A_provinces_included"],
                   checks["A_model_called_twice"]]), expect=True)
    v.require("(B) skip_continue=False, same veto firing: generation stops at the drop, the later sentence is "
              "never reached (matches the pre-existing conservative-truncate contract)",
              all([checks["B_one_attempt_traced"], checks["B_s1_dropped_stop"], checks["B_final_text_empty"],
                   checks["B_model_called_once_only"]]), expect=True)
    v.require("(C) the load-bearing lever: skip_continue is the ONLY varying term and it MOVES the final text "
              "(vary -> differ; (B) is the lesion -> vanish half)",
              checks["C_lever_moved"] and checks["C_diverge"] and checks["C_skip_has_what_stop_lacks"],
              expect=True)
    v.require("(D) flag OFF is byte-identical to the pre-existing untouched _generate_tokenid_continuation, "
              "AND to not passing skip_continue at all (proven by execution, not by comment)",
              checks["D_text_matches_pre_existing_fn"] and checks["D_trace_matches_pre_existing_fn"]
              and checks["D_default_param_matches_explicit_false"], expect=True)
    v.require("(E) sanity: with nothing for the veto to suppress, skip_continue has zero effect either way",
              checks["E_lesioned_facts_skip_and_stop_identical"] and checks["E_lesioned_toronto_survives_both"],
              expect=True)
    v.require("(F) a token budget smaller than the full script truncates cleanly under skip_continue=True, "
              "never raises, never loops unboundedly", checks["F_budget_exhaustion_no_crash"]
              and checks["F_budget_exhaustion_bounded"], expect=True)
    v.require("(G) BRAIN_HONESTY_SKIP_CONTINUE flag reader: default False, truthy strings -> True, '0' -> False",
              all([checks["G_flag_reader_default_false"], checks["G_flag_reader_true_on_1"],
                   checks["G_flag_reader_true_on_word"], checks["G_flag_reader_false_on_0"]]), expect=True)
    v.require("(H) answer_turn routes through the full stack (env var -> skip_continue_enabled() -> "
              "generate_with_generation_time_veto kwarg) without breaking the existing gen-time-honesty gate, "
              "and the env var lands as the EXACT skip_continue= kwarg the mechanism sees",
              all([checks["H_known_true_all_turns"], checks["H_gen_time_used_all_turns"],
                   checks["H_default_env_kwarg_false"], checks["H_flag_false_kwarg_false"],
                   checks["H_flag_true_kwarg_true"], checks["H_default_matches_explicit_false"]]), expect=True)

    go = all(checks.get(k) for k in (
        "A_two_attempts_traced", "A_s1_dropped_skip_not_stop", "A_s2_kept_unchanged", "A_final_text_is_s2_only",
        "A_toronto_excluded", "A_provinces_included", "A_model_called_twice",
        "B_one_attempt_traced", "B_s1_dropped_stop", "B_final_text_empty", "B_model_called_once_only",
        "C_lever_moved", "C_diverge", "C_skip_has_what_stop_lacks",
        "D_text_matches_pre_existing_fn", "D_trace_matches_pre_existing_fn",
        "D_default_param_matches_explicit_false",
        "E_lesioned_facts_skip_and_stop_identical", "E_lesioned_toronto_survives_both",
        "F_budget_exhaustion_no_crash", "F_budget_exhaustion_bounded",
        "G_flag_reader_default_false", "G_flag_reader_true_on_1", "G_flag_reader_true_on_word",
        "G_flag_reader_false_on_0", "H_known_true_all_turns", "H_gen_time_used_all_turns",
        "H_default_env_kwarg_false", "H_flag_false_kwarg_false", "H_flag_true_kwarg_true",
        "H_default_matches_explicit_false"))
    decided = v.decide(go=go)

    art = {
        "probe": "open_ended_gen_time_skip_continue_wiring_verify",
        "backend": "numpy(fake tokenizer/model, no GPU/Qwen/organs)",
        "checks": checks,
        "text_skip_continue_ON": text_skip, "trace_skip_continue_ON": trace_skip,
        "text_skip_continue_LESIONED_ie_OFF": text_stop, "trace_skip_continue_LESIONED_ie_OFF": trace_stop,
        "answer_default_env": r_default["answer"], "answer_flag_false": r_false["answer"],
        "answer_flag_true": r_true["answer"],
        "verdict": decided, "preconditions": decided.get("preconditions", []), "GO": bool(go),
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(art, indent=1))
    print(json.dumps(checks, indent=1, default=str))
    print(f"ON  (skip_continue=True ) accepted text: {text_skip!r}")
    print(f"OFF (skip_continue=False) accepted text: {text_stop!r}")
    print(f"wrote {OUT} -> {decided['status']}")
    return decided["status"]


if __name__ == "__main__":
    main()
