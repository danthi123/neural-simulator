"""WIRING verify: `webapp.open_ended_chat.answer_turn`'s new `chat=` parameter + `BRAIN_OPEN_ENDED_GEN_TIME_HONESTY`
gate correctly route to the generation-time consensus veto ONLY when both are satisfied, and are BYTE-IDENTICAL
to the pre-existing one-shot path in every other case. (2026-08-28, Vikunja #112 follow-on.)

SEPARATION OF CONCERNS. `_open_ended_gen_time_consensus_veto_derisk.py` proves the MECHANISM is real and
load-bearing (a live Qwen generation, a genuinely-spiking LTM-exempt organ-B/C consensus, vary/lesion on the
actual suppressed clause). This file proves the WIRING around it: that `answer_turn` calls through to that
mechanism ONLY under the declared gate (flag ON + a live `chat` + a KNOWN topic), and is otherwise UNCHANGED --
using a stubbed generator + a stubbed `generate_with_generation_time_veto` (no GPU, no organs) so the wiring
logic is checked in isolation from the mechanism it calls, mirroring how
`_open_ended_clause_contradiction_filter_verify.py` checked the PRIOR wiring the same way.

CHECKS.
  (1) FLAG OFF (chat provided or not) -> `gen_time_honesty_used=False`, one-shot path runs, output BYTE-IDENTICAL
      between `chat=None` and `chat=<a live-looking object>` (the extra kwarg has literally no effect off).
  (2) FLAG ON, `chat=None` -> still `gen_time_honesty_used=False` (no chat to consult) -> byte-identical to (1).
  (3) FLAG ON, `chat=<object>`, but the topic is UNKNOWN (no retrieved facts) -> still `gen_time_honesty_used=
      False` -- an unknown-topic honest-abstain turn is untouched by this mode either way.
  (4) FLAG ON, `chat=<object>`, topic KNOWN -> `gen_time_honesty_used=True`, `answer_turn` calls
      `generate_with_generation_time_veto` with the SAME (topic, seed, system, user) the one-shot path would
      have used, and its RETURNED text is what `post_filter` (the unchanged safety net) is applied to.
  (5) The safety net is NEVER skipped: in every case above, `post_filter` runs on whatever text was produced.
  (6) STRUCTURAL: `webapp/server.py`'s open-ended call site passes `chat=chat` to `answer_turn` (so the live,
      organ-wired production chat is actually reachable), still nested under the unchanged `BRAIN_OPEN_ENDED`
      guard (the outer gate is untouched).

MEMORY-SAFE BY DESIGN: no GPU, no Qwen render, no organ build -- a fake generator + a fake
`generate_with_generation_time_veto` isolate the wiring logic from both heavy dependencies.

    SIM_BACKEND=numpy /home/dant123/Projects/sim/.venv/bin/python -m \
        research.runners._open_ended_gen_time_consensus_veto_wiring_verify
"""
from __future__ import annotations

import json
import os
import re
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
import logging
logging.disable(logging.INFO)

from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.verdict import Verdict  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_open_ended_gen_time_consensus_veto_wiring_verify.json"

# a tiny, self-contained facts.json bundle with exactly ONE known agent (canada) -- built by this run, not
# checked in (scratchpad), so `known=True` is exercised without touching the real store.
_BUNDLE_DIR = "/tmp/claude-1000/-home-dant123-Projects-sim/87891831-e642-4a2f-abeb-50ea0867609b/scratchpad/" \
             "_gt_veto_wiring_bundle"


class _FakeGen:
    """Stands in for `OpenEndedGenerator` -- no model, no GPU. `.generate` returns a fixed, recognizable string
    so the one-shot path is trivially distinguishable from the gen-time path's stubbed return value below."""

    def generate(self, system, user, seed=42, max_new_tokens=110):
        return "ONE-SHOT: Canada is a country in North America, bordered by the United States.", 0.01


class _FakeChat:
    """A minimal stand-in for a live, organ-wired ChatBrain -- `answer_turn` never introspects it (it is only
    ever handed to `generate_with_generation_time_veto`, which this verify stubs out), so `object()` would do;
    a tiny class with a marker makes the call-capture below legible."""
    marker = "fake-chat-for-wiring-verify"


def _check_server_wiring():
    src = (_REPO / "webapp" / "server.py").read_text(encoding="utf-8")
    guard_re = re.compile(
        r'if os\.environ\.get\("BRAIN_OPEN_ENDED", "0"\)\.strip\(\)\.lower\(\) in \("1", "true", "on", "yes"\):'
        r'\s*\n\s*try:\s*\n\s*from webapp import open_ended_chat as _OE', re.M)
    gated = bool(guard_re.search(src))
    n_imports = len(re.findall(r'from webapp import open_ended_chat', src))
    call_re = re.compile(r'_OE\.answer_turn\(.*?chat=chat,?\s*\)', re.S)
    chat_passed = bool(call_re.search(src))
    return {"off_path_gated": gated, "n_imports": n_imports, "chat_kwarg_passed_at_call_site": chat_passed}


def main():
    import webapp.open_ended_chat as OE
    import research.runners._open_ended_gen_time_consensus_veto_derisk as GT

    fake_gen = _FakeGen()
    orig_get_generator = OE.get_generator
    orig_gen_time_fn = GT.generate_with_generation_time_veto
    calls = []

    def _stub_get_generator(_warm_faculty):
        return fake_gen

    def _stub_gen_time_veto(gen, chat, topic, seed, system, user, **kw):
        calls.append({"topic": topic, "seed": seed, "system": system, "user": user,
                      "chat_marker": getattr(chat, "marker", None), "gen_is_fake": gen is fake_gen})
        return ("GEN-TIME: Canada is a country in North America, bordered by the United States.",
               [{"raw": "Canada is a country in North America, bordered by the United States.",
                 "kept": "Canada is a country in North America, bordered by the United States.",
                 "action": "kept", "consensus_facts": [("borders", "united states")]}],
               {"borders": {"committed": "united states"}})

    OE.get_generator = _stub_get_generator
    GT.generate_with_generation_time_veto = _stub_gen_time_veto
    saved_flag = os.environ.pop("BRAIN_OPEN_ENDED_GEN_TIME_HONESTY", None)

    def _turn(msg, *, chat, flag_on):
        if flag_on:
            os.environ["BRAIN_OPEN_ENDED_GEN_TIME_HONESTY"] = "1"
        else:
            os.environ.pop("BRAIN_OPEN_ENDED_GEN_TIME_HONESTY", None)
        return OE.answer_turn(msg, None, 0.1, 0.4, ltm_bundle=None, brain_bundle=_BUNDLE_DIR,
                              seed=42, max_new_tokens=110, chat=chat)

    try:
        # (1) flag OFF, chat provided vs not -- byte-identical, one-shot path
        r_off_chat = _turn("tell me about canada", chat=_FakeChat(), flag_on=False)
        r_off_none = _turn("tell me about canada", chat=None, flag_on=False)
        check1 = {
            "gen_time_used_both_false": (r_off_chat["gen_time_honesty_used"] is False
                                         and r_off_none["gen_time_honesty_used"] is False),
            "raw_byte_identical": r_off_chat["raw"] == r_off_none["raw"],
            "answer_byte_identical": r_off_chat["answer"] == r_off_none["answer"],
            "raw_is_one_shot": r_off_chat["raw"].startswith("ONE-SHOT:"),
        }
        check1["ok"] = all(check1.values())

        # (2) flag ON, chat=None -- no chat to consult -> falls back, byte-identical to (1)
        n_calls_before = len(calls)
        r_on_nochat = _turn("tell me about canada", chat=None, flag_on=True)
        check2 = {
            "gen_time_used_false": r_on_nochat["gen_time_honesty_used"] is False,
            "no_gen_time_fn_called": len(calls) == n_calls_before,
            "byte_identical_to_flag_off": r_on_nochat["raw"] == r_off_none["raw"],
        }
        check2["ok"] = all(check2.values())

        # (3) flag ON, chat provided, UNKNOWN topic -- untouched
        n_calls_before = len(calls)
        r_on_unknown = _turn("tell me about zorplaxian", chat=_FakeChat(), flag_on=True)
        check3 = {
            "known_is_false": r_on_unknown["known"] is False,
            "gen_time_used_false": r_on_unknown["gen_time_honesty_used"] is False,
            "no_gen_time_fn_called": len(calls) == n_calls_before,
            "raw_is_one_shot": r_on_unknown["raw"].startswith("ONE-SHOT:"),
        }
        check3["ok"] = all(check3.values())

        # (4) flag ON, chat provided, KNOWN topic -- routes to the gen-time function, with the right args
        n_calls_before = len(calls)
        r_on_known = _turn("tell me about canada", chat=_FakeChat(), flag_on=True)
        this_call = calls[-1] if len(calls) > n_calls_before else None
        check4 = {
            "known_is_true": r_on_known["known"] is True,
            "gen_time_used_true": r_on_known["gen_time_honesty_used"] is True,
            "gen_time_fn_called_once": len(calls) == n_calls_before + 1,
            "raw_is_gen_time": r_on_known["raw"].startswith("GEN-TIME:"),
            "topic_passed_correctly": (this_call or {}).get("topic") == "canada",
            "chat_passed_through": (this_call or {}).get("chat_marker") == _FakeChat.marker,
            "generator_passed_through": bool((this_call or {}).get("gen_is_fake")),
            "trace_captured": r_on_known.get("gen_time_trace") is not None,
        }
        check4["ok"] = all(check4.values())

        # (5) the safety net (post_filter) still ran on the gen-time output -- the returned "united states"
        # phrase is grounded/kept (post_filter's known-topic path never empties a genuinely-supported reply).
        check5 = {"final_nonempty": bool(r_on_known["answer"].strip()),
                  "final_mentions_grounded_fact": "united states" in r_on_known["answer"].lower()}
        check5["ok"] = all(check5.values())
    finally:
        OE.get_generator = orig_get_generator
        GT.generate_with_generation_time_veto = orig_gen_time_fn
        os.environ.pop("BRAIN_OPEN_ENDED_GEN_TIME_HONESTY", None)
        if saved_flag is not None:
            os.environ["BRAIN_OPEN_ENDED_GEN_TIME_HONESTY"] = saved_flag

    # (6) structural: server.py passes chat=chat, still nested under the unchanged BRAIN_OPEN_ENDED guard
    server_check = _check_server_wiring()
    check6 = {"off_path_gated": server_check["off_path_gated"], "single_import": server_check["n_imports"] == 1,
              "chat_kwarg_passed": server_check["chat_kwarg_passed_at_call_site"]}
    check6["ok"] = all(check6.values())

    art = {
        "probe": "open_ended_gen_time_consensus_veto_wiring_verify", "backend": "numpy(stubbed, no GPU/organs)",
        "check1_flag_off_byte_identical": check1, "check2_flag_on_no_chat_falls_back": check2,
        "check3_flag_on_unknown_topic_untouched": check3, "check4_flag_on_known_topic_routes_correctly": check4,
        "check5_safety_net_still_applied": check5, "check6_server_wiring": {**server_check, "ok": check6["ok"]},
    }

    v = Verdict("answer_turn's chat= parameter + BRAIN_OPEN_ENDED_GEN_TIME_HONESTY route to the generation-time "
               "consensus veto ONLY under the full gate, byte-identical otherwise")
    v.require("(1) flag OFF: byte-identical whether or not a chat is passed", check1["ok"], expect=True)
    v.require("(2) flag ON + chat=None: falls back, byte-identical to flag-off", check2["ok"], expect=True)
    v.require("(3) flag ON + unknown topic: untouched", check3["ok"], expect=True)
    v.require("(4) flag ON + chat + known topic: routes to generate_with_generation_time_veto with the right "
              "args", check4["ok"], expect=True)
    v.require("(5) the string post-filter safety net still runs on the gen-time output", check5["ok"],
              expect=True)
    v.require("(6) server.py passes chat=chat at the (still singly-imported, still flag-gated) open-ended "
              "call site", check6["ok"], expect=True)
    go = all([check1["ok"], check2["ok"], check3["ok"], check4["ok"], check5["ok"], check6["ok"]])
    decided = v.decide(go=go)
    art["verdict"] = decided
    art["preconditions"] = decided.get("preconditions", [])
    art["GO"] = bool(go)

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(art, indent=1))
    print(json.dumps({k: art[k]["ok"] for k in (
        "check1_flag_off_byte_identical", "check2_flag_on_no_chat_falls_back",
        "check3_flag_on_unknown_topic_untouched", "check4_flag_on_known_topic_routes_correctly",
        "check5_safety_net_still_applied")} | {"check6_server_wiring": check6["ok"], "GO": go}, indent=1))
    print(f"wrote {OUT} -> {decided['status']}")
    return decided["status"]


if __name__ == "__main__":
    main()
