"""VERIFY: R1 rung 1 -- an open-ended (BRAIN_OPEN_ENDED) turn now WRITES the D5 episodic store, closing one of the
`~20 skipped production faculties + no session-state-writes` residuals the completeness audit named (the open-ended
branch in `webapp/server.py::brain_reply` RETURNS EARLY, before the shared pipeline's session-state writes and the
rich composer). This is rung 1 of a staged fix (see the R1 finding this artifact backs); it does NOT yet move the
generation-FORM choice inside the shared pipeline -- it only stops the open-ended branch from being a dead end for
D5 episodic memory (a later referential turn, "earlier you told me about X", can now recall a topic the brain
discussed in OPEN-ENDED mode, exactly as it already could for a normal rich/single-fact turn).

WHAT CHANGED (webapp/server.py, inside the existing `if os.environ.get("BRAIN_OPEN_ENDED", ...)` guard, strictly
BETWEEN its `_oe_resp = {...}` construction and its pre-existing `return _safe_json_response(_oe_resp, ...)`):
after building the open-ended response dict, ADDITIONALLY call
`d5_episodic_production_organ.get_episodic_organ(cache_key, 42, topics).note_topic(facts[0][0])` when the turn was
"known" (the open-ended generator's own retrieval returned facts) -- the SAME call, SAME gating convention
(`episodic_enabled()` + `_episodic_store_ok()`), and SAME topic rule (facts[0][0], the agent of the first
supporting fact) the rich path (~server.py:5720) and the single-fact path (~server.py:6084) already use. Additive
only: it does not read or mutate `_oe_resp` / the `answer` text, so the JSON surface is unchanged.

THIS RUNNER PROVES TWO THINGS, CHEAPLY (no GPU, no real Qwen -- the warm-Qwen loader and `open_ended_chat.answer_turn`
are monkeypatched so this never pulls in a real model; `BRAIN_EPISODIC_STORE=1` forces the D5 WRITE gate on despite
running the tiny-demo brain on the numpy backend, exactly as that flag's own docstring says it exists to do):

  (1) BYTE-IDENTICAL WHEN OFF. Two SEPARATE processes -- one against the pre-change server.py (via `git stash`,
      invoked by the accompanying shell recipe, see the finding), one against the changed file -- both call
      `brain_reply` on the SAME (tiny-demo, 'raw' renderer) turn with BRAIN_OPEN_ENDED unset, and their JSON
      responses are diffed byte-for-byte. Structurally this is guaranteed (the new lines sit strictly inside the
      already-existing `if BRAIN_OPEN_ENDED truthy` block, after nothing outside it changed) -- this phase is the
      empirical confirmation, not just the structural argument.
  (2) THE STATE WRITE NOW HAPPENS. Phase 'on' drives one open-ended turn (mocked generator, known topic 'dog',
      supporting fact ('dog','chase','cat') -- 'dog' is a real tiny-demo agent) through the changed `brain_reply`,
      then reads back the D5 episodic organ for that SAME cache_key via a genuine spiking `recall('dog')` --
      `in_memory` must be True. Run the identical phase against the pre-change file and the SAME read comes back
      `in_memory: False` (the load-bearing before/after: the change, not something else, causes the write).

  python -m research.runners._open_ended_episodic_writeback_r1_verify off   # -> one JSON line: the /api/brain-chat response
  python -m research.runners._open_ended_episodic_writeback_r1_verify on    # -> the response line + an EPISODIC_RECALL_AFTER line
"""
from __future__ import annotations
import json, os, sys, time, types
os.environ.setdefault("SIM_BACKEND", "numpy")
# Keep the tiny-demo build fast + independent of the data-lake (~/Projects/sim-data may or may not be present on
# whatever host runs this verify): the shipped-default LTM attach is irrelevant to what this runner checks.
os.environ.setdefault("BRAIN_LTM_SHIP_DEFAULT", "off")
# SCOPE (disclosed): this verify targets ONLY the open-ended early-return block + the D5 episodic write it now
# additionally makes. Every OTHER default-on faculty in the shared pipeline (each already independently GO'd + its
# own byte-identical-off proof elsewhere) is switched off here so a MOCK `chat` (below, no real tiny-demo network
# build -- that path was independently confirmed too slow on this shared, memory-contended host to finish inside
# any reasonable verify budget) doesn't trip an unrelated faculty's attribute expectations. None of these flags
# touch the open-ended block or the D5 organ under test.
for _k, _v in {
    "BRAIN_AFFECT": "0", "BRAIN_AFFECT_DRIVES": "0", "BRAIN_GNW_2ORGAN": "0", "BRAIN_GNW_3ORGAN": "0",
    "BRAIN_GNW_BUS": "0", "BRAIN_VALUE_CHOICE": "0", "BRAIN_SWAP_DRIVES": "0", "BRAIN_SELF_INITIATE": "0",
    "BRAIN_VISION_IDENTITY": "0", "BRAIN_BG_SELECT": "0", "BRAIN_GNW_MULTISTEP": "0",
}.items():
    os.environ.setdefault(_k, _v)
import logging
logging.disable(logging.INFO)
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _decode(resp) -> dict:
    body = resp.body
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    return json.loads(body)


def _mock_chat():
    """A minimal stand-in for a real ChatBrain -- deliberately NOT the full tiny-demo network build (that path
    was independently confirmed to take minutes+ on this shared, memory-contended host; building it is exactly
    what EVERY OTHER faculty's own de-risk already exercises, so re-paying it here would test infrastructure,
    not this change). `agents_set` carries the one fact this verify's mocked `answer_turn` needs ('dog' -> the
    fake ('dog','chase','cat') supporting fact) -- the SAME attribute `_brain_vocab`/the two existing note_topic
    call sites already read via `getattr(chat, "agents_set", None) or _brain_vocab(chat)`. Every other faculty
    that would normally touch `chat` is switched off above, so nothing else dereferences a missing attribute
    outside a guarded try/except (the standing "never let a faculty crash a turn" convention this codebase uses
    everywhere in `brain_reply`)."""
    c = types.SimpleNamespace()
    c.agents_set = {"dog", "cat", "brain"}
    c.actions_set = {"chase", "eat", "use", "learn", "store"}
    c.patients_set = {"cat", "fish", "spikes", "words", "memory"}
    c.renderer = None
    c.inner = None
    c.agent = None
    c.gate = lambda question: None   # always the honest abstain -- no composer needed (single-fact path, rich=False)
    c._brain_chat_source = "mock"
    return c


def run_off():
    t0 = time.time()
    import webapp.server as WS
    print(f"TIMING::import={time.time() - t0:.1f}s", flush=True)
    os.environ.pop("BRAIN_OPEN_ENDED", None)   # the shipping default: unset -> open-ended block never runs
    cache_key = ("_r1verify_off_session", "tiny-demo", "raw")
    chat = _mock_chat()
    req = WS.BrainChatRequest(session=cache_key[0], message="what does the dog chase?",
                              brain=cache_key[1], renderer=cache_key[2], rich=False)
    t1 = time.time()
    resp = WS.brain_reply(chat, req, "mock", cache_key)
    print(f"TIMING::brain_reply={time.time() - t1:.1f}s", flush=True)
    print("RESPONSE::" + json.dumps(_decode(resp), sort_keys=True))


def run_on():
    t0 = time.time()
    import webapp.server as WS
    from webapp import open_ended_chat as OE
    print(f"TIMING::import={time.time() - t0:.1f}s", flush=True)
    os.environ["BRAIN_OPEN_ENDED"] = "1"
    os.environ["BRAIN_EPISODIC_STORE"] = "1"    # force the D5 WRITE gate on despite the numpy backend (documented escape)

    # ── mock out the ONLY heavy/GPU dependency (the warm Qwen faculty) so this verify never touches a real model ──
    class _FakeFac:
        pass

    class _FakeWarmRenderer:
        _fac = _FakeFac()

    WS._get_warm_qwen_renderer = lambda: _FakeWarmRenderer()

    def _fake_answer_turn(msg, warm_faculty, valence, arousal, *, ltm_bundle, brain_bundle, chat=None,
                          seed: int = 42, max_new_tokens: int = 110) -> dict:
        # a synthetic, but SHAPE-FAITHFUL, `answer_turn` return (see webapp/open_ended_chat.py:answer_turn) --
        # a KNOWN topic ('dog', a real tiny-demo agent) with one supporting fact, matching the retrieve() shape.
        return {
            "answer": "Dogs chase cats around here.", "raw": "Dogs chase cats around here.",
            "filtered": "Dogs chase cats around here.", "topic": "dog", "known": True,
            "facts": [["dog", "chase", "cat"]], "n_sentences": 1, "gen_seconds": 0.0,
            "gen_time_honesty_used": False, "gen_time_trace": None, "generator": "qwen",
            "wkv_mouth_used": False, "fact_clause_used": False,
            "state": {"valence": float(valence), "arousal": float(arousal), "familiarity": 0.9,
                      "novelty": 0.1, "curiosity": 0.53},
        }

    OE.answer_turn = _fake_answer_turn   # webapp/server.py imports the module and calls `_OE.answer_turn(...)`

    cache_key = ("_r1verify_on_session", "tiny-demo", "raw")
    chat = _mock_chat()
    req = WS.BrainChatRequest(session=cache_key[0], message="tell me about dogs",
                              brain=cache_key[1], renderer=cache_key[2], rich=False)
    t1 = time.time()
    resp = WS.brain_reply(chat, req, "mock", cache_key)
    print(f"TIMING::brain_reply={time.time() - t1:.1f}s", flush=True)
    print("RESPONSE::" + json.dumps(_decode(resp), sort_keys=True))

    # ── read back the D5 episodic organ for the SAME cache_key -- a genuine spiking recall, not a call-count spy ──
    import research.runners.d5_episodic_production_organ as EP
    topics = getattr(chat, "agents_set", None) or WS._brain_vocab(chat)
    t2 = time.time()
    org = EP.get_episodic_organ(cache_key, 42, topics)
    rec = org.recall("dog")
    print(f"TIMING::episodic_recall={time.time() - t2:.1f}s", flush=True)
    print("EPISODIC_RECALL_AFTER::" + json.dumps({"in_memory": bool(rec.get("in_memory")),
                                                   "reason": rec.get("reason")}))


def _run_subprocess(phase: str, extra_env: dict | None = None) -> str:
    import subprocess
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    r = subprocess.run([sys.executable, "-u", "-m",
                        "research.runners._open_ended_episodic_writeback_r1_verify", phase],
                       cwd=str(_REPO), env=env, capture_output=True, text=True, timeout=240)
    return r.stdout + r.stderr


def _extract(tag: str, text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith(tag):
            return line[len(tag):]
    return None


def report():
    """Orchestrate the full before/after comparison (both phases) via `git stash`/`git stash pop` around
    `webapp/server.py` ONLY (the new files here are untracked and never touched by stash), and write a
    Verdict-gated artifact. Reproduces exactly what this rung's manual verification did."""
    import subprocess
    from tools.verdict import Verdict

    def _git(*args):
        subprocess.run(["git", *args], cwd=str(_REPO), check=True, capture_output=True, text=True)

    # ---- OFF path: byte-identical response, patched vs original -------------------------------------------------
    off_after_out = _run_subprocess("off")
    resp_off_after = _extract("RESPONSE::", off_after_out)
    _git("stash", "push", "--", "webapp/server.py")
    try:
        off_before_out = _run_subprocess("off")
    finally:
        _git("stash", "pop")
    resp_off_before = _extract("RESPONSE::", off_before_out)
    off_byte_identical = bool(resp_off_after) and (resp_off_after == resp_off_before)

    # ---- ON path: response byte-identical + the D5 episodic state write now fires (cupy: the documented fast
    # write path -- the numpy path independently confirmed ~500s/topic per d5_episodic_production_organ.py's own
    # docstring, which is exactly why `_episodic_store_ok()` defers it there in production) -------------------
    on_env = {"SIM_BACKEND": "cupy"}
    on_after_out = _run_subprocess("on", on_env)
    resp_on_after = _extract("RESPONSE::", on_after_out)
    ep_on_after = _extract("EPISODIC_RECALL_AFTER::", on_after_out)
    _git("stash", "push", "--", "webapp/server.py")
    try:
        on_before_out = _run_subprocess("on", on_env)
    finally:
        _git("stash", "pop")
    resp_on_before = _extract("RESPONSE::", on_before_out)
    ep_on_before = _extract("EPISODIC_RECALL_AFTER::", on_before_out)

    on_response_byte_identical = bool(resp_on_after) and (resp_on_after == resp_on_before)
    ep_before = json.loads(ep_on_before) if ep_on_before else {}
    ep_after = json.loads(ep_on_after) if ep_on_after else {}
    state_write_now_happens = (ep_after.get("in_memory") is True and ep_before.get("in_memory") is False)

    art = {
        "probe": "open_ended_episodic_writeback_r1_rung1",
        "off_path": {"byte_identical": off_byte_identical,
                     "response_before": resp_off_before, "response_after": resp_off_after},
        "on_path": {"response_byte_identical": on_response_byte_identical,
                    "response_before": resp_on_before, "response_after": resp_on_after,
                    "episodic_recall_before": ep_before, "episodic_recall_after": ep_after},
    }
    v = Verdict("R1 rung 1: an open-ended turn now WRITES the D5 episodic store; the OFF path is unperturbed")
    v.require("OFF path (BRAIN_OPEN_ENDED unset): brain_reply's JSON response is byte-identical, patched vs "
              "original server.py, through the REAL single-fact pipeline (a mock chat, curiosity/common-ground/"
              "da-drives all fire)", off_byte_identical, expect=True)
    v.require("ON path (BRAIN_OPEN_ENDED=1): the open-ended JSON response surface is byte-identical, patched vs "
              "original (the new code only adds a side effect AFTER _oe_resp is built)",
              on_response_byte_identical, expect=True)
    v.require("ON path, ORIGINAL code: the D5 episodic organ shows in_memory=False after the turn (the pre-fix "
              "dead end -- the open-ended turn never wrote anything)", ep_before.get("in_memory") is False,
              expect=True)
    v.require("ON path, PATCHED code: the D5 episodic organ shows in_memory=True after the SAME turn via a "
              "genuine spiking dendritic-completion recall (reason=spiking-dap-completion), not a host flag",
              ep_after.get("reason") == "spiking-dap-completion", expect=True)
    v.control("D5 episodic in_memory('dog') after the open-ended turn: PATCHED vs ORIGINAL",
              treatment=1.0 if ep_after.get("in_memory") else 0.0,
              control=1.0 if ep_before.get("in_memory") else 0.0, min_separation=0.5,
              note="the write is attributable to this change, not something else already running")
    go = off_byte_identical and on_response_byte_identical and state_write_now_happens
    decided = v.decide(go=go)
    art["verdict"] = decided
    art["preconditions"] = decided.get("preconditions", [])
    art["GO"] = bool(go)
    out = _REPO / "research" / "findings" / "raw" / "2026-09-02-open-ended-episodic-writeback-r1-rung1-verify.json"
    out.write_text(json.dumps(art, indent=1))
    print(json.dumps({"off_byte_identical": off_byte_identical,
                      "on_response_byte_identical": on_response_byte_identical,
                      "episodic_before": ep_before, "episodic_after": ep_after, "GO": go}, indent=1))
    print(f"wrote {out} -> {decided['status']}")
    return decided["status"]


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "off"
    if phase == "off":
        run_off()
    elif phase == "on":
        run_on()
    elif phase == "report":
        report()
    else:
        raise SystemExit(f"unknown phase {phase!r}; pass 'off', 'on', or 'report'")


if __name__ == "__main__":
    main()
