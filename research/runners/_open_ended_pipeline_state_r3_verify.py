"""VERIFY: R1 rung 3 -- an open-ended (BRAIN_OPEN_ENDED) turn now ALSO runs the DEEPER query-branch per-turn SESSION-
STATE FOLDS a normal turn runs inside its query-answer branches further down the pipeline, closing the rung-2 disclosed
residual (worldview E2 / multiref D6 / prospective-memory / activity-silent-WM).

WHAT CHANGED (webapp/server.py, inside the existing `if os.environ.get("BRAIN_OPEN_ENDED", ...)` guard, strictly AFTER
rung-2's per-turn writers and BEFORE the pre-existing `return _safe_json_response(_oe_resp, ...)`): additionally run the
WRITE side of four DEEPER folds a normal turn runs below the open-ended block --
  * E2 worldview affective forward-model UPDATE (`_SESSION_WORLDVIEW[cache_key]` context_sign/expected_sign; the normal
    else-branch update ~server.py 5073-5094),
  * D6 multi-referent WM MAINTAIN load+hold (`_SESSION_MULTIREF[cache_key]`; the normal MAINTAIN branch ~5206),
  * Gate-B prospective-memory intention LATCH on a formation / held-monitor ADVANCE (`_SESSION_PMEM[cache_key]`; the
    normal formation/monitor ~4940-4964),
  * Mongillo activity-silent WM MAINTAIN focus/distractor write (`_SESSION_SILENT_WM[cache_key]`; the normal MAINTAIN
    ~5245-5250).
Each reuses the SAME faculty function + gate + `cache_key`, runs for its STATE-WRITE side effect ONLY, and DISCARDS the
surprise-notice / reminder-prefix / read-out LEAD -> `_oe_resp` / the free-talk surface stay byte-identical. Each fold
lifts ONLY the NON-query WRITE branch: the specialist QUERY short-circuits (worldview expectation-query, D6 hold-query
read-out, pmem formation ACK surface, silent-WM temporal-recall read-out) are ROUTING, deferred to rung-4.

THIS RUNNER PROVES, CHEAPLY (no real Qwen -- the warm-Qwen loader + `open_ended_chat.answer_turn` are monkeypatched;
numpy backend), each PATCHED-vs-ORIGINAL (git stash of webapp/server.py ONLY):

  (1) BYTE-IDENTICAL WHEN OFF. `BRAIN_OPEN_ENDED` unset -> the single-fact JSON response is byte-identical (every new
      line sits inside the already-existing `if BRAIN_OPEN_ENDED truthy` block).
  (2) OPEN-ENDED SURFACE PRESERVED. `BRAIN_OPEN_ENDED=1` -> the open-ended JSON response is byte-identical patched vs
      original (the new code only adds STATE-WRITE side effects after `_oe_resp` is built).
  (3) THE FOUR DEEPER STATE WRITES NOW HAPPEN. Two open-ended turns (an affective 2-referent turn, then an intention-
      formation turn) drive the changed `brain_reply`; AFTER, the four session stores for that cache_key hold real
      content -- worldview expected_sign set, D6 >=2 referents held, pmem intention latched (held=True), silent-WM
      focus written. The SAME two turns on the ORIGINAL file leave ALL FOUR keys ABSENT (the deeper branches never ran)
      -- the load-bearing before/after, one control per store.

  python -m research.runners._open_ended_pipeline_state_r3_verify off      # -> RESPONSE:: line
  python -m research.runners._open_ended_pipeline_state_r3_verify on_deep  # -> RESPONSE:: + WORLDVIEW/MULTIREF/PMEM/SILENTWM_STATE::
  python -m research.runners._open_ended_pipeline_state_r3_verify report    # -> the full before/after artifact
"""
from __future__ import annotations
import json, os, sys, types

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_LTM_SHIP_DEFAULT", "off")
# Switch OFF every faculty that runs in the 4200-4598 PREFIX (before the open-ended block) so the MOCK chat never trips
# an unrelated faculty's attribute expectations -- EXACTLY as rung-1/rung-2's verify did. The rung-3 TARGETS
# (worldview/multiref/pmem/silent-wm) sit BELOW the open-ended block, so on an OPEN-ENDED turn ONLY the new state-write
# block reaches them; they are enabled per-phase below. (rung-2's targets -- affect-drives/cg/discourse -- stay OFF here
# so this runner isolates the rung-3 folds.)
for _k, _v in {
    "BRAIN_AFFECT": "0", "BRAIN_AFFECT_DRIVES": "0", "BRAIN_CG_DRIVES": "0", "BRAIN_DISCOURSE_REGISTER": "0",
    "BRAIN_AFFECTIVE_TOM": "0", "BRAIN_DA_DRIVES": "0", "BRAIN_GNW_2ORGAN": "0", "BRAIN_GNW_3ORGAN": "0",
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


def _mock_chat(agent=None):
    c = types.SimpleNamespace()
    c.agents_set = {"dog", "cat", "brain"}
    c.actions_set = {"chase", "eat", "use", "learn", "store", "run"}
    c.patients_set = {"cat", "fish", "spikes", "words", "memory"}
    c.renderer = None
    c.inner = None
    c.agent = agent
    c.gate = lambda question: None
    c._brain_chat_source = "mock"
    return c


def _install_fake_qwen(WS, OE):
    class _FakeFac:
        pass

    class _FakeWarmRenderer:
        _fac = _FakeFac()

    WS._get_warm_qwen_renderer = lambda: _FakeWarmRenderer()

    def _fake_answer_turn(msg, warm_faculty, valence, arousal, *, ltm_bundle, brain_bundle, chat=None,
                          seed: int = 42, max_new_tokens: int = 110) -> dict:
        return {
            "answer": "Dogs chase cats around here.", "raw": "Dogs chase cats around here.",
            "filtered": "Dogs chase cats around here.", "topic": "dog", "known": True,
            "facts": [["dog", "chase", "cat"]], "n_sentences": 1, "gen_seconds": 0.0,
            "gen_time_honesty_used": False, "gen_time_trace": None, "generator": "qwen",
            "wkv_mouth_used": False, "fact_clause_used": False,
            "state": {"valence": float(valence), "arousal": float(arousal), "familiarity": 0.9,
                      "novelty": 0.1, "curiosity": 0.53},
        }

    OE.answer_turn = _fake_answer_turn


def run_off():
    import webapp.server as WS
    os.environ.pop("BRAIN_OPEN_ENDED", None)
    cache_key = ("_r3verify_off_session", "tiny-demo", "raw")
    chat = _mock_chat()
    req = WS.BrainChatRequest(session=cache_key[0], message="what does the dog chase?",
                              brain=cache_key[1], renderer=cache_key[2], rich=False)
    resp = WS.brain_reply(chat, req, "mock", cache_key)
    print("RESPONSE::" + json.dumps(_decode(resp), sort_keys=True))


def run_on_deep():
    import webapp.server as WS
    from webapp import open_ended_chat as OE
    os.environ["BRAIN_OPEN_ENDED"] = "1"
    # the four rung-3 TARGET faculties ON (each is default-ON in production; set explicitly so the phase is unambiguous)
    os.environ["BRAIN_WORLDMODEL"] = "1"
    os.environ["BRAIN_MULTIREF"] = "1"
    os.environ["BRAIN_PMEM"] = "1"
    os.environ["BRAIN_SILENT_WM"] = "1"
    _install_fake_qwen(WS, OE)

    cache_key = ("_r3verify_deep_session", "tiny-demo", "raw")
    chat = _mock_chat()
    # Turn A: affective (worldview UPDATE writes an expected_sign) + 2 referents 'dog','cat' (D6 MAINTAIN load+hold) +
    #         a named referent for the silent-WM MAINTAIN focus write.
    reqA = WS.BrainChatRequest(session=cache_key[0], message="I'm thrilled about the dog and the cat!",
                               brain=cache_key[1], renderer=cache_key[2], rich=False)
    WS.brain_reply(chat, reqA, "mock", cache_key)
    # Turn B: an intention-FORMATION ('remind me to X when Y') -> pmem latches the deferred intention (held=True).
    reqB = WS.BrainChatRequest(session=cache_key[0], message="remind me to run when the fish appears",
                               brain=cache_key[1], renderer=cache_key[2], rich=False)
    respB = WS.brain_reply(chat, reqB, "mock", cache_key)
    print("RESPONSE::" + json.dumps(_decode(respB), sort_keys=True))

    # ── read back the four DEEPER session-state stores for THIS cache_key ──
    wv = WS._SESSION_WORLDVIEW.get(cache_key)
    print("WORLDVIEW_STATE::" + json.dumps({
        "key_present": wv is not None,
        "context_sign": (wv.get("context_sign") if wv else None),
        "expected_sign": (wv.get("expected_sign") if wv else None),
        "expected_sign_written": bool(wv is not None and wv.get("expected_sign") is not None),
    }))
    mr = WS._SESSION_MULTIREF.get(cache_key)
    _n_held = len(getattr(mr, "_slot_of_ref", {}) or {}) if mr is not None else 0
    print("MULTIREF_STATE::" + json.dumps({
        "key_present": mr is not None,
        "n_held": _n_held,
        "holds_two_plus": bool(mr is not None and _n_held >= 2),
    }))
    pm = WS._SESSION_PMEM.get(cache_key)
    print("PMEM_STATE::" + json.dumps({
        "key_present": pm is not None,
        "held": bool(getattr(pm, "held", False)) if pm is not None else False,
        "action_text": (getattr(pm, "action_text", None) if pm is not None else None),
    }))
    sw = WS._SESSION_SILENT_WM.get(cache_key)
    print("SILENTWM_STATE::" + json.dumps({
        "key_present": sw is not None,
        "focus": (getattr(sw, "_focus", None) if sw is not None else None),
        "focus_written": bool(sw is not None and getattr(sw, "_focus", None) is not None),
    }))


def _run_subprocess(phase: str, extra_env: dict | None = None) -> str:
    import subprocess
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    r = subprocess.run([sys.executable, "-u", "-m",
                        "research.runners._open_ended_pipeline_state_r3_verify", phase],
                       cwd=str(_REPO), env=env, capture_output=True, text=True, timeout=600)
    return r.stdout + r.stderr


def _extract(tag: str, text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith(tag):
            return line[len(tag):]
    return None


def report():
    import subprocess
    from tools.verdict import Verdict
    from tools.lab import attributable_to

    def _git(*args):
        subprocess.run(["git", *args], cwd=str(_REPO), check=True, capture_output=True, text=True)

    # ---- OFF: byte-identical response, patched vs original --------------------------------------------------------
    off_after = _run_subprocess("off")
    resp_off_after = _extract("RESPONSE::", off_after)
    _git("stash", "push", "--", "webapp/server.py")
    try:
        off_before = _run_subprocess("off")
    finally:
        _git("stash", "pop")
    resp_off_before = _extract("RESPONSE::", off_before)
    off_byte_identical = bool(resp_off_after) and (resp_off_after == resp_off_before)

    # ---- ON (deep folds): surface byte-identical + the four state writes now fire --------------------------------
    on_after = _run_subprocess("on_deep")
    resp_on_after = _extract("RESPONSE::", on_after)
    wv_after = _extract("WORLDVIEW_STATE::", on_after)
    mr_after = _extract("MULTIREF_STATE::", on_after)
    pm_after = _extract("PMEM_STATE::", on_after)
    sw_after = _extract("SILENTWM_STATE::", on_after)
    _git("stash", "push", "--", "webapp/server.py")
    try:
        on_before = _run_subprocess("on_deep")
    finally:
        _git("stash", "pop")
    resp_on_before = _extract("RESPONSE::", on_before)
    wv_before = _extract("WORLDVIEW_STATE::", on_before)
    mr_before = _extract("MULTIREF_STATE::", on_before)
    pm_before = _extract("PMEM_STATE::", on_before)
    sw_before = _extract("SILENTWM_STATE::", on_before)

    def _j(s):
        return json.loads(s) if s else {}

    wv_a, wv_b = _j(wv_after), _j(wv_before)
    mr_a, mr_b = _j(mr_after), _j(mr_before)
    pm_a, pm_b = _j(pm_after), _j(pm_before)
    sw_a, sw_b = _j(sw_after), _j(sw_before)

    on_surface_byte_identical = bool(resp_on_after) and (resp_on_after == resp_on_before)
    # each store: PATCHED writes real content, ORIGINAL leaves the key absent
    wv_now_writes = (wv_a.get("expected_sign_written") is True and wv_b.get("key_present") is False)
    mr_now_writes = (mr_a.get("holds_two_plus") is True and mr_b.get("key_present") is False)
    pm_now_writes = (pm_a.get("held") is True and pm_b.get("key_present") is False)
    sw_now_writes = (sw_a.get("focus_written") is True and sw_b.get("key_present") is False)

    art = {
        "probe": "open_ended_pipeline_state_r3_rung3_deeper_folds",
        "off_path": {"byte_identical": off_byte_identical,
                     "response_before": resp_off_before, "response_after": resp_off_after},
        "on_path_surface": {"response_byte_identical": on_surface_byte_identical,
                            "response_after": resp_on_after, "response_before": resp_on_before},
        "on_path_deep_folds": {
            "worldview": {"before": wv_b, "after": wv_a, "now_writes": wv_now_writes},
            "multiref": {"before": mr_b, "after": mr_a, "now_writes": mr_now_writes},
            "prospective_memory": {"before": pm_b, "after": pm_a, "now_writes": pm_now_writes},
            "silent_wm": {"before": sw_b, "after": sw_a, "now_writes": sw_now_writes},
        },
    }
    v = Verdict("R1 rung 3: an open-ended turn now runs the DEEPER query-branch per-turn SESSION-STATE folds "
                "(worldview/multiref/pmem/silent-wm); the OFF path + the open-ended surface are unperturbed")
    v.require("OFF path (BRAIN_OPEN_ENDED unset): brain_reply's JSON response is byte-identical, patched vs "
              "original server.py, through the REAL single-fact pipeline", off_byte_identical, expect=True)
    v.require("ON path: the open-ended JSON response surface is byte-identical patched vs original (the new code "
              "only adds state-write side effects after _oe_resp is built)", on_surface_byte_identical, expect=True)
    v.require("ON path, worldview E2: PATCHED writes _SESSION_WORLDVIEW[cache_key] with an expected_sign; ORIGINAL "
              "leaves the key ABSENT (the worldview branch never ran on the open-ended turn)", wv_now_writes, expect=True)
    v.require("ON path, multiref D6: PATCHED loads >=2 referents into _SESSION_MULTIREF[cache_key]; ORIGINAL leaves "
              "the key ABSENT (the MAINTAIN branch never ran)", mr_now_writes, expect=True)
    v.require("ON path, prospective memory: PATCHED latches the intention -> _SESSION_PMEM[cache_key].held=True; "
              "ORIGINAL leaves the key ABSENT (the formation branch never ran)", pm_now_writes, expect=True)
    v.require("ON path, activity-silent WM: PATCHED writes a focus -> _SESSION_SILENT_WM[cache_key]._focus set; "
              "ORIGINAL leaves the key ABSENT (the MAINTAIN branch never ran)", sw_now_writes, expect=True)
    # CONTROLS: each store's write is attributable to THIS change (ORIGINAL = server.py reverted, everything else
    # held fixed; key-present 1 vs 0 clears the min-separation).
    v.control("worldview _SESSION_WORLDVIEW key present after the open-ended turns: PATCHED vs ORIGINAL",
              treatment=1.0 if wv_a.get("key_present") else 0.0,
              control=1.0 if wv_b.get("key_present") else 0.0, min_separation=0.5,
              note="the worldview state write is attributable to this change")
    v.control("multiref _SESSION_MULTIREF held-referent count after the open-ended turns: PATCHED vs ORIGINAL",
              treatment=float(mr_a.get("n_held", 0)), control=float(mr_b.get("n_held", 0)), min_separation=1.5,
              note="the multiref hold write is attributable to this change")
    v.control("prospective-memory _SESSION_PMEM key present after the formation turn: PATCHED vs ORIGINAL",
              treatment=1.0 if pm_a.get("key_present") else 0.0,
              control=1.0 if pm_b.get("key_present") else 0.0, min_separation=0.5,
              note="the prospective-memory latch write is attributable to this change")
    v.control("silent-WM _SESSION_SILENT_WM key present after the open-ended turns: PATCHED vs ORIGINAL",
              treatment=1.0 if sw_a.get("key_present") else 0.0,
              control=1.0 if sw_b.get("key_present") else 0.0, min_separation=0.5,
              note="the silent-WM focus write is attributable to this change")
    art["attribution"] = {
        "worldview_key": attributable_to("worldview key present (patched vs original server.py)",
                                          1.0 if wv_a.get("key_present") else 0.0,
                                          1.0 if wv_b.get("key_present") else 0.0),
        "multiref_n_held": attributable_to("multiref held count (patched vs original server.py)",
                                            float(mr_a.get("n_held", 0)), float(mr_b.get("n_held", 0))),
        "pmem_key": attributable_to("pmem key present (patched vs original server.py)",
                                    1.0 if pm_a.get("key_present") else 0.0,
                                    1.0 if pm_b.get("key_present") else 0.0),
        "silent_wm_key": attributable_to("silent-wm key present (patched vs original server.py)",
                                          1.0 if sw_a.get("key_present") else 0.0,
                                          1.0 if sw_b.get("key_present") else 0.0),
    }
    go = (off_byte_identical and on_surface_byte_identical and wv_now_writes and mr_now_writes
          and pm_now_writes and sw_now_writes)
    decided = v.decide(go=go)
    art["verdict"] = decided
    art["preconditions"] = decided.get("preconditions", [])
    art["GO"] = bool(go)
    out = _REPO / "research" / "findings" / "raw" / "2026-09-02-open-ended-pipeline-state-r3-rung3-verify.json"
    out.write_text(json.dumps(art, indent=1))
    print(json.dumps({"off_byte_identical": off_byte_identical,
                      "on_surface_byte_identical": on_surface_byte_identical,
                      "wv_now_writes": wv_now_writes, "mr_now_writes": mr_now_writes,
                      "pm_now_writes": pm_now_writes, "sw_now_writes": sw_now_writes, "GO": go}, indent=1))
    print(f"wrote {out} -> {decided['status']}")
    return decided["status"]


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "off"
    if phase == "off":
        run_off()
    elif phase == "on_deep":
        run_on_deep()
    elif phase == "report":
        report()
    else:
        raise SystemExit(f"unknown phase {phase!r}; pass 'off', 'on_deep', or 'report'")


if __name__ == "__main__":
    main()
