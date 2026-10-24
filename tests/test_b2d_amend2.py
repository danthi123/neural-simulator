"""B2d Amendment 2 (research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md, "Amendment 2").

Pinned here without a brain, a GPU or a model:
1. The mouth trace is bookkeeping only: MoodConditionedRenderer returns exactly what it returned before and records
   each call (method, SVO, the manner clause it appended, generative or template, surface); the reply block is
   attached only under BRAIN_MOUTH_TRACE, is reset each turn, and marks which surfaces were spoken.
2. The route map's code facts (research/packets/b2d-mouth-renderer/QWEN-ROUTE-MAP.md): a long-term-memory fact is
   not frameable by the spiking recall mouth, so ChatBrain.render hands it to the renderer, with the manner clause
   in the Qwen prompt; the defaults the map relies on.
3. The probe (research/runners/b2d_amend2_probe.py): arms, candidate turns, the prompt capture and its per-turn split.
4. The scorer (tools/b2d_amend2.py): every cell state, the row readings, the outcome-blind dev selection, job lines.
"""
import json
import os
import subprocess
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("SIM_BACKEND", "numpy")

import research.runners.affect_production_organ as AO  # noqa: E402
import research.runners.b2d_amend2_probe as PR  # noqa: E402
import webapp.server as S  # noqa: E402
from tools import b2d_amend2 as A2  # noqa: E402

PIN = "a" * 40


class _FakeFac:
    CONSTRAIN_TEMPLATE = "Say {a} {v} {p}."

    def __init__(self):
        self.prompts = []

    def _generate(self, user_msg, seed=None):
        self.prompts.append(user_msg)
        return "<%s>" % user_msg, "full", 0.0

    def render_svo(self, a, v, p, seed=None):
        return self._generate(self.CONSTRAIN_TEMPLATE.format(a=a, v=v, p=p), seed=seed)


class _FakeQwen:
    name = "off-bridge Qwen-0.5B (spiking forward)"

    def __init__(self):
        self._fac = _FakeFac()

    def render_svo(self, a, v, p):
        s, _f, _t = self._fac.render_svo(a, v, p)
        return s, None

    def render_svo_regen(self, a, v, p):
        s, _f, _t = self._fac._generate("Exactly: %s %s %s." % (a, v, p))
        return s, None


# ── 1. the trace ─────────────────────────────────────────────────────────────────────────────────────────────────
def test_wrapper_output_unchanged_and_each_call_recorded():
    plain, traced = _FakeQwen(), AO.MoodConditionedRenderer(_FakeQwen())
    assert traced.render_svo("dog", "chase", "cat") == plain.render_svo("dog", "chase", "cat")
    assert traced.render_svo_regen("dog", "chase", "cat") == plain.render_svo_regen("dog", "chase", "cat")
    traced.manner = AO.manner_template_for(2)
    s, asserted = traced.render_svo("angora_turkey", "country", "turkey")
    filled = AO.manner_template_for(2).format(a="angora_turkey", v="country", p="turkey")
    assert asserted is None and traced.base._fac.prompts[-1].endswith(" " + filled)
    calls = traced.mouth_calls
    assert [c["method"] for c in calls] == ["render_svo", "render_svo_regen", "render_svo"]
    assert [c["manner"] for c in calls] == ["", "", filled]
    assert all(c["generative"] for c in calls) and calls[-1]["surface"] == s
    assert calls[-1]["svo"] == ["angora_turkey", "country", "turkey"]


def test_template_base_is_recorded_as_not_generative():
    from research.runners.brain_chat_tui import StubRenderer
    w = AO.MoodConditionedRenderer(StubRenderer())
    w.manner = AO.manner_template_for(2)
    out = w.render_svo("dog", "chase", "cat")
    assert out == StubRenderer().render_svo("dog", "chase", "cat")   # manner is a no-op on the template
    assert w.mouth_calls[-1]["generative"] is False and w.mouth_calls[-1]["manner"] == ""


def test_reply_block_is_flag_gated_reset_per_turn_and_marks_spoken(monkeypatch):
    chat = types.SimpleNamespace(renderer=AO.MoodConditionedRenderer(_FakeQwen()), _mood_tone_level=2)
    chat.renderer.manner = AO.manner_template_for(2)
    s, _ = chat.renderer.render_svo("a_b", "c", "d")
    chat.renderer.render_svo_regen("x_y", "z", "w")
    monkeypatch.delenv("BRAIN_MOUTH_TRACE", raising=False)
    assert S._mouth_trace(chat, "lead " + s) is None
    monkeypatch.setenv("BRAIN_MOUTH_TRACE", "1")
    mt = S._mouth_trace(chat, "lead " + s)
    assert mt["wrapped"] is True and mt["tone_level"] == 2 and mt["manner_template"] == AO.manner_template_for(2)
    assert [c["in_answer"] for c in mt["calls"]] == [True, False]
    S._mouth_trace_reset(chat)
    assert S._mouth_trace(chat, s)["calls"] == []
    bare = types.SimpleNamespace(renderer=_FakeQwen(), _mood_tone_level=0)
    S._mouth_trace_reset(bare)                          # no record on the renderer: left alone, never raises
    assert S._mouth_trace(bare, "x")["wrapped"] is False


# ── 2. route-map pins ────────────────────────────────────────────────────────────────────────────────────────────
def test_ltm_fact_bypasses_the_spiking_recall_mouth_and_reaches_the_renderer_with_its_manner():
    from research.runners.brain_chat_tui import ChatBrain
    from research.runners.spiking_mouth_recall_prod import frame_supported
    assert frame_supported("dog", "chase", "cat")
    for svo in (("angora_turkey", "country", "turkey"), ("sky", "is", "blue"), ("cat", "chase", "cat")):
        assert not frame_supported(*svo)
    fake = types.SimpleNamespace(raw_mode=False, renderer=AO.MoodConditionedRenderer(_FakeQwen()))
    fake.renderer.manner = AO.manner_template_for(-2)
    fake.spiking_recall_surface = lambda a, v, p: ChatBrain.spiking_recall_surface(fake, a, v, p)
    fake._verify = lambda surface, asserted, svo: True
    out = ChatBrain.render(fake, ["angora_turkey", "country", "turkey"])
    assert out.startswith("<Say angora_turkey country turkey.") and "blunt" in out
    assert len(fake.renderer.mouth_calls) == 1 and fake.renderer.mouth_calls[0]["generative"]


def test_route_map_defaults(monkeypatch):
    from research.runners.spiking_mouth_recall_prod import recall_mouth_enabled
    from webapp.affect_drives_chat import affect_marker_surface_enabled
    from webapp.open_ended_chat import open_ended_enabled
    for k in ("BRAIN_SPIKING_MOUTH", "BRAIN_SPIKING_MOUTH_RECALL", "BRAIN_AFFECT_MARKER_SURFACE", "BRAIN_OPEN_ENDED",
              "BRAIN_RICH", "BRAIN_CHAT_RENDERER", "BRAIN_MOUTH_TRACE"):
        monkeypatch.delenv(k, raising=False)
    assert recall_mouth_enabled() is True              # bounded transitive recalls go to the spiking mouth
    assert affect_marker_surface_enabled() is False    # the affect-drives marker is not glued onto the reply
    assert open_ended_enabled() is False               # the direct model.generate path is off
    assert S._brain_rich_default() is True             # an omitted `rich` is the rich path (API, OpenAI shim)
    assert S._mouth_trace_enabled() is False
    monkeypatch.setenv("SIM_BACKEND", "numpy")
    assert S._default_brain_renderer() == "stub"       # Qwen only on cupy with CUDA


# ── 3. the probe ─────────────────────────────────────────────────────────────────────────────────────────────────
def test_probe_arms_and_turns():
    assert set(PR.ARMS) == {"route", "intact_a", "intact_b", "les_coloring", "les_coloring_rep0", "les_appraisal",
                            "les_appraisal_rep0", "les_drives", "les_drives_rep0"}
    assert all(e["BRAIN_MOUTH_TRACE"] == "1" for e in PR.ARMS.values())
    assert PR.ARMS["route"] == PR.ARMS["intact_a"] == {"BRAIN_MOUTH_TRACE": "1"}
    assert PR.ARMS["les_appraisal"]["BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE_LESION"] == "1"
    table, order = PR.turn_table("EMO")
    assert order[:2] == ["a2_c1_mood", "a2_c1_q"] and len(order) == 2 * len(PR.CANDIDATES)
    assert table["a2_c1_mood"] == ("a2_c1_mood", "EMO", "a2_c1", True, None, False)
    assert table["a2_c3_q"][2:] == ("a2_c3", False, None, True)


# ── 3b. the arm's memory plan (infra only: nothing here changes a reply) ─────────────────────────────────────────
# The 2026-10-14 arm at the P2 export (8a969e7) never finished: that revision loads one private copy of the imported
# knowledge store per conversation (wikidata_100k: 78,857 one-per-fact bridges, ~5.4 GiB), the arm opens one
# conversation per candidate, and the old memcap's MemoryHigh then throttled every allocation (research/FAILURE_LOG.md
# 2026-10-17). The server now shares one store per process; the probe refuses to start without it.
_SHARED_ENV = ("BRAIN_SHARED_LTM", "BRAIN_LTM_RNG_ISOLATE", "BRAIN_LTM_CODEBOOK_CACHE", "BRAIN_LTM_DECODE_ESCALATION",
               "BRAIN_LTM_SUBSTRATE_STORE", "BRAIN_BATCHED_SUBSTRATE_SCAN", "BRAIN_SHARD_SPARSE_INDEX",
               "BRAIN_METACOG_SPIKING_MARGIN", "BRAIN_ONE_BRAIN", PR.PRIVATE_LTM_OK_ENV)


def _server_tree(tmp_path, body):
    (tmp_path / "webapp").mkdir()
    (tmp_path / "webapp" / "server.py").write_text(body)
    return str(tmp_path)


def test_preflight_passes_on_this_checkout_and_reads_the_server_source():
    assert PR.preflight_problems(environ={}) == []
    assert S._SHARED_LTM_DEFAULT_ON is True      # what the source read above stands for


def test_preflight_refuses_a_revision_without_the_shared_store(tmp_path):
    for i, body in enumerate(("X = 1\n", "_SHARED_LTM_DEFAULT_ON = False\n", "_SHARED_LTM_DEFAULT_ON = _flag()\n",
                              "def f(:\n")):                  # absent, off, not a literal, unparseable
        root = tmp_path / ("rev%d" % i)
        root.mkdir()
        why = PR.preflight_problems(root=_server_tree(root, body), environ={})
        assert len(why) == 1 and "absent or off in this revision" in why[0], (body, why)
    root = tmp_path / "ok"
    root.mkdir()
    assert PR.preflight_problems(root=_server_tree(root, "_SHARED_LTM_DEFAULT_ON = True\n"), environ={}) == []
    assert PR.preflight_problems(root=str(tmp_path / "no_such_tree"), environ={})       # no server source at all


def test_preflight_reads_the_sharing_escape_like_the_server_does(tmp_path):
    off = tmp_path / "off"
    off.mkdir()
    root = _server_tree(off, "_SHARED_LTM_DEFAULT_ON = False\n")
    for v in ("1", "true", "ON", " yes "):                   # the server's own truthy set turns it on, whatever the default
        assert PR.preflight_problems(root=root, environ={"BRAIN_SHARED_LTM": v}) == []
    for v in ("0", "false", "off", "no", "2"):
        why = PR.preflight_problems(environ={"BRAIN_SHARED_LTM": v})                    # this checkout's default is on
        assert len(why) == 1 and "turns the process-shared knowledge store off" in why[0]
    assert PR.preflight_problems(environ={"BRAIN_SHARED_LTM": "  "}) == []              # blank = unset = the default


def test_main_refuses_before_any_brain_work_and_the_escape_is_explicit(tmp_path, monkeypatch, capsys):
    import research.runners.onebrain_regression_battery as B
    for k in _SHARED_ENV:
        monkeypatch.delenv(k, raising=False)
    calls = []
    monkeypatch.setattr(B, "_collect_worker", lambda *a, **k: calls.append(a) or 3)
    argv = ["--arm", "route", "--seed", "1", "--pin", PIN, "--out", str(tmp_path / "o.json")]
    monkeypatch.setenv("BRAIN_SHARED_LTM", "0")
    assert PR.main(argv) == 4 and calls == []
    assert "REFUSED" in capsys.readouterr().err
    monkeypatch.setenv(PR.PRIVATE_LTM_OK_ENV, "1")            # dev escape: measuring the old behaviour on purpose
    assert PR.main(argv) == 3 and len(calls) == 1             # reached the (stubbed) worker, and returned its rc
    monkeypatch.delenv("BRAIN_SHARED_LTM")
    monkeypatch.delenv(PR.PRIVATE_LTM_OK_ENV)
    calls.clear()
    assert PR.main(argv) == 3 and len(calls) == 1             # the checked-out default passes with no escape


def test_one_arm_loads_the_knowledge_store_once_and_recall_builds_no_new_bridge(tmp_path, monkeypatch):
    """The arm's real conversation count (one per candidate) against a real, tiny, persisted store: only the first
    conversation builds bridges (one per fact), the others build none, and repeated recall through any view builds
    none either (no per-turn bridge rebuild)."""
    import sim.bridge as SB
    from research.runners.sharded_phasor_store import ShardedPhasorStore
    for k in _SHARED_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(S, "_SHARED_LTM", {})
    vocab = ["dog", "go", "north", "cat", "run", "south", "river", "look", "apple", "bird", "eat", "seed"]
    facts = [("dog", "go", "north"), ("cat", "run", "south"), ("river", "look", "apple"), ("bird", "eat", "seed")]
    st = ShardedPhasorStore(n_shards=2, seed=42, D=64, period=200, vocab=list(vocab), enable_substrate_store=True,
                            enable_batched_substrate_scan=True, enable_codebook_cache=True,
                            enable_decode_escalation=True)
    for a, v, p in facts:
        st.store(a, v, p, polarity="AFFIRM")
    bundle = tmp_path / "bundle"
    st.save(str(bundle))
    built = []
    orig = SB.SimulationBridge.__init__

    def counted(self, *a, **k):
        built.append(1)
        return orig(self, *a, **k)
    monkeypatch.setattr(SB.SimulationBridge, "__init__", counted)
    n0, views, marks = len(built), [], []
    for _ in PR.CANDIDATES:
        views.append(S._ltm_store_for_conversation(str(bundle), seed=7))
        marks.append(len(built))
    assert marks[0] - n0 >= len(facts)                                    # the one load: a bridge per stored fact
    assert marks[1:] == [marks[0]] * (len(PR.CANDIDATES) - 1)             # every later conversation: none
    assert len({id(v.shared_store) for v in views}) == 1
    for _ in range(2):                                                    # warm every lazily built, cached scan bridge
        for a, v, _p in facts:
            views[0].query_patient(a, v)
    warm = len(built)
    for i in range(24):
        a, v, _p = facts[i % len(facts)]
        views[i % len(views)].query_patient(a, v)
    assert len(built) == warm


def test_prompt_capture_is_transparent_and_split_by_counts():
    class F:
        def _generate(self, user_msg, seed=None):
            return ("g", user_msg, seed)

        def _generate_batch(self, user_msgs, seed=None):
            return [("b", m) for m in user_msgs]
    cap = PR.PromptCapture().install(F)
    try:
        f = F()
        assert f._generate("p1", seed=3) == ("g", "p1", 3)
        assert f._generate_batch(["p2", "p3"]) == [("b", "p2"), ("b", "p3")]
    finally:
        cap.uninstall(F)
    assert [c["prompts"] for c in cap.calls] == [["p1"], ["p2", "p3"]]
    resp = {"t1": {"_arm_renderer": {"qwen_generate_calls": 0}}, "t2": {"_arm_renderer": {"qwen_generate_calls": 2}}}
    split, left = PR.split_prompts(cap.calls, ["t1", "t2"], resp)
    assert split["t1"] == [] and len(split["t2"]) == 2 and left == 0
    assert PR.split_prompts(cap.calls, ["t1"], resp)[1] == 2


# ── 4. the scorer ────────────────────────────────────────────────────────────────────────────────────────────────
Q = PR.q_label("c1")


def _resp(spoken, manner="", tone=2, drives=1, answer=None, calls=None, extra_calls=(), renderer=A2.QWEN_NAME):
    gen = [{"method": "render_svo", "svo": list(svo), "manner": manner, "generative": True, "surface": s,
            "in_answer": True} for svo, s in spoken] + list(extra_calls)
    text = answer if answer is not None else " ".join(s for _svo, s in spoken)
    return {"answer": text, "abstained": False, "recalled_svo": None, "renderer": renderer,
            "_arm_renderer": {"mode": "production", "qwen_generate_calls": len(gen) if calls is None else calls},
            "affect": {"on": True, "tone_level": tone}, "affect_drives": {"on": True, "level": drives},
            "mouth_trace": {"wrapped": True, "tone_level": tone, "calls": gen}}


def _write(base, seed, arm, resp, pin=PIN, **over):
    gen = [c for c in resp["mouth_trace"]["calls"] if c["generative"]] if "mouth_trace" in resp else []
    doc = {"probe": "b2d-amend2", "arm": arm, "arm_env": PR.ARMS[arm], "seed": seed, "pin": pin,
           "renderer_mode": "production", "backend": "cupy",
           "revision": {"git_sha": pin, "source_kind": "git_archive", "manifest_verified": True},
           "valence_map": {"entries": 13000}, "responses": {PR.q_label(c["id"]): resp for c in PR.CANDIDATES},
           "qwen_prompts": {PR.q_label(c["id"]): [{"method": "_generate", "prompts": ["Say. " + g["manner"]]}
                                                  for g in gen] for c in PR.CANDIDATES},
           "qwen_prompts_unassigned": 0}
    doc.update(over)
    d = os.path.join(base, "s%d" % seed)
    os.makedirs(d, exist_ok=True)
    json.dump(doc, open(os.path.join(d, "%s.json" % arm), "w"))


WARM = AO.manner_template_for(2).format(a="x_y", v="r", p="z")
INTACT = _resp([(("x_y", "r", "z"), "The x y is warmly r z.")], manner=WARM)
LESION_MANNER = _resp([(("x_y", "r", "z"), "The x y r z.")], manner="", tone=0)


def _cell(tmp_path, arms, seed=42, row="affect-coloring", **kw):
    base = str(tmp_path)
    for arm, resp in arms.items():
        _write(base, seed, arm, resp, **kw.get(arm, {}))
    return A2.score_cell(base, PIN, seed, row, Q, kw.get("attempt1"))


def _std(lesion, rep=None, short="coloring", **intact_over):
    return {"route": INTACT, "intact_a": INTACT, "intact_b": INTACT, "les_%s" % short: lesion,
            "les_%s_rep0" % short: rep if rep is not None else lesion}


def test_defined_mouth_load_bearing_via_manner(tmp_path):
    c = _cell(tmp_path, _std(LESION_MANNER))
    assert c["state"] == "DEFINED" and c["mouth_load_bearing"] and c["via"] == ["manner"] and not c["host_only"]
    assert c["attribution"]["attributable"] == 1.0                    # clean null floor: all of it is the lesion's
    assert _cell(tmp_path / "n", _std(INTACT))["attribution"]["attributable"] is None   # no change: UNDEFINED


def test_plan_change_and_route_changed(tmp_path):
    more = _resp([(("x_y", "r", "z"), "The x y is warmly r z."), (("z", "s", "q_q"), "Z s q.")], manner=WARM)
    c = _cell(tmp_path / "a", {"route": more, "intact_a": more, "intact_b": more, "les_coloring": INTACT,
                               "les_coloring_rep0": INTACT})
    assert c["state"] == "DEFINED" and c["via"] == ["plan"]
    silent = _resp([], tone=0, answer="I don't know about that.")
    c = _cell(tmp_path / "b", _std(silent))
    assert c["state"] == "DEFINED" and c["mouth_load_bearing"] and c["via"] == ["route-changed"]


def test_host_only_change_is_not_mouth_load_bearing(tmp_path):
    led = dict(INTACT, answer="Oh! " + INTACT["answer"])
    c = _cell(tmp_path, _std(dict(led, mouth_trace=INTACT["mouth_trace"])))
    assert c["state"] == "DEFINED" and not c["mouth_load_bearing"] and c["host_only"]


def test_unexplained_same_manner_different_words(tmp_path):
    odd = _resp([(("x_y", "r", "z"), "Different words.")], manner=WARM)
    assert _cell(tmp_path, _std(odd))["via"] == ["unexplained"]


def test_silent_not_engaged_and_noise_are_never_zero(tmp_path):
    silent = _resp([], answer="I don't know about that.")
    arms = {k: silent for k in ("route", "intact_a", "intact_b", "les_coloring", "les_coloring_rep0")}
    assert _cell(tmp_path / "a", arms)["state"] == "MOUTH-SILENT"
    flat = _resp([(("x_y", "r", "z"), "The x y r z.")], tone=0)
    arms = {k: flat for k in ("route", "intact_a", "intact_b", "les_coloring", "les_coloring_rep0")}
    assert _cell(tmp_path / "b", arms)["state"] == "NOT-ENGAGED"
    other = _resp([(("x_y", "r", "z"), "Other.")], manner=WARM)
    arms = _std(LESION_MANNER)
    arms["intact_b"] = other
    assert _cell(tmp_path / "c", arms)["state"] == "UNDEFINED:null-noise"
    assert _cell(tmp_path / "d", _std(LESION_MANNER, rep=INTACT))["state"] == "UNDEFINED:lesion-noise"


def test_drives_engagement_reads_the_drives_level(tmp_path):
    flat_drives = _resp([(("x_y", "r", "z"), "The x y is warmly r z.")], manner=WARM, drives=0)
    arms = {k: flat_drives for k in ("route", "intact_a", "intact_b", "les_drives", "les_drives_rep0")}
    assert _cell(tmp_path, arms, row="affect-drives-response")["state"] == "NOT-ENGAGED"


def test_route_mismatch_trace_absent_and_invisible_manner(tmp_path):
    arms = _std(LESION_MANNER)
    arms["intact_a"] = _resp([(("x_y", "r", "z"), "The x y is warmly r z.")], manner=WARM, calls=2)
    assert _cell(tmp_path / "a", arms)["state"] == "ROUTE-MISMATCH"
    arms = _std(LESION_MANNER)
    arms["les_coloring"] = {k: v for k, v in LESION_MANNER.items() if k != "mouth_trace"}
    assert _cell(tmp_path / "b", arms)["state"] == "ROUTE-MISMATCH"
    base = str(tmp_path / "c")
    for arm, resp in _std(LESION_MANNER).items():
        _write(base, 42, arm, resp)
    doc = json.load(open(os.path.join(base, "s42", "intact_a.json")))
    doc["qwen_prompts"][Q][0]["prompts"] = ["Say. (no manner)"]   # the intact prompt lacks the manner clause
    json.dump(doc, open(os.path.join(base, "s42", "intact_a.json"), "w"))
    assert A2.score_cell(base, PIN, 42, "affect-coloring", Q)["state"] == "ROUTE-MISMATCH"


def test_validity_classes_rerun_once_then_fail(tmp_path):
    arms = _std(LESION_MANNER)
    c = _cell(tmp_path / "a", arms, les_coloring={"valence_map": {"entries": 0}})
    assert c["state"] == "RERUN" and "valence-map-absent" in c["arms"]["les_coloring"]
    a1 = tmp_path / "a1"
    os.makedirs(a1 / "s42")
    (a1 / "s42" / "les_coloring.json").write_text("{}")
    c = _cell(tmp_path / "b", arms, les_coloring={"valence_map": {"entries": 0}}, attempt1=str(a1))
    assert c["state"] == "FAIL"
    arms["intact_b"] = _resp([(("x_y", "r", "z"), "s")], renderer="template-stub (GPU-free)")
    assert _cell(tmp_path / "c", arms)["arms"]["intact_b"][0] == "M"
    err = dict(INTACT, affect={"on": True, "error": "ValueError: boom"})
    arms = _std(LESION_MANNER)
    arms["route"] = err
    assert _cell(tmp_path / "d", arms)["arms"]["route"] == ["E", "organ-error"]


def test_row_readings():
    D = lambda m: {"state": "DEFINED", "mouth_load_bearing": m}  # noqa: E731
    assert A2.row_reading({str(i): D(True) for i in range(6)}) == "MOUTH-LOAD-BEARING"
    assert A2.row_reading({str(i): D(False) for i in range(6)}) == "MOUTH-NOT-LOAD-BEARING"
    assert A2.row_reading({str(i): D(i < 2) for i in range(6)}) == "MOUTH-PARTIAL"
    assert A2.row_reading({str(i): {"state": "MOUTH-SILENT"} for i in range(6)}) == "MOUTH-SILENT"
    assert A2.row_reading({str(i): {"state": "NOT-ENGAGED"} for i in range(6)}) == "NOT-ENGAGED"
    cells = {str(i): D(True) for i in range(5)}
    cells["5"] = {"state": "NOT-ENGAGED"}
    assert A2.row_reading(cells) == "MIXED"
    cells["5"] = {"state": "RERUN"}
    assert A2.row_reading(cells) == "PENDING"


def _dev(base, seeds, cand_resp, lesion_resp=None):
    for s in seeds:
        for arm in A2.DEV_ARMS:
            _write(base, s, arm, cand_resp)
    for s in A2.DEV_LESION_SEEDS:
        for arm in A2.DEV_LESION_ARMS:
            _write(base, s, arm, lesion_resp or LESION_MANNER)


def test_dev_selection_is_outcome_blind_and_needs_six_of_eight(tmp_path):
    base = str(tmp_path / "a")
    _dev(base, A2.DEV_SEEDS, INTACT)
    sel = A2.score_dev(base, PIN)
    assert sel["status"] == "SELECTED" and sel["selected"] == "c1"   # all four pass; c1 is first in order
    base2 = str(tmp_path / "b")
    _dev(base2, A2.DEV_SEEDS, INTACT, lesion_resp=INTACT)             # a different lesion outcome ...
    assert A2.score_dev(base2, PIN)["selected"] == "c1"               # ... selects the same candidate
    base3 = str(tmp_path / "c")
    flat = _resp([(("x_y", "r", "z"), "The x y r z.")], tone=0)
    _dev(base3, A2.DEV_SEEDS[:5], INTACT)
    for s in A2.DEV_SEEDS[5:]:
        for arm in A2.DEV_ARMS:
            _write(base3, s, arm, flat)
    sel = A2.score_dev(base3, PIN)
    assert sel["candidates"]["c1"]["n_ok"] == 5 and sel["status"] == "NO-CANDIDATE"


def test_dev_is_incomplete_while_a_file_is_missing(tmp_path):
    base = str(tmp_path)
    _dev(base, A2.DEV_SEEDS[:-1], INTACT)
    assert A2.score_dev(base, PIN)["status"] == "INCOMPLETE"
    assert A2.score_dev(base, PIN, final=True)["status"] == "SELECTED"   # 7 of 8 after the one re-run


def test_job_lines_and_the_gate_seed_guard():
    from tools.gate_seed_guard import find_gate_hit_and_prereg
    pin = "b" * 40
    n = {p: len(A2.job_lines(pin, p)) for p in ("dev", "dev-lesion", "gate-route", "gate-rest")}
    assert n == {"dev": 24, "dev-lesion": 6, "gate-route": 6, "gate-rest": 48}
    for line in A2.job_lines(pin, "gate-rest"):
        assert line.startswith("cd ~/derisk-pool/revisions/%s && " % pin) and " ; " not in line
        assert "LB_ARM_RENDERER=production" in line and "SIM_BACKEND=cupy" in line
        hit, prereg = find_gate_hit_and_prereg(line)
        assert hit and prereg == A2.PREREG
    assert not any(find_gate_hit_and_prereg(ln)[0] for ln in A2.job_lines(pin, "dev"))
    with pytest.raises(SystemExit):
        A2.job_lines("short", "dev")


def test_route_check_reads_only_the_route_arm(tmp_path):
    base = str(tmp_path)
    for s in A2.GATE_SEEDS:
        _write(base, s, "route", INTACT)
    assert A2.route_check(base, PIN, "c1")["status"] == "ROUTE-OK"
    flat = _resp([(("x_y", "r", "z"), "The x y r z.")], tone=0)
    _write(base, 42, "route", flat)                                   # not engaged: reported, not a stop
    r = A2.route_check(base, PIN, "c1")
    assert r["status"] == "ROUTE-OK" and r["seeds"]["42"]["tone_level"] == 0
    _write(base, 43, "route", _resp([], answer="I don't know about that."))
    r = A2.route_check(base, PIN, "c1")
    assert r["status"] == "ROUTE-FAILED" and r["seeds"]["43"]["why"] == ["mouth-silent"]


def test_cli_refuses_a_selection_that_is_not_selected(tmp_path):
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"status": "NO-CANDIDATE", "pin": PIN}))
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "b2d_amend2.py"), "gate", "--pin", PIN, "--base",
                        str(tmp_path), "--selection", str(sel), "--out", str(tmp_path / "o.json")],
                       capture_output=True, text=True, env=dict(os.environ, SIM_NO_PROVENANCE="1"))
    assert p.returncode == 1 and "REFUSED" in p.stderr
