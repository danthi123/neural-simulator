"""Packet word-role-parse (session 2): the question parser's role map reads the WORDS on spiking neurons.

Pins, without building a brain unless stated:
  * the anchor `_WORD_ROLE_PARSE_DEFAULT_ON` is False and its reader's semantics (unset -> anchor, explicit values);
  * FLAG OFF is byte-identical: `_neural_question_parse` takes the positional BridgeParser read, never touches the
    organ, and the parse-lesion check still comes first when the flag is on;
  * FLAG ON routes to the organ, returns (agent, action) or None, and never raises;
  * the LBF row module (shape checks mirror the existing row selftests) and its registration;
  * the organ: GRID shape, the battery on ONE dev seed at op 0 completes with the documented keys (< 60 s), the cases
    the packet names pass, the facts transcription from a fake composer, and the trained-organ cache;
  * the CLI refuses gate seeds unless `--allow-gate-seed --prereg <path>` is given.
"""
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

DEV_SEED = 1
ENV_VALUES = [None, "", "0", "1", "true", "TRUE", " on ", "yes", "off", "no", "false", "garbage", "2"]


def _set(monkeypatch, name, v):
    if v is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, v)


def _tui():
    import research.runners.brain_chat_tui as T
    return T


class _FakeParser:
    """The positional BridgeParser stand-in: position k -> role k, as the trained parser reads it."""
    calls = 0

    def role_of(self, pos, voice=0):
        type(self).calls += 1
        return ("agent", "action", "patient")[pos]


def _stub_chat():
    T = _tui()
    chat = T.ChatBrain.__new__(T.ChatBrain)
    chat.inner = SimpleNamespace(composer=SimpleNamespace(parser=_FakeParser(), kb=[]), seed=7)
    chat.agent = SimpleNamespace(seed=7)
    return chat


# ── the anchor and its readers ─────────────────────────────────────────────────────────────────────────────────────
def test_anchor_is_false():
    assert _tui()._WORD_ROLE_PARSE_DEFAULT_ON is False


@pytest.mark.parametrize("v", ENV_VALUES)
def test_reader_semantics_anchor_off(monkeypatch, v):
    T = _tui()
    monkeypatch.setattr(T, "_WORD_ROLE_PARSE_DEFAULT_ON", False)
    _set(monkeypatch, "BRAIN_WORD_ROLE_PARSE", v)
    want = False if v is None else v.strip().lower() not in ("0", "false", "off", "no", "")
    assert T._word_role_parse_enabled() is want


def test_reader_semantics_anchor_on(monkeypatch):
    T = _tui()
    monkeypatch.setattr(T, "_WORD_ROLE_PARSE_DEFAULT_ON", True)
    monkeypatch.delenv("BRAIN_WORD_ROLE_PARSE", raising=False)
    assert T._word_role_parse_enabled() is True
    for off in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("BRAIN_WORD_ROLE_PARSE", off)
        assert T._word_role_parse_enabled() is False


@pytest.mark.parametrize("v,want", [(None, False), ("", False), ("0", False), ("1", True), ("true", True),
                                    (" ON ", True), ("yes", True), ("garbage", False)])
def test_lesion_reader(monkeypatch, v, want):
    T = _tui()
    _set(monkeypatch, "BRAIN_WORD_ROLE_LESION", v)
    assert T._word_role_lesioned() is want


@pytest.mark.parametrize("v,want", [(None, 0), ("3", 3), (" 2 ", 2), ("x", 0), ("", 0)])
def test_op_reader(monkeypatch, v, want):
    T = _tui()
    _set(monkeypatch, "BRAIN_WORD_ROLE_OP", v)
    assert T._word_role_op() == want


# ── the wiring: flag off is the positional path, byte for byte ─────────────────────────────────────────────────────
def test_flag_off_is_the_positional_path_and_never_touches_the_organ(monkeypatch):
    T = _tui()
    monkeypatch.delenv("BRAIN_WORD_ROLE_PARSE", raising=False)
    monkeypatch.delenv("BRAIN_SUBSTRATE_PARSE_LESION", raising=False)
    monkeypatch.setattr(T, "_WORD_ROLE_PARSE_DEFAULT_ON", False)
    import research.runners.word_role_parser_organ as W

    def boom(*a, **k):
        raise AssertionError("the organ must not be touched while the flag is off")
    monkeypatch.setattr(W, "parse_from_brain", boom)
    chat = _stub_chat()
    _FakeParser.calls = 0
    assert chat._neural_question_parse(["dog", "chase"]) == ("dog", "chase")
    assert _FakeParser.calls == 3                          # the three positional conjunction reads, as before
    # the positional map reads the first two content words whatever they are (the behaviour this packet replaces)
    assert chat._neural_question_parse(["brain", "usually", "use"]) == ("brain", "usually")
    assert chat._neural_question_parse(["dog"]) is None     # too short, as before


def test_flag_on_routes_to_the_organ(monkeypatch):
    T = _tui()
    monkeypatch.setenv("BRAIN_WORD_ROLE_PARSE", "1")
    monkeypatch.delenv("BRAIN_SUBSTRATE_PARSE_LESION", raising=False)
    monkeypatch.delenv("BRAIN_WORD_ROLE_LESION", raising=False)
    monkeypatch.setenv("BRAIN_WORD_ROLE_OP", "0")
    import research.runners.word_role_parser_organ as W
    seen = {}

    def fake(chat, content, *, seed, lesion=False):
        seen.update(content=list(content), seed=seed, lesion=lesion)
        return {"agent": "brain", "action": "use", "roles": {}, "counts": {}}
    monkeypatch.setattr(W, "parse_from_brain", fake)
    chat = _stub_chat()
    _FakeParser.calls = 0
    assert chat._neural_question_parse(["brain", "usually", "use"]) == ("brain", "use")
    assert _FakeParser.calls == 0                           # the positional read is not consulted
    assert seen == {"content": ["brain", "usually", "use"], "seed": 7, "lesion": False}
    monkeypatch.setenv("BRAIN_WORD_ROLE_LESION", "1")
    chat._neural_question_parse(["brain", "use"])
    assert seen["lesion"] is True


def test_flag_on_decline_and_error_return_none(monkeypatch):
    monkeypatch.setenv("BRAIN_WORD_ROLE_PARSE", "1")
    monkeypatch.delenv("BRAIN_SUBSTRATE_PARSE_LESION", raising=False)
    import research.runners.word_role_parser_organ as W
    chat = _stub_chat()
    monkeypatch.setattr(W, "parse_from_brain", lambda *a, **k: {"agent": "brain", "action": None})
    assert chat._neural_question_parse(["brain", "drink"]) is None
    monkeypatch.setattr(W, "parse_from_brain", lambda *a, **k: {"agent": "x", "action": "x"})
    assert chat._neural_question_parse(["x", "x"]) is None

    def raises(*a, **k):
        raise RuntimeError("organ down")
    monkeypatch.setattr(W, "parse_from_brain", raises)
    assert chat._neural_question_parse(["brain", "use"]) is None      # never raises out


def test_parse_lesion_check_still_comes_first(monkeypatch):
    monkeypatch.setenv("BRAIN_WORD_ROLE_PARSE", "1")
    monkeypatch.setenv("BRAIN_SUBSTRATE_PARSE_LESION", "1")
    import research.runners.word_role_parser_organ as W
    monkeypatch.setattr(W, "parse_from_brain", lambda *a, **k: (_ for _ in ()).throw(AssertionError("reached")))
    assert _stub_chat()._neural_question_parse(["dog", "chase"]) is None


# ── the LBF row module ─────────────────────────────────────────────────────────────────────────────────────────────
def test_row_module_selftest_and_shape():
    from research.runners.lbf_rows import word_role_parse as R
    st = R.selftest()
    assert st["ok"], st["checks"]
    assert R.REQUIRED_ENV == {"BRAIN_NEURAL_SELFID": "1", "BRAIN_WORD_ROLE_PARSE": "1"}
    assert [t[1] for t in R.EXTRA_TURNS] == ["what do you usually use?", "what do you usually drink?"]
    assert set(R.EXTRA_LESIONS) == {"word-role-selfid", "word-role-nofact-ctl"}
    assert all(v["flag"] == "BRAIN_WORD_ROLE_LESION" for v in R.EXTRA_LESIONS.values())


def test_row_expectations():
    from research.runners.lbf_rows import word_role_parse as R
    ok = R.expectation_holds("word-role-selfid", {"answer": "I use spikes.", "abstained": False},
                             {"answer": "I do not know.", "abstained": True})
    assert ok["held"] is True and ok["moved"]
    bad = R.expectation_holds("word-role-selfid", {"answer": "I do not know.", "abstained": True},
                              {"answer": "I do not know.", "abstained": True})
    assert bad["held"] is False
    ctl = R.expectation_holds("word-role-nofact-ctl", {"answer": "I do not know.", "abstained": True},
                              {"answer": "I do not know.", "abstained": True})
    assert ctl["held"] is True
    moved = R.expectation_holds("word-role-nofact-ctl", {"answer": "I do not know.", "abstained": True},
                                {"answer": "I drink water.", "abstained": False})
    assert moved["held"] is False
    assert R.expectation_holds("word-role-selfid", None, None)["held"] is False      # never raises


def test_row_modules_register_when_env_is_set(monkeypatch):
    from research.runners import lbf_rows
    from research.runners.lbf_rows import word_role_parse as R
    monkeypatch.setenv("BRAIN_NEURAL_SELFID", "1")
    monkeypatch.setenv("BRAIN_WORD_ROLE_PARSE", "1")
    lesions, probes = {}, []
    rep = lbf_rows.merge_lbf_rows(lesions, probes, _mods_override=[R])
    assert set(rep["keys_added"]) == {"word-role-selfid", "word-role-nofact-ctl"}
    assert {p[0] for p in probes} == {"word-role-selfid", "word-role-nofact-ctl"}
    monkeypatch.delenv("BRAIN_WORD_ROLE_PARSE")
    lesions2, probes2 = {}, []
    rep2 = lbf_rows.merge_lbf_rows(lesions2, probes2, _mods_override=[R])
    assert not lesions2 and not probes2 and rep2["parked"]            # opt-in: parked without the flags


# ── the organ ──────────────────────────────────────────────────────────────────────────────────────────────────────
def test_grid_shape():
    import research.runners.word_role_parser_organ as W
    assert 6 <= len(W.GRID) <= 12
    assert W.PARAMS == W.GRID[0]
    keys = set(W.GRID[0])
    assert all(set(g) == keys for g in W.GRID)
    assert not set(W.GATE_SEEDS) & set(W.DEV_SEEDS)
    with pytest.raises(ValueError):
        W.set_operating_point(len(W.GRID))
    try:
        W.set_operating_point(1)
        assert W.PARAMS == W.GRID[1]
    finally:
        W.set_operating_point(0)
    assert W.PARAMS == W.GRID[0]


def test_organ_battery_one_dev_seed_op0():
    import research.runners.word_role_parser_organ as W
    t0 = time.time()
    res = W.organ_battery(DEV_SEED, 0)
    assert time.time() - t0 < 60
    assert {"seed", "op_index", "params", "cases", "all_pass", "backend"} <= set(res)
    assert res["seed"] == DEV_SEED and res["op_index"] == 0 and res["params"] == W.GRID[0]
    assert set(res["cases"]) == {"svo", "self", "adverb", "swapped", "novel", "lesion_self"}
    assert all(isinstance(c["pass"], bool) for c in res["cases"].values())
    assert res["all_pass"] is all(c["pass"] for c in res["cases"].values())
    json.dumps(res, default=str)                                       # serializable
    # the cases the packet names, on this dev seed at the frozen default operating point
    assert res["all_pass"], {k: (c["got"], c["roles"]) for k, c in res["cases"].items() if not c["pass"]}
    assert res["cases"]["adverb"]["roles"]["usually"] is None          # a never-heard modifier gets no role
    assert res["cases"]["swapped"]["got"] == ["dog", "chase"]          # word order reversed: not a positional read
    assert res["cases"]["lesion_self"]["got"] is None                  # the word -> role cut declines


def test_facts_from_chat_skips_non_str_and_caches(monkeypatch):
    import research.runners.word_role_parser_organ as W
    kb = [({"agent": "brain", "action": "use", "patient": "spikes"}, None),
          ({"agent": "dog", "action": "chase", "patient": "cat"}, None),
          ({"agent": None, "action": "x", "patient": "y"}, None),
          ({"agent": "a", "action": 3, "patient": "y"}, None),
          "not-a-pair"]
    chat = SimpleNamespace(inner=SimpleNamespace(composer=SimpleNamespace(kb=kb)))
    assert W.facts_from_chat(chat) == [("brain", "use", "spikes"), ("dog", "chase", "cat")]
    assert W.facts_from_chat(SimpleNamespace()) == []
    W._ORGAN_CACHE.clear()
    r1 = W.parse_from_brain(chat, ["brain", "usually", "use"], seed=DEV_SEED)
    assert (r1["agent"], r1["action"]) == ("brain", "use")
    assert len(W._ORGAN_CACHE) == 1
    organ = next(iter(W._ORGAN_CACHE.values()))
    r2 = W.parse_from_brain(chat, ["dog", "chase"], seed=DEV_SEED)
    assert (r2["agent"], r2["action"]) == ("dog", "chase") and next(iter(W._ORGAN_CACHE.values())) is organ
    assert len(W._ORGAN_CACHE) == 1
    r3 = W.parse_from_brain(chat, ["brain", "use"], seed=DEV_SEED, lesion=True)
    assert (r3["agent"], r3["action"]) == (None, None)
    W._ORGAN_CACHE.clear()


# ── the CLI ────────────────────────────────────────────────────────────────────────────────────────────────────────
def _cli(*args):
    env = dict(os.environ, SIM_BACKEND="numpy", SIM_NO_PROVENANCE="1", PYTHONPATH=REPO)
    return subprocess.run([sys.executable, "-m", "research.runners.word_role_parser_organ", *args], cwd=REPO,
                          capture_output=True, text=True, env=env, timeout=120)


@pytest.mark.parametrize("seed", [42, 43, 44, 100, 101, 102])
def test_cli_refuses_gate_seeds(seed, tmp_path):
    out = tmp_path / "x.json"
    p = _cli("--seed", str(seed), "--op", "0", "--out", str(out))
    assert p.returncode != 0 and "gate seed" in (p.stderr + p.stdout)
    assert not out.exists()


def test_cli_gate_seed_needs_prereg_too(tmp_path):
    out = tmp_path / "x.json"
    p = _cli("--seed", "42", "--allow-gate-seed", "--out", str(out))
    assert p.returncode != 0 and not out.exists()
    p = _cli("--sweep", "--seeds", "1", "44", "--out-dir", str(tmp_path / "d"))
    assert p.returncode != 0 and not (tmp_path / "d").exists()


def test_cli_dev_seed_writes_the_battery_json(tmp_path):
    out = tmp_path / "b.json"
    p = _cli("--seed", str(DEV_SEED), "--op", "0", "--out", str(out))
    assert p.returncode == 0, p.stderr[-400:]
    data = json.loads(out.read_text())
    assert {"seed", "op_index", "params", "cases", "all_pass", "backend"} <= set(data)
    assert data["seed"] == DEV_SEED and data["op_index"] == 0
