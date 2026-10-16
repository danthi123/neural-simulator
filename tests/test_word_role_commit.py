"""Packet word-role-commit (docs/drafts/spiking-commit/PLAN.md, application A1): the word-role parse decided by a
spiking commit stage instead of the host ROLE_MIN / ROLE_MARGIN rule and per-role argmax.

Pins (numpy, CPU; organ builds take ~2 s each):
  * the chat flags: BRAIN_WORD_ROLE_COMMIT default ON since 2026-10-16 flip (RULING wrc-tests-1016) and its
    readers; with the escape BRAIN_WORD_ROLE_COMMIT=0 the word-role route is unchanged
    (same route name, same parse_from_brain call, no commit keys in the record);
  * with it on the route records `word_role_commit`, the commit row, the commit lesion and the commit read, and a
    swallowed organ error is recorded;
  * the count-mode organ is unchanged: its battery cases on dev seed 1 at GRID 0 hash to the value captured on main
    864e8af33 (the same hash the branch was checked against before commit);
  * the commit organ: COMMIT_GRID shape; the battery passes on dev seed 1 at row 0 with a threshold-invariant read;
    the commit lesion leaves the role-ensemble counts identical and only the commit is lost; the host count rule
    cannot change a commit parse (its output is a record only); the CLI refuses gate seeds;
  * the LBF row module and the scorer packet.
"""
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

DEV_SEED = 1
COUNT_MODE_CASES_SHA256 = "8c6c0ecb21fa6e30157f285eab0a85580dbb17c614ba3eaf7ea849f1c55bb898"


def _tui():
    import research.runners.brain_chat_tui as T
    return T


def _set(monkeypatch, name, v):
    if v is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, v)


# ── the chat flags ─────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("v,want", [(None, True), ("", False), ("0", False), ("off", False), ("1", True),
                                    ("true", True), (" on ", True)])
def test_commit_flag_default_on(monkeypatch, v, want):
    """DEFAULT ON since 2026-10-16 (flip, RULING wrc-tests-1016); BRAIN_WORD_ROLE_COMMIT=0 is the escape."""
    T = _tui()
    assert T._WORD_ROLE_COMMIT_DEFAULT_ON is True
    _set(monkeypatch, "BRAIN_WORD_ROLE_COMMIT", v)
    assert T._word_role_commit_enabled() is want


@pytest.mark.parametrize("v,want", [(None, False), ("0", False), ("1", True), ("yes", True), ("garbage", False)])
def test_commit_lesion_reader(monkeypatch, v, want):
    _set(monkeypatch, "BRAIN_WORD_ROLE_COMMIT_LESION", v)
    assert _tui()._word_role_commit_lesioned() is want


@pytest.mark.parametrize("v,want", [(None, 0), ("3", 3), ("x", 0)])
def test_commit_op_reader(monkeypatch, v, want):
    _set(monkeypatch, "BRAIN_WORD_ROLE_COMMIT_OP", v)
    assert _tui()._word_role_commit_op() == want


def _stub_chat():
    T = _tui()
    chat = T.ChatBrain.__new__(T.ChatBrain)
    chat.inner = SimpleNamespace(composer=SimpleNamespace(parser=object(), kb=[]), seed=7)
    chat.agent = SimpleNamespace(seed=7)
    return chat


def _recording_parse(calls, result):
    def fake(chat, content, **kw):
        calls.append(dict(kw))
        if isinstance(result, Exception):
            raise result
        return result
    return fake


def test_flag_off_route_is_unchanged(monkeypatch):
    import research.runners.word_role_parser_organ as W
    monkeypatch.setenv("BRAIN_WORD_ROLE_PARSE", "1")
    monkeypatch.setenv("BRAIN_WORD_ROLE_COMMIT", "0")       # escape (default is ON since the 2026-10-16 flip):
    monkeypatch.delenv("BRAIN_SUBSTRATE_PARSE_LESION", raising=False)  # off -> pre-flip route, no commit kwargs
    monkeypatch.delenv("BRAIN_WORD_ROLE_LESION", raising=False)
    calls = []
    monkeypatch.setattr(W, "parse_from_brain", _recording_parse(calls, {"agent": "brain", "action": "use"}))
    chat = _stub_chat()
    assert chat._neural_question_parse(["brain", "use"]) == ("brain", "use")
    assert calls == [{"seed": 7, "lesion": False}]                      # no commit kwargs reach the organ
    m = chat.last_word_role_meta
    assert m["route"] == "word_role" and "commit_lesioned" not in m and "commit" not in m


def test_flag_on_routes_to_the_commit_stage(monkeypatch):
    import research.runners.word_role_parser_organ as W
    monkeypatch.setenv("BRAIN_WORD_ROLE_PARSE", "1")
    monkeypatch.setenv("BRAIN_WORD_ROLE_COMMIT", "1")
    monkeypatch.setenv("BRAIN_WORD_ROLE_COMMIT_OP", "2")
    monkeypatch.setenv("BRAIN_WORD_ROLE_COMMIT_LESION", "1")
    monkeypatch.delenv("BRAIN_SUBSTRATE_PARSE_LESION", raising=False)
    monkeypatch.delenv("BRAIN_WORD_ROLE_LESION", raising=False)
    calls = []
    res = {"agent": None, "action": None, "roles": {"brain": None}, "counts": {"brain": {"agent": 30}},
           "commit": {"per_item": [{"new": [], "role": None}], "slot_counts": [{"agent": 0}], "threshold": 18.0,
                      "read_variants": {"x0.5": [None, None], "x2": [None, None]}, "read_invariant": True,
                      "co_commit": 0, "relatch": 0, "host_rule_roles": {"brain": "agent"}}}
    monkeypatch.setattr(W, "parse_from_brain", _recording_parse(calls, res))
    chat = _stub_chat()
    assert chat._neural_question_parse(["brain", "use"]) is None
    assert calls == [{"seed": 7, "lesion": False, "commit_op": 2, "commit_lesion": True}]
    m = chat.last_word_role_meta
    assert m["route"] == "word_role_commit" and m["commit_lesioned"] is True and m["lesioned"] is False
    assert m["commit_op"] == 2 and m["error"] is None and m["commit"]["read_invariant"] is True
    json.dumps(m)                                                        # the reply trace stays JSON-safe


def test_flag_on_records_a_swallowed_error(monkeypatch):
    import research.runners.word_role_parser_organ as W
    monkeypatch.setenv("BRAIN_WORD_ROLE_PARSE", "1")
    monkeypatch.setenv("BRAIN_WORD_ROLE_COMMIT", "1")
    monkeypatch.delenv("BRAIN_SUBSTRATE_PARSE_LESION", raising=False)
    monkeypatch.setattr(W, "parse_from_brain", _recording_parse([], RuntimeError("organ down")))
    chat = _stub_chat()
    assert chat._neural_question_parse(["brain", "use"]) is None
    assert "organ down" in chat.last_word_role_meta["error"]


# ── the organ ──────────────────────────────────────────────────────────────────────────────────────────────────────
def test_count_mode_battery_is_unchanged():
    import research.runners.word_role_parser_organ as W
    a = W.organ_battery(DEV_SEED, 0)
    cases = {k: [v["got"], v["counts"]] for k, v in a["cases"].items()}
    assert hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest() == COUNT_MODE_CASES_SHA256


def test_commit_grid_shape():
    import research.runners.word_role_parser_organ as W
    assert len(W.COMMIT_GRID) == 6 and W.COMMIT_PARAMS == W.COMMIT_GRID[0]
    for row in W.COMMIT_GRID:
        assert 0 <= row["BASE_OP"] < len(W.GRID)
        assert set(row) >= {"NG", "NGI", "NS", "W_RG", "W_GG", "W_GGI", "W_GIG", "W_GS", "W_SS", "HOLD_STEPS",
                            "READ_STEPS", "READ_FRAC"}
        assert row["READ_STEPS"] <= row["HOLD_STEPS"]
    params, commit = W.commit_organ_params(0)
    assert params == W.GRID[W.COMMIT_GRID[0]["BASE_OP"]] and commit == W.COMMIT_GRID[0]


@pytest.fixture(scope="module")
def commit_organ():
    import research.runners.word_role_parser_organ as W
    params, commit = W.commit_organ_params(0)
    return W.WordRoleParserOrgan(W.BATTERY_FACTS, seed=DEV_SEED, params=params, commit=commit).train()


def test_commit_battery_passes_on_a_dev_seed():
    import research.runners.word_role_commit_organ as C
    r = C.commit_battery(DEV_SEED, 0)
    assert r["all_pass"], {k: (v["got"], v["want"]) for k, v in r["cases"].items() if not v["pass"]}
    assert r["read_invariant"] and r["mode"] == "commit" and r["op_index"] == 0
    d = r["diagnostics"]
    assert d["latched_min"] > 4 * d["unlatched_max"] + 4                 # the slot read is bimodal on this seed
    assert d["relatch_total"] == 0 and d["co_commit_total"] == 0


def test_commit_lesion_cuts_only_the_commit(commit_organ):
    intact = commit_organ.parse(["brain", "use"])
    cut = commit_organ.parse(["brain", "use"], commit_lesion=True)
    assert (intact["agent"], intact["action"]) == ("brain", "use")
    assert (cut["agent"], cut["action"]) == (None, None)
    assert cut["counts"] == intact["counts"]                             # the role ensembles fire exactly as intact
    assert all(v == 0 for g in cut["commit"]["gate_counts"] for v in g.values())


def test_host_rule_cannot_change_a_commit_parse(commit_organ, monkeypatch):
    base = commit_organ.parse(["brain", "usually", "use"])
    monkeypatch.setattr(commit_organ, "_rule", lambda counts: "patient")  # a host rule that would wreck the parse
    again = commit_organ.parse(["brain", "usually", "use"])
    assert (again["agent"], again["action"]) == (base["agent"], base["action"]) == ("brain", "use")
    assert again["roles"] == base["roles"]
    assert set(again["commit"]["host_rule_roles"].values()) == {"patient"}   # recorded, not used


def test_first_word_to_fill_a_role_keeps_it(commit_organ):
    r = commit_organ.parse(["dog", "chase", "bird"])                   # bird is an agent word at position 2
    assert r["agent"] == "dog"
    roles = [pi["role"] for pi in r["commit"]["per_item"]]
    assert roles[0] == "agent" and roles[2] != "agent"


def test_commit_lesions_need_a_commit_organ():
    import research.runners.word_role_parser_organ as W
    o = W.WordRoleParserOrgan(W.BATTERY_FACTS, seed=DEV_SEED, params=dict(W.GRID[0]))
    with pytest.raises(ValueError):
        o.parse(["brain", "use"], commit_lesion=True)


def test_cli_refuses_gate_seeds():
    import research.runners.word_role_commit_organ as C
    with pytest.raises(SystemExit):
        C.main(["--seed", "42", "--op", "0"])


# ── the LBF rows and the scorer packet ─────────────────────────────────────────────────────────────────────────────
def test_row_module_selftest_and_shape():
    from research.runners.lbf_rows import word_role_commit as R
    st = R.selftest()
    assert st["ok"], st["checks"]
    assert set(R.EXTRA_LESIONS) == {"word-role-commit-selfid", "word-role-commit-nofact-ctl"}


def test_row_modules_register_with_and_without_the_commit_flag(monkeypatch):
    from research.runners import lbf_rows
    from research.runners.lbf_rows import word_role_commit as R
    for k in ("BRAIN_NEURAL_SELFID", "BRAIN_WORD_ROLE_PARSE", "BRAIN_WORD_ROLE_COMMIT"):
        monkeypatch.setenv(k, "1")
    lesions, probes = {}, []
    rep = lbf_rows.merge_lbf_rows(lesions, probes, _mods_override=[R])
    assert set(rep["keys_added"]) == {"word-role-commit-selfid", "word-role-commit-nofact-ctl"}
    monkeypatch.delenv("BRAIN_WORD_ROLE_COMMIT")
    lesions2, probes2 = {}, []
    rep2 = lbf_rows.merge_lbf_rows(lesions2, probes2, _mods_override=[R])
    # REQUIRED_ENV lifted at the flip: the row joins at the default, nothing parked
    assert set(rep2["keys_added"]) == {"word-role-commit-selfid", "word-role-commit-nofact-ctl"}
    assert not rep2["parked"]


def test_scorer_packet_matches_the_chat_flags():
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import s2p_score as S
    import research.runners.word_role_parser_organ as W
    spec = S.PACKETS["word-role-commit"]
    assert spec["grid_len"] == len(W.COMMIT_GRID)
    src = open(os.path.join(REPO, "research", "runners", "brain_chat_tui.py"), encoding="utf8").read()
    for flag in spec["flags"] + spec["lesion_flags"] + [spec["op_env"]]:
        assert flag in src, flag
