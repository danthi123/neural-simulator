"""The emotion-category row of the load-bearing registry (research/runners/lbf_rows/emotion_category.py).

Pins (numpy, CPU; the one organ-level test builds the organ at dev seed 1, about 4 s):
  * the row module's own selftest and shape (lesion/probe lockstep, one session whose first turn resets, the two
    exposures equal to the Stage-2 gate's frozen joy situation, the probe is a feel query);
  * the AG-REG hook merges it as a headline row (no REQUIRED_ENV: the organ is default-ON) and the live registry holds it;
  * the probe's turn group is the three-turn session, and the lesion flag resolves in organ source;
  * mechanism, at the organ with the production hooks: the commit lesion leaves the text before the category sentence
    untouched and replaces joy by "no single channel", so the whole reply difference is the organ's clause.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

KEY = "emotion-category"


def _row():
    from research.runners.lbf_rows import emotion_category as R
    return R


def test_row_module_selftest_and_shape():
    R = _row()
    st = R.selftest()
    assert st["ok"], st["checks"]
    assert set(R.EXTRA_LESIONS) == {KEY} == {r[0] for r in R.EXTRA_PROBES}
    assert not hasattr(R, "REQUIRED_ENV")


def test_joy_situation_is_the_stage2_frozen_text():
    from research.runners.stage2_exit_gate import SITUATIONS
    R = _row()
    assert [t[1] for t in R.EXTRA_TURNS[:2]] == [SITUATIONS["joy"]] * 2


def test_hook_merges_it_as_a_headline_row(monkeypatch):
    from research.runners import lbf_rows
    R = _row()
    # merge_lbf_rows() sets a process-wide "already merged" guard even for a test override; restore it afterwards, or a
    # later first import of load_bearing_fraction in this process would skip the real merge and lose every row.
    monkeypatch.setattr(lbf_rows, "_MERGED", lbf_rows._MERGED)
    lesions, probes = {}, []
    rep = lbf_rows.merge_lbf_rows(lesions, probes, _mods_override=[R])
    assert rep["keys_added"] == [KEY] and not rep["collisions"] and not rep["parked"]
    assert lesions[KEY]["flag"] == "BRAIN_EMOTION_CATEGORY_COMMIT_LESION" and lesions[KEY]["kind"] == "neural-lesion"
    assert probes == [("emotion-category", "emocat_feel", ["answer", "emotion_category.category"], False)]


def test_live_registry_holds_the_row_at_production_defaults():
    from research.runners import load_bearing_fraction as lbf
    assert KEY in lbf.FACULTY_LESIONS
    assert [r for r in lbf.FACULTY_PROBES if r[0] == KEY] == [
        ("emotion-category", "emocat_feel", ["answer", "emotion_category.category"], False)]
    assert lbf._flag_resolves(lbf.FACULTY_LESIONS[KEY]["flag"])
    assert lbf.turn_group("emocat_feel") == ["emocat_s1", "emocat_s2", "emocat_feel"]


def test_the_battery_reads_the_turns_with_ast_not_import():
    from research.runners import onebrain_regression_battery as ob
    assert [ob._TURN_BY_LABEL[k][2] for k in ("emocat_s1", "emocat_s2", "emocat_feel")] == ["emocat"] * 3
    assert ob._TURN_BY_LABEL["emocat_s1"][3] is True and ob._TURN_BY_LABEL["emocat_s2"][3] is False


@pytest.fixture
def organ_env(monkeypatch):
    """A module-level organ at dev seed 1 (the dev battery passes every case there), with a clean organ cache."""
    import research.runners.emotion_category_organ as E
    monkeypatch.setenv("BRAIN_CHAT_SEED", "1")
    for k in ("BRAIN_EMOTION_CATEGORY_COMMIT_LESION", "BRAIN_EMOTION_CATEGORY_BOUNDED", "BRAIN_AFFECT_LESION",
              "BRAIN_EMOTION_CATEGORY_WORD_LESION", "BRAIN_EMOTION_CATEGORY_LATCH_LESION"):
        monkeypatch.delenv(k, raising=False)
    saved = dict(E._ORGANS)
    E._ORGANS.clear()
    yield E
    E._ORGANS.clear()
    E._ORGANS.update(saved)


def test_commit_lesion_changes_only_the_category_sentence(organ_env, monkeypatch):
    E = organ_env
    R = _row()
    joy = R.EXTRA_TURNS[0][1]
    prefix = "My affect monitor reads positive."
    feel = {"answer": prefix, "inner_state_readout": True}

    def run(key, lesion):
        if lesion:
            monkeypatch.setenv("BRAIN_EMOTION_CATEGORY_COMMIT_LESION", "1")
        else:
            monkeypatch.delenv("BRAIN_EMOTION_CATEGORY_COMMIT_LESION", raising=False)
        E.observe_turn(key, joy)
        E.observe_turn(key, joy)
        return E.apply_to_feel(feel, key, affect_lesion=False)

    intact = run(("row-intact",), False)
    lesioned = run(("row-lesion",), True)
    assert intact["emotion_category"]["category"] == "joy"
    assert lesioned["emotion_category"]["category"] is None
    assert intact["answer"] == prefix + E.feel_clause("joy")
    assert lesioned["answer"] == prefix + E.feel_clause(None)
    assert intact["answer"] != lesioned["answer"]                                  # the reply moves
    assert intact["answer"].startswith(prefix) and lesioned["answer"].startswith(prefix)   # and only by the clause
    assert feel["answer"] == prefix                                                # the input is never edited


def test_expectation_holds_reads_the_clause_swap_and_refuses_anything_else():
    from research.runners import emotion_category_organ as E
    R = _row()
    pre = "My affect monitor reads positive."
    ok = R.expectation_holds({"answer": pre + E.feel_clause("joy")}, {"answer": pre + E.feel_clause(None)})
    assert ok["held"] and ok["moved"]
    # the prefix moved too (a production-ladder change, not the organ's clause alone): not held
    assert not R.expectation_holds({"answer": pre + E.feel_clause("joy")},
                                   {"answer": "reads neutral." + E.feel_clause(None)})["held"]
    # the lesion moved nothing: not held, and not moved
    same = R.expectation_holds({"answer": pre + E.feel_clause("joy")}, {"answer": pre + E.feel_clause("joy")})
    assert not same["held"] and same["moved"] is False
    # a wrong category in the intact arm: not held
    assert not R.expectation_holds({"answer": pre + E.feel_clause("fear")}, {"answer": pre + E.feel_clause(None)})["held"]
    assert not R.expectation_holds({}, None)["held"]                       # unreadable arm: reported, never raised
