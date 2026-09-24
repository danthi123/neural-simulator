"""The adjudication rule of the closed-class parse instrument (`_lexicon_closed_class_parse_diag.adjudicate`) and its
independent POS ground truth. Pure; no corpus or lexicon build."""
from research.runners import _lexicon_closed_class_parse_diag as P

_NOUNS = {"owl", "apple", "marble", "basket", "room", "box", "leaves"}
_isn = _NOUNS.__contains__


def test_new_noun_without_drop_matches():
    r = P.adjudicate(["wolf"], ["wolf", "owl"], 5, _isn)
    assert r["match"] and r["changed"] and r["admitted"] == ["owl"]


def test_closed_class_admission_is_a_mismatch():
    r = P.adjudicate(["dog", "bird"], ["dog", "when", "bird"], 5, _isn)
    assert not r["match"] and r["bad_admits"] == ["when"]


def test_cap_displacement_by_genuine_nouns_is_explained():
    r = P.adjudicate(["sally", "anne", "box"], ["marble", "basket", "sally", "leaves", "room"], 5, _isn)
    assert r["match"] and r["dropped"] == ["anne", "box"] and r["unexplained_drops"] == []


def test_cap_displacement_with_a_closed_class_word_is_not_explained():
    r = P.adjudicate(["sally", "anne"], ["marble", "sally", "when", "room", "basket"], 5, _isn)
    assert not r["match"] and r["unexplained_drops"] == ["anne"]


def test_drop_without_full_cap_is_not_explained():
    r = P.adjudicate(["sally", "anne"], ["sally", "owl"], 5, _isn)
    assert not r["match"] and r["unexplained_drops"] == ["anne"]


def test_order_change_is_a_mismatch():
    assert not P.adjudicate(["a1", "b1"], ["b1", "a1"], 5, _isn)["match"]


def test_ground_truth_maps():
    is_noun, gt_pos = P.load_gt_noun()
    assert is_noun("owl") and is_noun("marble") and is_noun("apple")          # fixture / broad map nouns
    for w in ("what", "who", "when", "before", "the", "most", "wonderful"):   # closed class / adjectives
        assert not is_noun(w), w
    assert gt_pos("most") == "ADJ" and gt_pos("what") is None
