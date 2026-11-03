"""Abstention switch in the live brain (mock cortex from test_brain_sleep): default OFF changes nothing; ON gates by
the feeling-of-knowing read-out (hippocampal familiarity, or the cortex's cue response with the hippocampus off)."""
from tests.test_brain_sleep import _brain, _talk, cortex  # noqa: F401  (fixture)


def test_abstain_off_is_the_default_and_on_with_bar_minus_inf_changes_nothing(tmp_path, cortex):  # noqa: F811
    plain = _talk(_brain(tmp_path / "a", cortex))
    b = _brain(tmp_path / "b", cortex)
    assert b.abstain is False
    b.abstain = True
    b.know_bar = {"hippocampus": -1e9, "cortex": -1e9}   # everything familiar enough: the unchanged path
    assert _talk(b) == plain


def test_below_the_bar_no_retrieval_is_attempted(tmp_path, cortex, monkeypatch):  # noqa: F811
    b = _brain(tmp_path, cortex, abstain=True)
    b.know_bar = {"hippocampus": 1e9, "cortex": 1e9}
    b.hear("My cat is called Pixel.")                     # a statement: never gated, stored as usual
    b.sleep()
    assert b.cortex.active
    seen = []
    orig = b.cortex.read
    monkeypatch.setattr(b.cortex, "read", lambda text, state=None: (seen.append((text, b.cortex.active)), orig(text, state))[1])
    _, info = b.hear("What is my cat called?")
    assert info["known"] is False and info["fok_source"] == "hippocampus" and info["recalled"] is None
    assert [a for t, a in seen if t.endswith("Assistant:")] == [False] and b.cortex.active  # reply spoken without the consolidated synapses, then restored
    b.set_lesions(hippocampus_off=True)
    _, info = b.hear("What is my cat called?")
    assert info["fok_source"] == "cortex" and info["known"] is False
