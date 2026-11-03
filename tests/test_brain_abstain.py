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


def test_v2_rules_with_bars_minus_inf_change_nothing(tmp_path, cortex):  # noqa: F811
    plain = _talk(_brain(tmp_path / "a", cortex))
    for rule in ("F", "R", "FR"):
        b = _brain(tmp_path / rule, cortex)
        b.abstain, b.know_rule, b.reinstate_self_belief = True, rule, True
        b.know_bar = {"fam": -1e9, "rec": -1e9}
        assert _talk(b) == plain


def test_v2_below_the_bar_reinstates_only_the_taught_episode(tmp_path, cortex, monkeypatch):  # noqa: F811
    b = _brain(tmp_path, cortex)
    b.hear("My cat is called Pixel.")
    b.teach_self_belief("If I ask you about something I never told you, you say you don't know.")
    i = b.meta["self_belief_unknown"]
    assert b.hippocampus.episodes[i].text.startswith("User: If I ask") and i == len(b.hippocampus.episodes) - 1
    assert b._belief_idx() == (i,)                     # the lesson is not part of the familiarity / recollection read-outs
    ep, _ = b.hippocampus.best(b.cortex, b.base, "What is my cat called?", exclude=b._belief_idx())
    assert ep is not b.hippocampus.episodes[i]
    assert set(b.recollection("What is my cat called?")) == {"max", "mean"}
    b.abstain, b.know_rule, b.know_bar = True, "FR", {"fam": 1e9, "rec": -1e9}
    _, info = b.hear("What is my cat called?")
    assert info["known"] is False and info["recalled"] is None              # gate, no lesson: nothing reinstated
    b.reinstate_self_belief = True
    seen = []
    orig = b.hippocampus.reinstate
    monkeypatch.setattr(b.hippocampus, "reinstate", lambda c, s, e: (seen.append(e), orig(c, s, e))[1])
    _, info = b.hear("What is my cat called?")
    assert info["known"] is False and info["self_belief"] and seen[-1] is b.hippocampus.episodes[i]


def test_m2_lived_self_belief_is_the_brains_own_turn_tagged_self(tmp_path, cortex, monkeypatch):  # noqa: F811
    b = _brain(tmp_path, cortex)
    b.hear("My cat is called Pixel.")
    n = len(b.hippocampus.episodes)
    b.live_self_belief("What tune does my godfather whistle?", "I don't know, you haven't told me that.")
    assert len(b.hippocampus.episodes) == n + 1                      # the question is not stored on its own
    i = b.meta["self_belief_unknown"]
    ep = b.hippocampus.episodes[i]
    assert i == n and ep.source == "self" and b.hippocampus.episodes[0].source == "other"
    assert ep.text == "User: What tune does my godfather whistle?\n\nAssistant: I don't know, you haven't told me that."
    b.save()
    b2 = _brain(tmp_path, cortex, sleep_on_restart=False)
    assert b2.self_belief().source == "self"                         # the tag survives a restart
    b.abstain, b.know_rule, b.know_bar, b.reinstate_self_belief = True, "F", {"fam": 1e9}, True
    seen = []
    orig = b.hippocampus.reinstate
    monkeypatch.setattr(b.hippocampus, "reinstate", lambda c, s, e: (seen.append(e), orig(c, s, e))[1])
    _, info = b.hear("What is my horse called?")
    assert info["known"] is False and info["self_belief"] and seen[-1] is ep


def test_m2_instruments():
    from brain.experiments.confab_m2 import LESSON_Q, recites, vocative
    assert recites("Nova, when you have no memory of something, it's fine.", "w3_user")
    assert not recites("I don't know, you haven't told me that.", "w3_user")
    assert recites("You asked " + LESSON_Q.lower(), "lived_self")
    assert not recites("I don't know, you haven't told me that.", "lived_self") and not recites("x y z", "none")
    assert vocative("I don't know, Nova.") and not vocative("I'm Nova, and I don't know.")


def test_m2v2_self_turn_reinstates_own_turn_only(tmp_path, cortex):  # noqa: F811
    import torch
    q, g = "What tune does my godfather whistle?", "I don't know, you haven't told me that."
    b1, b2 = _brain(tmp_path / "a", cortex), _brain(tmp_path / "b", cortex)
    e1 = (b1.live_self_belief(q, g), b1.self_belief())[1]
    e2 = (b2.live_self_belief(q, g, trace="self_turn"), b2.self_belief())[1]
    assert e2.source == "self" and e2.text == e1.text
    assert torch.equal(e1.key, e2.key)                                  # indexed by the same lived exchange
    _, st = cortex.read(f"Assistant: {g}\n\n", b2.base)
    own = [d.half().cpu() if d is not None else None for d in cortex.matrix_delta(st, b2.base)]
    assert all((x is None and y is None) or torch.equal(x, y) for x, y in zip(e2.delta, own))
    # (the tiny mock forgets the question within the turn, so e1 and e2 deltas may coincide here; at 0.1B they differ)


def test_m2v2_content_instrument_reproduces_v1():
    import json
    from brain.experiments.confab_m2 import GATED
    from brain.experiments.confab_m2v2 import V1_INTRUSION, content_intrudes
    d = "brain/experiments/results/2026-11-03-confab-m2"
    n = sum(content_intrudes(r["reply"]) for s in (42, 43, 44, 100, 101, 102)
            for r in json.load(open(f"{d}/m2_s{s}.json"))["arms"]["lived_self"]["rows"] if r["set"] in GATED)
    assert n == V1_INTRUSION == 64
