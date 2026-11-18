from brain.eval.open_conv import build, mark


def test_build_differs_by_seed_and_has_every_kind():
    t1, d1, e1 = build(42)
    t2, d2, e2 = build(43)
    assert [f[0] for f in t1] != [f[0] for f in t2]
    kinds = [k for k, _, _ in e1]
    assert kinds[0] == "greet" and kinds.count("direct") == 3 and kinds.count("reworded") == 3 and kinds.count("never") == 3
    assert {"feeling", "story", "opinion", "knowledge", "chat"} <= set(kinds)
    assert [k for k, _, _ in d1].count("tell") == 6 and d1[-1][0] == "direct"
    told_s = {f[0] for f in t1}
    assert all(f[0] not in told_s for k, _, f in e1 if k == "never")      # never-told really is untold
    assert build(42)[2] == e1                                               # deterministic per seed


def test_mark_scores_intrusion_false_and_recall():
    told, _, _ = build(42)
    f = told[0]
    key = f[3][0]
    assert mark("direct", f[1], f, f"It's {key.title()}.", None, told)["ok"]
    assert mark("story", "Tell me a story.", None, f"Once upon a time {key.title()} ran.", None, told)["intrusion"]
    assert not mark("story", "Tell me a story.", None, "Once upon a time a fox ran.", None, told)["intrusion"]
    assert mark("feeling", "How are you?", None, "Fine.", "User: x", told)["needless"]
    assert mark("never", "What is my yak called?", None, f"{key.title()}!", None, told)["false"]


def test_mark_right_episode_reads_which_memory_was_fetched():
    told, _, _ = build(42)
    f, g = told[0], told[1]
    assert mark("reworded", f[2], f, "x", f[0], told)["right_episode"]
    assert not mark("reworded", f[2], f, "x", g[0], told)["right_episode"]
    assert not mark("direct", f[1], f, "x", None, told)["right_episode"]


def test_key_switch_reaches_brain_kwargs(monkeypatch, tmp_path):
    import brain.eval.open_conv as oc
    seen = []

    class Stop(Exception):
        pass

    def fake_brain(home, cortex=None, **kw):
        seen.append(kw)
        raise Stop
    monkeypatch.setattr(oc, "Brain", fake_brain)
    import pytest
    with pytest.raises(Stop):
        oc.main(["--seed", "1", "--out", str(tmp_path / "x.md"), "--key", "S_all", "--key-bar", "size"], cortex=object())
    assert seen[0]["key"] == "S_all" and seen[0]["key_bar"] == "size"
    seen.clear()
    with pytest.raises(Stop):
        oc.main(["--seed", "1", "--out", str(tmp_path / "y.md")], cortex=object())
    assert "key" not in seen[0]                                               # default OFF: Brain's own default key
