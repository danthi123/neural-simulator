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
