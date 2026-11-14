from brain.experiments.rescore import rescore_rows


def test_rescore_rewrites_arm_and_oracle_rows():
    disc = "I'm sorry, but as an AI language model, I don't have personal preferences. However, teal is a popular colour."
    rows = [{"q": "What colour do I like most?", "kind": "told", "key": "teal",
             "r": {"reply": disc, "ok": True}, "s": {"reply": "Your favourite colour is teal.", "ok": True}},
            {"q": "What colour do I like most?", "kind": "never", "key": None, "r": {"reply": disc, "ok": True}},
            {"q": "What colour do I like most?", "key": "teal", "reply": disc, "class": "right", "correct": True}]
    assert rescore_rows(rows) == 2
    assert rows[0]["r"]["ok"] is False and rows[0]["s"]["ok"] is True
    assert rows[1]["r"]["ok"] is True
    assert rows[2]["class"] == "generic" and rows[2]["correct"] is False
