from brain.experiments.mixed_contrast_table import verdict


def run(told_new, told_slept=4, contrast=0, false_r=6):
    rows = [{"kind": "told", "set": "new", "r": {"ok": i < told_new, "recalled": "e"}} for i in range(6)]
    rows += [{"kind": "told", "set": "slept", "r": {"ok": i < told_slept, "recalled": "e"}} for i in range(6)]
    rows += [{"kind": "never", "set": None, "r": {"ok": i < false_r, "recalled": None}} for i in range(12)]
    rows += [{"kind": "self", "set": None, "r": {"ok": True, "recalled": None}} for i in range(3)]
    return {"speaker_contrast": contrast, "rows": rows}


S = (42, 43, 44, 100, 101, 102)


def test_verdicts():
    T = {s: run(3) for s in S}
    assert verdict(T, {s: run(4, contrast=3) for s in S})["verdict"] == "GO"
    assert verdict(T, {s: run(5, told_slept=2, contrast=3) for s in S})["verdict"] == "NO-GO"
    assert verdict(T, {s: run(4, contrast=3, false_r=8) for s in S})["verdict"] == "NO-GO"
