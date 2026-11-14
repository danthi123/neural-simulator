from brain.experiments.sall_contrast_table import verdict


def run(told_r, contrast=0, false_r=0, self_r=1, rec="ep"):
    rows = [{"kind": "told", "r": {"ok": i < told_r, "recalled": rec}} for i in range(12)]
    rows += [{"kind": "never", "r": {"ok": i < false_r, "recalled": None}} for i in range(12)]
    rows += [{"kind": "self", "r": {"ok": i < self_r, "recalled": None}} for i in range(3)]
    return {"speaker_contrast": contrast, "rows": rows}


S = (42, 43, 44, 100, 101, 102)


def test_verdicts():
    T = {s: run(6) for s in S}
    assert verdict(T, {s: run(8, 3) for s in S})["verdict"] == "GO"
    assert verdict(T, {s: run(6, 3) for s in S})["verdict"] == "NO-GO"
    assert verdict(T, {s: run(8, 3, rec="other") for s in S})["verdict"] == "VOID"
