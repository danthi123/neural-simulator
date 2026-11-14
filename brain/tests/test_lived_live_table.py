from brain.experiments.lived_live_table import verdict


def run(ep, told, false=1, self_=1):
    rows = [{"kind": "told", "key": "pixel", "s": {"ok": i < told, "reply": ""}} for i in range(12)]
    rows += [{"kind": "never", "key": None, "s": {"ok": i < false, "reply": ""}} for i in range(12)]
    rows += [{"kind": "self", "key": "nova", "s": {"ok": i < self_, "reply": ""}} for i in range(3)]
    stored = ["User: My cat is called Pixel.\n\nAssistant: Nice!"] if ep == "lived" else ["User: My cat is called Pixel."]
    return {"episode": ep, "stored": stored, "rows": rows}


S = (42, 43, 44, 100, 101, 102)


def test_go_nogo_void():
    A = {42: 3, 43: 2, 44: 6, 100: 6, 101: 6, 102: 6}
    T = {s: run("tell", A[s]) for s in S}
    assert verdict(T, {s: run("lived", A[s] + 2) for s in S})["verdict"] == "GO"
    assert verdict(T, {s: run("lived", A[s]) for s in S})["verdict"] == "NO-GO"
    assert verdict(T, {s: run("lived", A[s] + 2, false=3) for s in S})["verdict"] == "NO-GO"
    assert verdict(T, {s: run("tell", 7) for s in S})["verdict"] == "VOID"
