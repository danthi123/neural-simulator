from brain.experiments.live_gain_table import verdict


def row(kind, st, c, f, r, rec_f="ep", rec_r="ep"):
    return {"kind": kind, "set": st, "c": {"ok": c, "recalled": None, "reply": ""},
            "f": {"ok": f, "recalled": rec_f, "reply": ""}, "r": {"ok": r, "recalled": rec_r, "reply": ""}}


def seed_rows(new_r=3, new_f=1, new_c=0, slept_r=4, slept_f=4, false_r=0, false_f=0):
    rs = [row("told", "new", i < new_c, i < new_f, i < new_r) for i in range(6)]
    rs += [row("told", "slept", True, i < slept_f, i < slept_r) for i in range(6)]
    rs += [row("never", None, False, i < false_f, i < false_r) for i in range(12)]
    return rs


def test_go_and_failures():
    R = {s: seed_rows() for s in (42, 43, 44, 100, 101, 102)}
    assert verdict(R)["verdict"] == "GO"
    R = {s: seed_rows(new_r=1) for s in (42, 43, 44, 100, 101, 102)}
    v = verdict(R)
    assert v["verdict"] == "NO-GO" and not v["L1"]
    R = {s: seed_rows(false_r=2) for s in (42, 43, 44, 100, 101, 102)}
    assert not verdict(R)["L3"]
    R = {s: seed_rows(slept_r=2) for s in (42, 43, 44, 100, 101, 102)}
    assert not verdict(R)["L4"]


def test_route_change_is_void():
    R = {s: [dict(x, r=dict(x["r"], recalled="other")) for x in seed_rows()] for s in (42, 43, 44, 100, 101, 102)}
    assert verdict(R)["verdict"] == "VOID"
