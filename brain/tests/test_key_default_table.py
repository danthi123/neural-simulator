from brain.experiments.key_default_table import verdict

SEEDS = (42, 43, 44, 100, 101, 102)


def arm(ok, gain=0.5, key=None):
    d = {"ok": ok, "reply": "", "gain": gain}
    if key:
        d["key"] = key
    return d


def rows(told_s=5, told_r=7, slept_s=4, slept_r=4, false_s=2, false_r=2, self_s=3, self_r=3, sets=False):
    out = []
    for i in range(12):
        st = ("slept" if i < 6 else "new") if sets else None
        out.append({"kind": "told", "key": "pixel", "set": st, "c": arm(False, 4.0), "s": arm(i < told_s, key="S0"),
                    "r": arm(i < told_r, key="S_all")})
    if sets:   # slept rows are the first six told rows
        for i, r in enumerate(out[:6]):
            r["s"]["ok"], r["r"]["ok"] = i < slept_s, i < slept_r
    out += [{"kind": "never", "key": None, "c": arm(False, 4.0), "s": arm(i < false_s, key="S0"),
             "r": arm(i < false_r, key="S_all")} for i in range(12)]
    out += [{"kind": "self", "key": "nova", "c": arm(False, 4.0), "s": arm(i < self_s, key="S0"),
             "r": arm(i < self_r, key="S_all")} for i in range(3)]
    return out


def test_flip_and_blocks():
    A = {s: rows() for s in SEEDS}
    B = {s: rows(sets=True) for s in SEEDS}
    assert verdict(A, B)["verdict"] == "FLIP"
    A2 = {s: rows(self_r=1) for s in SEEDS}
    assert verdict(A2, B)["verdict"].startswith("NOT YET")
    A3 = {s: rows(told_r=5) for s in SEEDS}
    assert verdict(A3, B)["verdict"] == "KEEP S0"
    B4 = {s: rows(sets=True, slept_r=2) for s in SEEDS}
    assert not verdict(A, B4)["K3"]


def test_void_when_gain_wrong():
    A = {s: rows() for s in SEEDS}
    B = {s: rows(sets=True) for s in SEEDS}
    for r in B[42]:
        r["r"]["gain"] = 4.0
    assert verdict(A, B)["verdict"] == "VOID"
