from brain.experiments.near_miss_table import verdict


def _x(seed, never_ok, told_ok, nm):
    rows = [{"q": f"t{i}", "kind": "told", "set": "slept", "s": {"ok": i < told_ok, "reply": ""}, "r": {"ok": False, "reply": ""}}
            for i in range(12)]
    rows += [{"q": f"n{i}", "kind": "never", "s": {"ok": i < never_ok, "reply": ""}, "r": {"ok": False, "reply": ""}}
             for i in range(12)]
    log = [{"cues": ["a?"], "kept": ["a?"]}] if nm else []
    return {"seed": seed, "near_miss": nm, "near_miss_log": log, "rows": rows}


SEEDS = (42, 43, 44, 100, 101, 102)


def test_go_lead_and_void():
    off = {s: _x(s, 7, 7, 0) for s in SEEDS}
    assert verdict(off, {s: _x(s, 3, 7, 4) for s in SEEDS})["verdict"] == "GO"
    assert verdict(off, {s: _x(s, 5, 7, 4) for s in SEEDS})["verdict"] == "LEAD"
    assert verdict(off, {s: _x(s, 3, 4, 4) for s in SEEDS})["verdict"] == "NOT A LEAD"   # told cost
    assert verdict(off, {s: _x(s, 3, 7, 0) for s in SEEDS})["verdict"] == "VOID"         # flag not recorded
    ref = {s: _x(s, 6, 7, 0) for s in SEEDS}
    assert verdict(off, {s: _x(s, 3, 7, 4) for s in SEEDS}, ref)["verdict"] == "VOID"    # does not reproduce
