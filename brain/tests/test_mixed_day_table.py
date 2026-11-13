"""mixed_day_table: the fixed-bar gate takes c on confident slept facts and f on unsure new facts."""
import json
import sys

from brain.experiments import mixed_day_table as T


def _row(kind, st, lp, c_ok, f_ok, fired):
    rep = lambda ok: "right" if ok else "wrong"
    return {"q": "q", "kind": kind, "key": "k", "set": st, "mean_logp": lp, "n_tok": 3,
            "c": {"reply": rep(c_ok), "recalled": None, "ok": c_ok},
            "f": {"reply": rep(f_ok) + ("!" if fired else ""), "recalled": "ep" if fired else None, "ok": f_ok}}


def test_gate_beats_both_arms(tmp_path, capsys):
    for s in (42, 43, 44, 100, 101, 102):
        rows = [_row("told", "slept", -0.01, True, False, True) for _ in range(6)] + \
               [_row("told", "new", -2.0, False, True, True) for _ in range(6)] + \
               [_row("never", None, -1.0, False, False, False) for _ in range(3)]
        json.dump({"seed": s, "slept": "even", "rows": rows}, open(tmp_path / f"mixday_s{s}.json", "w"))
    T.N_PERM = 200
    sys.argv = ["x", str(tmp_path)]
    T.main()
    out = capsys.readouterr().out
    assert "mean h    told 12.00" in out
    assert "M1 instrument (new facts: f - c >= 1.5): holds" in out
    assert "M5 (permutation p < 0.05)" in out and "VERDICT: LEAD" in out
