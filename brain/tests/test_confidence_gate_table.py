"""confidence_gate_table: leave-one-seed-out gate picks c's reply when confident and f's otherwise."""
import json

from brain.experiments import confidence_gate_table as T


def _row(kind, shift, c_ok, f_ok, fired):
    rep = lambda ok: "right" if ok else "wrong"
    return {"q": "q", "kind": kind, "key": "k", "shift": shift, "cue": shift, "mean_logp": shift, "first_logp": shift,
            "min_logp": shift, "c": {"reply": rep(c_ok), "recalled": None, "ok": c_ok},
            "f": {"reply": rep(f_ok) + ("!" if fired else ""), "recalled": "ep" if fired else None, "ok": f_ok}}


def test_gate_recovers_c_when_confident(tmp_path, capsys):
    # confident rows (shift 2): c right, reinstated f wrong; unsure rows (shift 0): c wrong, f right
    for s in (42, 43, 44, 100, 101, 102):
        rows = [_row("told", 2.0, True, False, True) for _ in range(6)] + \
               [_row("told", 0.0, False, True, True) for _ in range(6)] + [_row("never", 1.0, False, False, False)]
        json.dump({"seed": s, "rows": rows}, open(tmp_path / f"cgate_s{s}.json", "w"))
    import sys
    sys.argv = ["x", str(tmp_path)]
    T.main()
    out = capsys.readouterr().out
    assert "mean told: c 6.00  f 6.00  h 12.00" in out
    assert "G1 (pooled >= 0.70 and per-seed mean >= 0.65): holds" in out
    assert "G2 (h told >= c - 0.5): holds" in out and "G3 (h - f told >= 1, h >= f on >= 5/6 seeds): holds" in out
    assert "G0 f==c when nothing fired: 6/6" in out


def test_fit_ties_take_highest_bar():
    rows = [_row("told", 1.0, True, True, True), _row("told", 2.0, True, True, True)]
    assert T.fit(rows, "shift") == float("inf")
