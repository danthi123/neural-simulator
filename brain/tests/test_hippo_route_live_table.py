"""hippo_route_live_table: the pre-registered bars are compared on exact means (a mean difference of exactly 2.0
over six seeds must pass `>= 2.0`; float means give 1.9999999999999996)."""
import json

from brain.experiments import hippo_route_live_table as t
from brain.experiments.facts import FACTS


def _arm(recall, false=0, routed_right=0):
    rows = [{"kind": "told", "recalled": f"User: {s}" if i < routed_right else None} for i, (s, _, _) in enumerate(FACTS)]
    return {"recall_correct": recall, "paraphrase_correct": 0, "false_recall": false, "echo": 0, "rows": rows}


def test_exact_bar_at_two(tmp_path, capsys):
    g = [4, 5, 5, 4, 6, 5]   # the 11-03 AWS run: sum 29
    b = [2, 3, 3, 2, 4, 3]   # sum 17; (29 - 17) / 6 == 2 exactly
    for i, s in enumerate([42, 43, 44, 100, 101, 102]):
        arms = {a: _arm(0) for a in t.ARMS}
        arms["g_fix_nosleep"] = _arm(g[i], routed_right=9)
        arms["b_intact_nosleep"] = _arm(b[i])
        (tmp_path / f"hroute_s{s}.json").write_text(json.dumps({"seed": s, "arms": arms}))
    t.main(str(tmp_path), "hroute_s*.json")
    assert "F5 (g - b >= 2, 5/6) True" in capsys.readouterr().out
