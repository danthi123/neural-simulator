"""The BG neural seam (flag BRAIN_BG_NEURAL_SEAM, region `neural-seam` of bg_action_selection_production_organ):
byte-identical when off, and with it on the race commits with no host score, no admission pre-filter and a latched
commit. numpy, dev seed 7 (a CI guard, not a result)."""
import ast
import hashlib
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners import bg_action_selection_production_organ as BG

ROOT = Path(__file__).resolve().parents[1]

# sha256 of the flag-off decide_action / select_once record below, captured on main 3c2907b85 (the organ file there is
# unmodified; the same script on this branch gives the same hash). Recapture only if the flag-OFF path changes on
# purpose, which would itself need a byte-identity argument.
GOLDEN_OFF_SHA256 = "4bbe13667b97f8007420b2944ad845af23932583cf30e21bef71a5305d3ea330"


def _off_record():
    msgs = ["...", "?", "what does the dog chase", "hi", "..", "!"]
    out = []
    for m in msgs * 2:
        r = BG.decide_action(m, seed=7)
        out.append(None if r is None else {k: r[k] for k in ("action", "winner", "speak_salience", "silent_salience",
                                                             "decision_step", "loser_ratio", "lesion")})
    org = BG.get_organ(seed=7)
    raw = [org.select_once(1.0, 0.0), org.select_once(0.0, 1.0), org.select_once(0.0, 0.0)]
    return json.dumps({"decide": out, "raw": raw}, sort_keys=True)


def test_flag_off_is_byte_identical(monkeypatch):
    monkeypatch.delenv("BRAIN_BG_NEURAL_SEAM", raising=False)
    monkeypatch.delenv("BRAIN_BG_SELECT_LESION", raising=False)
    BG.reset_organs()
    BG._NEURAL_ORGANS.clear()
    assert hashlib.sha256(_off_record().encode()).hexdigest() == GOLDEN_OFF_SHA256
    assert BG._NEURAL_ORGANS == {}          # the off path never builds the neural seam


@pytest.mark.parametrize("val,on", [(None, False), ("", False), ("0", False), ("off", False), ("1", True),
                                    ("on", True), ("true", True)])
def test_flag_parsing(monkeypatch, val, on):
    if val is None:
        monkeypatch.delenv("BRAIN_BG_NEURAL_SEAM", raising=False)
    else:
        monkeypatch.setenv("BRAIN_BG_NEURAL_SEAM", val)
    assert BG.neural_seam_enabled() is on


def test_server_dispatches_on_the_flag_only():
    """webapp/server.py: `decide_action_neural` is called only under `if _BG.neural_seam_enabled():`, and the else
    branch is the unchanged `decide_action(msg)` call."""
    tree = ast.parse((ROOT / "webapp" / "server.py").read_text())
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "neural_seam_enabled" in ast.unparse(node.test):
            body, orelse = ast.unparse(ast.Module(node.body, [])), ast.unparse(ast.Module(node.orelse, []))
            hits.append(("decide_action_neural(msg)" in body, "_BG.decide_action(msg)" in orelse
                         and "neural" not in orelse))
    assert hits == [(True, True)]


def test_neural_seam_reads_no_host_score_and_has_no_prefilter(monkeypatch):
    def _forbidden(*a, **k):
        raise AssertionError("the host salience was read")
    monkeypatch.setattr(BG, "salience", _forbidden)
    BG._NEURAL_ORGANS.clear()
    r = BG.decide_action_neural("what does the dog chase", seed=7, lesion=None)
    trial = BG.decide_action_neural.last
    assert trial is not None and trial["heard_groups"] == 5    # the race ran on a content turn (no pre-filter)
    assert r is None and trial["winner"] == BG.SPEAK


def test_neural_seam_behaviour_dev_seed7():
    BG._NEURAL_ORGANS.clear()
    org = BG.NeuralSeamSelector(seed=7)
    empty = [org.select_once("...") for _ in range(6)]
    content = [org.select_once("what does the dog chase") for _ in range(6)]
    assert sum(1 for r in empty if r["winner"] == BG.STAY_SILENT) >= 5
    assert all(r["winner"] == BG.SPEAK for r in content)
    for r in empty + content:
        assert -150.0 <= r["v_min"] and r["v_max"] <= 100.0
    # the committed latch is still active after the drive was withdrawn (the commit holds on its own)
    assert all(max(r["latch_rates"]) >= BG.LATCH_READ_RATE for r in content)


def test_value_lesion_removes_the_state_dependence():
    BG._NEURAL_ORGANS.clear()
    org = BG.NeuralSeamSelector(seed=7, lesion="value")
    empty = [org.select_once("...")["winner"] for _ in range(6)]
    content = [org.select_once("what does the dog chase")["winner"] for _ in range(6)]
    # with heard/hold -> D1 closed the message no longer reaches the race: the two classes are no longer separated
    assert not (all(w == BG.STAY_SILENT for w in empty) and all(w == BG.SPEAK for w in content))


# ── the ignition-bus read (research/runners/_gnw_norgan_bus_derisk.py, region `neural-seam read`) ────────────────
# sha256 of the flag-off norgan_hop record below, captured on main 3c2907b85 (file unmodified there).
GOLDEN_BUS_OFF_SHA256 = "0e831b244bc8805974efa9f2d1d5f6a252b1c57b57845b3bcfb17e3e0eb7087e"
_BUS_CASES = (["cat", "cat", "cat"], ["cat", None, "cat"], ["cat", "cat", None], ["cat", "dog", "cat"], [None, None, None])


def _bus_record():
    import numpy as np
    from research.runners._gnw_norgan_bus_derisk import norgan_hop, D_SUB_UNANIMITY
    from research.runners._p1_2_workspace_deliberation_loop_derisk import build_workspace_bridge
    b, xp, slots, snap = build_workspace_bridge(7)
    rec = []
    for cands in _BUS_CASES:
        c, rates, w, n = norgan_hop(b, xp, slots, snap, cands, "ball", D_SUB_UNANIMITY[3], rng=np.random.default_rng(3))
        rec.append([c, [round(float(r), 6) for r in rates], int(w), int(n)])
    return rec


def test_bus_read_flag_off_is_byte_identical(monkeypatch):
    monkeypatch.delenv("BRAIN_GNW_UNIQUE_IGNITION_READ", raising=False)
    blob = json.dumps(_bus_record(), sort_keys=True)
    assert hashlib.sha256(blob.encode()).hexdigest() == GOLDEN_BUS_OFF_SHA256


def test_unique_read_has_no_argmax():
    from research.runners._gnw_norgan_bus_derisk import THR, unique_ignition_read, readout_variants
    order = ["cat", "ball"]
    assert unique_ignition_read([THR + 0.1, 0.0, 0.0, 0.0], order, 4)[0] == "cat"
    # two co-ignited slots: the argmax read picks the larger, the unique read does not commit
    co = [THR + 0.2, THR + 0.1, 0.0, 0.0]
    v = readout_variants(co, order, 4)
    assert v["argmax_thr (production)"] == "cat" and v["unique"] is None
    assert unique_ignition_read([0.0, 0.0, THR + 0.1, 0.0], order, 4)[0] is None   # an unoccupied slot: no concept
