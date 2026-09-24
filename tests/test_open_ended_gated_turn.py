"""BRAIN_OPEN_ENDED_GATED (default OFF) -- pure checks, no brain build.

* the module's own selftest (route labels, fresh-vs-stale gate traces, salience transduction, the honesty regex);
* the LBF row module has exactly the FACULTY_LESIONS / FACULTY_PROBES entry shapes, over existing turn labels, with
  lesion flags that resolve in organ source;
* every server.py hook is guarded by `_oeg_on()` (flag off -> the module is never imported).
"""
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_selftest_passes():
    from webapp import open_ended_gated_turn as m
    r = m.selftest()
    assert r["all_pass"], r


def test_row_module_shapes():
    from research.runners.lbf_rows import open_ended_gated as rows
    from research.runners.onebrain_regression_battery import _TURN_BY_LABEL
    kinds = {"neural-lesion", "whether-disable", "in-process", "thin", "mechanism-only", "proposed"}
    assert isinstance(rows.EXTRA_LESIONS, dict) and rows.EXTRA_LESIONS
    for key, ent in rows.EXTRA_LESIONS.items():
        assert set(ent) == {"flag", "value", "kind", "note"}, key
        assert ent["kind"] in kinds
    assert isinstance(rows.EXTRA_PROBES, list)
    keys = set()
    for row in rows.EXTRA_PROBES:
        assert isinstance(row, tuple) and len(row) == 4
        key, turn, fields, thin = row
        assert key in rows.EXTRA_LESIONS
        assert turn in _TURN_BY_LABEL, turn
        assert fields and all(f.startswith("open_ended_gated.") for f in fields)
        # trace copies of an organ read are never decision fields (PREREG pass-by-construction audit)
        assert not any(f.endswith(("valence_sign", "familiarity_band")) for f in fields)
        assert thin is False
        # PREREG Amendment 3 (review 2026-09-24): marker_level changes BY CONSTRUCTION under the row lesion (the marker
        # WTA's own lesion mode returns None) and is structurally 0 on a neutral turn -> never a decision field
        assert not any(f.endswith("marker_level") for f in fields)
        keys.add(key)
    assert keys == set(rows.EXTRA_LESIONS)
    by_key = {r[0]: r for r in rows.EXTRA_PROBES}
    # gnw-drive: ONE decision (the route); bg_action / reply_kind follow from it by host tables (one cause, not three)
    assert by_key["open-ended-turn-gnw-drive"][2] == ["open_ended_gated.route"]
    # the single-shot faculty-drive row is declared non-discriminating (reported, never counted)
    assert set(rows.DESCRIPTIVE_ONLY) == {"open-ended-turn-faculty-drive"}
    assert set(rows.SHARED_LESION_WITH) <= set(rows.EXTRA_LESIONS)
    assert tuple(rows.FACULTIES) == tuple(r[0] for r in rows.EXTRA_PROBES)


def test_score_row_marks_descriptive_rows():
    from research.runners.lbf_rows import open_ended_gated as rows
    arm = lambda action, kind, route: {"unknown": {"open_ended_gated": {"bg_action": action, "reply_kind": kind,
                                                                        "route": route}},
                                       "chase": {"open_ended_gated": {"bg_action": action, "reply_kind": kind,
                                                                      "route": route}}}
    ia, ib = arm("STAY_SILENT", "hold", "offkb"), arm("STAY_SILENT", "hold", "offkb")
    le = arm("SPEAK", "conditioned_generation", "offkb")
    r = rows.score_row("open-ended-turn-faculty-drive", ia, ib, le)
    assert r["load_bearing"] is True and r["counts_toward_claim"] is False and r["descriptive_only_reason"]
    g = rows.score_row("open-ended-turn-gnw-drive", arm("SPEAK", "grounded", "grounded"),
                       arm("SPEAK", "grounded", "grounded"), arm("STAY_SILENT", "withheld_abstain", "withheld"))
    assert g["counts_toward_claim"] is True and len(g["treatment_diffs"]) == 1   # route only, not three fields


def test_row_lesion_flags_resolve_in_source():
    from research.runners.lbf_rows import open_ended_gated as rows
    for ent in rows.EXTRA_LESIONS.values():
        p = subprocess.run(["grep", "-rqlE", "--include=*.py", "--exclude-dir=lbf_rows", ent["flag"],
                            os.path.join(ROOT, "webapp"), os.path.join(ROOT, "research", "runners")])
        assert p.returncode == 0, ent["flag"]


def test_server_hooks_are_flag_guarded():
    src = open(os.path.join(ROOT, "webapp", "server.py"), encoding="utf-8").read().splitlines()
    uses = [i for i, ln in enumerate(src) if "_oeg_mod()" in ln and not ln.lstrip().startswith("def ")]
    assert len(uses) == 5, uses          # install + single apply + single attach + rich apply + rich attach
    for i in uses:
        prev = src[i - 1]
        assert re.search(r"if _oeg_on\(\):", prev), (i, prev)
    # the flag reader is a pure env read defaulting OFF
    body = "\n".join(src)
    assert 'os.environ.get("BRAIN_OPEN_ENDED_GATED", "0")' in body


def test_flag_default_off():
    from webapp import open_ended_gated_turn as m
    old = os.environ.pop("BRAIN_OPEN_ENDED_GATED", None)
    try:
        assert m.gated_enabled() is False
        os.environ["BRAIN_OPEN_ENDED_GATE_LESION"] = "1"
        assert m.gate_lesioned() is False          # the lesion is only read when the flag is on
    finally:
        os.environ.pop("BRAIN_OPEN_ENDED_GATE_LESION", None)
        if old is not None:
            os.environ["BRAIN_OPEN_ENDED_GATED"] = old


def test_race_and_marker_reads_hold_the_shared_organ_lock(monkeypatch):
    """review 2026-09-24 (flag-ON readiness): the BG selector and the marker reader are process-wide singletons shared
    by every chat session; select_once / select_valence must run under _RACE_LOCK (not only their creation)."""
    from webapp import open_ended_gated_turn as m
    seen = []

    class _Org:
        def select_once(self, speak, silent):
            seen.append(("bg", m._RACE_LOCK.locked()))
            return {"winner": 0, "committed": True}

        def select_valence(self, mood, lesion=False):
            seen.append(("marker", m._RACE_LOCK.locked()))
            return 2, None, {"margin": 1.0}

        warmup = 500

    monkeypatch.setattr(m, "_bg_organ", lambda seed: _Org())
    monkeypatch.setattr(m, "_marker_reader", lambda seed: _Org())

    class _Chat:
        pass
    c = _Chat()
    m.bg_race(c, 1.0, 0.0, 7)
    m.settle_marker(c, {"level": 2, "mood": 0.5}, 7, False)
    assert seen == [("bg", True), ("marker", True)], seen
    assert not m._RACE_LOCK.locked()
