# Capability: Self and identity
# Record: brain/research/2026-11-26-source-tag-prereg-DRAFT.md
# Expected-fail: none (harness plumbing on the stub cortex; no model weights)
"""voice_tag on a stub life: the told questions are deduplicated told probes; the hybrid opener takes the earlier
night's slow store; a battery asks every question; the table reads synthetic outs."""
import json
import os

from brain.experiments import voice_tag as VT
from brain.tests.test_brain_key_switch import StubCortex
from brain.tests.test_life import BKW, _env  # noqa: F401  (autouse fixture: fake night, 3 turns a day)


def _life(tmp_path):
    from brain.experiments import life_kw
    snaps = tmp_path / "snaps"
    argv = ["--seed", "7", "--days", "3", "--full-probe-nights", "1", "--frozen-probe", "--homes", str(tmp_path / "h"),
            "--out", str(tmp_path / "o" / "life_s7"), "--night-snapshots", str(snaps)]
    life_kw.main(argv, cortex=StubCortex(), brain_kw=BKW)
    return json.load(open(str(tmp_path / "o" / "life_s7") + ".json")), str(snaps), str(tmp_path / "h" / "trunk")


def test_hybrid_open_and_battery_on_a_stub_life(tmp_path, monkeypatch):
    res, snaps, trunk = _life(tmp_path)
    qs = VT.told_questions(res, night=2)
    assert qs and len({q["text"] for q in qs}) == len(qs)
    prev = json.load(open(os.path.join(snaps, "n1", "meta.json")))
    b = VT._open(os.path.join(snaps, "n2"), {"meta.json": os.path.join(snaps, "n1", "meta.json")},
                 str(tmp_path / "w" / "x"), StubCortex(), dict(res["brain_kw"]), trunk)
    try:
        assert b.meta.get("sleeps") == prev.get("sleeps")
        assert len(b.hippocampus.episodes) == len(json.load(open(os.path.join(snaps, "n2", "episodes.json"))))
    finally:
        b.release()
    monkeypatch.setattr(VT, "_onset_ids", lambda pipe: (1, 2))
    out = VT.battery(res, snaps, trunk, StubCortex(), str(tmp_path / "w"), qs, None, None, 1, "A0", night=2)
    assert set(out) == {"hoff", "intact"}
    assert len(out["hoff"]["rows"]) == len(qs) and all(r["onset"] is not None for r in out["hoff"]["rows"])
    assert all(r["onset"] is None for r in out["intact"]["rows"])


def _row(item, brain, onset):
    return {"kind": "H", "item": item, "you": f"What is {item}?", "brain": brain, "onset": onset}


def test_table_reads_synthetic_outs():
    led = {"pixel": {"telling": "My cat is called Pixel.", "keys": ["Pixel"], "told": True}}
    def arm(brain, onset):
        return {"hoff": {"rows": [_row("pixel", brain, onset)]}, "intact": {"rows": [_row("pixel", brain, None)]}}
    outs = [{"seed": s, "reproduction": {"rel_diff": 0.1, "cosine": 0.99},
             "engagement": {"TAG": [0.5], "SHAM": [0.01]},
             "batteries": {"A0snap": {"hoff": {"rows": [_row("pixel", "My cat is called Pixel.", -1.0)]}},
                           "A0": arm("My cat is called Pixel.", -1.0), "TAG": arm("Your cat is called Pixel.", 1.0),
                           "LESION": arm("My cat is called Pixel.", -0.9), "SHAM": arm("My cat is Pixel.", -1.0)}}
            for s in (1, 2)]
    t = VT.table(outs, {1: led, 2: led})
    assert t["verdict"] in ("GO", "NEGATIVE", "MIXED", "NO-ENGAGEMENT", "UNDEFINED")
    assert t["onset_TAG-A0"]["n"] in (0, 2)
