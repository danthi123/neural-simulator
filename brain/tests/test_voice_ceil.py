# Capability: Self and identity
# Record: brain/research/2026-11-26-voice-ceiling-prereg.md (prereg)
# Expected-fail: none (harness plumbing on the stub cortex; no model weights)
"""voice_ceil on a stub life: every arm runs the whole battery; A0 reproduces the voice split's hippocampus-off B rows;
training arms log targets and losses and change the adapters; the sham targets carry no context; the table reads."""
import json

from brain.experiments import voice_ceil as VC
from brain.experiments import voice_split as VS
from brain.tests.test_brain_key_switch import StubCortex
from brain.tests.test_life import BKW, _env  # noqa: F401  (autouse fixture: fake night, 3 turns a day)


def test_voice_ceil_on_a_stub_life(tmp_path, monkeypatch):
    from brain.experiments import life_kw

    seen = []
    def fake_distill(b, targets, steps=VC.STEPS, **kw):   # the stub _plastic has no weights (test_life fixture)
        seen.append(targets)
        return [(0.0, 0.0)] * steps, (0.5, 1.0)
    monkeypatch.setattr(VC, "distill", fake_distill)

    snaps = tmp_path / "snaps"
    argv = ["--seed", "7", "--days", "3", "--full-probe-nights", "1", "--frozen-probe", "--homes", str(tmp_path / "h"),
            "--out", str(tmp_path / "o" / "life_s7"), "--night-snapshots", str(snaps)]
    life_kw.main(argv, cortex=StubCortex(), brain_kw=BKW)
    res = json.load(open(str(tmp_path / "o" / "life_s7") + ".json"))
    trunk = str(tmp_path / "h" / "trunk")
    nights = (2, 3)
    out = VC.run(res, str(snaps), trunk, StubCortex(), str(tmp_path / "w"), weekend=False, nights=nights, steps=3,
                 log=lambda *a: None)
    vs = VS.run(res, str(snaps), trunk, StubCortex(), str(tmp_path / "w2"), weekend=False, nights=nights,
                log=lambda *a: None)
    assert out["cues"] and not set(out["cues"]) & set(VC.HELD)   # trained items never include a test item
    for n, vn in zip(out["nights"], vs["nights"]):
        n_q = len(vn["arms"]["hippocampus-off"]["B"]["rows"])
        assert set(n["arms"]) == {a for a, _, _ in VC.ARMS}
        assert all(len(v["rows"]) == n_q for v in n["arms"].values())
        assert [r["brain"] for r in n["arms"]["A0"]["rows"]] == [r["brain"] for r in vn["arms"]["hippocampus-off"]["B"]["rows"]]
        for a, trains, ctx in VC.ARMS:
            if trains:
                v = n["arms"][a]
                assert len(v["losses"]) == 3 and len(v["targets"]) == sum(len(q) for q in out["cues"].values())
                assert {x for x, _, _ in v["targets"]} == set(out["cues"])
    ctx_n = {len(t) for tg in seen for t in tg}
    assert ctx_n == {2, 3}   # sham targets are (prefix, reply) pairs; teacher and lesion targets carry the context
    t = VC.table([dict(out, seed=s) for s in (1, 2)], [dict(vs, seed=s) for s in (1, 2)],
                 {s: res["ledger"] for s in (1, 2)})
    assert t["verdict"] in ("TRANSFER", "NO-TRANSFER", "ABSTAINS", "UNDEFINED", "VOID")
    assert not t["void_nights"]


def test_engaged_rule():
    good = {"losses": [(1.0, 0.0)] * 10 + [(0.5, 0.0)] * 100 + [(0.2, 0.0)] * 10, "agreement": [0.4, 0.95]}
    assert VC.engaged(good)
    assert not VC.engaged({**good, "agreement": [0.4, 0.85]})
    assert not VC.engaged({**good, "losses": [(1.0, 0.0)] * 120})


def test_context_for_uses_the_ledger_telling():
    led = {"pixel": {"telling": "My cat is called Pixel."}}
    c = VC.context_for(led, "pixel")
    assert c.startswith("User: My cat is called Pixel.\n\nAssistant: ") and c.endswith("\n\n")
