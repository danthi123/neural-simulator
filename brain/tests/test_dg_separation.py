# Capability: Memory selection (pattern completion and separation)
# Record: brain/research/2026-11-27-dg-separation-port-prereg-DRAFT.md
# Expected-fail: none (mechanism and harness plumbing on the stub cortex; no model weights, no claim)
"""brain/dg_separation.py and brain/experiments/dg_screen.py on the stub cortex: the default Brain is untouched (the
module is not even imported), the code is k-sparse, the lesion is the dense centred cosine, the shuffled control
destroys content, Brain(key="dg") needs an explicit bar and persists, and the offline screen reproduces the banked
fetches on a stub life and scores."""
import json
import subprocess
import sys

import pytest
import torch

from brain import clock
from brain import dg_separation as DG
from brain.brain import BIRTH, Brain
from brain.experiments import dg_screen as DS
from brain.hippocampus import Hippocampus
from brain.hippocampus_keys import KeyedHippocampus
from brain.tests.test_brain_key_switch import KW, StubCortex, TALK, _meta, _talk
from brain.tests.test_life import BKW, _env  # noqa: F401  (autouse fixture: fake night, 3 turns a day)
from brain.tests.test_routing_replay import CUES, TELLINGS, T

SMALL = dict(E=512, fan_in=16)


@pytest.fixture
def store():
    clock.set(lambda: T)
    cx = StubCortex()
    _, base = cx.read(BIRTH, None)
    hc = KeyedHippocampus(key="S_all", familiarity=0.0, margin=0.0)
    for t in TELLINGS:
        hc.store(cx, base, t)
    yield hc, cx, base
    clock.set(None)


def test_off_path_never_imports_the_module():
    code = ("import sys; from brain.brain import Brain; import brain.brain as b; "
            "assert 'brain.dg_separation' not in sys.modules; "
            "assert Brain.DG_BAR is None and Brain.DG_KW == {}")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=sys.modules["brain"].__path__[0] + "/..")
    assert r.returncode == 0, r.stderr


def test_default_brain_is_unchanged(tmp_path):
    b = Brain(str(tmp_path / "h"), cortex=StubCortex(), **KW)
    assert type(b.hippocampus) is Hippocampus and b.key == "S0"
    assert "key" not in (_talk(b) and (b.save() or _meta(str(tmp_path / "h"))))


def test_dg_needs_a_bar(tmp_path):
    with pytest.raises(ValueError, match="no fitted bar"):
        Brain(str(tmp_path / "h"), cortex=StubCortex(), **KW, key="dg")


def test_kwta_code_is_k_sparse_and_overlap_is_a_fraction(store):
    hc, cx, base = store
    sep = DG.DGSeparator(frac=0.1, **SMALL)
    H = torch.stack([sep.episode_h(e) for e in hc.episodes])
    codes = DG._kwta(H - H.mean(0), sep.k)
    assert sep.k == 51 and (codes.sum(1) == sep.k).all()
    ov = DG.code_similarity(sep, H, H[2])
    assert ov.shape == (len(hc.episodes),) and ov.min() >= 0 and ov.max() <= 1
    assert int(ov.argmax()) == 2                                  # an episode's own projection is nearest to itself


def test_projection_is_linear_so_centring_can_follow_it(store):
    hc, _, _ = store
    sep = DG.DGSeparator(**SMALL)
    d = hc.episodes[0].delta
    a = sep.project(DG.matrix_blocks(d))
    b = sep.project({l: 3.0 * x for l, x in DG.matrix_blocks(d).items()})      # blocks are L2-normalised first
    assert torch.allclose(a, b, atol=2e-2)   # the stub deltas are fp16: 3x rounds
    assert sep.project(DG.matrix_blocks(d)).shape == (512,)


def test_lesion_is_the_dense_centred_cosine_and_tracks_S_all(store):
    hc, cx, base = store
    dgh = DG.DGHippocampus(familiarity=0.0, margin=0.0, E=4096, fan_in=32, lesion=True)
    dgh.episodes = hc.episodes
    for cue in CUES[:4]:
        want, got = hc.similarities(cx, base, cue), dgh.similarities(cx, base, cue)
        assert torch.corrcoef(torch.stack([want, got]))[0, 1] > 0.9   # a random projection keeps the cosine (JL)


def test_shuffled_code_loses_the_content(store):
    hc, cx, base = store
    plain = DG.DGHippocampus(familiarity=0.0, margin=0.0, frac=0.1, **SMALL)
    shuf = DG.DGHippocampus(familiarity=0.0, margin=0.0, frac=0.1, shuffle=True, sep=plain.sep)
    plain.episodes = shuf.episodes = hc.episodes
    own, sh = [], []
    for i, t in enumerate(TELLINGS):
        own.append(float(plain.similarities(cx, base, t)[i]))
        sh.append(float(shuf.similarities(cx, base, t)[i]))
    assert sum(own) / len(own) > 2 * (sum(sh) / len(sh))
    assert abs(sum(sh) / len(sh) - 0.1) < 0.1                       # the chance floor is the active fraction


def test_ca3_completion_off_equals_plain_and_on_changes_scores(store):
    hc, cx, base = store
    a = DG.DGHippocampus(familiarity=0.0, margin=0.0, frac=0.1, **SMALL)
    b = DG.DGHippocampus(familiarity=0.0, margin=0.0, sep=a.sep, ca3_beta=DS.CA3_BETA)
    a.episodes = b.episodes = hc.episodes
    x, y = a.similarities(cx, base, CUES[0]), b.similarities(cx, base, CUES[0])
    assert x.shape == y.shape and not torch.equal(x, y)
    assert int(y.argmax()) == int(x.argmax())                       # completion sharpens, it does not invent a winner here


def test_similarities_from_delta_is_similarities(store):
    hc, cx, base = store
    h = DG.DGHippocampus(familiarity=0.0, margin=0.0, **SMALL)
    h.episodes = hc.episodes
    _, st = cx.read(f"User: {CUES[1]}\n\n", base)
    assert torch.equal(h.similarities_from_delta(cx.matrix_delta(st, base)), h.similarities(cx, base, CUES[1]))


def test_brain_key_dg_stores_no_key_persists_and_reloads(tmp_path, monkeypatch):
    monkeypatch.setattr(Brain, "DG_BAR", (0.05, 0.01))
    monkeypatch.setattr(Brain, "DG_KW", dict(**SMALL, frac=0.1))
    home, c = str(tmp_path / "h"), StubCortex()
    b = Brain(home, cortex=c, **KW, key="dg")
    h = b.hippocampus
    assert isinstance(h, DG.DGHippocampus) and b.key == "dg" and (h.FAMILIARITY, h.MARGIN) == (0.05, 0.01)
    assert (Hippocampus.FAMILIARITY, Hippocampus.MARGIN) == (0.12, 0.02)
    _talk(b)
    assert h.episodes and all(e.key is None for e in h.episodes)
    reply, info = b.hear("What is my cat called?")
    assert isinstance(reply, str) and "similarity" in info
    b.save()
    assert _meta(home)["key"] == "dg"
    b2 = Brain(home, cortex=c, **KW)
    assert b2.key == "dg" and len(b2.hippocampus.episodes) == len(b.hippocampus.episodes)
    b3 = Brain(home, cortex=c, **KW, key="S0")
    assert type(b3.hippocampus) is Hippocampus and all(e.key is not None for e in b3.hippocampus.episodes)


# ------------------------------------------------------------------ the offline screen
def test_apply_bar_is_compete_and_fit_bar_reads_fire_rows_only():
    rec = {"wbest": 3, "score_best": 0.30, "margin_w": 0.05, "question": True}
    assert DS.apply_bar(rec, 0.25) == 3 and DS.apply_bar(rec, 0.35) is None
    assert DS.apply_bar({**rec, "margin_w": 0.01}, 0.25) is None
    assert DS.apply_bar({**rec, "question": False}, 0.25) is None          # statements keep the class bar ratio
    rows = [{"arms": {"X": {**rec, "score_best": s}}} for s in (0.1, 0.2, 0.3, 0.4)]
    b = DS.fit_bar(rows, "X", 2)
    assert sum(DS.apply_bar(r["arms"]["X"], b) is not None for r in rows) <= 2
    assert sum(DS.apply_bar(r["arms"]["X"], b - 0.005) is not None for r in rows) > 2 or b == 0.0


def test_auroc():
    assert DS.auroc([2, 3], [0, 1]) == 1.0 and DS.auroc([0, 1], [2, 3]) == 0.0 and DS.auroc([1], [1]) == 0.5
    assert DS.auroc([], [1]) is None


def test_dg_screen_on_a_stub_life(tmp_path, monkeypatch):
    from brain.experiments import life_kw

    monkeypatch.setattr(DS, "E", 256)
    monkeypatch.setattr(DS, "FAN_IN", 8)
    snaps = tmp_path / "snaps"
    argv = ["--seed", "7", "--days", "3", "--full-probe-nights", "1", "--frozen-probe", "--homes", str(tmp_path / "h"),
            "--out", str(tmp_path / "o" / "life_s7"), "--night-snapshots", str(snaps)]
    life_kw.main(argv, cortex=StubCortex(), brain_kw=BKW)
    res = json.load(open(str(tmp_path / "o" / "life_s7") + ".json"))
    out = DS.run(res, str(snaps), str(tmp_path / "h" / "trunk"), StubCortex(), str(tmp_path / "w"), weekend=False,
                 nights=range(1, 4), log=lambda *a: None, size=None)
    assert [n["night"] for n in out["nights"]] == [1, 2, 3]
    names = {"DG:0.02", "DG:0.05", "DGcc:0.05", "DGshuf:0.05", "DGlesion", "DG:0.05:102"}
    for n in out["nights"]:
        assert not n["void"], n["void_rows"]
        for r in n["rows"]:
            assert r["A"]["idx"] == r["banked_idx"] and names <= set(r["arms"]) and "K_rec" in r
    json.dumps(out)                                                  # records are plain data
    v = DS.score([dict(out, seed=s) for s in (1, 2, 3)], {})
    assert v["status"] == "READ"
    pv = v["verdict"]
    assert pv["primary"]["status"] and set(pv["bars"]) >= {"K", "K_registered", "DGlesion", "DG:0.05"}
    assert pv["arms"]["DG:0.05"]["seeds"]["1"]["told_n"] >= 0
    bad = DS.score([dict(out, seed=s) for s in (1, 2, 3)], {"1": ["x"]})
    assert bad["status"].startswith("INCONCLUSIVE")
