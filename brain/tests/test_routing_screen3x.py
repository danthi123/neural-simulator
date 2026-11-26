# Capability: Memory selection (pattern completion and separation)
# Record: brain/research/2026-11-26-routing-screen3-prereg.md (part X)
# Expected-fail: none (harness plumbing on the stub cortex; no model weights)
"""routing_screen3x: KX is K's competition with the permanent episodes excluded (never fetched, otherwise K's choice
when K's winner is not permanent); the scorer applies X1-X7 and the engagement rule."""
import pytest

from brain import clock
from brain.brain import BIRTH
from brain.experiments import routing_screen2 as S2
from brain.experiments import routing_screen3x as S3
from brain.hippocampus import Hippocampus
from brain.tests.test_brain_key_switch import StubCortex
from brain.tests.test_routing_replay import CUES, TELLINGS, T
from brain.tests.test_life import _env  # noqa: F401  (autouse fixture: fake night, 3 turns a day)


@pytest.fixture
def store():
    clock.set(lambda: T)
    cx = StubCortex()
    _, base = cx.read(BIRTH, None)
    hc = Hippocampus()
    for i, t in enumerate(TELLINGS):
        e = hc.store(cx, base, t, permanent=(i in (1, 3)))
        e.strength, e.last_used = [1.0, 9.0, 0.5, 8.5, 1.0, 2.0][i], T - 86400.0 * [10, 1, 6, 0, 3, 12][i]
    yield hc, cx, base
    clock.set(None)


def test_kx_never_fetches_permanent(store):
    hc, cx, base = store
    k = S2.keyed_from(hc)
    k.FAMILIARITY = k.FAMILIARITY_SPONTANEOUS = -1.0
    k.MARGIN = -1.0
    perm = tuple(i for i, e in enumerate(hc.episodes) if e.permanent)
    saw_perm = False
    for cue in CUES + TELLINGS:
        dk = S3.decide(k, cx, base, cue, T)
        dx = S3.decide(k, cx, base, cue, T, exclude=perm)
        assert dx["idx"] not in perm
        saw_perm |= dk["idx"] in perm
        if dk["idx"] not in perm:
            assert dx["idx"] == dk["idx"]
        assert dk == S3.decide(k, cx, base, cue, T)          # deterministic, nothing changed
    assert saw_perm, "the stub store lets a permanent episode win somewhere (engagement)"
    # K's decide equals screen 2's decide_K
    for cue in CUES:
        assert S3.decide(k, cx, base, cue, T)["idx"] == S2.decide_K(k, cx, base, cue, T)["idx"]


def _row(cls, K, KX, wperm=False):
    def arm(x):
        idx, right, perm = x
        return {"idx": idx, "right_episode": right if cls in ("told", "identity") else None, "perm": perm, "text": ""}
    return {"cls": cls, "K_weighted_winner_perm": wperm, "K": arm(K), "KX": arm(KX)}


def _out(seed, rows):
    return {"seed": seed, "nights": [{"night": 5, "rows": rows, "dropped_rows": [], "void": False}]}


def test_scorer_pass_and_kills():
    good = ([_row("told", (1, False, True), (2, True, False), True)] * 5
            + [_row("told", (3, True, False), (3, True, False))] * 10
            + [_row("told", (None, False, False), (None, False, False))] * 2)
    v = S3.score([_out(s, good) for s in (1, 2, 3)])
    assert v["status"].startswith("PASS"), v
    bad = ([_row("told", (1, False, True), (None, False, False), True)] * 7
           + [_row("told", (3, True, False), (3, True, False))] * 10)
    v = S3.score([_out(s, bad) for s in (1, 2, 3)])
    assert v["status"] == "KILLED" and any(k.startswith("X7") for k in v["X"]["kills"])
    none = [_row("told", (3, True, False), (3, True, False))] * 10
    assert S3.score([_out(s, none) for s in (1, 2, 3)])["status"] == "UNDEFINED"


def test_open_snapshot_of_an_sall_life(tmp_path):
    """An S_all life keeps no S0 key (keys.pt holds None): open_snapshot with deltas accepts None == None (review 11-26)."""
    import json
    from brain.experiments import life_kw
    from brain.experiments import routing_replay as RR
    from brain.tests.test_life import BKW

    snaps = tmp_path / "snaps"
    argv = ["--seed", "7", "--days", "2", "--full-probe-nights", "1", "--frozen-probe", "--homes", str(tmp_path / "h"),
            "--out", str(tmp_path / "o" / "life_s7"), "--night-snapshots", str(snaps), "--key", "S_all"]
    life_kw.main(argv, cortex=StubCortex(), brain_kw=BKW)
    res = json.load(open(str(tmp_path / "o" / "life_s7") + ".json"))
    assert res["brain_kw"]["key"] == "S_all"
    b = RR.open_snapshot(str(snaps / "n1"), str(tmp_path / "home"), StubCortex(), res["brain_kw"],
                         trunk=str(tmp_path / "h" / "trunk"), with_deltas=True)
    try:
        eps = b.hippocampus.episodes
        assert eps and all(e.key is None and e.delta is not None for e in eps)
    finally:
        b.release()
