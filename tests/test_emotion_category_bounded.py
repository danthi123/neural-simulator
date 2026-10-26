"""BRAIN_EMOTION_CATEGORY_BOUNDED (default OFF): the emotion-category organ built with the engine's own bounded
conductance integration (cfg.enable_bounded_conductance_integration), plus the engagement fields every read carries
(research/findings/2026-10-26-emotion-organ-bounded-conductance-derisk.md).

Pins (numpy, CPU; one organ build + development ~5 s each):
  * the flag is default OFF and reads 1/true/yes/on;
  * flag unset: the organ's named categories and its latch / readout / gate spike counts at dev seed 1 equal the values
    recorded from main (011fa0036, before this flag existed) EXACTLY, and every pre-existing field of the read record
    is untouched by the added engagement fields;
  * flag unset == flag explicitly "0" == bounded=False, record for record;
  * flag "1": the bridge's own core_config reads the guard back as True (bounded_readback), the flag and the raw env
    string are reported, the four situations are still named, and no organ neuron leaves the physiological range;
  * the organ cache keys on the flag (a bounded and an unbounded organ for one seed are different organs);
  * the out-of-range census sees an excursion below -110 mV and ignores a quiet step;
  * the production hook `read_turn` carries the engagement fields into the feel block.

The golden counts are spike counts of a seeded Izhikevich run, recorded on the workstation (numpy CPU). They are exact on
a machine with the same numpy float behaviour; if ONLY the golden-count test fails on another machine, compare the flag-off
organ against the pre-flag file (`git show 011fa0036:research/runners/emotion_category_organ.py`) before believing it.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

ENV = "BRAIN_EMOTION_CATEGORY_BOUNDED"
CATS = ("joy", "fear", "anger", "sadness")
SITUATIONS = {
    "joy": "you win a gift and you are happy and glad and delighted",
    "fear": "a wolf is attacking you and you are scared and afraid",
    "anger": "someone cheats you and you are angry and mad and furious",
    "sadness": "you are alone and lonely and sad and you cry",
}
# recorded from main 011fa0036 (the organ before the flag existed): dev seed 1, GRID row 0, each situation heard twice
# then one read. (latch slot_counts, readout_counts, gate_counts) per channel, in CATS order.
GOLDEN_SEED1 = {
    "joy": ("joy", [93, 0, 0, 0], [75, 0, 0, 0], [60, 0, 0, 0]),
    "fear": ("fear", [0, 84, 0, 0], [0, 57, 0, 0], [0, 56, 0, 0]),
    "anger": ("anger", [0, 0, 84, 0], [0, 0, 69, 0], [0, 0, 59, 0]),
    "sadness": ("sadness", [0, 0, 0, 89], [0, 0, 0, 76], [0, 0, 0, 60]),
}
PRE_EXISTING = ("readout_counts", "gate_counts", "slot_counts", "lesioned", "commit_lesioned", "latch_lesioned",
                "category", "latched", "co_commit", "threshold", "read_variants", "read_invariant")
ENGAGEMENT = ("bounded_flag", "bounded_env", "bounded_readback", "oor_count", "oor_count_session")


def _E():
    import research.runners.emotion_category_organ as E
    return E


def _reads(org, key=("t",)):
    """The four situations in the gate's order, each twice, a read after each: {channel: read record}."""
    out = {}
    for c in CATS:
        for _ in range(2):
            org.observe(key, SITUATIONS[c])
        out[c] = org.read(key)
    return out


def _vec(d):
    return [d[c] for c in CATS]


@pytest.fixture(scope="module")
def off_reads():
    os.environ.pop(ENV, None)
    return _reads(_E().EmotionCategoryOrgan(seed=1).train())


@pytest.fixture(scope="module")
def on_reads():
    old = os.environ.get(ENV)
    os.environ[ENV] = "1"
    try:
        org = _E().EmotionCategoryOrgan(seed=1).train()
    finally:
        if old is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = old
    return _reads(org)


@pytest.mark.parametrize("v,want", [(None, False), ("", False), ("0", False), ("off", False), ("1", True),
                                    ("true", True), (" on ", True), ("yes", True)])
def test_flag_default_off(monkeypatch, v, want):
    if v is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, v)
    assert _E().bounded_requested() is want


def test_flag_off_matches_main_exactly(off_reads):
    for c in CATS:
        cat, slots, outs, gates = GOLDEN_SEED1[c]
        r = off_reads[c]
        assert r["category"] == cat and r["co_commit"] is False and r["read_invariant"] is True, c
        assert _vec(r["slot_counts"]) == slots, c
        assert _vec(r["readout_counts"]) == outs, c
        assert _vec(r["gate_counts"]) == gates, c


def test_flag_off_engagement_fields_added_not_changed(off_reads):
    for c in CATS:
        r = off_reads[c]
        assert all(k in r for k in PRE_EXISTING + ENGAGEMENT), c
        assert r["bounded_flag"] is False and r["bounded_env"] is None and r["bounded_readback"] is False


def test_unset_equals_explicit_zero_equals_bounded_false(off_reads, monkeypatch):
    monkeypatch.setenv(ENV, "0")
    org0 = _E().EmotionCategoryOrgan(seed=1).train()
    assert org0.bounded is False and org0.bounded_env == "0"
    zero = _reads(org0)
    monkeypatch.delenv(ENV, raising=False)
    explicit = _reads(_E().EmotionCategoryOrgan(seed=1, bounded=False).train())
    for c in CATS:
        for k in PRE_EXISTING + ("oor_count", "oor_count_session", "bounded_flag", "bounded_readback"):
            assert zero[c][k] == off_reads[c][k] == explicit[c][k], (c, k)


def test_flag_on_is_read_back_from_the_bridge(on_reads):
    for c in CATS:
        r = on_reads[c]
        assert r["bounded_flag"] is True and r["bounded_env"] == "1" and r["bounded_readback"] is True, c
        assert r["category"] == c and r["read_invariant"] is True and r["co_commit"] is False, c
        assert r["oor_count"] == 0 and r["oor_count_session"] == 0, c


def test_flag_on_organ_config_is_the_engine_guard(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    org = _E().EmotionCategoryOrgan(seed=2)
    org._ensure()
    cfg = org._bridge.core_config
    assert cfg.enable_bounded_conductance_integration is True
    assert cfg.neuron_model_type == "IZHIKEVICH"
    org2 = _E().EmotionCategoryOrgan(seed=2, bounded=False)
    org2._ensure()
    assert org2._bridge.core_config.enable_bounded_conductance_integration is False


def test_organ_cache_keys_on_the_flag(monkeypatch):
    E = _E()
    monkeypatch.delenv(ENV, raising=False)
    a = E.get_organ(seed=3)
    monkeypatch.setenv(ENV, "1")
    b = E.get_organ(seed=3)
    assert a is not b and a.bounded is False and b.bounded is True
    monkeypatch.delenv(ENV, raising=False)
    assert E.get_organ(seed=3) is a


class _StubBridge:
    def __init__(self, v):
        self.cp_membrane_potential_v = np.asarray(v, dtype=np.float32)

    def _run_one_simulation_step(self):
        pass


def test_oor_census_sees_excursion_and_ignores_quiet_step():
    E = _E()
    org = E.EmotionCategoryOrgan(seed=1)
    seen = set()
    org._bridge = _StubBridge([-65.0, -80.0, 30.0, -109.9])
    org._step(seen)
    assert seen == set()
    org._bridge = _StubBridge([-65.0, -137.0, 30.0, -110.5])
    org._step(seen)
    assert seen == {1, 3}


def test_read_turn_carries_engagement_fields(monkeypatch):
    E = _E()
    monkeypatch.setenv("BRAIN_CHAT_SEED", "1")
    monkeypatch.setenv(ENV, "1")
    key = ("t-hook",)
    E.observe_turn(key, SITUATIONS["joy"])
    E.observe_turn(key, SITUATIONS["joy"])
    ec = E.read_turn(key, affect_lesion=False)
    assert ec["category"] == "joy"
    assert ec["bounded_flag"] is True and ec["bounded_readback"] is True and ec["oor_count_session"] == 0
    out = E.apply_to_feel({"answer": "x"}, key, False)
    assert out["emotion_category"]["bounded_readback"] is True and out["emotion_category"]["error"] is None
    E.forget_session(key)


def test_server_never_reads_the_bounded_flag():
    """Flag off must leave the handler alone: the env flag is read only inside the organ module."""
    src = open(os.path.join(REPO, "webapp", "server.py")).read()
    assert "EMOTION_CATEGORY_BOUNDED" not in src
