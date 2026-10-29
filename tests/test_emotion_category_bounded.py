"""BRAIN_EMOTION_CATEGORY_BOUNDED (default ON since the flip this file pins; prereg
research/findings/2026-10-27-emo-bounded-flip-PREREGISTRATION.md): the emotion-category organ built with the engine's own
bounded conductance integration (cfg.enable_bounded_conductance_integration), plus the engagement fields every read
carries (research/findings/2026-10-26-emotion-organ-bounded-conductance-derisk.md).

Pins (numpy, CPU; one organ build + development ~5 s each):
  * the flag is default ON (`_BOUNDED_DEFAULT_ON`): unset and blank read True, 1/true/yes/on read True, and an explicit
    0/false/no/off (or any value that is not an ON spelling) reads False;
  * flag "0": the organ's named categories and its latch / readout / gate spike counts at dev seed 1 equal the values
    recorded from main (011fa0036, before this flag existed) EXACTLY, and every pre-existing field of the read record
    is untouched by the added engagement fields; this is the byte-identical escape back to the pre-flip default;
  * flag "0" == bounded=False == the old unset default, record for record (the engagement fields aside: bounded_env is
    the raw string "0" here, None when unset);
  * flag unset == flag "1" == bounded=True, record for record (bounded_env aside: None when unset, "1" when set), and
    the unset read reports bounded_flag True, bounded_env None and bounded_readback True;
  * flag on (unset or "1"): the bridge's own core_config reads the guard back as True (bounded_readback), the four
    situations are still named, and no organ neuron leaves the physiological range;
  * the organ cache keys on the RESOLVED flag (a bounded and an unbounded organ for one seed are different organs, and
    unset and "1" share one);
  * the out-of-range census sees an excursion below -110 mV and ignores a quiet step;
  * the production hook `read_turn` carries the engagement fields into the feel block, with the env unset and with "1".

The golden counts are spike counts of a seeded Izhikevich run, recorded on the workstation (numpy CPU). They are exact on
a machine with the same numpy float behaviour; if ONLY the golden-count test fails on another machine, compare the
flag-off (=0) organ against the pre-flag file (`git show 011fa0036:research/runners/emotion_category_organ.py`) before
believing it.
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


def _reads_with_env(value, seed=1):
    """Build + develop + read the dev organ with BRAIN_EMOTION_CATEGORY_BOUNDED set to `value` (None = unset)."""
    old = os.environ.get(ENV)
    if value is None:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = value
    try:
        org = _E().EmotionCategoryOrgan(seed=seed).train()
    finally:
        if old is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = old
    return org, _reads(org)


@pytest.fixture(scope="module")
def off_reads():
    """The escape: BRAIN_EMOTION_CATEGORY_BOUNDED=0 (the pre-flip default's behaviour)."""
    return _reads_with_env("0")[1]


@pytest.fixture(scope="module")
def unset_reads():
    """The new production default: the variable is absent."""
    return _reads_with_env(None)[1]


@pytest.fixture(scope="module")
def on_reads():
    """Explicitly on (BRAIN_EMOTION_CATEGORY_BOUNDED=1), as the 10-27 gate's ON lines ran it."""
    return _reads_with_env("1")[1]


def test_default_constant_is_on():
    assert _E()._BOUNDED_DEFAULT_ON is True


def test_offarm_staleness_gate_resolves_the_default_on():
    """tools/gates/flip_offarm_staleness.py (CLASS OS) must see this default as ON, so a future soak/flip/verify runner
    that pops the flag as its OFF arm (the 2026-08-27 class) blocks. It resolves a bare literal
    os.environ.get("BRAIN_EMOTION_CATEGORY_BOUNDED") gated on a `_*DEFAULT_ON` constant; a read through a variable
    resolves UNKNOWN (None) and the gate skips the flag."""
    sys.path.insert(0, os.path.join(REPO, "tools", "gates"))
    import flip_offarm_staleness as gate
    assert gate._default_state(ENV, REPO) == "on"


@pytest.mark.parametrize("v,want", [(None, True), ("", True), ("   ", True), ("0", False), ("off", False),
                                    ("false", False), ("no", False), (" 0 ", False), ("banana", False),
                                    ("1", True), ("true", True), (" on ", True), ("yes", True), ("TRUE", True)])
def test_flag_default_on(monkeypatch, v, want):
    if v is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, v)
    assert _E().bounded_requested() is want


def test_flag_off_matches_main_exactly(off_reads):
    """=0 reproduces the organ as it was before the flag existed (golden counts from main 011fa0036), spike for spike."""
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
        assert r["bounded_flag"] is False and r["bounded_env"] == "0" and r["bounded_readback"] is False


def test_zero_equals_bounded_false_record_for_record(off_reads):
    """=0 and bounded=False are the same organ: every field, the engagement fields included except the raw env string."""
    explicit = _reads(_E().EmotionCategoryOrgan(seed=1, bounded=False).train())
    for c in CATS:
        for k in PRE_EXISTING + ("oor_count", "oor_count_session", "bounded_flag", "bounded_readback"):
            assert off_reads[c][k] == explicit[c][k], (c, k)
        assert off_reads[c]["bounded_env"] == "0" and explicit[c]["bounded_env"] is None


@pytest.mark.parametrize("spelling", ["off", "false", "no"])
def test_every_off_spelling_resolves_to_the_unbounded_organ(monkeypatch, spelling):
    """Construction reads the env once and builds nothing (the build is lazy), so this is cheap."""
    monkeypatch.setenv(ENV, spelling)
    org = _E().EmotionCategoryOrgan(seed=1)
    assert org.bounded is False and org.bounded_env == spelling


def test_unset_is_on_and_equals_explicit_one_equals_bounded_true(unset_reads, on_reads):
    """The flip: the variable absent builds the SAME organ as =1 and as bounded=True (record for record; only the raw
    env string differs, None vs "1"), and it is not the =0 organ at the engagement fields."""
    explicit = _reads(_E().EmotionCategoryOrgan(seed=1, bounded=True).train())
    for c in CATS:
        u = unset_reads[c]
        assert u["bounded_flag"] is True and u["bounded_env"] is None and u["bounded_readback"] is True, c
        assert on_reads[c]["bounded_env"] == "1" and explicit[c]["bounded_env"] is None
        for k in PRE_EXISTING + ("oor_count", "oor_count_session", "bounded_flag", "bounded_readback"):
            assert u[k] == on_reads[c][k] == explicit[c][k], (c, k)
        assert u["category"] == c and u["read_invariant"] is True and u["co_commit"] is False, c
        assert u["oor_count"] == 0 and u["oor_count_session"] == 0, c


def test_flag_on_is_read_back_from_the_bridge(on_reads):
    for c in CATS:
        r = on_reads[c]
        assert r["bounded_flag"] is True and r["bounded_env"] == "1" and r["bounded_readback"] is True, c
        assert r["category"] == c and r["read_invariant"] is True and r["co_commit"] is False, c
        assert r["oor_count"] == 0 and r["oor_count_session"] == 0, c


@pytest.mark.parametrize("value,want", [(None, True), ("1", True), ("0", False)])
def test_organ_config_is_the_engine_guard_per_flag_state(monkeypatch, value, want):
    """Unset builds the engine guard (the new default), =1 builds it, =0 leaves the engine's own default (off)."""
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    org = _E().EmotionCategoryOrgan(seed=2)
    org._ensure()
    cfg = org._bridge.core_config
    assert cfg.enable_bounded_conductance_integration is want
    assert cfg.neuron_model_type == "IZHIKEVICH"
    assert org.bounded is want and org.bounded_env == value
    org2 = _E().EmotionCategoryOrgan(seed=2, bounded=not want)
    org2._ensure()
    assert org2._bridge.core_config.enable_bounded_conductance_integration is (not want)


def test_organ_cache_keys_on_the_resolved_flag(monkeypatch):
    E = _E()
    monkeypatch.setenv(ENV, "0")
    a = E.get_organ(seed=3)
    monkeypatch.delenv(ENV, raising=False)
    b = E.get_organ(seed=3)
    assert a is not b and a.bounded is False and b.bounded is True
    monkeypatch.setenv(ENV, "1")
    assert E.get_organ(seed=3) is b                    # unset and "1" resolve to one organ (the cache keys on the bool)
    monkeypatch.setenv(ENV, "0")
    assert E.get_organ(seed=3) is a


class _StubBridge:
    def __init__(self, v):
        self.cp_membrane_potential_v = np.asarray(v, dtype=np.float32)

    def _run_one_simulation_step(self):
        pass


def test_oor_census_sees_excursion_and_ignores_quiet_step():
    E = _E()
    org = E.EmotionCategoryOrgan(seed=1)
    cen = org._census()      # the on-device census (tests/test_emotion_census_device.py pins it against the old host census)
    org._bridge = _StubBridge([-65.0, -80.0, 30.0, -109.9])
    org._step(cen)
    assert cen.members() == set()
    org._bridge = _StubBridge([-65.0, -137.0, 30.0, -110.5])
    org._step(cen)
    assert cen.members() == {1, 3}


@pytest.mark.parametrize("value,env_seen,flag", [(None, None, True), ("1", "1", True), ("0", "0", False)])
def test_read_turn_carries_engagement_fields(monkeypatch, value, env_seen, flag):
    E = _E()
    monkeypatch.setenv("BRAIN_CHAT_SEED", "1")
    monkeypatch.setattr(E, "_ORGANS", {})              # a fresh process-shared cache: each state builds its own organ
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    key = ("t-hook",)
    E.observe_turn(key, SITUATIONS["joy"])
    E.observe_turn(key, SITUATIONS["joy"])
    ec = E.read_turn(key, affect_lesion=False)
    assert ec["category"] == "joy"
    assert ec["bounded_flag"] is flag and ec["bounded_readback"] is flag and ec["bounded_env"] == env_seen
    assert ec["oor_count_session"] == 0 or flag is False     # the unbounded organ may legitimately excurse (seed-dependent)
    out = E.apply_to_feel({"answer": "x"}, key, False)
    assert out["emotion_category"]["bounded_readback"] is flag and out["emotion_category"]["error"] is None
    E.forget_session(key)


def test_server_never_reads_the_bounded_flag():
    """Flag off must leave the handler alone: the env flag is read only inside the organ module."""
    src = open(os.path.join(REPO, "webapp", "server.py")).read()
    assert "EMOTION_CATEGORY_BOUNDED" not in src
