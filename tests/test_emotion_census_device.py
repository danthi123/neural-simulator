"""The emotion-category organ's out-of-range census, moved on-device (`DeviceCensus`): a running per-neuron minimum of v
folded in with one elementwise fmin per simulation step, read back to the host ONCE per observe / read. On CuPy the
former per-step host copy (`to_host(v)` + `v.min()`) cost 8-12% per step (workstation, organ seed 1: 7.8-19.2% over five
reps); the device census is meant to be a few percent at most. The counts it returns must not change.

Pins (numpy, CPU; one organ build + development ~5 s per dev seed):
  * LOCKSTEP equivalence: on dev seeds 1 and 7, over a full feel read sequence (the four situations, each heard twice,
    a read after each, then the affect_out-lesioned read), the old per-step host census (kept below as `_old_census_step`,
    the body of main cadf0f40f's `_step` after the simulation step) and the device census see the SAME v after every
    step. The distinct-neuron sets agree at every observe and read, and so do `oor_count` / `oor_count_session` against
    the old bookkeeping, at the real -110 mV threshold (a few session hits: seed 1 one neuron, seed 7 two) and at a
    sensitive -70 mV threshold (36-38 of 920 neurons per read, sets that grow over the session), so the check is not
    vacuous;
  * GOLDEN: the same sequence's (oor_count, oor_count_session) per read recorded from main cadf0f40f (the old code, run
    unmodified, byte-identical full read records to this code on dev seeds 1 and 7 at -110 / -70 / -66 mV);
  * the census does ONE host transfer per read, none per step (`to_host` is counted), and one device op per step;
  * a NaN membrane value is ignored (fmin) and hides no other neuron; a quiet step adds nothing; an excursion is seen.

The golden values are counts from a seeded Izhikevich run (numpy CPU, workstation); if ONLY the golden test fails on
another machine, trust the lockstep test (it needs no recorded numbers).
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import emotion_category_organ as E  # noqa: E402
from sim.backend import to_host  # noqa: E402

CATS = E.CATS
SITUATIONS = {
    "joy": "you win a gift and you are happy and glad and delighted",
    "fear": "a wolf is attacking you and you are scared and afraid",
    "anger": "someone cheats you and you are angry and mad and furious",
    "sadness": "you are alone and lonely and sad and you cry",
}
DEV_SEEDS = (1, 7)
THRESHOLDS = (-110.0, -70.0)     # the real one; a sensitive one at which 36-38 of 920 neurons cross per read

# (oor_count, oor_count_session) of the five reads (joy, fear, anger, sadness, lesioned), recorded from main cadf0f40f
# (the per-step host census), same sequence as `_sequence` below, session key ("seq",).
GOLDEN_FROM_MAIN = {
    (1, -110.0): [(0, 0), (0, 0), (0, 1), (0, 1), (0, 1)],
    (1, -70.0): [(38, 276), (36, 361), (37, 381), (36, 404), (0, 368)],
    (7, -110.0): [(0, 0), (0, 0), (0, 0), (0, 2), (0, 2)],
    (7, -70.0): [(36, 277), (37, 367), (36, 392), (36, 414), (0, 378)],
}


def _old_census_step(v, oor):
    """The REFERENCE: the per-step host census of main cadf0f40f, verbatim (a host copy of the membrane array and a min
    each step; every neuron now below OOR_MV joins `oor`)."""
    v = to_host(v)
    if float(v.min()) < E.OOR_MV:
        oor.update(int(i) for i in np.flatnonzero(v < E.OOR_MV))


class PairedCensus(E.DeviceCensus):
    """The device census plus the old host census fed the SAME v each step (`ref`), so the two are compared on identical
    state with no determinism assumption."""

    made = []

    def __init__(self, xp=None):
        super().__init__(xp)
        self.ref = set()
        PairedCensus.made.append(self)

    def update(self, v):
        super().update(v)
        _old_census_step(v, self.ref)


@pytest.fixture(scope="module", params=DEV_SEEDS)
def organ(request):
    return request.param, E.EmotionCategoryOrgan(seed=request.param).train()


def _sequence(org, key):
    """The full feel read sequence on one session: every situation twice then a read, then the lesioned read. Returns
    [(kind, record)] in order."""
    out = []
    for c in CATS:
        for _ in range(2):
            out.append(("observe", org.observe(key, SITUATIONS[c])))
        out.append(("read", org.read(key)))
    out.append(("read", org.read(key, lesion=True)))
    return out


@pytest.mark.parametrize("mv", THRESHOLDS)
def test_lockstep_device_census_equals_old_host_census(organ, monkeypatch, mv):
    seed, org = organ
    monkeypatch.setattr(E, "OOR_MV", mv)
    monkeypatch.setattr(org, "_census", lambda: PairedCensus(org._xp))
    PairedCensus.made.clear()
    key = ("lockstep", seed, mv)
    ref_session = set()                     # the old bookkeeping: the union of the session's observes
    golden = GOLDEN_FROM_MAIN[(seed, mv)]
    reads = []
    seq = _sequence(org, key)
    assert len(PairedCensus.made) == len(seq) == 13, "each observe / read builds exactly one census"
    for (kind, rec), cen in zip(seq, PairedCensus.made):
        assert cen.members() == cen.ref, (seed, mv, kind, len(cen.members()), len(cen.ref))
        if kind == "observe":
            ref_session |= cen.ref
            assert rec["oor_count_session"] == len(ref_session), (seed, mv, rec["oor_count_session"], len(ref_session))
        else:
            assert rec["oor_count"] == len(cen.ref)
            assert rec["oor_count_session"] == len(cen.ref | ref_session)
            reads.append((rec["oor_count"], rec["oor_count_session"]))
    assert reads == golden[:len(reads)]
    if mv == -70.0:      # non-vacuous: partial, growing sets
        assert all(0 < n < 920 for n, _ in reads[:4]) and reads[3][1] > reads[0][1]


def test_forget_session_clears_the_census_record(organ):
    seed, org = organ
    key = ("forget", seed)
    org.observe(key, SITUATIONS["joy"])
    org.forget(key)
    assert key not in org._oor_sessions


class _StubBridge:
    def __init__(self, v):
        self.cp_membrane_potential_v = np.asarray(v, dtype=np.float32)

    def _run_one_simulation_step(self):
        pass


def test_quiet_step_adds_nothing_and_excursion_is_seen():
    org = E.EmotionCategoryOrgan(seed=1)
    cen = org._census()
    org._bridge = _StubBridge([-65.0, -80.0, 30.0, -109.9])
    org._step(cen)
    assert cen.members() == set()
    org._bridge = _StubBridge([-65.0, -137.0, 30.0, -110.5])
    org._step(cen)
    assert cen.members() == {1, 3}
    org._bridge = _StubBridge([-65.0, -60.0, 30.0, -60.0])     # a neuron that left the range and came back stays counted
    org._step(cen)
    assert cen.members() == {1, 3}
    assert E.DeviceCensus().members() == set()                  # no step seen


def test_nan_is_ignored_and_hides_no_other_neuron():
    cen = E.DeviceCensus()
    cen.update(np.array([-65.0, np.nan, -120.0, -90.0], dtype=np.float32))
    cen.update(np.array([-65.0, -111.0, np.nan, -90.0], dtype=np.float32))
    assert cen.members() == {1, 2}


class _CountingXP:
    """numpy with the census's device ops counted."""

    def __init__(self):
        self.n_fmin = self.n_full = 0

    def __getattr__(self, name):
        return getattr(np, name)

    def fmin(self, *a, **k):
        self.n_fmin += 1
        return np.fmin(*a, **k)

    def full_like(self, *a, **k):
        self.n_full += 1
        return np.full_like(*a, **k)


def test_one_device_op_per_step_and_one_host_transfer_per_read(monkeypatch):
    calls = []
    real = E.to_host
    monkeypatch.setattr(E, "to_host", lambda a: (calls.append(1), real(a))[1])
    xp = _CountingXP()
    org = E.EmotionCategoryOrgan(seed=1)
    org._bridge = _StubBridge([-65.0, -120.0, 30.0, -50.0])
    cen = E.DeviceCensus(xp)
    for _ in range(50):
        org._step(cen)
    assert xp.n_fmin == 50 and xp.n_full == 1 and calls == [], "no host transfer while stepping"
    assert cen.members() == {1} and len(calls) == 1, "exactly one host transfer to build the read's fields"
