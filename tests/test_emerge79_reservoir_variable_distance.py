"""CPU tests for EMERGE-79 -- the UNCONTINGENT variable-distance reservoir-necessity test.

Verifies the construction generators + the local-sanity shape, and (slow) the seed-42 de-risk: a REAL discovered voice
marker flips a far word's role across a VARIABLE distance; the reservoir beats every fixed window (uncontingent -- no OOV
trick), the window CAN do the task locally (fails only from distance), and the role is mark-determined. CPU/numpy, offline.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import pytest

import research.runners._emerge79_reservoir_variable_distance_derisk as m79


def test_mark_flips_the_far_word_role():
    rng = np.random.default_rng(0)
    subj, obj = ["dog", "cat"], ["ball", "bone"]
    ta, oa, ra = m79._make("act", 3, rng, subj, obj)
    tp, op, rp = m79._make("pas", 3, rng, subj, obj)
    assert ra == "THEME" and rp == "AGENT"          # same shape, the START marker flips the END word's role
    assert ta[0] == "act" and tp[0] == "pas"
    assert oa == len(ta) - 1                          # the scored word is the LAST token
    # more fillers -> the marker sits further from the scored word (variable distance)
    t5, _o5, _r5 = m79._make("act", 5, rng, subj, obj)
    assert t5.count(m79._FILLER) == 5 and ta.count(m79._FILLER) == 3


def test_local_sanity_shape_puts_mark_adjacent():
    rng = np.random.default_rng(0)
    toks, o_idx, role = m79._make_local("pas", rng, ["dog"], ["ball"])
    assert toks[-1] == "pas" and o_idx == 4          # mark adjacent to the scored word -> a +-2 window can see it
    assert role == "AGENT"


def test_roles_are_the_flipping_binary():
    assert m79._ROLES == ["AGENT", "THEME"] and m79._MARKS == ["act", "pas"]


@pytest.mark.slow
def test_seed42_derisk_uncontingent_go():
    d = m79._derisk_one(42)
    # the marker is a REAL discovered closed cue (no OOV trick -- resolves the EMERGE-78 contingency)
    assert d["marks_only_discovered"] is True
    # at every tested distance the fixed +-2 window is at chance while the reservoir resolves it
    for dd in m79._TEST_DISTS:
        assert d["by_distance"][dd]["window2"] <= 0.65, (dd, d["by_distance"][dd])
        assert d["by_distance"][dd]["reservoir"] >= 0.90, (dd, d["by_distance"][dd])
    # the window CAN do the role task LOCALLY -> it fails on the distal case only because of distance
    assert d["window2_local_sanity"] >= 0.90
    # the role is genuinely marker-determined
    assert (d["res_ctl"] - d["res_mark_lesion"]) >= 0.30
