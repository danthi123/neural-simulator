"""R3 (scale-validate the R1 fused fabric at PRODUCTION V=320): the FUSED-REDUCED fabric folds the FULL-V divnorm
score pool + the REDUCED (V'_A/V'_X) K-way sequencer onto ONE Izhikevich bridge and routes the cleanup membrane ->
score pool DEVICE-RESIDENT (no to_host of the cleanup score), at the consolidated-320 production scale (320 stream-
learned cortex codes + the grounded projection). These tests pin the load-bearing R3 properties at V=320 (CPU/numpy
oracle, D=128 -- the production CPU dim; the GPU 6-seed at D=2048 is the controller's run):

  * the fused-reduced path's per-query decision == the host `_scan` oracle (==host) at V=320;
  * the no-confab MOAT holds (an absent/cross cue abstains, 0 false-accepts) at V=320 -- the HARD gate;
  * the cleanup-score `to_host` is GONE from the fused query path (the R1 seam stays closed AT SCALE);
  * lesion fails safe (sever the cleanup->score drive -> abstain, never confabulate);
  * the reduced cue vocab stays MODEST (V'_A / V'_X are the distinct stored agents/actions, NOT 320) -- the
    scale-wiring that makes the fused path tractable at V=320 (12,730 vs ~225K neurons full-V).

These build two D=128/V=320 composers (~36K-neuron RF bridges) + the fused-reduced fabric, so each test is heavier
than the V=72 R1 gate; they are marked `slow`. Skips gracefully if the 320 stream codes are absent.

Run on CPU: SIM_BACKEND=numpy pytest tests/test_seq_fused_fabric_320.py -v
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.consolidated_320_conversation_demo import FACTS, ABSENT_WHAT
from research.runners._seq_fused_fabric_320 import (
    _load_codes, _build_composers, fused_reduced_query_patient, fused_seq_block_reduced,
    fused_reduced_query_patient_lesioned, build_fused_fabric_reduced_bridge,
)
from research.runners._phaseB_onebrain_sequencerK_derisk import host_scan_block, patient_of

D = 128
SEED = 42
_CODES = _load_codes(SEED, "neural")
_HAVE_CODES = _CODES is not None
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not _HAVE_CODES, reason="320 stream-learned codes (_phaseB_stream_codes_320_*) absent")]

# Build ONE pair of composers for the module (the build is the expensive part); the tests share it.
_COMPOSERS = None


def _composers():
    global _COMPOSERS
    if _COMPOSERS is None:
        c_host, c_fused, _vocab = _build_composers(SEED, D, _CODES, list(FACTS))
        _COMPOSERS = (c_host, c_fused)
    return _COMPOSERS


# A small representative cue sample (the full 8-fact battery + GPU 6 seeds is the controller's run).
_PRESENT = list(FACTS)[:3]                 # dog->apple, cat->ball, bird->tree
_ABSENT = list(ABSENT_WHAT)[:3]            # (dog,sing) (cat,run) (bird,eat) -- never stored together


def test_reduced_fabric_cue_vocab_is_modest():
    """The scale-wiring claim: at V=320 with the 8-fact production store the reduced fabric builds the sequencer over
    only the DISTINCT stored agents/actions (V'_A / V'_X), NOT 320 -- so K stays modest per cue and the fabric is
    tractable (~12.7K neurons vs ~225K full-V)."""
    c_host, c_fused = _composers()
    # touch a query so the fused-reduced fabric is built (lazily) and the reduced maps populated.
    _ = fused_seq_block_reduced(c_fused, _PRESENT[0][0], _PRESENT[0][1])
    assert c_fused.V == 320, f"expected the production V=320, got {c_fused.V}"
    assert len(c_fused._fused320_mapA) <= 8 and len(c_fused._fused320_mapX) <= 8, (
        f"reduced cue vocab not modest: VA={len(c_fused._fused320_mapA)} VX={len(c_fused._fused320_mapX)} "
        f"(should be the distinct stored agents/actions, not 320)")
    sb, _meta = c_fused._fused320_seq
    assert sb.core_config.num_neurons < 30000, (
        f"reduced fabric too large at V=320 ({sb.core_config.num_neurons:,} neurons) -- the shrink did not apply")


def test_fused_reduced_never_confabulates_at_v320():
    """The R3 answer gate at the CPU/D=128 scale: the fused-reduced path's query_patient is ALWAYS host-or-abstain --
    it NEVER emits a WRONG patient (the no-confab guarantee, independent of D fidelity). A present cue MAY over-abstain
    at low D (the documented code-fidelity miss, the SAFE direction -- both the fused AND the production separate-bridge
    spiking path do this at D=128; the consolidated_320 R3 note: "over-abstains 2/8 on the K=8 demo set"). The STRICT
    ==host bar is the GPU production run at D=2048 (high fidelity). Here we pin: never a wrong answer + the moat. The
    smoke (`_seq_fused_fabric_320 --smoke --dim 256`) confirmed strict ==host at D=256."""
    c_host, c_fused = _composers()
    for (qa, qx, want) in _PRESENT:
        h = patient_of(c_host, host_scan_block(c_host, qa, qx))
        f = fused_reduced_query_patient(c_fused, qa, qx)
        # never wrong: the answer is host (==want here) OR an abstention; NEVER a different (confabulated) patient.
        assert f == h or f is None, f"fused-reduced CONFABULATED {f!r} (host {h!r}) for present cue {(qa, qx)} @ V=320"
    for (qa, qx) in _ABSENT:
        h = patient_of(c_host, host_scan_block(c_host, qa, qx))   # host abstains on these (None)
        f = fused_reduced_query_patient(c_fused, qa, qx)
        assert f == h, f"fused-reduced {f!r} != host {h!r} for moat cue {(qa, qx)} @ V=320 (host abstains)"


def test_fused_reduced_moat_zero_false_accepts_at_v320():
    """THE MOAT (HARD) at V=320: every absent/cross cue abstains on the fused-reduced path -- 0 false-accepts. A single
    false-accept is a FAIL (never traded for a pass)."""
    _c_host, c_fused = _composers()
    fa = sum(1 for (qa, qx) in _ABSENT if fused_reduced_query_patient(c_fused, qa, qx) is not None)
    assert fa == 0, f"MOAT BREACH at V=320: {fa} false-accept(s) on the fused-reduced path"


def test_cleanup_score_to_host_eliminated_at_v320():
    """R1 preserved AT SCALE: the cleanup -> SEQUENCER hand-off (`fused_seq_block_reduced` -> the device-resident
    fused_block_drives_reduced) does ZERO `to_host` of the cleanup membrane at V=320. (The firing-state body reads on
    the fabric bridge -- which score-pool word fired -- are the placed-rheobase body read, NOT the cleanup score.)"""
    import sim.backend as backend
    from research.runners import _seq_fused_fabric as sff
    from research.runners import _seq_fused_fabric_320 as sff320

    _c_host, c_fused = _composers()
    cleanup_membrane = c_fused.b.cp_membrane_potential_v
    calls = {"cleanup_membrane_reads": 0, "total": 0}
    real = backend.to_host

    def _spy(arr):
        calls["total"] += 1
        if arr is cleanup_membrane:
            calls["cleanup_membrane_reads"] += 1
        return real(arr)

    backend.to_host = _spy
    sff.to_host = _spy
    sff320.to_host = _spy
    try:
        idx = fused_seq_block_reduced(c_fused, _PRESENT[0][0], _PRESENT[0][1])
    finally:
        backend.to_host = real
        sff.to_host = real
        sff320.to_host = real

    assert idx is not None, "sanity: the fused-reduced sequencer should select a block for a present cue"
    assert calls["cleanup_membrane_reads"] == 0, (
        f"R1 NOT closed at V=320: the cleanup membrane was read to host {calls['cleanup_membrane_reads']} time(s) in "
        f"the cleanup->sequencer hand-off (total to_host calls {calls['total']})")


def test_fused_reduced_lesion_fails_safe_at_v320():
    """Sever the cleanup->score drive on a present cue at V=320 -> the score pool gets nothing -> the decoded lines
    stay silent -> the fused-reduced sequencer must ABSTAIN (None), never confabulate a wrong block."""
    _c_host, c_fused = _composers()
    for (a, x, _p) in _PRESENT:
        assert fused_reduced_query_patient_lesioned(c_fused, a, x) is None, (
            f"lesion not fail-safe at V=320: severed cleanup->score drive still answered for cue {(a, x)}")


def test_reduced_fabric_builds_at_v320_k8():
    """The reduced fused fabric builds at the production V=320/K=8 store (8 agents / 7 actions) without the full-V cost
    (the un-shrunk full-V fabric is ~225K neurons; this is ~12.7K)."""
    sb, meta = build_fused_fabric_reduced_bridge(seed=SEED, V=320, VA=8, VX=7, K=8)
    n = sb.core_config.num_neurons
    assert 5000 < n < 30000, f"reduced V=320/K=8 fabric neuron count off: {n:,}"
    assert meta["V"] == 320 and meta["VA"] == 8 and meta["VX"] == 7


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
