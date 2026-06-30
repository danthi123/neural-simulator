"""R1 close (the device-resident cleanup->score handoff): the fused fabric folds the divnorm-score pool + the K-way
sequencer onto ONE Izhikevich bridge and routes the cleanup membrane -> score pool DEVICE-RESIDENT (no to_host of the
cleanup score). These tests pin the load-bearing R1 properties at the validated V=72/K=2 scale (CPU/numpy oracle):

  * the fused path's per-query decision == the separate-bridge spiking path == the host `_scan` oracle (==host);
  * the no-confab MOAT holds (an absent/cross cue abstains, 0 false-accepts);
  * the cleanup-score `to_host` is GONE from the fused query path (the SEAM is closed -- the whole point of R1);
  * OFF == byte-identical (integrated_loop=False/True produce the SAME answers as before; the fused path is additive);
  * lesion fails safe (sever the cleanup->score drive -> abstain, never confabulate).

Run on CPU: SIM_BACKEND=numpy pytest tests/test_seq_fused_fabric.py -v
(The GPU 6-seed K{2,4,8} is the controller's run; these tests are the CPU correctness gate.)
"""
import os

import pytest

# The validated V=72 production fact table (the divnorm op-point is V-dependent and tuned for V=72/320; a 12-word toy
# vocab over-suppresses the divnorm winner below rheobase -- so the parity scale MUST be the validated V=72 table).
from research.runners._phaseB_onebrain_sequencerK_k32_margin_derisk import ALL_FACTS, VOCAB

OP = dict(sequencer_match_thresh=0.06, sequencer_gain=0.1, sequencer_sigma=1.0, sequencer_input_gain=1.0)
D = 128
SEED = 42
K = 2


def _build(integrated_loop, **extra):
    from research.runners.one_brain_composer import OneBrainComposer
    kw = dict(seed=SEED, D=D, vocab=VOCAB, k_max=max(8, K), enable_batched=False, enable_rf_cudagraph=False)
    if integrated_loop:
        kw.update(OP)
    c = OneBrainComposer(integrated_loop=integrated_loop, **kw, **extra)
    for (a, x, p) in ALL_FACTS[:K]:
        c.store(a, x, p)
    return c


def _battery():
    """Present cues (each answers its block) + moat cues (absent agent / absent action / cross)."""
    facts = ALL_FACTS[:K]
    present = [((a, x), p) for (a, x, p) in facts]
    agents = {a for (a, x, p) in facts}
    actions = {x for (a, x, p) in facts}
    pairs = {(a, x) for (a, x, p) in facts}
    absent_agent = next(w for w in VOCAB if w not in agents)
    absent_action = next(w for w in VOCAB if w not in actions)
    a0, x0 = facts[0][0], facts[0][1]
    cross_action = next((x for (a, x, p) in facts if (a0, x) not in pairs), absent_action)
    moat = [((absent_agent, x0), None), ((a0, absent_action), None), ((a0, cross_action), None)]
    return present, moat


def test_fused_equals_host_and_separate_bridge():
    """The fused (device-resident) path's query_patient == the host `_scan` oracle AND == the separate-bridge spiking
    path, on every present + moat cue (V=72/K=2). The cleanup->sequencer DATA hand-off being device-resident must NOT
    change the decision -- only WHERE the score lives (host array vs device)."""
    c_host = _build(False)
    c_sep = _build(True)
    c_fused = _build("fused")
    present, moat = _battery()
    for (qa, qx), want in present + moat:
        h = c_host.query_patient(qa, qx)
        s = c_sep.query_patient(qa, qx)
        f = c_fused.query_patient(qa, qx)
        assert f == h, f"fused {f} != host {h} for cue {(qa, qx)}"
        assert s == h, f"separate-bridge {s} != host {h} for cue {(qa, qx)} (op-point sanity)"
        if want is not None:
            assert f == want, f"fused {f} != expected {want} for present cue {(qa, qx)}"


def test_fused_moat_zero_false_accepts():
    """THE MOAT (HARD): every absent/cross cue abstains on the fused path -- 0 false-accepts. A single false-accept is
    a FAIL (never traded for a pass)."""
    c_fused = _build("fused")
    _present, moat = _battery()
    false_accepts = sum(1 for (cue, _want) in moat if c_fused.query_patient(*cue) is not None)
    assert false_accepts == 0, f"moat breach: {false_accepts} false-accept(s) on the fused path"


def test_cleanup_score_to_host_eliminated():
    """R1 (the precise residual): the cleanup -> SEQUENCER hand-off reads the cleanup score (the composer RF bridge's
    cp_membrane_potential_v) to host and re-drives it onto a separate divnorm-score bridge. R1 is closed iff THAT
    hand-off -- `_seq_block` -> `fused_seq_block` -> `ensure_fused_fabric` -> `fused_block_drives` (the S4->S5 cleanup
    -> score-pool path) -- does ZERO `to_host` of the cleanup membrane (it stays device-resident the whole way from the
    RF cleanup to the score-pool drive). We instrument `sim.backend.to_host` and count reads of the cleanup membrane
    DURING ONLY that hand-off (`_seq_block`).

    NOTE on the patient body-read (R5, NOT R1): the SEPARATE downstream `query_patient` step `got =
    self._read_blocks()[idx]` re-decodes the SELECTED block to EMIT its patient word -- that read happens AFTER the
    sequencer has chosen the block (S7, the answer body-read), is the SAME read every composer path does, and is the
    documented legitimate "which concept-neuron won" boundary (scoping R5, "effectively closed" under
    enable_spiking_cleanup). It is NOT the cleanup->sequencer DATA seam R1 targets. So we scope the assert to the
    hand-off, not the whole query. The remaining to_host calls inside the hand-off (firing-state body reads on the
    fabric bridge -- "which score-pool word fired") are the placed-rheobase body read, NOT the cleanup score. This is a
    CODE-PATH property (holds on numpy + cupy); on numpy to_host is a passthrough but the call-site that marshalled the
    cleanup score to host is GONE."""
    import sim.backend as backend
    from research.runners import one_brain_composer as obc

    c_fused = _build("fused")
    cleanup_membrane = c_fused.b.cp_membrane_potential_v   # the RF bridge's membrane = the cleanup-score carrier

    calls = {"cleanup_membrane_reads": 0, "total": 0}
    real_to_host = backend.to_host

    def _spy(arr):
        calls["total"] += 1
        if arr is cleanup_membrane:
            calls["cleanup_membrane_reads"] += 1
        return real_to_host(arr)

    # Patch BOTH the backend symbol and the one already imported into one_brain_composer's namespace, and instrument
    # ONLY the cleanup->sequencer hand-off (_seq_block), which is exactly the S4->S5 path R1 lives in.
    backend.to_host = _spy
    obc.to_host = _spy
    try:
        idx = c_fused._seq_block("dog", "go")   # the cue-match CONTROL hand-off (S4 cleanup -> S5 score -> S6 select)
    finally:
        backend.to_host = real_to_host
        obc.to_host = real_to_host

    assert idx == 0, f"sanity: the fused sequencer should select block 0 for (dog,go), got {idx}"
    assert calls["cleanup_membrane_reads"] == 0, (
        f"R1 NOT closed: the cleanup membrane was read to host {calls['cleanup_membrane_reads']} time(s) in the "
        f"cleanup->sequencer hand-off (total to_host calls {calls['total']})")


def test_off_byte_identical():
    """The fused path is purely ADDITIVE: integrated_loop=False (host) and integrated_loop=True (separate-bridge
    spiking) produce the SAME answers whether or not the fused code exists (regression guard). Here we assert the two
    legacy paths still agree with each other on the full battery (they were GO before this change)."""
    c_host = _build(False)
    c_sep = _build(True)
    present, moat = _battery()
    for (qa, qx), _want in present + moat:
        assert c_sep.query_patient(qa, qx) == c_host.query_patient(qa, qx), \
            f"legacy separate-bridge != host for {(qa, qx)} -- the fused change perturbed a legacy path"


def test_fused_lesion_fails_safe():
    """Sever the cleanup->score drive on a present cue -> the score pool gets nothing -> the decoded lines stay silent
    -> the fused sequencer must ABSTAIN (None), never confabulate a wrong block."""
    from research.runners._seq_fused_fabric import fused_query_patient_lesioned
    c_fused = _build("fused")
    for (a, x, p) in ALL_FACTS[:K]:
        assert fused_query_patient_lesioned(c_fused, a, x) is None, \
            f"lesion not fail-safe: severed cleanup->score drive still answered for cue {(a, x)}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
