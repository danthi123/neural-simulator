"""Q2R Task-0 grounding pin. RED until Task 1/2 ship the net-new modules."""
import importlib


def test_reused_q2_mechanism_and_instrument_present():
    cg = importlib.import_module("research.runners.constrained_decode_gate")
    assert hasattr(cg, "_GroundedConstrainedLM")
    cc = importlib.import_module("research.runners.constrained_decode_core")
    assert callable(cc.cdc_verdict)
    assert cc._CDC_MIN_GROUNDED_ANSWER_RATE == 0.50  # the value _Q2R_TOP_MIN equals


def test_generator_f_artifact_present():
    import os
    b = "research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real"
    assert os.path.exists(b + ".pt") and os.path.exists(b + ".bpe.json")


def test_q2r_core_frozen_and_pure():
    c = importlib.import_module("research.runners.q2r_core")
    assert c._Q2R_LADDER == (12, 24, 48, 96)
    assert c._Q2R_SCALE_TOL == 0.10
    assert c._Q2R_TOP_MIN == 0.50
    assert c._Q2R_MIN_SEEDS == 3
    assert callable(c.q2r_scale_confidence)
    import inspect
    src = inspect.getsource(c)
    assert "import torch" not in src and "backward(" not in src
    assert "constrained_decode_core" not in src  # owns its own bars


def test_q2r_gate_importable_and_reuses_byte_unmodified():
    g = importlib.import_module("research.runners.q2r_gate")
    from research.runners.constrained_decode_gate import _GroundedConstrainedLM
    from research.runners.constrained_decode_core import cdc_verdict
    assert g._GroundedConstrainedLM is _GroundedConstrainedLM
    assert g.cdc_verdict is cdc_verdict
    assert len(g._Q2R_GROUNDED) >= 96
