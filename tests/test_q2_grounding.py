"""Q2 Task-0 grounding pin. RED until Task 1/2 ship the net-new modules."""
import importlib


def test_reused_no_confab_moat_and_metrics_present():
    ag = importlib.import_module("research.runners.abstention_gate")
    assert ag.DEFAULT_THRESHOLD == 650.0
    gg = importlib.import_module("research.runners.generator_g_core")
    assert callable(gg.ungrounded_entity_rate) and callable(gg.is_answered)
    assert "the" in gg.FUNCTION_WORDS
    assert callable(importlib.import_module("sim.grounded_decode").grounded_decode)


def test_generator_f_artifact_present():
    import os
    base = "research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real"
    assert os.path.exists(base + ".pt") and os.path.exists(base + ".bpe.json")


def test_constrained_decode_core_frozen_and_pure():
    c = importlib.import_module("research.runners.constrained_decode_core")
    assert c._CDC_FAITHFUL_MAX == 0.20
    assert c._CDC_MIN_GROUNDED_CONTENT == 2
    assert c._CDC_MIN_GROUNDED_ANSWER_RATE == 0.5
    assert c._CDC_MIN_SEEDS == 3
    assert c._CDC_SCALE_LADDER == (6, 12, 24)
    assert c._CDC_SCALE_TOL == 0.10
    assert callable(c.cdc_verdict) and callable(c.cdc_scale_confidence)
    import inspect
    src = inspect.getsource(c)
    assert "backward(" not in src and "import torch" not in src


def test_constrained_decode_gate_importable():
    g = importlib.import_module("research.runners.constrained_decode_gate")
    assert hasattr(g, "_GroundedConstrainedLM") and hasattr(g, "main")
