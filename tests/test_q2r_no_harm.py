import subprocess, sys, importlib, inspect


def test_protected_byte_empty():
    protected = [
      "research/runners/constrained_decode_gate.py",
      "research/runners/constrained_decode_core.py",
      "research/runners/engram_bootstrap_gate.py",
      "research/runners/abstention_gate.py","tests/test_abstention_gate.py",
      "sim/grounded_decode.py","research/runners/generator_g_core.py",
      "research/runners/compose_bridge_core.py",
      "research/runners/compose_bind_core.py",
      "research/runners/td_critic_core.py",
      "research/runners/dendritic_fair_core.py",
      "sim/tiny_transformer.py","sim/bpe_tokenizer.py","sim/bridge.py",
      "sim/td_value_critic.py","sim/compose_temporal_bind.py",
      "sim/kernels.py","sim/neuromodulators.py","sim/train_checkpoint.py",
      "sim/backend.py","sim/dendritic_plasticity.py",
      "research/runners/text_minimal_isolation.py"]
    d = subprocess.run(["git","diff","a1035cf..HEAD","--",*protected],
                        capture_output=True, text=True)
    assert d.stdout.strip() == "", "PROTECTED changed:\n"+d.stdout


def test_no_confab_moat_7_of_7():
    r = subprocess.run([sys.executable,"-m","pytest",
                        "tests/test_abstention_gate.py","-q"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]


def test_q2r_core_pure_and_imports_no_core():
    c = importlib.import_module("research.runners.q2r_core")
    src = inspect.getsource(c)
    assert "import torch" not in src and "backward(" not in src
    assert "constrained_decode_core" not in src


def test_q2r_gate_reuses_byte_unmodified_and_no_new_training():
    g = importlib.import_module("research.runners.q2r_gate")
    from research.runners.constrained_decode_gate import _GroundedConstrainedLM
    from research.runners.constrained_decode_core import cdc_verdict
    assert g._GroundedConstrainedLM is _GroundedConstrainedLM
    assert g.cdc_verdict is cdc_verdict
    src = inspect.getsource(g)
    assert "backward(" not in src and ".step()" not in src
    assert "optimizer" not in src.lower() and "loss" not in src.lower()
