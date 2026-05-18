"""Phase B no-harm: the net-new gate harms NOTHING protected."""
import subprocess, sys, importlib, inspect


def test_protected_set_byte_empty_diff():
    protected = [
        "research/runners/abstention_gate.py",
        "tests/test_abstention_gate.py",
        "sim/td_value_critic.py", "sim/compose_temporal_bind.py",
        "sim/kernels.py", "sim/bridge.py", "sim/neuromodulators.py",
        "sim/train_checkpoint.py", "sim/backend.py",
        "sim/dendritic_plasticity.py",
        "research/runners/text_minimal_isolation.py",
        "research/runners/compose_bridge_core.py",
        "research/runners/compose_bind_core.py",
        "research/runners/td_critic_core.py",
        "research/runners/dendritic_fair_core.py"]
    d = subprocess.run(["git", "diff", "bda6e46..HEAD", "--", *protected],
                        capture_output=True, text=True)
    assert d.stdout.strip() == "", "PROTECTED set changed:\n" + d.stdout


def test_no_confab_moat_still_7_of_7():
    r = subprocess.run([sys.executable, "-m", "pytest",
                        "tests/test_abstention_gate.py", "-q"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]


def test_no_autograd_in_shipped_path():
    g = importlib.import_module("research.runners.engram_bootstrap_gate")
    src = inspect.getsource(g)
    assert "autograd" not in src and "import torch" not in src


def test_cbr_verdict_reused_byte_identical():
    from research.runners.compose_bridge_core import cbr_verdict
    g = importlib.import_module("research.runners.engram_bootstrap_gate")
    assert g.cbr_verdict is cbr_verdict
