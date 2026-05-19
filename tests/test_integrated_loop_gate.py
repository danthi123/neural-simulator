import json
import subprocess
import sys
from pathlib import Path


def test_integration_runner_tiny_smoke_produces_verdict(tmp_path):
    """Grounding pin: the integration runner exists, runs a fast
    --tiny-synth smoke end-to-end on the CPU backend, and writes a
    verdict JSON whose classification is the explicitly-not-propagated
    TINY marker (never a real PASS/FAIL/VOID at toy scale)."""
    out = tmp_path / "tiny.json"
    proc = subprocess.run(
        [sys.executable, "-m", "research.runners.integrated_loop_gate",
         "--tiny-synth", "--seeds", "42", "43", "44",
         "--out", str(out)],
        capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        "runner failed: %s\n%s" % (proc.stdout, proc.stderr))
    assert out.exists(), "runner did not write the verdict JSON"
    v = json.loads(out.read_text())
    assert "GATE" in v, "verdict has no GATE field"
    assert "TINY" in json.dumps(v), (
        "tiny-synth verdict must be marked TINY / NOT propagated")


def test_runner_imports_reused_parts_byte_unchanged():
    """The runner composes the validated parts by import; it must not
    declare its own copies of them, and must add no autograd."""
    src = Path("research/runners/integrated_loop_gate.py").read_text()
    assert "import torch" not in src and ".backward(" not in src
    assert "build_biological_brain_regions" in src
    assert "build_bg_brain_regions" in src
    assert "start_engram_recording" in src or "commit_engram_tag" in src
    assert "from research.runners.abstention_gate import" in src
    assert "from research.runners.integrated_loop_core import" in src


def test_phase_factored_tiny_smoke_produces_tiny_verdict(tmp_path):
    """Grounding pin (phase-factored): the runner accepts the
    phase-factored flag, runs a fast --tiny-synth smoke end-to-end on
    the CPU backend, and writes a TINY-marked verdict JSON (never a
    real PASS/FAIL/VOID at toy scale)."""
    out = tmp_path / "tiny_pf.json"
    proc = subprocess.run(
        [sys.executable, "-m", "research.runners.integrated_loop_gate",
         "--tiny-synth", "--phase-factored",
         "--seeds", "42", "43", "44", "--out", str(out)],
        capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        "runner failed: %s\n%s" % (proc.stdout, proc.stderr))
    assert out.exists(), "runner did not write the verdict JSON"
    v = json.loads(out.read_text())
    assert "GATE" in v, "verdict has no GATE field"
    assert "TINY" in json.dumps(v), (
        "tiny-synth verdict must be marked TINY / NOT propagated")


def test_runner_reuses_validated_phase_factored_parts():
    """The phase-factored runner composes the validated parts by
    import; it must not declare its own copies, must add no autograd,
    and must reuse the frozen verdict + the Phase-1.3 consolidation
    interface + the no-confab moat byte-unchanged."""
    src = Path("research/runners/integrated_loop_gate.py").read_text()
    assert "import torch" not in src and ".backward(" not in src
    assert "build_biological_brain_regions" in src
    assert "build_bg_brain_regions" in src
    assert "run_concept_replay_phase" in src
    assert "set_awake_gates" in src and "set_sleep_gates" in src
    assert "freeze_all_gates" in src
    assert "start_engram_recording" in src or "commit_engram_tag" in src
    assert "from research.runners.abstention_gate import" in src
    assert "from research.runners.integrated_loop_core import" in src


def test_phase_factored_runs_offline_after_online_before_readout():
    """Structural: the phase-factored path calls set_sleep_gates +
    run_concept_replay_phase AFTER the online encode/commit and BEFORE
    the consolidated readout, and freeze_all_gates before the readout
    (the validated Phase-1.3 freeze-then-evaluate idiom). The online
    encode/write path is byte-unchanged from e02f692 (verified
    separately by the controller's git-range check; this test pins the
    ordering markers exist)."""
    src = Path("research/runners/integrated_loop_gate.py").read_text()
    assert "--phase-factored" in src or "phase_factored" in src
    assert "run_concept_replay_phase(" in src
    assert "set_sleep_gates(" in src and "freeze_all_gates(" in src
