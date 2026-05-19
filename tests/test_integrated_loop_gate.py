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
    declare its own copies of them, and must add no autograd. The
    acceptance instrument is the NEW frozen integrated_loop_core_v2
    (the original integrated_loop_core is NEVER imported here and is
    NEVER edited; its VOID stands as the honest record)."""
    src = Path("research/runners/integrated_loop_gate.py").read_text()
    assert "import torch" not in src and ".backward(" not in src
    assert "build_biological_brain_regions" in src
    assert "build_bg_brain_regions" in src
    assert "start_engram_recording" in src or "commit_engram_tag" in src
    assert "from research.runners.abstention_gate import" in src
    assert "integrated_loop_verdict_v2" in src
    assert "from research.runners.integrated_loop_core import" not in src
    assert "import integrated_loop_core\n" not in src


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
    and must reuse the NEW frozen verdict + the Phase-1.3 consolidation
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
    assert "integrated_loop_verdict_v2" in src
    assert "from research.runners.integrated_loop_core import" not in src


def test_distinct_pathways_reuses_parts_byte_unchanged_and_new_core():
    """The distinct-pathways mode composes the validated parts by
    import, adds no autograd, scores via the NEW core (not the
    original), and never imports the original frozen core (Task 4
    Step 1 structural pin)."""
    src = Path("research/runners/integrated_loop_gate.py").read_text()
    assert "import torch" not in src and ".backward(" not in src
    assert "build_biological_brain_regions" in src
    assert "run_concept_replay_phase" in src
    assert "start_engram_recording" in src or "commit_engram_tag" in src
    assert "from research.runners.abstention_gate import" in src
    assert "integrated_loop_verdict_v2" in src
    assert "from research.runners.integrated_loop_core import" not in src
    assert "import integrated_loop_core\n" not in src
    # The distinct-pathways episodic readout must be the ONLINE
    # trisynaptic completion taken BEFORE the offline consolidation
    # (NOT post-consolidation); the structural markers for the
    # order-preserving online path + the order-invariant offline
    # consolidation must both be present.
    assert "_DISTINCT_PATHWAYS" in src
    assert "--distinct-pathways" in src
    assert "_episodic_order_readout" in src


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


def test_distinct_pathways_tiny_smoke_produces_tiny_verdict(tmp_path):
    """Grounding pin (Task 3): the runner's distinct-readout-pathways
    mode runs a fast --tiny-synth smoke end-to-end on the CPU backend
    and writes a verdict JSON marked TINY (never propagated at toy
    scale), and that mode scores via the NEW core module
    integrated_loop_verdict_v2 (not the original frozen core). This pin
    is intentionally red until the --distinct-pathways mode lands."""
    out = tmp_path / "tiny_dp.json"
    proc = subprocess.run(
        [sys.executable, "-m", "research.runners.integrated_loop_gate",
         "--distinct-pathways", "--tiny-synth",
         "--seeds", "42", "43", "44", "--out", str(out)],
        capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, (proc.stdout + "\n" + proc.stderr)
    assert out.exists()
    v = json.loads(out.read_text())
    assert "GATE" in v
    assert "TINY" in json.dumps(v)


# --------------------------------------------------------------------
# Task 0 -- REMOTE/consolidated-memory-regime grounding pins (Design B).
# RED until the --remote-regime path lands (plan Task 1.3). Mirrors the
# established grounding-pin discipline (the --phase-factored /
# --distinct-pathways pins above). The new mode is selected by a new
# argv flag --remote-regime read BEFORE argparse exactly like the
# existing _PHASE_FACTORED / _DISTINCT_PATHWAYS idiom (no extra rng
# draw, no signature churn). The toy verdict is ALWAYS marked TINY and
# is NEVER propagated.
# --------------------------------------------------------------------
def test_remote_regime_tiny_smoke_produces_tiny_verdict(tmp_path):
    """Task 0.1 grounding pin (remote regime): the runner accepts the
    --remote-regime flag, runs a fast --tiny-synth smoke end-to-end on
    the CPU backend, returns 0, and writes a verdict JSON that has a
    GATE field and is marked TINY (the toy verdict is NEVER propagated).
    RED until the --remote-regime mode lands (plan Task 1.3)."""
    out = tmp_path / "tiny_rr.json"
    proc = subprocess.run(
        [sys.executable, "-m", "research.runners.integrated_loop_gate",
         "--remote-regime", "--tiny-synth",
         "--seeds", "42", "43", "44", "--out", str(out)],
        capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, (proc.stdout + "\n" + proc.stderr)
    assert out.exists(), "runner did not write the verdict JSON"
    v = json.loads(out.read_text())
    assert "GATE" in v, "verdict has no GATE field"
    assert "TINY" in json.dumps(v), (
        "remote-regime tiny-synth verdict must be marked TINY / NOT "
        "propagated")


def test_remote_regime_reuses_validated_parts_byte_unchanged():
    """Task 0.2 structural pin (remote regime). Mirrors
    test_runner_reuses_validated_phase_factored_parts: the runner
    composes the validated parts by import (online encode/write +
    offline Phase-1.3 consolidation + the no-confab moat + the NEW
    frozen v2 verdict), adds no autograd, and never imports the
    original frozen core. Adds the remote-regime-specific markers that
    pin the reuse map: the _REMOTE_REGIME flag, the --remote-regime
    argv selector, and the VALIDATED strict-silence / hippocampus-OFF
    idiom by name (evaluate_with_hippo_off or HIPPO_REGIONS) with the
    validated 3/3-strict configuration strength as a literal (-2000).
    RED until the --remote-regime mode lands (plan Task 1.3)."""
    src = Path("research/runners/integrated_loop_gate.py").read_text()
    assert "import torch" not in src and ".backward(" not in src
    assert "build_biological_brain_regions" in src
    assert "build_bg_brain_regions" in src
    assert "run_concept_replay_phase" in src
    assert "set_awake_gates" in src and "set_sleep_gates" in src
    assert "freeze_all_gates" in src
    assert "start_engram_recording" in src or "commit_engram_tag" in src
    assert "from research.runners.abstention_gate import" in src
    assert "integrated_loop_verdict_v2" in src
    assert "from research.runners.integrated_loop_core import" not in src
    assert "import integrated_loop_core\n" not in src
    # Remote-regime-specific reuse markers (the validated Phase-1.3
    # strict-silence / hippocampus-OFF mechanism reused byte-unchanged).
    assert "_REMOTE_REGIME" in src
    assert "--remote-regime" in src
    assert ("evaluate_with_hippo_off" in src
            or "HIPPO_REGIONS" in src)
    assert "-2000" in src
