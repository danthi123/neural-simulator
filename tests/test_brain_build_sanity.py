"""Structural tests for tools/brain_build_sanity.py -- the POST-PROVISION SANITY CHECK instrument
(tools/aws_brain_sanity_check.sh) that builds the tiny-demo brain and reports its neuron/synapse counts.

A REAL build is heavy (multiple SimulationBridge instances, ~1-2 min, needs the project .venv's h5py/scipy/
cupy stack) -- inappropriate for a fast unit-test pass. These tests instead verify the JSON CONTRACT
(`main()`'s ok/error shape) via a monkeypatched `build_and_measure`, so a broken contract is caught without
paying the build cost every CI run. The real build itself was exercised manually while landing this fix
(SIM_BACKEND=numpy, tiny-demo, seed 42): n_neurons=8382, n_synapses=520536 across 9 summed bridges -- a sane,
strongly non-degenerate reference, in contrast to the board's reported AWS "2-neuron/0-synapse" build."""
import io
import json
import os
import sys
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import brain_build_sanity as bbs  # noqa: E402


def test_main_prints_ok_json_and_returns_0(monkeypatch):
    monkeypatch.setattr(bbs, "build_and_measure",
                        lambda: {"ok": True, "n_neurons": 8382, "n_synapses": 520536, "n_bridges": 9,
                                 "n_facts": 5, "backend": "numpy", "source": "tiny-demo"})
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = bbs.main()
    assert rc == 0
    out = json.loads(buf.getvalue().strip())
    assert out["ok"] is True
    assert out["n_neurons"] == 8382 and out["n_synapses"] == 520536


def test_main_reports_degenerate_build_without_crashing(monkeypatch):
    """The exact 2026-09-22 symptom shape -- a build that 'succeeds' (no exception) but is empty."""
    monkeypatch.setattr(bbs, "build_and_measure",
                        lambda: {"ok": True, "n_neurons": 2, "n_synapses": 0, "n_bridges": 1,
                                 "n_facts": 5, "backend": "cupy", "source": "tiny-demo"})
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = bbs.main()
    out = json.loads(buf.getvalue().strip())
    assert rc == 0                       # ok=True still exits 0 -- the CALLER (aws_brain_sanity_check.sh)
    assert out["n_neurons"] == 2         # does the degenerate-vs-reference COMPARISON, not this script.
    assert out["n_synapses"] == 0


def test_main_never_lets_an_exception_escape_as_a_bare_traceback(monkeypatch):
    def _boom():
        raise ModuleNotFoundError("No module named 'h5py'")
    monkeypatch.setattr(bbs, "build_and_measure", _boom)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = bbs.main()
    assert rc == 1
    out = json.loads(buf.getvalue().strip())
    assert out["ok"] is False
    assert "ModuleNotFoundError" in out["error"]
