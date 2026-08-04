"""Public package imports must not depend on which subsystem loads first."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _clean_import(statement: str) -> None:
    subprocess.run(
        [sys.executable, "-c", statement],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_lightweight_experiment_submodule_imports_first():
    _clean_import("import experiment.inhibitory_conductance_clamp")


def test_experiment_public_api_imports_first():
    _clean_import("from experiment import ExperimentEngine, ExperimentPresets")


def test_sim_public_api_imports_first():
    _clean_import("from sim import SimulationBridge, CoreSimConfig, NeuronModel")
