"""Sanity test for the --seed flag added to run_experiment_headless.py (D.C.1).

Verifies two things:
  1. `python run_experiment_headless.py --seed 42` runs without crashing on a
     minimal configuration.
  2. Two runs with the same seed produce the same CS->US mean weight at
     the training phase entry (deterministic pipeline).

This is a coarse check — it doesn't exercise the whole Pavlovian preset to
completion (too slow for CI). Just verifies the seed plumbing is wired.
"""
import json
import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_TESTS"),
    reason="Runs run_experiment_headless.py end-to-end (~60s). Set RUN_SLOW_TESTS=1.",
)
def test_headless_seed_is_deterministic(tmp_path):
    pytest.importorskip("cupy")

    out1 = tmp_path / "associative_seed42_run1.json"
    out2 = tmp_path / "associative_seed42_run2.json"

    # Minimal pavlovian run: 20 trials, smaller network, seed=42.
    cmd_base = [
        sys.executable, "run_experiment_headless.py",
        "--experiment", "associative",
        "--num-trials", "20",
        "--num-neurons", "1000",
        "--seed", "42",
    ]
    subprocess.run(cmd_base + ["--output", str(out1)], check=True, timeout=600)
    subprocess.run(cmd_base + ["--output", str(out2)], check=True, timeout=600)

    d1 = json.load(open(out1))
    d2 = json.load(open(out2))

    # Find first intergroup_weights event in each (at training phase entry)
    def first_weight(d):
        for e in d.get("log", d):
            if isinstance(e, dict) and e.get("event") == "intergroup_weights":
                return e.get("mean_weight")
        return None

    w1 = first_weight(d1)
    w2 = first_weight(d2)
    assert w1 is not None and w2 is not None, "No intergroup_weights event found"
    # Allow tiny floating-point diff but expect bit-identical within 1e-5
    assert abs(w1 - w2) < 1e-5, (
        f"seed=42 non-determinism: run1 mean_weight={w1}, run2 mean_weight={w2}"
    )
