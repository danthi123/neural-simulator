"""Read-only localizer for V13 NumPy/CuPy Izhikevich arithmetic.

This helper consumes already-sealed state-transplant evidence. It has no RNG,
accepts no seed, writes no artifact, and does not construct a SimulationBridge.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = ROOT / "research/findings/raw/v13_backend_state_transplant"
EVIDENCE_SHA256 = {
    "bundle-numpy.json": "5f9750ff1f26c38df8676525bc698571102e5aeefeae7d3787b8f2b3f6aa943b",
    "run-numpy-on-numpy-default.json": "e205b02693646d2529644b82a9d82d1952a4e8c1d62122ddf1852d0c339f313c",
    "run-numpy-on-cupy-default.json": "1228bd4946e2e20803a650d81069697bc167143486b260ee0847d577f5dac7ae",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decode_array(record: dict[str, Any]) -> np.ndarray:
    value = np.frombuffer(
        base64.b64decode(record["data_base64"]),
        dtype=np.dtype(record["dtype"]),
    ).reshape(record["shape"])
    return value.copy()


def load_evidence() -> dict[str, Any]:
    """Load only the sealed inputs needed by the arithmetic probe."""
    records: dict[str, Any] = {}
    for name, expected in EVIDENCE_SHA256.items():
        path = EVIDENCE_DIR / name
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"sealed evidence hash mismatch for {name}: {observed}")
        records[name] = json.loads(path.read_text())

    bundle = records["bundle-numpy.json"]
    numpy_run = records["run-numpy-on-numpy-default.json"]
    cupy_run = records["run-numpy-on-cupy-default.json"]
    arrays = {name: _decode_array(value) for name, value in bundle["cp_arrays"].items()}
    numpy_trajectories = {
        name: _decode_array(value) for name, value in numpy_run["trajectories"].items()
    }
    cupy_trajectories = {
        name: _decode_array(value) for name, value in cupy_run["trajectories"].items()
    }
    return {
        "arrays": arrays,
        "numpy": numpy_trajectories,
        "cupy": cupy_trajectories,
    }


def strict_izhikevich2007_update(
    xp: Any,
    v: Any,
    u: Any,
    C: Any,
    k: Any,
    vr: Any,
    vt: Any,
    a: Any,
    b: Any,
    total_current: Any,
    dt: Any,
) -> tuple[Any, Any]:
    """Evaluate the existing Euler equations with a fixed float32 round order.

    Every primitive is materialized before the next primitive. This is the
    backend-neutral reference contract, not the proposed GPU implementation.
    The implementation correction should preserve this order in one device
    kernel with explicit round-to-nearest CUDA intrinsics.
    """
    f32 = xp.float32

    def rounded(value: Any) -> Any:
        return value.astype(f32, copy=False)

    C_safe = rounded(xp.where(C == f32(0.0), f32(1.0), C))
    v_minus_vr = rounded(v - vr)
    v_minus_vt = rounded(v - vt)

    k_times_vr = rounded(k * v_minus_vr)
    quadratic = rounded(k_times_vr * v_minus_vt)
    dv_numerator = rounded(rounded(quadratic - u) + total_current)
    dv = rounded(dv_numerator / C_safe)
    v_new = rounded(v + rounded(dv * f32(dt)))

    b_times_vr = rounded(b * v_minus_vr)
    du_inner = rounded(b_times_vr - u)
    du = rounded(a * du_inner)
    u_new = rounded(u + rounded(du * f32(dt)))
    return v_new, u_new


def probe_inputs(xp: Any, evidence: dict[str, Any], row: int) -> tuple[Any, ...]:
    """Return identical sealed inputs for one isolated neuron update."""
    arrays = evidence["arrays"]
    trajectory = evidence["numpy"]
    names = (
        "cp_izh_C", "cp_izh_k", "cp_izh_vr", "cp_izh_vt",
        "cp_izh_a", "cp_izh_b", "cp_intrinsic_current_pA",
    )
    params = tuple(xp.asarray(arrays[name]) for name in names)
    return (
        xp.asarray(trajectory["v"][row]),
        xp.asarray(trajectory["u"][row]),
        *params,
        xp.float32(1.0),
    )


def _host(value: Any) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.ascontiguousarray(np.asarray(value), dtype=np.float32)


def differing_cells(left: Any, right: Any) -> list[int]:
    left_host = _host(left)
    right_host = _host(right)
    return np.flatnonzero(left_host.view(np.uint32) != right_host.view(np.uint32)).tolist()


def run_probe() -> dict[str, Any]:
    """Reproduce the two sealed first-divergence cases on the active backend."""
    from sim.backend import get_backend
    from sim.kernels import fused_izhikevich2007_dynamics_update

    xp, backend = get_backend()
    evidence = load_evidence()
    cases = {
        "u": {"input_row": 1, "output_row": 2},
        "v": {"input_row": 9, "output_row": 10},
    }
    report: dict[str, Any] = {
        "schema": "v13-izh-arithmetic-localizer-v1",
        "backend": backend,
        "rng_used": False,
        "seed": None,
        "cases": {},
    }
    for variable, case in cases.items():
        inputs = probe_inputs(xp, evidence, case["input_row"])
        production = fused_izhikevich2007_dynamics_update(*inputs)
        strict = strict_izhikevich2007_update(xp, *inputs)
        index = 0 if variable == "v" else 1
        expected_numpy = evidence["numpy"][variable][case["output_row"]]
        expected_cupy = evidence["cupy"][variable][case["output_row"]]
        report["cases"][variable] = {
            **case,
            "sealed_numpy_cupy_differing_cells": differing_cells(expected_numpy, expected_cupy),
            "production_vs_sealed_numpy": differing_cells(production[index], expected_numpy),
            "production_vs_sealed_cupy": differing_cells(production[index], expected_cupy),
            "strict_vs_sealed_numpy": differing_cells(strict[index], expected_numpy),
            "strict_vs_sealed_cupy": differing_cells(strict[index], expected_cupy),
            "production_vs_strict": differing_cells(production[index], strict[index]),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run_probe(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
