from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners._v13_izh_arithmetic_localizer import (  # noqa: E402
    differing_cells,
    load_evidence,
    probe_inputs,
    strict_izhikevich2007_update,
)


def test_sealed_evidence_has_the_preregistered_first_difference_cases():
    evidence = load_evidence()
    assert differing_cells(evidence["numpy"]["u"][2], evidence["cupy"]["u"][2]) == [30, 33, 41, 57]
    assert differing_cells(evidence["numpy"]["v"][10], evidence["cupy"]["v"][10]) == [40]
    assert differing_cells(evidence["numpy"]["u"][1], evidence["cupy"]["u"][1]) == []
    assert differing_cells(evidence["numpy"]["v"][9], evidence["cupy"]["v"][9]) == []
    assert (
        evidence["numpy"]["u"][9, 40].view(np.uint32)
        == evidence["cupy"]["u"][9, 40].view(np.uint32)
    )


@pytest.mark.parametrize(
    ("variable", "input_row", "output_row", "output_index"),
    (("u", 1, 2, 1), ("v", 9, 10, 0)),
)
def test_strict_float32_order_matches_sealed_numpy(variable, input_row, output_row, output_index):
    evidence = load_evidence()
    inputs = probe_inputs(np, evidence, input_row)
    output = strict_izhikevich2007_update(np, *inputs)[output_index]
    assert differing_cells(output, evidence["numpy"][variable][output_row]) == []


def _usable_cupy():
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _usable_cupy(), reason="requires a usable CUDA device")
@pytest.mark.parametrize(
    ("variable", "input_row", "output_row", "output_index", "expected_fma_cells"),
    (
        ("u", 1, 2, 1, [30, 33, 41, 57]),
        ("v", 9, 10, 0, [40]),
    ),
)
def test_gpu_strict_order_removes_exact_recorded_fma_residual(
    variable, input_row, output_row, output_index, expected_fma_cells
):
    import cupy as cp

    evidence = load_evidence()
    inputs = probe_inputs(cp, evidence, input_row)
    strict = strict_izhikevich2007_update(cp, *inputs)[output_index]
    expected_numpy = evidence["numpy"][variable][output_row]
    expected_cupy = evidence["cupy"][variable][output_row]

    assert differing_cells(strict, expected_numpy) == []
    assert differing_cells(strict, expected_cupy) == expected_fma_cells


@pytest.mark.skipif(not _usable_cupy(), reason="requires a usable CUDA device")
def test_nvrtc_inner_fma_matches_recorded_cupy_u_and_fmad_false_matches_numpy():
    import cupy as cp

    evidence = load_evidence()
    v, u, _C, _k, vr, _vt, a, b, _current, dt = probe_inputs(cp, evidence, 1)
    default_kernel = cp.ElementwiseKernel(
        "float32 u, float32 a, float32 b, float32 v, float32 vr, float32 dt",
        "float32 out",
        "out = u + a * (b * (v - vr) - u) * dt;",
        "v13_u_contraction_default_test",
    )
    no_fma_kernel = cp.ElementwiseKernel(
        "float32 u, float32 a, float32 b, float32 v, float32 vr, float32 dt",
        "float32 out",
        "out = u + a * (b * (v - vr) - u) * dt;",
        "v13_u_contraction_disabled_test",
        options=("--fmad=false",),
    )
    inner_fma_kernel = cp.ElementwiseKernel(
        "float32 u, float32 a, float32 b, float32 v, float32 vr, float32 dt",
        "float32 out",
        (
            "float x = __fsub_rn(v, vr);"
            "float inner = fmaf(b, x, -u);"
            "float du = __fmul_rn(a, inner);"
            "out = __fadd_rn(u, __fmul_rn(du, dt));"
        ),
        "v13_u_inner_fma_test",
    )

    observed_cupy = evidence["cupy"]["u"][2]
    observed_numpy = evidence["numpy"]["u"][2]
    assert differing_cells(default_kernel(u, a, b, v, vr, dt), observed_cupy) == []
    assert differing_cells(inner_fma_kernel(u, a, b, v, vr, dt), observed_cupy) == []
    assert differing_cells(no_fma_kernel(u, a, b, v, vr, dt), observed_numpy) == []


@pytest.mark.skipif(not _usable_cupy(), reason="requires a usable CUDA device")
def test_nvrtc_inner_fma_matches_recorded_cupy_v_and_fmad_false_matches_numpy():
    import cupy as cp

    evidence = load_evidence()
    v, u, C, k, vr, vt, _a, _b, current, dt = probe_inputs(cp, evidence, 9)
    expression = "out = v + ((k * (v - vr) * (v - vt) - u + current) / C) * dt;"
    params = (
        "float32 v, float32 u, float32 C, float32 k, float32 vr, "
        "float32 vt, float32 current, float32 dt"
    )
    default_kernel = cp.ElementwiseKernel(
        params, "float32 out", expression, "v13_v_contraction_default_test"
    )
    no_fma_kernel = cp.ElementwiseKernel(
        params,
        "float32 out",
        expression,
        "v13_v_contraction_disabled_test",
        options=("--fmad=false",),
    )
    inner_fma_kernel = cp.ElementwiseKernel(
        params,
        "float32 out",
        (
            "float x1 = __fsub_rn(v, vr);"
            "float x2 = __fsub_rn(v, vt);"
            "float first_product = __fmul_rn(k, x1);"
            "float numerator = __fadd_rn(fmaf(first_product, x2, -u), current);"
            "float dv = __fdiv_rn(numerator, C);"
            "out = __fadd_rn(v, __fmul_rn(dv, dt));"
        ),
        "v13_v_inner_fma_test",
    )

    args = (v, u, C, k, vr, vt, current, dt)
    observed_cupy = evidence["cupy"]["v"][10]
    observed_numpy = evidence["numpy"]["v"][10]
    assert differing_cells(default_kernel(*args), observed_cupy) == []
    assert differing_cells(inner_fma_kernel(*args), observed_cupy) == []
    assert differing_cells(no_fma_kernel(*args), observed_numpy) == []


def test_probe_has_no_seed_or_artifact_output_interface():
    from research.runners import _v13_izh_arithmetic_localizer as localizer

    assert not hasattr(localizer, "LOCKED_SEED")
    assert not hasattr(localizer, "OUTPUT_PATH")
