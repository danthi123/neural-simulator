"""Tests for sim.backend — pluggable numpy-like backend.

These tests cover the abstraction layer itself (get_backend, fuse, etc.)
NOT the eventual NumPy bridge implementation (Phase 2 of the tiering
design). Phase 1 of the tiering design = just the abstraction.

All tests are CPU-only. They exercise the NumPy code path; the CuPy
path is also tested when CuPy is available on the host, but those
tests are skipped on CPU-only runners.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.backend import (
    get_backend, get_sparse_module, is_gpu_backend,
    fuse, synchronize, to_host, from_host,
    get_memory_pool_used_mb, _reset_cache_for_tests,
    set_device, get_device_mem_info, get_device_properties,
    get_memory_pool, get_pinned_memory_pool,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    """Reset the backend cache between tests so SIM_BACKEND switches take."""
    _reset_cache_for_tests()
    yield
    _reset_cache_for_tests()


def _has_cupy() -> bool:
    try:
        import cupy  # noqa: F401
        return True
    except (ImportError, RuntimeError):
        return False


# ──────────────────────────────────────────────────────────────────────
# get_backend — basic resolution
# ──────────────────────────────────────────────────────────────────────


def test_get_backend_numpy_explicit():
    """Explicit numpy backend returns NumPy module."""
    xp, name = get_backend("numpy")
    assert name == "numpy"
    assert xp is np


def test_get_backend_auto_picks_numpy_when_cupy_unavailable(monkeypatch):
    """When CuPy is mocked-unavailable, auto falls back to numpy."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("mocked: cupy not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    xp, backend_name = get_backend("auto")
    assert backend_name == "numpy"
    assert xp is np


def test_get_backend_caches_subsequent_calls():
    """Repeated get_backend("numpy") returns the same module object."""
    xp1, _ = get_backend("numpy")
    xp2, _ = get_backend("numpy")
    assert xp1 is xp2


def test_get_backend_rejects_unknown():
    """Unknown backend name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("totally_made_up")


def test_get_backend_mlx_raises_not_implemented():
    """MLX backend is reserved but not yet implemented."""
    with pytest.raises(NotImplementedError, match="MLX"):
        get_backend("mlx")


def test_get_backend_reads_env_var(monkeypatch):
    """SIM_BACKEND env var sets the default when name=None."""
    monkeypatch.setenv("SIM_BACKEND", "numpy")
    xp, name = get_backend()  # no arg
    assert name == "numpy"


# ──────────────────────────────────────────────────────────────────────
# is_gpu_backend
# ──────────────────────────────────────────────────────────────────────


def test_is_gpu_backend_numpy_false():
    """NumPy backend reports is_gpu_backend == False."""
    get_backend("numpy")
    assert is_gpu_backend() is False


@pytest.mark.skipif(not _has_cupy(), reason="cupy not available")
def test_is_gpu_backend_cupy_true():
    """CuPy backend reports is_gpu_backend == True."""
    get_backend("cupy")
    assert is_gpu_backend() is True


# ──────────────────────────────────────────────────────────────────────
# fuse decorator
# ──────────────────────────────────────────────────────────────────────


def test_fuse_decorator_with_parens_numpy_noop():
    """@fuse() on numpy backend leaves the function unchanged."""
    get_backend("numpy")

    @fuse()
    def add(a, b):
        return a + b

    # NumPy backend: function should work directly on numpy arrays
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    result = add(a, b)
    np.testing.assert_array_equal(result, [5, 7, 9])


def test_fuse_decorator_without_parens_numpy_noop():
    """@fuse (no parens) on numpy backend also leaves function unchanged."""
    get_backend("numpy")

    @fuse
    def mul(a, b):
        return a * b

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    result = mul(a, b)
    np.testing.assert_array_equal(result, [4.0, 10.0, 18.0])


def test_fuse_with_kwargs_numpy_noop():
    """@fuse(kernel_name=...) on numpy backend ignores kwargs."""
    get_backend("numpy")

    @fuse(kernel_name="my_kernel")
    def sub(a, b):
        return a - b

    a = np.array([10, 20, 30])
    b = np.array([1, 2, 3])
    result = sub(a, b)
    np.testing.assert_array_equal(result, [9, 18, 27])


@pytest.mark.skipif(not _has_cupy(), reason="cupy not available")
def test_fuse_delegates_to_cupy_when_available():
    """@fuse() on cupy backend wraps with cupy.fuse()."""
    get_backend("cupy")

    @fuse()
    def add(a, b):
        return a + b

    import cupy as cp
    a = cp.array([1, 2, 3])
    b = cp.array([4, 5, 6])
    result = add(a, b)
    cp.testing.assert_array_equal(result, cp.array([5, 7, 9]))


# ──────────────────────────────────────────────────────────────────────
# synchronize
# ──────────────────────────────────────────────────────────────────────


def test_synchronize_numpy_noop():
    """synchronize() on numpy backend returns without error."""
    get_backend("numpy")
    # Should not raise
    synchronize()


@pytest.mark.skipif(not _has_cupy(), reason="cupy not available")
def test_synchronize_cupy_runs():
    """synchronize() on cupy backend invokes cuda synchronize."""
    get_backend("cupy")
    # If this raises, the test fails — but on most setups it just succeeds
    synchronize()


# ──────────────────────────────────────────────────────────────────────
# to_host / from_host transfers
# ──────────────────────────────────────────────────────────────────────


def test_to_host_numpy_passthrough():
    """to_host(arr) on numpy backend returns the array unchanged."""
    get_backend("numpy")
    a = np.array([1, 2, 3])
    result = to_host(a)
    assert result is a or np.array_equal(result, a)


def test_from_host_numpy_passthrough():
    """from_host(arr) on numpy backend returns a numpy array."""
    get_backend("numpy")
    a = [1, 2, 3]
    result = from_host(a)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1, 2, 3])


def test_from_host_with_dtype():
    """from_host respects dtype on numpy backend."""
    get_backend("numpy")
    a = [1.0, 2.0, 3.0]
    result = from_host(a, dtype=np.float32)
    assert result.dtype == np.float32


@pytest.mark.skipif(not _has_cupy(), reason="cupy not available")
def test_to_host_cupy_returns_numpy():
    """to_host(cupy_arr) returns a numpy array."""
    get_backend("cupy")
    import cupy as cp
    a = cp.array([1, 2, 3])
    result = to_host(a)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1, 2, 3])


@pytest.mark.skipif(not _has_cupy(), reason="cupy not available")
def test_from_host_cupy_returns_cupy():
    """from_host([..]) under cupy backend returns a cupy array."""
    get_backend("cupy")
    import cupy as cp
    result = from_host([1.0, 2.0, 3.0], dtype=np.float32)
    assert isinstance(result, cp.ndarray)
    assert result.dtype == np.float32


# ──────────────────────────────────────────────────────────────────────
# get_sparse_module
# ──────────────────────────────────────────────────────────────────────


def test_get_sparse_module_numpy_returns_scipy():
    """get_sparse_module() on numpy backend returns scipy.sparse."""
    get_backend("numpy")
    sp = get_sparse_module()
    import scipy.sparse as ref
    assert sp is ref


def test_get_sparse_module_numpy_csr_works():
    """scipy.sparse.csr_matrix works through the abstraction."""
    get_backend("numpy")
    sp = get_sparse_module()
    M = sp.csr_matrix(np.eye(3))
    assert M.shape == (3, 3)
    assert M.nnz == 3


@pytest.mark.skipif(not _has_cupy(), reason="cupy not available")
def test_get_sparse_module_cupy_returns_cupyx_sparse():
    """get_sparse_module() on cupy backend returns cupyx.scipy.sparse."""
    get_backend("cupy")
    sp = get_sparse_module()
    import cupyx.scipy.sparse as ref
    assert sp is ref


# ──────────────────────────────────────────────────────────────────────
# get_memory_pool_used_mb
# ──────────────────────────────────────────────────────────────────────


def test_get_memory_pool_used_mb_numpy_returns_none():
    """NumPy backend has no memory pool; returns None."""
    get_backend("numpy")
    assert get_memory_pool_used_mb() is None


@pytest.mark.skipif(not _has_cupy(), reason="cupy not available")
def test_get_memory_pool_used_mb_cupy_returns_float():
    """CuPy backend returns a non-negative float."""
    get_backend("cupy")
    val = get_memory_pool_used_mb()
    assert val is not None
    assert val >= 0.0


# ──────────────────────────────────────────────────────────────────────
# Round-trip: a small computation works the same on both backends
# ──────────────────────────────────────────────────────────────────────


def test_round_trip_simple_op_numpy():
    """A simple vector op produces correct result on numpy backend."""
    xp, _ = get_backend("numpy")
    a = xp.array([1.0, 2.0, 3.0, 4.0])
    b = xp.array([5.0, 6.0, 7.0, 8.0])
    result = xp.dot(a, b)  # 1*5 + 2*6 + 3*7 + 4*8 = 70
    assert float(result) == pytest.approx(70.0)


@pytest.mark.skipif(not _has_cupy(), reason="cupy not available")
def test_round_trip_simple_op_cupy_matches_numpy():
    """Same op on cupy backend gives matching result (numerical parity)."""
    xp_cp, _ = get_backend("cupy")
    a_cp = xp_cp.array([1.0, 2.0, 3.0, 4.0])
    b_cp = xp_cp.array([5.0, 6.0, 7.0, 8.0])
    result_cp = float(xp_cp.dot(a_cp, b_cp))
    assert result_cp == pytest.approx(70.0)


# ──────────────────────────────────────────────────────────────────────
# Backwards-compat: existing code (sim.kernels) still works on cupy when
# `from sim.backend import ...` is added alongside the unchanged `import cupy`
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _has_cupy(), reason="cupy not available")
def test_existing_sim_modules_still_import_under_default_backend():
    """Importing sim.kernels still works (CuPy path preserved)."""
    # Force cupy backend to match the current default
    get_backend("cupy")
    # If this import raises, the abstraction broke existing code
    import sim.kernels  # noqa: F401


# ──────────────────────────────────────────────────────────────────────
# Device-level helpers (set_device, get_device_mem_info, etc.)
# ──────────────────────────────────────────────────────────────────────


def test_set_device_numpy_noop():
    """set_device() on numpy backend is a silent no-op (no error)."""
    get_backend("numpy")
    set_device(0)  # should not raise


@pytest.mark.skipif(not _has_cupy(), reason="cupy not available")
def test_set_device_cupy_runs():
    """set_device() on cupy backend invokes cupy.cuda.Device.use()."""
    get_backend("cupy")
    set_device(0)  # should not raise


def test_get_device_mem_info_numpy_returns_tuple():
    """get_device_mem_info() on numpy returns (free_bytes, total_bytes) > 0."""
    get_backend("numpy")
    mem = get_device_mem_info()
    assert isinstance(mem, tuple)
    assert len(mem) == 2
    free, total = mem
    assert isinstance(free, int)
    assert isinstance(total, int)
    assert free > 0
    assert total > 0
    assert free <= total


@pytest.mark.skipif(not _has_cupy(), reason="cupy not available")
def test_get_device_mem_info_cupy_returns_gpu_mem():
    """get_device_mem_info() on cupy returns GPU VRAM info."""
    get_backend("cupy")
    free, total = get_device_mem_info()
    # RTX 3090 has 24 GB ~ 24 * 1024^3 ~ 2.6e10. Other GPUs differ.
    # Sanity: at least 1 GB total VRAM.
    assert total > 1 * 1024**3, f"total VRAM seems too small: {total}"
    assert free <= total


def test_get_device_properties_numpy_synthetic_dict():
    """get_device_properties() on numpy returns synthetic dict with key fields."""
    get_backend("numpy")
    props = get_device_properties(0)
    assert isinstance(props, dict)
    assert "totalGlobalMem" in props
    assert "name" in props
    # Bridge reads: gpu_name = dev_props.get('name', b'Unknown').decode()
    name = props.get("name", b"Unknown")
    assert isinstance(name, bytes)
    assert name.decode().startswith("CPU")
    # totalGlobalMem must be positive int (used in pool.set_limit calculation)
    assert isinstance(props["totalGlobalMem"], int)
    assert props["totalGlobalMem"] > 0


@pytest.mark.skipif(not _has_cupy(), reason="cupy not available")
def test_get_device_properties_cupy_returns_real_props():
    """get_device_properties() on cupy returns the real GPU props dict."""
    get_backend("cupy")
    props = get_device_properties(0)
    assert "totalGlobalMem" in props
    assert "name" in props
    # GPU has totalGlobalMem in bytes; should be >= 1 GB
    assert props["totalGlobalMem"] >= 1 * 1024**3


def test_get_memory_pool_numpy_returns_none():
    """get_memory_pool() on numpy returns None (no pool concept)."""
    get_backend("numpy")
    assert get_memory_pool() is None


def test_get_pinned_memory_pool_numpy_returns_none():
    """get_pinned_memory_pool() on numpy returns None."""
    get_backend("numpy")
    assert get_pinned_memory_pool() is None


@pytest.mark.skipif(not _has_cupy(), reason="cupy not available")
def test_get_memory_pool_cupy_returns_pool():
    """get_memory_pool() on cupy returns a MemoryPool with usable API."""
    get_backend("cupy")
    pool = get_memory_pool()
    assert pool is not None
    # API check: pool should support used_bytes()
    assert pool.used_bytes() >= 0


def test_bridge_synthetic_props_compatible_with_call_pattern():
    """The synthetic props dict supports the exact bridge.py read pattern.

    Bridge does:
      dev_props = get_device_properties(0)
      total_mem = dev_props['totalGlobalMem']
      gpu_name = dev_props.get('name', b'Unknown').decode()
    """
    get_backend("numpy")
    dev_props = get_device_properties(0)
    total_mem = dev_props['totalGlobalMem']  # __getitem__
    gpu_name = dev_props.get('name', b'Unknown').decode()  # .get + .decode
    assert isinstance(total_mem, int) and total_mem > 0
    assert isinstance(gpu_name, str)
