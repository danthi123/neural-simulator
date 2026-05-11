"""Pluggable numpy-like backend for the simulator.

Design doc: docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md (Phase 1)
Strategic context: docs/plans/2026-05-11-strategic-reevaluation.md

Goal: make every `cp.*` call in `sim/bridge.py` and friends route through
an `xp` variable that points to either CuPy or NumPy. This unlocks:

- CPU-only execution on hardware without NVIDIA (Mac M-series, Linux
  servers, Windows boxes without RTX cards)
- Algorithmic verification at toy scale (NumPy reference matches CuPy GPU)
- CI tests without GPU
- Foundation for SSD synapse paging (NumPy backend = easier RAM<->SSD
  transitions; CuPy backend can still page out to RAM<->SSD via .get())

Usage:
    from sim.backend import get_backend
    xp, backend_name = get_backend()
    # All cp.array() -> xp.array(), all cp.zeros() -> xp.zeros(), etc.

The backend is selected at import time:
- `SIM_BACKEND=cupy` -> force CuPy (raises ImportError if unavailable)
- `SIM_BACKEND=numpy` -> force NumPy
- `SIM_BACKEND=auto` or unset (default) -> CuPy if available, else NumPy

Backwards compat: with no environment variable set, behavior is identical
to today (CuPy if installed). This module is additive; it does not
change runtime behavior for existing users.

Note on sparse matrices: scipy.sparse is the NumPy-side analog of
cupyx.scipy.sparse. Both expose `csr_matrix`, `csc_matrix`, etc. with
mostly drop-in API. Edge cases are noted in
docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md Phase 2.
"""
from __future__ import annotations

import os
from typing import Tuple, Any

# Module-level cache so repeated get_backend() calls don't re-import.
_cached_backend: Any = None
_cached_name: str | None = None
_cached_sparse: Any = None


def _detect_default() -> str:
    """Auto-detect: CuPy if available, else NumPy."""
    try:
        import cupy  # noqa: F401
        return "cupy"
    except (ImportError, RuntimeError):
        # ImportError: cupy not installed
        # RuntimeError: cupy installed but no CUDA driver (rare; e.g. on
        # a Linux box with cupy in venv but no nvidia-smi)
        return "numpy"


def get_backend(name: str | None = None) -> Tuple[Any, str]:
    """Return (xp_module, backend_name) for the requested backend.

    Args:
        name: "cupy" | "numpy" | "auto" | None. If None, reads SIM_BACKEND
            env var; falls back to "auto" if env var unset.

    Returns:
        (xp, backend_name) where xp is the numpy-like module (cupy or
        numpy) and backend_name is the canonical string identifier.

    Raises:
        ImportError: if the requested backend is unavailable.
        ValueError: if name is not one of the known backends.

    Caches: subsequent calls with the same name return the same module
    object (no repeated import). Pass a different name to switch (useful
    in tests).
    """
    global _cached_backend, _cached_name, _cached_sparse

    # Resolve name. Order:
    #   1. Explicit arg (e.g. get_backend("numpy"))
    #   2. SIM_BACKEND env var (only relevant on first call; subsequent
    #      calls with no arg return the cached backend)
    #   3. Cached backend (so once we've picked one, sticky)
    #   4. Auto-detect (CuPy if available, else NumPy)
    if name is None:
        # Honour an explicit env var even if a cache exists (so a fresh
        # process picks the env var); otherwise fall back to the cache.
        env_choice = os.environ.get("SIM_BACKEND")
        if env_choice is not None:
            name = env_choice
        elif _cached_name is not None:
            return _cached_backend, _cached_name
        else:
            name = "auto"
    if name == "auto":
        name = _detect_default()

    # Cache hit
    if _cached_name == name and _cached_backend is not None:
        return _cached_backend, _cached_name

    # Load the requested backend
    if name == "cupy":
        import cupy
        _cached_backend = cupy
        _cached_name = "cupy"
    elif name == "numpy":
        import numpy
        _cached_backend = numpy
        _cached_name = "numpy"
    elif name == "mlx":
        # Apple Silicon / MLX is reserved for a future phase. The
        # detection logic + this stub document the extensibility point.
        raise NotImplementedError(
            "MLX backend not yet implemented. See "
            "docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md "
            "Phase 5 (post-MVP)."
        )
    else:
        raise ValueError(
            f"Unknown backend '{name}'. Valid: 'cupy', 'numpy', 'auto'."
        )

    # Also resolve sparse module lazily
    _cached_sparse = None

    return _cached_backend, _cached_name


def get_sparse_module():
    """Return the sparse-matrix module matching the active backend.

    For CuPy: cupyx.scipy.sparse
    For NumPy: scipy.sparse

    Both expose csr_matrix, csc_matrix, coo_matrix, etc. with API
    parity for the operations the sim uses (matvec, addition, slicing).

    Returns the module; use as e.g. `sp = get_sparse_module();
    M = sp.csr_matrix(...)`.

    Raises:
        ImportError: if the matching sparse module is unavailable
            (notably: scipy required for NumPy backend).
    """
    global _cached_sparse
    if _cached_sparse is not None:
        return _cached_sparse
    _, name = get_backend()
    if name == "cupy":
        import cupyx.scipy.sparse as sp  # type: ignore[import-not-found]
        _cached_sparse = sp
    elif name == "numpy":
        import scipy.sparse as sp
        _cached_sparse = sp
    else:
        raise RuntimeError(
            f"get_sparse_module: no sparse module for backend '{name}'"
        )
    return _cached_sparse


def is_gpu_backend() -> bool:
    """True if the active backend runs on GPU (CuPy); False for CPU (NumPy)."""
    _, name = get_backend()
    return name == "cupy"


def fuse(func=None, **kwargs):
    """Backend-aware decorator equivalent to `cp.fuse()`.

    On CuPy: delegates to cupy.fuse(), enabling kernel fusion.
    On NumPy: returns the function unchanged (no-op).

    Usage:
        from sim.backend import fuse

        @fuse()
        def my_op(a, b):
            return a + b

    Works as both `@fuse()` and `@fuse` (with or without parens). When
    used without parens, returns the wrapped function directly; with
    parens, returns the decorator factory.
    """
    # Distinguish the two call forms:
    #   @fuse           -> func is the function being decorated
    #   @fuse()         -> func is None; returns the actual decorator
    #   @fuse(arg=...)  -> func is None; returns the actual decorator with kwargs
    if func is not None and callable(func):
        # @fuse (no parens) - func is the target
        _, name = get_backend()
        if name == "cupy":
            import cupy
            return cupy.fuse()(func)
        return func

    # @fuse() or @fuse(arg=...) - return the decorator
    def _decorator(f):
        _, name = get_backend()
        if name == "cupy":
            import cupy
            return cupy.fuse(**kwargs)(f)
        return f
    return _decorator


def synchronize() -> None:
    """Backend-aware synchronization barrier.

    On CuPy: cp.cuda.Stream.null.synchronize() — waits for GPU work to
    complete before returning.
    On NumPy: no-op (CPU operations are synchronous).

    Use this where the sim explicitly needs to ensure GPU work has
    completed before measuring latency, writing checkpoints, etc.
    """
    _, name = get_backend()
    if name == "cupy":
        import cupy
        cupy.cuda.Stream.null.synchronize()
    # else: no-op


def to_host(arr):
    """Copy an array from device memory to host (NumPy) memory.

    On CuPy: arr.get() — D-to-H transfer
    On NumPy: returns the array unchanged (already on host)

    Used at lineage save/load boundaries and any time we need to hand
    data off to non-CuPy code (e.g. h5py, msgpack, plotting libraries).
    """
    xp, name = get_backend()
    if name == "cupy":
        # Guard against numpy arrays sneaking through (e.g. result of a
        # cpu-side computation in cupy mode)
        if hasattr(arr, "get"):
            return arr.get()
        return arr  # already a numpy array
    return arr  # numpy backend: already on host


def from_host(arr, dtype=None):
    """Copy a NumPy array to the active backend's device.

    On CuPy: cp.asarray(arr) — H-to-D transfer
    On NumPy: asnumpy with optional dtype cast

    Useful when constructing arrays from disk / Python lists and need
    them in the active backend's address space.
    """
    xp, name = get_backend()
    if name == "cupy":
        import cupy
        return cupy.asarray(arr, dtype=dtype)
    import numpy
    return numpy.asarray(arr, dtype=dtype)


def get_memory_pool_used_mb() -> float | None:
    """Return current memory pool usage in MB, or None for backends that
    don't expose pool stats.

    On CuPy: cp.get_default_memory_pool().used_bytes() / 1024^2
    On NumPy: None (CPU process memory isn't pool-managed)
    """
    _, name = get_backend()
    if name == "cupy":
        import cupy
        used = cupy.get_default_memory_pool().used_bytes()
        return used / (1024 * 1024)
    return None


def _reset_cache_for_tests():
    """Test-only helper: clear the cached backend so tests can switch.

    Production code should not call this. Tests that want to verify both
    backends call this between get_backend(name=...) invocations.
    """
    global _cached_backend, _cached_name, _cached_sparse
    _cached_backend = None
    _cached_name = None
    _cached_sparse = None
