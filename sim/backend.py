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

import logging
import os
from typing import Tuple, Any

_log = logging.getLogger(__name__)

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
        # WARN LOUDLY when NumPy is selected on a box that HAS a usable GPU. This is a perf cliff that is otherwise
        # invisible: the process stays alive, burns 100% CPU, grows its log -- every liveness signal says healthy --
        # while running 10-50x slower on the wrong device.
        # HOW THIS BIT US (2026-07-16): 327 research runners do os.environ.setdefault("SIM_BACKEND", "numpy").
        # That was HARMLESS for months only because `scipy` was MISSING: get_sparse_module() raised
        # ModuleNotFoundError, sim/bridge.py:45's `except ImportError:` swallowed it, and bridge fell back to
        # `import cupy as cp`. So those runners ASKED for numpy and silently got the GPU. Installing scipy made the
        # request REAL -- and a 4-arm sweep then ground ~50 min on CPU before anyone noticed. The runners were not
        # broken by the fix; the fix started HONORING what they literally asked for, and revealed the ask was wrong.
        # A warning cannot fix their default, but it makes the choice VISIBLE at the one place every caller passes.
        try:
            import cupy as _cp  # noqa: F401
            if _cp.cuda.runtime.getDeviceCount() > 0:
                _log.warning(
                    "SIM_BACKEND=numpy selected, but a CUDA GPU + CuPy ARE available -> this run is on the CPU "
                    "(typically 10-50x slower). If you did not mean that, pass SIM_BACKEND=cupy explicitly: many "
                    "research runners do os.environ.setdefault('SIM_BACKEND','numpy'), which silently wins unless "
                    "the caller overrides it."
                )
        except ImportError:
            pass   # no cupy installed => numpy is the correct and only choice; stay quiet.
        except Exception as _e:
            # NARROW, and never silent: a bare `except Exception: pass` here previously swallowed a NameError in
            # this very warning (a missing `_log`), so the warning never fired and nothing said so -- the exact
            # failure mode this warning exists to surface.
            _log.debug("backend GPU-availability probe failed (%s: %s)", type(_e).__name__, _e)
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
    # Convert by ARRAY TYPE, not the active backend. A CuPy (device) array must be brought to host even if the active
    # backend was later switched to numpy mid-process (e.g. a cupy-built bridge whose arrays outlive a
    # get_backend("numpy") when a CPU sub-brain is created) -- otherwise the device array reaches np.asarray() and
    # raises "Implicit conversion to a NumPy array is not allowed". A numpy array has NO `.get()`, so this is
    # byte-identical to the prior logic in BOTH consistent-backend cases (cupy backend -> arr.get(); numpy backend ->
    # numpy array returned unchanged); only the backend-mismatch case changes (from a broken cupy array to a correct
    # host copy). Matches the module's own defensive fallback (`arr.get() if hasattr(arr, "get") else arr`).
    if hasattr(arr, "get"):
        return arr.get()
    return arr


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


# ── Device-level helpers (GPU queries with CPU fallbacks) ───────────────


def set_device(device_id: int = 0) -> None:
    """Set the active device on backends that have a device concept.

    On CuPy: cupy.cuda.Device(device_id).use()
    On NumPy: no-op (CPU has no addressable device)
    """
    _, name = get_backend()
    if name == "cupy":
        import cupy
        cupy.cuda.Device(device_id).use()


def get_device_mem_info() -> tuple[int, int]:
    """Return (free_bytes, total_bytes) for the active device.

    On CuPy: cupy.cuda.Device().mem_info (GPU free + total VRAM)
    On NumPy: psutil.virtual_memory() free + total (system RAM)
              Falls back to (sys.maxsize, sys.maxsize) if psutil
              unavailable — i.e. "assume unlimited" so VRAM-aware
              allocation decisions on the call side don't refuse work.
    """
    _, name = get_backend()
    if name == "cupy":
        import cupy
        return cupy.cuda.Device().mem_info
    # NumPy backend: prefer real system RAM info if psutil available
    try:
        import psutil
        vm = psutil.virtual_memory()
        return (int(vm.available), int(vm.total))
    except ImportError:
        import sys
        return (sys.maxsize, sys.maxsize)


def get_device_properties(device_id: int = 0) -> dict:
    """Return device properties dict.

    On CuPy: cupy.cuda.runtime.getDeviceProperties(device_id)
             (dict with 'name' (bytes), 'totalGlobalMem', etc.)
    On NumPy: synthetic dict mirroring the key fields the bridge reads
              (name as bytes, totalGlobalMem from psutil or sys.maxsize).

    The bridge's usage pattern is:
        dev_props = cp.cuda.runtime.getDeviceProperties(0)
        total_mem = dev_props['totalGlobalMem']
        gpu_name = dev_props.get('name', b'Unknown').decode()

    So the synthetic dict must support both __getitem__ and .get().
    """
    _, name = get_backend()
    if name == "cupy":
        import cupy
        return cupy.cuda.runtime.getDeviceProperties(device_id)
    # NumPy backend: synthetic dict
    free, total = get_device_mem_info()
    return {
        "name": b"CPU (NumPy backend)",
        "totalGlobalMem": total,
        "freeGlobalMem": free,
        "is_gpu": False,
    }


def get_memory_pool():
    """Return the active backend's default memory pool, or None.

    On CuPy: cupy.get_default_memory_pool()
    On NumPy: None (no pool)

    Caller should `if pool := get_memory_pool(): pool.foo()` or similar
    None-check before using.
    """
    _, name = get_backend()
    if name == "cupy":
        import cupy
        return cupy.get_default_memory_pool()
    return None


def get_pinned_memory_pool():
    """Return the active backend's pinned memory pool, or None.

    On CuPy: cupy.get_default_pinned_memory_pool()
    On NumPy: None
    """
    _, name = get_backend()
    if name == "cupy":
        import cupy
        return cupy.get_default_pinned_memory_pool()
    return None


# ── Random state save/restore (different method names on CuPy/NumPy) ────


def get_random_state():
    """Return the active backend's RNG state in an opaque container.

    On CuPy: cupy.random.get_random_state() (a RandomState object)
    On NumPy: numpy.random.get_state() (a tuple)

    Use with set_random_state() in a save-state / seed / restore pattern.
    """
    _, name = get_backend()
    if name == "cupy":
        import cupy
        return cupy.random.get_random_state()
    import numpy
    return numpy.random.get_state()


def set_random_state(state) -> None:
    """Restore RNG state saved by get_random_state().

    On CuPy: cupy.random.set_random_state(state)
    On NumPy: numpy.random.set_state(state)
    """
    _, name = get_backend()
    if name == "cupy":
        import cupy
        cupy.random.set_random_state(state)
        return
    import numpy
    numpy.random.set_state(state)


def _reset_cache_for_tests():
    """Test-only helper: clear the cached backend so tests can switch.

    Production code should not call this. Tests that want to verify both
    backends call this between get_backend(name=...) invocations.
    """
    global _cached_backend, _cached_name, _cached_sparse
    _cached_backend = None
    _cached_name = None
    _cached_sparse = None
