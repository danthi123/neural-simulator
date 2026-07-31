"""Regression tests for the CHUNKED spatial connectivity generator.

WHY THIS FILE EXISTS (2026-07-31). `generate_spatial_connections_chunked` contained a `del` of four names that do
not exist on its path, so it raised NameError on the FIRST chunk. Every spatial network above the n>15000
chunking threshold (sim/connectivity.py:132) therefore crashed -- the entire large-network path, in a project
whose stated scope is 10K-100K+ neurons and whose shipped profiles declare 50,000. The bug predates the module
extraction (it is in the pre-split monolith), so this path had never executed successfully in the repo's history.

NOTHING CAUGHT IT because nothing tests any spatial generator. tests/test_numpy_backend_integration.py claims to
catch "a future CuPy-only call sneaking into connectivity.py", but it builds 50-neuron Watts-Strogatz networks
and so can never reach a spatial generator (which needs n>1000) -- a check that passes without determining
anything.

AND THE FIRST FIX-VERIFICATION WAS INADEQUATE, which is the reason these tests force the chunk count. Calling the
function at n=300 and n=1200 looked like a verification but wasn't: chunk_size is
`min(n, int(target_mem / (60*n)))`, which on a mostly-free card evaluates to n, so num_chunks == 1 and every
behaviour chunking exists for -- cross-chunk row offsets, accumulate-and-concatenate, the `num_chunks > 1`
progress branch -- was skipped. These tests shim the VRAM query to force >= 3 chunks.
"""
import numpy as np
import pytest

from sim.backend import get_backend, is_gpu_backend, to_host

xp, _BACKEND = get_backend()

# The chunked generator used to be CuPy-only (cp.cuda.mem_info / cp.asnumpy / cp.random.uniform(dtype=)), so
# this file skipped on numpy. Those are fixed, so the structural test now runs on BOTH backends -- which is the
# point: a CPU-runnable test is one CI can actually execute. Only the chunk-forcing shim needs CuPy, since it
# patches cp.cuda.Device; on numpy the generator takes the large-budget path and runs single-chunk.

N = 900
K = 12


# Positions must span a range where the distance term actually BITES. connection_distance_decay_factor
# defaults to 0.01, so in a unit cube exp(-0.01*d) ~ 0.993 and distance is irrelevant -- a locality assertion on
# unit-cube positions is vacuous and passes/fails on noise. The engine places neurons over a box of order 100.
POS_SPAN = 100.0


def _positions_and_traits(n, seed=0):
    rng = np.random.default_rng(seed)
    pos = xp.asarray((rng.random((n, 3)) * POS_SPAN).astype(np.float32))
    traits = xp.asarray((rng.random(n) < 0.5).astype(np.int32))
    return pos, traits


def _force_chunks(monkeypatch, n_chunks):
    """Shrink the reported free VRAM so chunk_size forces >= n_chunks chunks.

    chunk_size = max(64, int(0.35 * free_mem / (60 * n))), so a small `free` yields a small chunk.
    """
    import cupy as cp
    target_rows = max(64, N // n_chunks)
    free_bytes = int(target_rows * 60 * N / 0.35)

    _RealDevice = cp.cuda.Device

    class _Dev(_RealDevice):
        # Subclass the REAL Device so everything else (.id, .use(), the RNG's _random_states lookup) still
        # works -- a bare stub broke cupy.random, which also calls cuda.Device(). Only mem_info is overridden.
        @property
        def mem_info(self):
            return (free_bytes, free_bytes * 4)

    monkeypatch.setattr(cp.cuda, "Device", _Dev)


@pytest.mark.skipif(not is_gpu_backend(), reason="chunk-forcing shim patches cp.cuda.Device")
def test_chunked_runs_multi_chunk_and_is_structurally_correct(monkeypatch):
    """The regression: it must not raise, and must produce a correct graph ACROSS chunk boundaries."""
    from sim.connectivity import generate_spatial_connections_chunked
    from sim.config import CoreSimConfig

    _force_chunks(monkeypatch, 6)
    pos, traits = _positions_and_traits(N)
    m = generate_spatial_connections_chunked(N, K, pos, traits, CoreSimConfig(),
                                             log_fn=lambda *a, **k: None)
    coo = m.tocoo()
    rows, cols = to_host(coo.row), to_host(coo.col)

    assert m.shape == (N, N)
    # exactly k out-edges per neuron, so no chunk was dropped or double-counted
    assert m.nnz == N * K, f"expected {N * K} edges, got {m.nnz}"
    deg = np.bincount(rows, minlength=N)
    assert deg.min() == K and deg.max() == K, f"degree not uniform: {deg.min()}..{deg.max()}"
    # every row index appears -> the per-chunk `start_idx` offset is applied correctly
    assert set(np.unique(rows).tolist()) == set(range(N))
    # self-connections are masked per chunk at start_idx + i; a wrong offset shows up here
    assert not np.any(rows == cols), "self-connections present -> chunk row offset is wrong"


@pytest.mark.skipif(not is_gpu_backend(), reason="chunk-forcing shim patches cp.cuda.Device")
def test_chunked_matches_the_non_chunked_generator_statistically(monkeypatch):
    """Chunking must not change WHICH connections are made, only how they are computed.

    The generator is stochastic (Gumbel top-k), so this compares distributions, not exact edges: both paths must
    select neighbours far closer than chance and biased toward same-trait partners.
    """
    from sim.connectivity import generate_spatial_connections_chunked, generate_spatial_connections_gpu
    from sim.config import CoreSimConfig

    pos, traits = _positions_and_traits(N, seed=1)
    cfg = CoreSimConfig()
    p_host, t_host = to_host(pos), to_host(traits)

    _force_chunks(monkeypatch, 5)
    a = generate_spatial_connections_chunked(N, K, pos, traits, cfg, log_fn=lambda *a, **k: None)
    monkeypatch.undo()
    b = generate_spatial_connections_gpu(N, K, pos, traits, cfg, log_fn=lambda *a, **k: None)

    def stats(m):
        coo = m.tocoo()
        r, c = to_host(coo.row), to_host(coo.col)
        d = np.linalg.norm(p_host[r] - p_host[c], axis=1)
        return float(d.mean()), float((t_host[r] == t_host[c]).mean())

    d_chunk, same_chunk = stats(a)
    d_full, same_full = stats(b)

    # chance level for a random pair
    rng = np.random.default_rng(7)
    i, j = rng.integers(0, N, 20000), rng.integers(0, N, 20000)
    d_chance = float(np.linalg.norm(p_host[i] - p_host[j], axis=1).mean())

    # The GROUND TRUTH for "how local should this be" is the non-chunked generator on the same inputs, not a
    # constant. At the default decay of 0.01 the Gumbel noise (sigma ~1.28) is comparable to the log-prob
    # gradient, so selection is deliberately weak -- an absolute "X% better than chance" threshold is arbitrary
    # and would fail on a correct implementation (it did: 8.6% observed against a made-up 15% bar).
    assert d_chunk < d_chance, "chunked path shows NO spatial locality at all"
    assert d_full < d_chance, "reference path shows no locality -- the test inputs cannot detect the effect"
    assert abs(d_chunk - d_full) < 0.25 * d_full, (
        f"chunked mean distance {d_chunk:.3f} disagrees with full path {d_full:.3f}")
    assert abs(same_chunk - same_full) < 0.15, (
        f"chunked trait bias {same_chunk:.3f} disagrees with full path {same_full:.3f}")


def test_dispatcher_routes_large_n_to_the_chunked_path():
    """The threshold that made this bug reachable: n > 15000 goes to the chunked generator.

    Asserted so that if the routing changes, the test that guards the chunked path is known to have stopped
    guarding the production route.
    """
    import inspect
    from sim import connectivity

    src = inspect.getsource(connectivity.generate_spatial_connections_gpu)
    assert "generate_spatial_connections_chunked" in src
    assert "15000" in src, "the chunking threshold moved; update this test and the chunk-forcing shim"


def test_chunked_runs_on_either_backend_single_chunk():
    """Backend-portability guard: the generator must at least RUN under SIM_BACKEND=numpy.

    Before 2026-07-31 every spatial generator in this module died on numpy -- cp.cuda.Device().mem_info,
    cp.asnumpy, and cp.random.uniform(dtype=) are all CuPy-only, and the first of them fired before any maths.
    test_numpy_backend_integration.py claimed to guard this but used 50-neuron networks that never reach a
    spatial generator.
    """
    from sim.connectivity import generate_spatial_connections_chunked
    from sim.config import CoreSimConfig

    pos, traits = _positions_and_traits(400, seed=3)
    m = generate_spatial_connections_chunked(400, 8, pos, traits, CoreSimConfig(),
                                             log_fn=lambda *a, **k: None)
    rows = to_host(m.tocoo().row)
    assert m.nnz == 400 * 8
    assert not np.any(rows == to_host(m.tocoo().col))
