"""Regression gate for the cupy-hybrid `.tocoo()` crash in set_pathway_weights.

WHY (2026-08-19, a PRODUCTION-DOWN bug found while re-verifying an affect faculty on
the GPU). On the cupy backend `self.cp_connections` can arrive at
`set_pathway_weights(..., add_missing=True)` as a SciPy CSR whose `.data` is a *cupy*
array (a hybrid container built by the onebrain-parser pool bind). SciPy's pure-Python
`.tocoo()` then does `np.array(self.data)` on that cupy array and raises
`TypeError: Implicit conversion to a NumPy array is not allowed`. That 400'd EVERY
production GPU chat build (tiny-demo -> qwen -> make_pool1_onebrain_composer ->
_bind_parser_onto_pool -> set_pathway_weights). The main faculty audit ran on numpy, so
the cupy path was never exercised and the crash stayed invisible.

The fix rebuilds the existing COO from the host CSR arrays the function already decodes
(indptr/indices/data) instead of calling the fragile `.tocoo()`, so it is correct for a
SciPy-hybrid container as well as a cupyx one.

This gate reproduces the crash on a CPU-only CI by wrapping cp_connections in a proxy
whose `.tocoo()` raises the exact cupy TypeError while every other attribute forwards to
a real numpy CSR. On the OLD code the add_missing branch calls `.tocoo()` and the test
raises; on the fixed code the branch never touches `.tocoo()` and the new edges land.
NB: the fix is verified end-to-end on a real GPU separately (production chat loads +
answers on cupy) — this test guards the code path so it cannot silently regress in CI.
"""
import numpy as np
import pytest

from sim import (
    SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig,
)


class _TocooRaisingCSR:
    """Forwards everything to a real numpy CSR but makes .tocoo() raise the cupy error.

    Mimics a SciPy CSR holding cupy .data: reads of indptr/indices/data succeed (the
    host-decode path in set_pathway_weights uses those), but any `.tocoo()` dies exactly
    as cupy data does under SciPy's np.array()."""

    __slots__ = ("_csr",)

    def __init__(self, csr):
        object.__setattr__(self, "_csr", csr)

    def tocoo(self, *a, **k):  # the exact failure the fix removes from the hot path
        raise TypeError(
            "Implicit conversion to a NumPy array is not allowed. "
            "Please use `.get()` to construct a NumPy array explicitly."
        )

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_csr"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_csr"), name, value)


def _fresh_bridge(n=60, seed=42):
    b = SimulationBridge(
        core_config=CoreSimConfig(num_neurons=n, seed=seed),
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(enable_profiling=False),
    )
    b._initialize_simulation_data()
    return b


def test_set_pathway_weights_add_missing_survives_tocoo_hostile_container():
    """add_missing must not depend on cp_connections.tocoo() (the cupy-hybrid crash)."""
    b = _fresh_bridge()
    # Wrap the live CSR so .tocoo() raises like a cupy-hybrid container would.
    b.cp_connections = _TocooRaisingCSR(b.cp_connections)

    pre = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    post = np.array([40, 41, 42, 43, 44], dtype=np.int64)
    w = np.array([0.11, 0.22, 0.33, 0.44, 0.55], dtype=np.float32)

    # On the pre-fix code this raises TypeError from the wrapped .tocoo(); on the fix it
    # completes because the COO is rebuilt from host indptr/indices/data.
    n_updated = b.set_pathway_weights("gate_probe", pre, post, w, add_missing=True)
    assert n_updated == 5

    # The five new edges must be present with the correct weights.
    csr = b.cp_connections
    coo = csr.tocoo() if hasattr(csr, "tocoo") else csr
    row = np.asarray(coo.row)
    col = np.asarray(coo.col)
    dat = np.asarray(coo.data)
    for p, q, wv in zip(pre, post, w):
        m = (row == p) & (col == q)
        assert m.any(), f"edge ({p},{q}) missing after add_missing"
        assert np.isclose(dat[m][0], wv, atol=1e-6), f"edge ({p},{q}) wrong weight"


def test_set_pathway_weights_add_missing_correct_on_plain_csr():
    """No-regression: the normal (non-hostile) numpy path still adds edges correctly."""
    b = _fresh_bridge()
    pre = np.array([2, 6], dtype=np.int64)
    post = np.array([50, 51], dtype=np.int64)
    w = np.array([0.7, 0.9], dtype=np.float32)
    n_updated = b.set_pathway_weights("plain_probe", pre, post, w, add_missing=True)
    assert n_updated == 2
    coo = b.cp_connections.tocoo()
    row, col, dat = np.asarray(coo.row), np.asarray(coo.col), np.asarray(coo.data)
    for p, q, wv in zip(pre, post, w):
        m = (row == p) & (col == q)
        assert m.any() and np.isclose(dat[m][0], wv, atol=1e-6)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
