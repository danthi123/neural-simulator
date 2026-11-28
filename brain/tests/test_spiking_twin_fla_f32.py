# Capability: Language
# Record: brain/experiments/results/2026-11-27-spiking-twin-e1/run/parity/bias_diagnosis.md
# Expected-fail: none (a kernel-numerics guard, not a capability claim)
"""fla's fp32 dots run in IEEE fp32, not Triton's default TF32 (brain.spiking_twin.fla_f32_precision). CPU only: fla
itself is never imported (its import probes CUDA); a stub package stands in for it where the import order is checked.

The GPU diagnosis (bias_diagnosis.md) found TF32 inside chunk_rwkv7 shrinks its fp32-input output toward zero (slope
-3e-4 to -9e-4 per layer, 63% of o-site count flips downward); TRITON_F32_DEFAULT=ieee removes it (error 1e-6)."""
from __future__ import annotations

import sys
import types

import pytest

import brain.spiking_twin as st


def test_sets_ieee_when_unset(monkeypatch):
    monkeypatch.delenv("TRITON_F32_DEFAULT", raising=False)
    assert st.fla_f32_precision() == "ieee"
    import os
    assert os.environ["TRITON_F32_DEFAULT"] == "ieee"


def test_accepts_explicit_ieee_and_refuses_other_settings(monkeypatch):
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    assert st.fla_f32_precision() == "ieee"
    for bad in ("tf32", "tf32x3"):
        monkeypatch.setenv("TRITON_F32_DEFAULT", bad)
        with pytest.raises(RuntimeError, match="TRITON_F32_DEFAULT"):
            st.fla_f32_precision()


def test_triton_reads_the_setting_and_a_python_override_is_refused(monkeypatch):
    triton = pytest.importorskip("triton")
    if not hasattr(triton, "knobs"):
        pytest.skip("this triton has no knobs module")
    monkeypatch.delenv("TRITON_F32_DEFAULT", raising=False)
    st.fla_f32_precision()
    assert triton.knobs.language.fp32_default == "ieee"        # what tl.dot reads at compile time
    triton.knobs.language.fp32_default = "tf32"                # also writes the environment (monkeypatch restores it)
    try:
        monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")       # environment fine, Python-level override not
        with pytest.raises(RuntimeError, match="knobs"):
            st.fla_f32_precision()
    finally:
        del triton.knobs.language.fp32_default


def test_fla_import_sets_ieee_before_importing_fla(monkeypatch):
    """The precision must be in place before any fla module loads (kernels compile on first call; Triton's in-process
    cache is not keyed on the setting), so fla_import sets it before its import."""
    monkeypatch.delenv("TRITON_F32_DEFAULT", raising=False)
    monkeypatch.setattr(st, "fla_version", lambda: st.FLA_VERSION)
    seen = {}

    class _Ops(types.ModuleType):
        def __getattr__(self, name):
            import os
            if name.startswith("__"):
                raise AttributeError(name)
            seen[name] = os.environ.get("TRITON_F32_DEFAULT")
            return object()

    for name in ("fla", "fla.ops"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "fla.ops.rwkv7", _Ops("fla.ops.rwkv7"))
    st.fla_import()
    assert seen == {"chunk_rwkv7": "ieee", "fused_recurrent_rwkv7": "ieee"}


def test_fla_import_refuses_a_tf32_setting(monkeypatch):
    monkeypatch.setenv("TRITON_F32_DEFAULT", "tf32")
    monkeypatch.setattr(st, "fla_version", lambda: st.FLA_VERSION)
    with pytest.raises(RuntimeError):
        st.fla_import()
