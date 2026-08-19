"""Independent correctness oracle: cross-validate the vanilla spiking core against Brian2.

WHY THIS EXISTS
    The project has no independent check that its core spiking engine is numerically
    correct. A wrong-mechanism bug can still produce clean-looking numbers (an
    unseeded-substrate confound once voided a whole research arc here). Brian2
    (Stimberg, Brette & Goodman, "Brian 2, an intuitive and efficient neural
    simulator", eLife 8:e47314, 2019; https://doi.org/10.7554/eLife.47314) shares
    ZERO code with our engine, so it is a far stronger cross-check than our own
    numpy path. This oracle rebuilds a well-defined VANILLA subset in BOTH
    simulators at the same seed/dt/params and asserts agreement.

SCOPE (deliberately narrow)
    Validates the VANILLA point-neuron core only:
      (a) Izhikevich-2007 point neuron  (production default neuron model)
      (b) AdEx point neuron             (second neuron model)
      (c) COBA synapse conductance kinetics + driving-force current (voltage clamp)
      (d) pair-based STDP weight-change curve (soft multiplicative bounds)
    Our load-bearing CUSTOM mechanisms (multicompartment dendritic credit, BTSP
    plateaus, neuromodulator-gated plasticity, HTM/BDSP rules, the VSA composer)
    have NO Brian2 equivalent and are explicitly OUT of scope.

TEST-ONLY, CPU-ONLY
    Brian2 never touches the production substrate, the GPU path, or the bridge.
    We load ONLY the engine's pure-math kernels (sim/kernels.py + sim/backend.py)
    via a file-based import, with SIM_BACKEND=numpy forced, and drive them in a
    minimal loop that mirrors the bridge's documented per-step order (see the exact
    bridge.py line references in the harness helpers). The kernels ARE the real
    production dynamics/plasticity math; nothing here re-expresses the biology.

ACHIEVED PARITY (measured on this box; see the finding doc for the full table)
    IZH   : spike steps BIT-EXACT; float64 traj <=1.8e-11 mV; float32 <=0.045 mV
    AdEx  : spike steps BIT-EXACT; float64 traj <=1e-12 mV;   float32 <=1e-3 mV
    COBA  : conductance <=1e-15 nS, current <=1e-13 pA (machine precision)
    STDP  : worst |dw| diff 4.8e-8 across a delta_t x w0 sweep

RUNNING IT
    Brian2 2.9.0 (latest) requires numpy < 2, while the repo pins numpy 2.4.x, so
    brian2 cannot be installed into the repo venv. Run this oracle in a dedicated
    numpy<2 venv (see requirements-dev.txt). Without brian2 importable the module
    SKIPS cleanly (never hard-fails CI).
"""
import os
import sys
import types
import importlib.util
from pathlib import Path

import numpy as np
import pytest

# --- brian2 import guard: skip the whole module if brian2 cannot be imported for
#     ANY reason (absent, or the numpy>=2 incompatibility). importorskip only
#     catches ImportError; the numpy-2 breakage raises AttributeError, so guard broadly.
try:
    import brian2 as b2
except Exception as _brian_err:  # pragma: no cover - environment dependent
    pytest.skip(
        f"brian2 unavailable/incompatible ({type(_brian_err).__name__}); "
        "run the oracle in a numpy<2 venv (see requirements-dev.txt)",
        allow_module_level=True,
    )

b2.prefs.codegen.target = "numpy"          # single-threaded CPU codegen; no compilation
b2.BrianLogger.suppress_name("resolution_conflict")


# --------------------------------------------------------------------------------------
# Load the ENGINE's real pure-math kernels (sim/kernels.py) on the numpy/CPU backend,
# WITHOUT importing the heavy sim package __init__ (which pulls in the GPU bridge).
# This keeps the oracle test-only and minimal-dep while running genuine engine code.
# --------------------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
os.environ["SIM_BACKEND"] = "numpy"        # FORCE cpu even where cupy is installed


def _load_engine_kernels():
    if "sim" not in sys.modules:
        pkg = types.ModuleType("sim")
        pkg.__path__ = [str(_REPO / "sim")]
        sys.modules["sim"] = pkg

    def _load(mod, path):
        spec = importlib.util.spec_from_file_location(mod, str(path))
        m = importlib.util.module_from_spec(spec)
        sys.modules[mod] = m
        spec.loader.exec_module(m)
        return m

    backend = _load("sim.backend", _REPO / "sim" / "backend.py")
    kernels = _load("sim.kernels", _REPO / "sim" / "kernels.py")
    return backend, kernels


_BACKEND, K = _load_engine_kernels()


def test_engine_backend_is_numpy_cpu():
    """Anti-cheat: the oracle must never run the engine kernels on the GPU."""
    _xp, name = _BACKEND.get_backend()
    assert name == "numpy", f"engine backend must be numpy for this CPU oracle, got {name!r}"


# --------------------------------------------------------------------------------------
# Production parameters (read from sim/config.py + sim/enums.py, verified at pin time):
#   Izhikevich-2007 RS cortical pyramidal (default_neuron_type_izh):
#     C=100 pF, k=0.7, vr=-60, vt=-40, vpeak=35, a=0.03, b=-2, c_reset=-50, d=100
#     init u = izh_b*(v-vr)  (bridge.py:1789);  dt_ms=1.0
#   AdEx (sim/config.py): C=281, g_L=30, E_L=-70.6, V_T=-50.4, Delta_T=2.0, a=4,
#     tau_w=144, b=80.5, V_peak=-40, V_r=-70.6
#   COBA: syn_tau_g_e=5, syn_tau_g_i=10, E_e=0, E_i=-75; decay=exp(-dt/tau) (bridge.py:2034)
#   STDP: a_plus=0.012, a_minus=0.01, tau_plus=tau_minus=20, w_min=0, w_max=2
# Refractory: production default refractory_period_steps=2 blocks spike DETECTION only
#   (voltage keeps integrating; bridge.py:7577,7595). It is NON-BINDING at the firing
#   regimes below (min ISI >> 2 steps), so we disable it to isolate the integrator; the
#   traces/spikes are identical to production. STDP/COBA kernels are refractory-free.
# --------------------------------------------------------------------------------------
IZH = dict(C=100.0, k=0.7, vr=-60.0, vt=-40.0, vpeak=35.0, a=0.03, b=-2.0, c=-50.0, d=100.0)
ADEX = dict(C=281.0, gL=30.0, EL=-70.6, VT=-50.4, DT=2.0, a=4.0, tauw=144.0,
            b=80.5, Vpeak=-40.0, Vr=-70.6)
STDP = dict(Aplus=0.012, Aminus=0.01, tauplus=20.0, tauminus=20.0, wmin=0.0, wmax=2.0)
DT_MS = 1.0


# ================================ engine-side harnesses ================================
# Each drives the REAL sim/kernels.py fused kernel in a loop mirroring the bridge's
# per-step order.  Returns the membrane trace (per step, post-reset) and spike step
# indices, matching Brian2's StateMonitor(when='end') convention (verified: offset 0).

def _engine_izhikevich(i_pA, n_steps, dtype):
    p = {kk: dtype(vv) for kk, vv in IZH.items()}
    dt = dtype(DT_MS)
    v = np.array([p["vr"]], dtype=dtype)
    u = np.array([p["b"] * (v[0] - p["vr"])], dtype=dtype)   # bridge.py:1789 init
    cur = np.array([i_pA], dtype=dtype)
    trace, spikes = [], []
    for i in range(n_steps):
        # bridge.py:7571 fused_izhikevich2007_dynamics_update
        vn, un = K.fused_izhikevich2007_dynamics_update(
            v, u, p["C"], p["k"], p["vr"], p["vt"], p["a"], p["b"], cur, dt)
        if vn[0] >= p["vpeak"]:                              # bridge.py:7595 threshold
            vn[0] = p["c"]                                   # bridge.py:7603 reset v=c
            un[0] = un[0] + p["d"]                           # bridge.py:7604 reset u+=d
            spikes.append(i)
        v[:], u[:] = vn, un
        trace.append(float(v[0]))
    return np.asarray(trace), spikes


def _engine_adex(i_pA, n_steps, dtype):
    p = {kk: dtype(vv) for kk, vv in ADEX.items()}
    dt = dtype(DT_MS)
    v = np.array([p["EL"]], dtype=dtype)
    w = np.array([0.0], dtype=dtype)
    cur = np.array([i_pA], dtype=dtype)
    trace, spikes = [], []
    for i in range(n_steps):
        # bridge.py:7737 fused_adex_dynamics_update
        vn, wn = K.fused_adex_dynamics_update(
            v, w, cur, dt, p["C"], p["gL"], p["EL"], p["VT"], p["DT"], p["a"], p["tauw"])
        if vn[0] >= p["Vpeak"]:                              # bridge.py:7744 threshold
            vn[0] = p["Vr"]                                  # bridge.py:7752 reset v=V_r
            wn[0] = wn[0] + p["b"]                            # bridge.py:7753 reset w+=b
            spikes.append(i)
        v[:], w[:] = vn, wn
        trace.append(float(v[0]))
    return np.asarray(trace), spikes


def _engine_coba(g0_nS, tau_ms, E_mV, v_clamp_mV, n_steps, dtype):
    # bridge.py:6985 fused_conductance_decay_and_current; decay=exp(-dt/tau) bridge.py:2034
    decay_e = dtype(np.exp(-DT_MS / tau_ms))
    decay_i = dtype(np.exp(-DT_MS / 10.0))
    ge = np.array([g0_nS], dtype=dtype)
    gi = np.array([0.0], dtype=dtype)
    v = np.array([v_clamp_mV], dtype=dtype)
    Ee, Ei = dtype(E_mV), dtype(-75.0)
    g_trace, i_trace = [], []
    for _ in range(n_steps):
        gen, gin, isyn = K.fused_conductance_decay_and_current(ge, gi, decay_e, decay_i, v, Ee, Ei)
        ge[:], gi[:] = gen, gin
        g_trace.append(float(ge[0]))
        i_trace.append(float(isyn[0]))
    return np.asarray(g_trace), np.asarray(i_trace)


def _engine_stdp(delta_t_ms, w0, dtype):
    # bridge.py:8076 fused_stdp_weight_update, single last-spike pair
    return float(K.fused_stdp_weight_update(
        dtype(delta_t_ms), dtype(w0),
        dtype(STDP["Aplus"]), dtype(STDP["Aminus"]),
        dtype(STDP["tauplus"]), dtype(STDP["tauminus"]),
        dtype(STDP["wmin"]), dtype(STDP["wmax"])))


# ================================ Brian2-side builders =================================
# Fully independent: Brian2's own unit system, code generation, ODE state updaters and
# event-driven synapse machinery.  Namespace constants use unique tokens so no name
# collides with the module/global namespace (silences Brian2 resolution warnings).

def _brian_izhikevich(i_pA, n_steps):
    b2.start_scope()
    b2.defaultclock.dt = DT_MS * b2.ms
    ns = dict(C0=100 * b2.pF, k0=0.7 * b2.pA / b2.mV**2, vr0=-60 * b2.mV, vt0=-40 * b2.mV,
              a0=0.03 / b2.ms, b0=-2 * b2.nS, Iin=i_pA * b2.pA)
    eqs = """dv/dt = (k0*(v-vr0)*(v-vt0) - u + Iin)/C0 : volt
             du/dt = a0*(b0*(v-vr0) - u) : amp"""
    G = b2.NeuronGroup(1, eqs, threshold="v>=35*mV", reset="v=-50*mV; u+=100*pA",
                       method="euler", namespace=ns)
    G.v = -60 * b2.mV
    G.u = "-2*nS*(v-(-60*mV))"
    mon = b2.StateMonitor(G, "v", record=True, when="end")
    spk = b2.SpikeMonitor(G)
    b2.run(n_steps * b2.ms)
    return np.asarray(mon.v[0] / b2.mV), np.round(np.asarray(spk.t / b2.ms)).astype(int).tolist()


def _brian_adex(i_pA, n_steps):
    b2.start_scope()
    b2.defaultclock.dt = DT_MS * b2.ms
    ns = dict(C0=281 * b2.pF, gL0=30 * b2.nS, EL0=-70.6 * b2.mV, VT0=-50.4 * b2.mV,
              DT0=2 * b2.mV, a0=4 * b2.nS, tauw0=144 * b2.ms, Iin=i_pA * b2.pA)
    # exp arg clipped to [-20, 5] exactly as sim/kernels.py fused_adex_dynamics_update.
    eqs = """dv/dt = (-gL0*(v-EL0) + gL0*DT0*exp(clip((v-VT0)/DT0,-20,5)) - w + Iin)/C0 : volt
             dw/dt = (a0*(v-EL0) - w)/tauw0 : amp"""
    G = b2.NeuronGroup(1, eqs, threshold="v>=-40*mV", reset="v=-70.6*mV; w+=80.5*pA",
                       method="euler", namespace=ns)
    G.v = -70.6 * b2.mV
    G.w = 0 * b2.pA
    mon = b2.StateMonitor(G, "v", record=True, when="end")
    spk = b2.SpikeMonitor(G)
    b2.run(n_steps * b2.ms)
    return np.asarray(mon.v[0] / b2.mV), np.round(np.asarray(spk.t / b2.ms)).astype(int).tolist()


def _brian_coba(g0_nS, tau_ms, E_mV, v_clamp_mV, n_steps):
    b2.start_scope()
    b2.defaultclock.dt = DT_MS * b2.ms
    ns = dict(tau0=tau_ms * b2.ms, Ee0=E_mV * b2.mV)
    eqs = """dge/dt = -ge/tau0 : siemens
             v : volt (constant)
             Isyn = ge*(Ee0 - v) : amp"""
    G = b2.NeuronGroup(1, eqs, method="exact", namespace=ns)   # exact exp decay
    G.ge = g0_nS * b2.nS
    G.v = v_clamp_mV * b2.mV
    mon = b2.StateMonitor(G, ["ge", "Isyn"], record=True, when="end")
    b2.run(n_steps * b2.ms)
    return np.asarray(mon.ge[0] / b2.nS), np.asarray(mon.Isyn[0] / b2.pA)


def _brian_stdp(delta_t_ms, w0):
    """Independent soft-bound pair-STDP via Brian2 event-driven traces.

    For a single pre/post pair separated by delta_t this reproduces the engine's
    delta_t-parameterised soft-bound kernel:
        causal (post after pre): dw = +Aplus *(wmax-w)*exp(-delta_t/tauplus)
        anti-causal (pre after post): dw = -Aminus*(w-wmin)*exp(+delta_t/tauminus)
    """
    b2.start_scope()
    b2.defaultclock.dt = DT_MS * b2.ms
    t_pre = 100.0
    t_post = 100.0 + delta_t_ms
    pre = b2.SpikeGeneratorGroup(1, [0], [t_pre] * b2.ms)
    post = b2.SpikeGeneratorGroup(1, [0], [t_post] * b2.ms)
    ns = dict(Aplus=STDP["Aplus"], Aminus=STDP["Aminus"],
              tauplus=STDP["tauplus"] * b2.ms, tauminus=STDP["tauminus"] * b2.ms,
              wmin=STDP["wmin"], wmax=STDP["wmax"])
    S = b2.Synapses(
        pre, post,
        model="""w : 1
                 dapre/dt  = -apre/tauplus  : 1 (event-driven)
                 dapost/dt = -apost/tauminus : 1 (event-driven)""",
        on_pre="""apre = 1
                  w = clip(w - Aminus*(w-wmin)*apost, wmin, wmax)""",
        on_post="""apost = 1
                   w = clip(w + Aplus*(wmax-w)*apre, wmin, wmax)""",
        namespace=ns,
    )
    S.connect(i=0, j=0)
    S.w = w0
    b2.run((max(t_pre, t_post) + 50.0) * b2.ms)
    return float(S.w[0])


# ==================================== the oracle tests =================================

@pytest.mark.parametrize("i_pA,n_steps,min_spikes", [(100.0, 400, 3), (300.0, 400, 10), (600.0, 400, 20)])
def test_izhikevich_point_neuron(i_pA, n_steps, min_spikes):
    # float64: isolates the integration SCHEME (should be algorithmically identical)
    ev64, es64 = _engine_izhikevich(i_pA, n_steps, np.float64)
    bv, bs = _brian_izhikevich(i_pA, n_steps)
    assert len(bs) >= min_spikes, "sanity: neuron must actually fire (no vacuous pass)"
    assert es64 == bs, "spike step indices must be bit-exact (float64)"
    assert np.max(np.abs(ev64 - bv)) < 1e-4, "float64 scheme identity"
    # float32: production dtype -> quantify the precision gap; spikes stay bit-exact
    ev32, es32 = _engine_izhikevich(i_pA, n_steps, np.float32)
    assert es32 == bs, "spike step indices bit-exact at production float32"
    assert np.max(np.abs(ev32 - bv)) < 0.2, "float32 production trajectory within tolerance"


@pytest.mark.parametrize("i_pA,n_steps,expect_spikes", [(400.0, 400, False), (700.0, 400, True), (1000.0, 400, True)])
def test_adex_point_neuron(i_pA, n_steps, expect_spikes):
    ev64, es64 = _engine_adex(i_pA, n_steps, np.float64)
    bv, bs = _brian_adex(i_pA, n_steps)
    if expect_spikes:
        assert len(bs) >= 3, "sanity: AdEx must actually fire"
    else:
        assert len(bs) == 0 and len(es64) == 0, "sanity: subthreshold case must not fire"
    assert es64 == bs, "spike step indices must be bit-exact (float64)"
    assert np.max(np.abs(ev64 - bv)) < 1e-3, "float64 scheme identity"
    ev32, es32 = _engine_adex(i_pA, n_steps, np.float32)
    assert es32 == bs, "spike step indices bit-exact at production float32"
    assert np.max(np.abs(ev32 - bv)) < 0.1, "float32 production trajectory within tolerance"


@pytest.mark.parametrize("tau_ms,E_mV,v_clamp_mV", [(5.0, 0.0, -60.0), (10.0, -75.0, -55.0)])
def test_coba_synapse_current(tau_ms, E_mV, v_clamp_mV):
    ge, isyn = _engine_coba(5.0, tau_ms, E_mV, v_clamp_mV, 120, np.float64)
    bge, bisyn = _brian_coba(5.0, tau_ms, E_mV, v_clamp_mV, 120)
    assert ge[0] > 0 and abs(isyn[0]) > 0, "sanity: conductance/current must be nonzero"
    assert np.max(np.abs(ge - bge)) < 1e-6, "conductance kinetics match"
    assert np.max(np.abs(isyn - bisyn)) < 1e-6, "driving-force current matches"


def test_pair_stdp_weight_curve():
    worst = 0.0
    checked = 0
    for delta_t in (-40, -20, -10, -5, 5, 10, 20, 40):
        for w0 in (0.2, 1.0, 1.8):
            e = _engine_stdp(delta_t, w0, np.float32)
            b = _brian_stdp(delta_t, w0)
            worst = max(worst, abs(e - b))
            checked += 1
            # sanity: causal must potentiate, anti-causal must depress (non-vacuous)
            if delta_t > 0:
                assert e > w0, "causal pairing must potentiate"
            else:
                assert e < w0, "anti-causal pairing must depress"
    assert checked == 24
    assert worst < 1e-5, f"STDP weight-change curve must match Brian2 (worst |dw| diff={worst:.2e})"
