"""Fused kernels for neural simulation dynamics.

All functions are decorated with @fuse() which is `cp.fuse()` on the CuPy
backend (element-wise GPU execution) and a no-op on the NumPy backend
(pure-Python fallback for CPU-only runs). They are pure math: arrays
in, arrays out, no simulator-specific dependencies.

`cp` here is the active backend module (CuPy or NumPy) — call sites
like `cp.where(...)`, `cp.exp(...)`, `cp.clip(...)` work identically
on both because the NumPy API matches.

See sim/backend.py + docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md.
"""

import math

# Route through the backend abstraction so this module works on both
# CuPy and NumPy. `cp` is the active backend module; @fuse() is the
# backend-aware kernel-fusion decorator.
from sim.backend import get_backend, fuse
cp, _backend_name = get_backend()
_BACKEND_ARRAY_TYPE = cp.ndarray
_FLOAT32_DTYPE = cp.dtype(cp.float32)
_REAL_SCALAR_TYPES = (int, float, cp.integer, cp.floating)
_STRICT_IZH_ARRAY_NAMES = (
    "v", "u", "C_param", "k_param", "vr_param", "vt_param", "a_param",
    "b_param", "total_synaptic_current",
)
_STRICT_IZH_IEEE_PREAMBLE = r"""
__device__ __forceinline__ float sim_fadd_rn(float lhs, float rhs) {
    float result;
    asm volatile("add.rn.f32 %0, %1, %2;"
                 : "=f"(result) : "f"(lhs), "f"(rhs));
    return result;
}

__device__ __forceinline__ float sim_fsub_rn(float lhs, float rhs) {
    float result;
    asm volatile("sub.rn.f32 %0, %1, %2;"
                 : "=f"(result) : "f"(lhs), "f"(rhs));
    return result;
}

__device__ __forceinline__ float sim_fmul_rn(float lhs, float rhs) {
    float result;
    asm volatile("mul.rn.f32 %0, %1, %2;"
                 : "=f"(result) : "f"(lhs), "f"(rhs));
    return result;
}

__device__ __forceinline__ float sim_fdiv_rn(float lhs, float rhs) {
    float result;
    asm volatile("div.rn.f32 %0, %1, %2;"
                 : "=f"(result) : "f"(lhs), "f"(rhs));
    return result;
}
"""

# --- CuPy Fused Kernels ---
@fuse()
def fused_izhikevich_legacy_dynamics_update(v, u, a, b, total_I, dt):
    """Fused kernel for legacy Izhikevich model dynamics."""
    dv = (0.04 * v**2 + 5 * v + 140 - u + total_I)
    du = a * (b * v - u)
    v_new = v + dv * dt
    u_new = u + du * dt
    return v_new, u_new

@fuse()
def fused_izhikevich2007_dynamics_update(v, u, C_param, k_param, vr_param, vt_param, a_param, b_param, total_synaptic_current, dt):
    """Fused kernel for Izhikevich 2007 model dynamics."""
    # Ensure C_param is not zero to prevent division by zero errors.
    C_param_safe = cp.where(C_param == 0.0, 1.0, C_param) # Use 1.0 as a safe non-zero default if C is 0

    # Differential equation for membrane potential v
    dv_dt = (k_param * (v - vr_param) * (v - vt_param) - u + total_synaptic_current) / C_param_safe
    # Differential equation for recovery variable u
    du_dt = a_param * (b_param * (v - vr_param) - u)

    # Euler integration to update v and u
    v_new = v + dv_dt * dt
    u_new = u + du_dt * dt
    return v_new, u_new


if _backend_name == "cupy":
    _strict_izhikevich2007_gpu_kernel = cp.ElementwiseKernel(
        (
            "float32 v, float32 u, float32 C, float32 k, float32 vr, "
            "float32 vt, float32 a, float32 b, float32 total_current, "
            "float32 dt"
        ),
        "float32 v_new, float32 u_new",
        r"""
        const float C_safe = C == 0.0f ? 1.0f : C;
        const float v_minus_vr = sim_fsub_rn(v, vr);
        const float v_minus_vt = sim_fsub_rn(v, vt);

        const float k_times_vr = sim_fmul_rn(k, v_minus_vr);
        const float quadratic = sim_fmul_rn(k_times_vr, v_minus_vt);
        const float dv_minus_u = sim_fsub_rn(quadratic, u);
        const float dv_numerator = sim_fadd_rn(dv_minus_u, total_current);
        const float dv = sim_fdiv_rn(dv_numerator, C_safe);
        const float dv_dt = sim_fmul_rn(dv, dt);
        v_new = sim_fadd_rn(v, dv_dt);

        const float b_times_vr = sim_fmul_rn(b, v_minus_vr);
        const float du_inner = sim_fsub_rn(b_times_vr, u);
        const float du = sim_fmul_rn(a, du_inner);
        const float du_dt = sim_fmul_rn(du, dt);
        u_new = sim_fadd_rn(u, du_dt);
        """,
        "strict_izhikevich2007_float32_update",
        preamble=_STRICT_IZH_IEEE_PREAMBLE,
    )
else:
    _strict_izhikevich2007_gpu_kernel = None


def _require_float32_c_arrays(arrays):
    expected_shape = None
    for name, value in zip(_STRICT_IZH_ARRAY_NAMES, arrays):
        if not isinstance(value, _BACKEND_ARRAY_TYPE):
            raise TypeError(f"{name} must be an {_backend_name} array")
        if value.dtype != _FLOAT32_DTYPE:
            raise TypeError(f"{name} must have dtype float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        if expected_shape is None:
            expected_shape = value.shape
        elif value.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {value.shape}"
            )


def strict_izhikevich2007_dynamics_update(
    v, u, C_param, k_param, vr_param, vt_param, a_param, b_param,
    total_synaptic_current, dt,
):
    """Izhikevich-2007 Euler update with explicit float32 rounding points.

    CuPy executes one device-resident elementwise kernel with explicit IEEE
    round-to-nearest operations that preserve subnormal values. NumPy
    materializes the same primitive operations in the same order. The caller
    must supply C-contiguous float32 arrays so this opt-in correction cannot
    silently promote or copy state.
    """
    arrays = (
        v, u, C_param, k_param, vr_param, vt_param, a_param, b_param,
        total_synaptic_current,
    )
    _require_float32_c_arrays(arrays)
    if not isinstance(dt, _REAL_SCALAR_TYPES):
        raise TypeError("dt must be a real scalar")
    dt_f32 = cp.float32(dt)
    if not math.isfinite(float(dt_f32)):
        raise ValueError("dt must be finite")

    if _backend_name == "cupy":
        return _strict_izhikevich2007_gpu_kernel(
            v, u, C_param, k_param, vr_param, vt_param, a_param, b_param,
            total_synaptic_current, dt_f32,
        )

    C_safe = cp.where(C_param == cp.float32(0.0), cp.float32(1.0), C_param)
    v_minus_vr = cp.subtract(v, vr_param)
    v_minus_vt = cp.subtract(v, vt_param)
    k_times_vr = cp.multiply(k_param, v_minus_vr)
    quadratic = cp.multiply(k_times_vr, v_minus_vt)
    dv_minus_u = cp.subtract(quadratic, u)
    dv_numerator = cp.add(dv_minus_u, total_synaptic_current)
    dv = cp.divide(dv_numerator, C_safe)
    dv_dt = cp.multiply(dv, dt_f32)
    v_new = cp.add(v, dv_dt)

    b_times_vr = cp.multiply(b_param, v_minus_vr)
    du_inner = cp.subtract(b_times_vr, u)
    du = cp.multiply(a_param, du_inner)
    du_dt = cp.multiply(du, dt_f32)
    u_new = cp.add(u, du_dt)
    return v_new, u_new


def izhikevich2007_dynamics_update(
    v, u, C_param, k_param, vr_param, vt_param, a_param, b_param,
    total_synaptic_current, dt, *, backend_neutral_arithmetic=False,
):
    """Dispatch the opt-in correction without changing the legacy path."""
    if type(backend_neutral_arithmetic) is not bool:
        raise TypeError("backend_neutral_arithmetic must be a boolean")
    if backend_neutral_arithmetic:
        return strict_izhikevich2007_dynamics_update(
            v, u, C_param, k_param, vr_param, vt_param, a_param, b_param,
            total_synaptic_current, dt,
        )
    return fused_izhikevich2007_dynamics_update(
        v, u, C_param, k_param, vr_param, vt_param, a_param, b_param,
        total_synaptic_current, dt,
    )

@fuse()
def fused_hodgkin_huxley_dynamics_update(V, m, h, n, I_syn, dt, C_m, g_Na_max, g_K_max, g_L, E_Na, E_K, E_L, phi_m, phi_h, phi_n):
    """Fused kernel for Hodgkin-Huxley model dynamics with per-gate Q10.

    Per-gate phi (φ_m, φ_h, φ_n) values precomputed by caller from
    temperature and per-gate Q10 values. This replaces the previous
    uniform-Q10 implementation, which over-compressed dynamics at body
    temperature (37°C) and prevented APs from firing — see
    `research/findings/2026-04-25-hh-temperature-bug.md`.

    Biological default Q10 values at body temperature:
      Q10_m ≈ 3.0   (fast activation)
      Q10_h ≈ 1.5   (slower inactivation; preserves spike width)
      Q10_n ≈ 1.5   (slower recovery; preserves AP duration)
    """
    # Rate functions (alpha, beta) for gating variables m, h, n
    # Original HH equations, adjusted for V in mV.
    # Handling for V = -40 (for alpha_m) and V = -55 (for alpha_n) to avoid division by zero in expm1.
    # expm1(x) = exp(x) - 1. For small x, expm1(x) approx x.
    # If V = -40, then -(V+40)/10 = 0. The limit of -0.1*x / (exp(-x/10)-1) as x->-40 is 1.0.
    # (Using L'Hopital's rule: d/dx (-0.1(x+40)) / d/dx (exp(-(x+40)/10)-1) = -0.1 / (-0.1 * exp(-(x+40)/10)) = exp((x+40)/10) -> exp(0) = 1)

    v_plus_40 = V + 40.0 # For m-gate alpha expression
    alpha_m_orig = cp.where(v_plus_40 == 0, 1.0 * 0.1 * 10.0 , -0.1 * v_plus_40 / cp.expm1(-v_plus_40 / 10.0)) # Corrected limit handling
    beta_m_orig  = 4.0 * cp.exp(-(V + 65.0) / 18.0)

    alpha_h_orig = 0.07 * cp.exp(-(V + 65.0) / 20.0)
    beta_h_orig  = 1.0 / (cp.exp(-(V + 35.0) / 10.0) + 1.0)

    v_plus_55 = V + 55.0 # For n-gate alpha expression
    alpha_n_orig = cp.where(v_plus_55 == 0, 0.1 * 0.01 * 10.0, -0.01 * v_plus_55 / cp.expm1(-v_plus_55 / 10.0)) # Corrected limit handling
    beta_n_orig  = 0.125 * cp.exp(-(V + 65.0) / 80.0)

    # Per-gate temperature correction
    alpha_m = alpha_m_orig * phi_m; beta_m  = beta_m_orig  * phi_m
    alpha_h = alpha_h_orig * phi_h; beta_h  = beta_h_orig  * phi_h
    alpha_n = alpha_n_orig * phi_n; beta_n  = beta_n_orig  * phi_n

    # Update gating variables using analytical solution for first-order kinetics (assuming V is constant during dt)
    # m_new = m_inf - (m_inf - m_old) * exp(-dt / tau_m)
    # where m_inf = alpha_m / (alpha_m + beta_m) and tau_m = 1 / (alpha_m + beta_m)

    # Epsilon-based safe division eliminates branching overhead from cp.where()
    # For biophysically valid voltages, alpha+beta > 0 always; epsilon is only a numerical guard.
    # This avoids 6 cp.where() calls and 3 cp.isinf() calls per step (3-5% HH speedup).
    _EPS_GATE = 1e-12  # Small enough to not affect dynamics, large enough for float32 safety
    sum_alpha_beta_m = alpha_m + beta_m + _EPS_GATE
    m_inf = alpha_m / sum_alpha_beta_m
    m_new = m_inf + (m - m_inf) * cp.exp(-dt * sum_alpha_beta_m)

    sum_alpha_beta_h = alpha_h + beta_h + _EPS_GATE
    h_inf = alpha_h / sum_alpha_beta_h
    h_new = h_inf + (h - h_inf) * cp.exp(-dt * sum_alpha_beta_h)

    sum_alpha_beta_n = alpha_n + beta_n + _EPS_GATE
    n_inf = alpha_n / sum_alpha_beta_n
    n_new = n_inf + (n - n_inf) * cp.exp(-dt * sum_alpha_beta_n)

    # Clip gating variables to be between 0 and 1
    m_new = cp.clip(m_new, 0.0, 1.0); h_new = cp.clip(h_new, 0.0, 1.0); n_new = cp.clip(n_new, 0.0, 1.0)

    # Ionic currents
    I_Na = g_Na_max * (m_new**3) * h_new * (V - E_Na) # Sodium current
    I_K  = g_K_max * (n_new**4) * (V - E_K)   # Potassium current
    I_L  = g_L * (V - E_L)                    # Leak current
    I_ion = I_Na + I_K + I_L                  # Total ionic current

    # Membrane potential update
    dV_dt = (I_syn - I_ion) / C_m # dV/dt = (I_external - I_ionic) / C_m
    V_new = V + dV_dt * dt        # Euler integration
    return V_new, m_new, h_new, n_new

@fuse()
def fused_hh_m_current_update(V, p_old, dt, g_M_max, E_K, tau_m_ms, phi):
    """Optional slow K+ M-current for extended HH models.

    Uses a simple sigmoidal steady-state activation with a first-order time course.
    g_M_max = 0.0 disables the current without branching.
    phi: Q10 temperature correction factor (same as main HH kinetics).
    """
    # Steady-state activation (approximate; centered around -35 mV)
    p_inf = 1.0 / (1.0 + cp.exp(-(V + 35.0) / 10.0))
    # Time constant (ms) with Q10 temperature correction — faster at higher temperatures
    # Literature: M-current tau ranges 30-200ms depending on cell type and temperature
    tau_safe = cp.maximum(tau_m_ms / phi, 1e-3)
    # First-order update assuming V is approximately constant over dt
    p_new = p_inf + (p_old - p_inf) * cp.exp(-dt / tau_safe)
    # M-current (K+): uses potassium reversal potential E_K
    I_M = g_M_max * p_new * (V - E_K)
    return p_new, I_M

@fuse()
def fused_hh_CaT_current_update(V, m_old, h_old, dt, g_CaT_max, E_CaT, phi):
    """Low-threshold T-type Ca2+ current for extended HH models.

    Uses simple sigmoidal steady-state activation/inactivation with Q10-corrected time constants.
    phi: Q10 temperature correction factor.
    """
    # Steady-state activation/inactivation (approximate, thalamic-like)
    m_inf = 1.0 / (1.0 + cp.exp(-(V + 50.0) / 7.4))
    h_inf = 1.0 / (1.0 + cp.exp((V + 80.0) / 5.0))
    # Temperature-corrected time constants (Q10 ~3-4 for Ca2+ channels)
    tau_m = 5.0 / phi   # ms, fast activation (scaled by temperature)
    tau_h = 20.0 / phi  # ms, slower inactivation (scaled by temperature)
    m_new = m_inf + (m_old - m_inf) * cp.exp(-dt / tau_m)
    h_new = h_inf + (h_old - h_inf) * cp.exp(-dt / tau_h)
    I_CaT = g_CaT_max * (m_new ** 2) * h_new * (V - E_CaT)
    return m_new, h_new, I_CaT

@fuse()
def fused_hh_h_current_update(V, q_old, dt, g_h_max, E_h, phi):
    """Hyperpolarization-activated mixed cation current (I_h) for extended HH models.

    phi: Q10 temperature correction factor. I_h has Q10 ~3-4 (Magee 1998).
    """
    # Steady-state activation: more active at hyperpolarized voltages
    q_inf = 1.0 / (1.0 + cp.exp((V + 75.0) / 5.5))
    # Temperature-corrected time constant
    tau_q = 100.0 / phi  # ms, slow activation (faster at mammalian temperatures)
    q_new = q_inf + (q_old - q_inf) * cp.exp(-dt / tau_q)
    I_h = g_h_max * q_new * (V - E_h)
    return q_new, I_h

@fuse()
def fused_hh_NaP_current_update(V, p_old, dt, g_NaP_max, E_Na, phi):
    """Persistent Na+ current for extended HH models.

    phi: Q10 temperature correction factor. NaP kinetics scale similarly to transient Na+ (Q10 ~3).
    """
    p_inf = 1.0 / (1.0 + cp.exp(-(V + 55.0) / 5.0))
    # Temperature-corrected time constant
    tau_p = 5.0 / phi  # ms, relatively fast activation (faster at mammalian temperatures)
    p_new = p_inf + (p_old - p_inf) * cp.exp(-dt / tau_p)
    I_NaP = g_NaP_max * p_new * (V - E_Na)
    return p_new, I_NaP


@fuse()
def fused_snr_conductance_update(
    V,
    nap_activation_old,
    nap_inactivation_old,
    ca_activation_old,
    ca_inactivation_old,
    calcium_old,
    sk_activation_old,
    h_activation_old,
    dt,
    g_nalcn_max,
    g_nap_max,
    g_ca_max,
    g_sk_max,
    g_h_max,
    E_nalcn,
    E_Na,
    E_Ca,
    E_K,
    E_h,
    calcium_baseline,
    calcium_influx_scale,
    calcium_decay_tau_ms,
    sk_half_activation,
    sk_hill_coefficient,
    sk_tau_ms,
):
    """Advance the explicit SNr pacemaker bundle by one timestep.

    Conductances are in units compatible with voltage so each current is
    ``g * gates * (V - E)``. Negative currents are inward. Calcium uses an
    arbitrary nonnegative concentration unit whose conversion from inward
    calcium current is set by ``calcium_influx_scale``.

    Returns the seven updated dynamic states followed by total ionic current.
    The voltage-gate kinetics are the evidence-center Stage-A values from the
    V14 SNr fallback review; calcium coupling and SK concentration kinetics are
    explicit because the available evidence does not identify their units.
    """
    # Persistent sodium: fast activation and slow inactivation.
    nap_activation_inf = 1.0 / (1.0 + cp.exp(-(V + 50.0) / 4.5))
    nap_inactivation_inf = 1.0 / (1.0 + cp.exp((V + 57.0) / 6.0))
    nap_activation = nap_activation_inf + (
        nap_activation_old - nap_activation_inf
    ) * cp.exp(-dt / 0.1)
    nap_inactivation = nap_inactivation_inf + (
        nap_inactivation_old - nap_inactivation_inf
    ) * cp.exp(-dt / 20.0)
    nap_activation = cp.clip(nap_activation, 0.0, 1.0)
    nap_inactivation = cp.clip(nap_inactivation, 0.0, 1.0)

    # Cav2.2-like high-threshold calcium current.
    ca_activation_inf = 1.0 / (1.0 + cp.exp(-(V + 27.5) / 3.0))
    ca_inactivation_inf = 1.0 / (1.0 + cp.exp((V + 52.5) / 5.2))
    ca_activation = ca_activation_inf + (
        ca_activation_old - ca_activation_inf
    ) * cp.exp(-dt / 0.5)
    ca_inactivation = ca_inactivation_inf + (
        ca_inactivation_old - ca_inactivation_inf
    ) * cp.exp(-dt / 18.0)
    ca_activation = cp.clip(ca_activation, 0.0, 1.0)
    ca_inactivation = cp.clip(ca_inactivation, 0.0, 1.0)

    # Ih is optional through g_h_max=0 and primarily supports recovery from
    # hyperpolarization rather than baseline SNr pacemaking.
    h_activation_inf = 1.0 / (1.0 + cp.exp((V + 75.0) / 5.5))
    h_activation = h_activation_inf + (
        h_activation_old - h_activation_inf
    ) * cp.exp(-dt / 100.0)
    h_activation = cp.clip(h_activation, 0.0, 1.0)

    I_nalcn = g_nalcn_max * (V - E_nalcn)
    I_nap = (
        g_nap_max * nap_activation * nap_inactivation * (V - E_Na)
    )
    I_ca = (
        g_ca_max * ca_activation * ca_activation * ca_inactivation
        * (V - E_Ca)
    )

    # Exact first-order calcium update for constant influx over this step.
    calcium_tau = cp.maximum(calcium_decay_tau_ms, 1e-6)
    calcium_floor = cp.maximum(calcium_baseline, 0.0)
    calcium_influx = cp.maximum(calcium_influx_scale, 0.0) * cp.maximum(-I_ca, 0.0)
    calcium_target = calcium_floor + calcium_tau * calcium_influx
    calcium = calcium_target + (calcium_old - calcium_target) * cp.exp(
        -dt / calcium_tau
    )
    calcium = cp.maximum(calcium, 0.0)

    sk_half = cp.maximum(sk_half_activation, 1e-12)
    sk_hill = cp.maximum(sk_hill_coefficient, 1e-6)
    calcium_hill = cp.power(calcium, sk_hill)
    sk_half_hill = cp.power(sk_half, sk_hill)
    sk_activation_inf = calcium_hill / (
        calcium_hill + sk_half_hill
    )
    sk_tau = cp.maximum(sk_tau_ms, 1e-6)
    sk_activation = sk_activation_inf + (
        sk_activation_old - sk_activation_inf
    ) * cp.exp(-dt / sk_tau)
    sk_activation = cp.clip(sk_activation, 0.0, 1.0)

    I_sk = g_sk_max * sk_activation * (V - E_K)
    I_h = g_h_max * h_activation * (V - E_h)
    total_ionic_current = I_nalcn + I_nap + I_ca + I_sk + I_h
    return (
        nap_activation,
        nap_inactivation,
        ca_activation,
        ca_inactivation,
        calcium,
        sk_activation,
        h_activation,
        total_ionic_current,
    )


@fuse()
def fused_adex_dynamics_update(V, w, I_syn, dt, C, g_L, E_L, V_T, Delta_T, a, tau_w):
    """Fused kernel for Adaptive Exponential Integrate-and-Fire (AdEx) dynamics.

    All parameters can be either scalars or arrays broadcastable to V.
    Units are assumed to be consistent with the calling code (pF, nS, mV, ms, pA).
    """
    C_safe = cp.where(C == 0.0, 1.0, C)
    tau_w_safe = cp.maximum(tau_w, 1e-9)
    Delta_T_safe = cp.maximum(Delta_T, 1e-9)  # Prevent division by zero

    # Clamp exponential argument to prevent overflow. For float32:
    # exp(-20) ≈ 2e-9 (underflows gracefully), exp(5) ≈ 148 (safe with g_L*Delta_T scaling)
    # Wider range improves subthreshold accuracy near threshold without numerical risk.
    exp_arg = cp.clip((V - V_T) / Delta_T_safe, -20.0, 5.0)

    # Membrane equation: C dV/dt = -g_L (V - E_L) + g_L * Delta_T * exp((V - V_T)/Delta_T) - w + I_syn
    dV_dt = (-g_L * (V - E_L) + g_L * Delta_T * cp.exp(exp_arg) - w + I_syn) / C_safe
    # Adaptation variable: tau_w dw/dt = a (V - E_L) - w
    dw_dt = (a * (V - E_L) - w) / tau_w_safe
    V_new = V + dV_dt * dt
    w_new = w + dw_dt * dt
    return V_new, w_new

@fuse()
def fused_conductance_decay_and_current(g_e, g_i, decay_e, decay_i, v, E_e, E_i):
    """Fused kernel for synaptic conductance decay and calculating synaptic current."""
    # Decay conductances
    g_e_new = g_e * decay_e # Excitatory conductance decay
    g_i_new = g_i * decay_i # Inhibitory conductance decay
    # Calculate total synaptic current based on new conductances
    I_syn = g_e_new * (E_e - v) + g_i_new * (E_i - v) # I_syn = g_e*(E_e - V) + g_i*(E_i - V)
    return g_e_new, g_i_new, I_syn

@fuse()
def fused_readonly_izh_step(
    g_e, g_i, g_e_increase, g_i_increase, decay_e, decay_i, E_e, E_i,
    v, u, external_current, ou_current,
    C_param, k_param, vr_param, vt_param, a_param, b_param,
    vpeak, c_reset, d_increment, refractory, refractory_reset, dt,
):
    """Fused READ-ONLY inference megastep for the Izhikevich-2007 model (opt-in
    enable_step_megakernel). Collapses the per-neuron ELEMENT-WISE chain of
    `_run_one_simulation_step` -- conductance decay + synaptic current +
    (pre-computed) E/I matvec increment + total-input assembly + Izhikevich-2007
    dynamics + threshold-select + fast_spike_reset -- into ONE kernel launch. The
    math + op ORDER are byte-faithful to the separate ops so the spike raster
    matches (a neuron on threshold can flip under FMA/summation reordering, so
    the ordering is load-bearing): I_syn is read from the DECAYED (pre-increment)
    conductances exactly like fused_conductance_decay_and_current, then the matvec
    increment (already scaled by propagation_strength / inhibitory_propagation_
    strength) is applied to give the NEXT step's conductance; the total input is
    assembled `(I_syn + external) + ou` left-to-right like the step; the dynamics
    replicate fused_izhikevich2007_dynamics_update; the reset replicates the
    fast_spike_reset cp.where path (v/u reset + refractory update).

    The cuSPARSE E/I-split matvec (g_e_increase/g_i_increase) and the OU-noise
    cp.random.randn draw stay OUTSIDE this kernel (keeps the RNG stream + sparse
    summation bit-faithful). `E_i` is a scalar or a per-neuron array; `ou_current`
    is the updated OU current (or a zeros array when OU is off); `g_i_increase` is
    a zeros array when inhibitory neurons are disabled; `refractory_reset` is the
    fired-neuron refractory value (max(0, refractory_period_steps - 1), int32).

    Returns (g_e_new, g_i_new, v_new, u_new, fired, refractory_new).
    """
    # 1-2. Conductance decay + synaptic current from the DECAYED (pre-increment)
    # conductances -- identical to fused_conductance_decay_and_current.
    g_e_dec = g_e * decay_e
    g_i_dec = g_i * decay_i
    I_syn = g_e_dec * (E_e - v) + g_i_dec * (E_i - v)
    # 3. Apply the (already propagation-scaled) matvec increment for the NEXT step
    # (I_syn above used the pre-increment conductances -- matches the step order).
    g_e_new = g_e_dec + g_e_increase
    g_i_new = g_i_dec + g_i_increase
    # 5. Total input assembly: (I_syn + external) + ou (left-to-right, matching the step).
    total_I = I_syn + external_current + ou_current
    # 6. Izhikevich-2007 dynamics -- identical to fused_izhikevich2007_dynamics_update.
    C_safe = cp.where(C_param == 0.0, 1.0, C_param)
    dv_dt = (k_param * (v - vr_param) * (v - vt_param) - u + total_I) / C_safe
    du_dt = a_param * (b_param * (v - vr_param) - u)
    v_dyn = v + dv_dt * dt
    u_dyn = u + du_dt * dt
    # 7. Threshold-select (vpeak) gated by the refractory timer.
    fired = (v_dyn >= vpeak) & (refractory <= 0)
    # 8. fast_spike_reset (the cp.where masked-update path): v/u reset + refractory update.
    v_new = cp.where(fired, c_reset, v_dyn)
    u_new = cp.where(fired, u_dyn + d_increment, u_dyn)
    refractory_new = cp.where(fired, refractory_reset, cp.maximum(refractory - 1, 0))
    return g_e_new, g_i_new, v_new, u_new, fired, refractory_new

@fuse()
def fused_gabab_decay_and_current(g_gabab, decay_gabab, v, E_gabab):
    """Slow GABA_B -> GIRK K+ inhibitory conductance (E_gabab ~ -90 mV, the
    potassium reversal). Metabotropic/slow: decay_gabab = exp(-dt/tau) with
    tau ~150 ms, far slower than GABA_A (~10 ms). Mirrors the AMPA/NMDA pattern
    inverted: a hyperpolarizing K+ current independent of the chloride gradient,
    so it strongly inhibits KCC2-lacking DA cells where GABA_A is weak/shunting."""
    g_gabab_new = g_gabab * decay_gabab
    I_gabab = g_gabab_new * (E_gabab - v)
    return g_gabab_new, I_gabab

@fuse()
def fused_nmda_update_and_current(g_nmda, g_nmda_rise, decay_nmda, decay_nmda_rise, v, E_nmda, mg_conc):
    """Fused kernel for NMDA conductance with voltage-dependent Mg2+ block.

    Implements the Jahr & Stevens (1990) Mg2+ block:
        B(V) = 1 / (1 + [Mg2+]_o/3.57 * exp(-0.062 * V))

    Uses dual-exponential kinetics: g_NMDA = g_slow - g_rise for realistic
    rise/decay dynamics. The Mg2+ block factor B(V) produces the characteristic
    voltage-dependent nonlinearity that gates Ca2+ influx and is critical
    for coincidence detection in STDP and associative learning.
    """
    # Dual-exponential decay
    g_nmda_new = g_nmda * decay_nmda
    g_nmda_rise_new = g_nmda_rise * decay_nmda_rise
    # Effective NMDA conductance (difference of exponentials)
    g_eff = g_nmda_new - g_nmda_rise_new
    g_eff = cp.maximum(g_eff, 0.0)
    # Voltage-dependent Mg2+ block (Jahr & Stevens 1990)
    mg_block = 1.0 / (1.0 + (mg_conc / 3.57) * cp.exp(-0.062 * v))
    # NMDA current with Mg2+ gating
    I_nmda = g_eff * mg_block * (E_nmda - v)
    return g_nmda_new, g_nmda_rise_new, I_nmda

@fuse()
def fused_coincidence_plateau(g, g_rise, decay, decay_rise, v, E_e, mg_conc,
                              c_count, k_thresh, gain, plateau_strength,
                              self_regen=0.0, v_hold=-35.0, v_hold_k=0.2):
    """Dendritic-COINCIDENCE (NMDA-spike) plateau. A per-neuron SUPRALINEAR, all-or-none switch on the
    COUNT of SYNCHRONOUS clustered inputs c_count: >= k_thresh coincident inputs this step -> a
    regenerative plateau conductance increment; fewer -> ~0. The plateau then decays with the SAME
    dual-exponential idiom as fused_nmda_update_and_current (slow ~80ms tail = the 50-100ms NMDA spike)
    and produces an Mg2+-self-limiting (Jahr-Stevens) current, so it is genuinely NMDA-like (voltage-
    gated, self-limiting). This is the FIRST non-linear-summation element in the engine: it makes a
    handful of SIMULTANEOUS clustered inputs trigger the soma where the same inputs spread in time
    (c_count < k_thresh each step) cannot -- the inverse of the point-neuron rate-coding wall.
    (Poirazi-Brannon-Mel 2003 two-layer subunit; Major-Larkum-Schiller 2013 NMDA spike; Branco-Clark-
    Hausser 2010 temporal sensitivity -> the jitter anti-cheat.) Called ONLY from the guarded
    coincidence block in bridge._run_one_simulation_step, so its presence is byte-inert when
    cfg.enable_coincidence_detection is False (the block is unreached and this kernel is never invoked)."""
    # All-or-none sigmoid switch on the coincidence count (the supralinear subunit) = the INPUT TRIGGER.
    g_inc = plateau_strength / (1.0 + cp.exp(-gain * (c_count - k_thresh)))
    # v-GATED SELF-REGENERATING SUSTAIN (default self_regen=0 -> zero -> byte-identical). Once the compartment is
    # depolarized past v_hold, this REPLENISHES the SLOW reservoir g each step (weighted by the Mg-unblock, computed
    # below) so the plateau HOLDS after the input volley ends -- the intrinsic bistable up state (Antic 2010). Added to
    # the SLOW g only (NOT g_rise), so it does not cancel in g_eff = g - g_rise; needs a KIR down-state stabilizer
    # (apical ODE) for a robust bistable band (Sanders 2013). mg_block is recomputed here to gate the self-drive.
    _mg = 1.0 / (1.0 + (mg_conc / 3.57) * cp.exp(-0.062 * v))
    sustain = self_regen * _mg / (1.0 + cp.exp(-v_hold_k * (v - v_hold)))
    # Dual-exponential plateau (g_slow - g_rise), mirroring the NMDA dual-exp kinetics. The trigger drives both (rise);
    # the sustain replenishes only the slow reservoir (hold).
    g_new = g * decay + g_inc + sustain
    g_rise_new = g_rise * decay_rise + g_inc
    g_eff = g_new - g_rise_new
    g_eff = cp.maximum(g_eff, 0.0)
    # Voltage-dependent Mg2+ block (Jahr & Stevens 1990) -- regenerative as V depolarizes.
    mg_block = 1.0 / (1.0 + (mg_conc / 3.57) * cp.exp(-0.062 * v))
    # Plateau current (conductance form: driving force toward E_e = 0 mV).
    I = g_eff * mg_block * (E_e - v)
    return g_new, g_rise_new, I

@fuse()
def fused_graded_dendritic_plateau(g, g_rise, decay, decay_rise, v, E_e, mg_conc,
                                   c_weighted, center, slope, plateau_strength):
    """GRADED dendritic-plateau READ-OUT (Stage 1) -- the SMOOTH, non-saturating sibling of
    fused_coincidence_plateau. The dendrite's ONE genuine unlock (de-risk A GO): a GRADED ANALOG
    read-out of a distributed code (Mikulasch-Priesemann) the point-neuron soma provably cannot be
    (sub-rheobase 0, or all-or-none saturated -- never the graded middle).

    Identical kinetics to fused_coincidence_plateau (dual-exponential g_slow - g_rise; Jahr-Stevens
    voltage-dependent Mg2+ block -> a regenerative, self-limiting NMDA-spike-like current toward
    E_e = 0 mV). The ONLY difference is the transfer function on the per-neuron drive:
      * fused_coincidence_plateau: a STEEP all-or-none switch  1/(1+exp(-gain*(c-k_thresh)))  with
        gain ~2 (~0.88/0.12 at K+/-1) -- a binary subunit (the value snaps 0/1; the graded middle is LOST).
      * here (graded): a GENTLE, CENTERED logistic  V = 1/(1+exp(-slope*(c_weighted-center)))  with a
        SMALL slope (~0.33 = 1/dend_slope) -- so V varies smoothly and non-saturatingly across the
        active range of c_weighted (the WEIGHTED coincident drive Sum_j w_eff_j*x_j, the learned
        place->value synaptic value). A high-value (NEAR) ensemble gives a high-but-not-saturated V,
        a mid-value ensemble an intermediate V, a low-value (FAR) ensemble a low V -> the continuum
        V(near) > V(mid) > V(far) the all-or-none switch cannot express.

    This is the on-substrate realization of the Stage-0 numpy value read-out V = sigmoid((v_basal-
    theta)/slope) (`_dendrite_deriskA_graded_plateau_readout.py`), produced by the spiking-bridge
    dendrite (c_weighted is the on-bridge restricted matvec of the learned routed synapses against the
    prior firing). Called ONLY from the guarded graded-plateau block in
    bridge._run_one_simulation_step, so its presence is byte-inert when cfg.enable_graded_dendritic_
    plateau is False (the block is unreached and this kernel is never invoked).

    NO-DRIVE FLOOR SUBTRACTION (biologically: no synaptic drive => no NMDA plateau): a bare logistic has
    a non-zero left-tail value at c_weighted=0 (sigmoid(-slope*center) > 0), so an UNDRIVEN neuron (a
    target with NO routed coincident input, c_weighted=0) would still accumulate a resting plateau
    conductance over the slow ~80ms tau and depolarize -- physiologically wrong (an NMDA spike requires
    real clustered drive) and, on-bridge, it would flood non-target neurons (e.g. the downstream SNc) with
    a resting current. We subtract that c_weighted=0 floor so V(0)=0 exactly: an undriven neuron gets ZERO
    plateau, the graded MIDDLE is preserved (the relative grading of driven c_weighted is unchanged up to
    the floor shift), and the on-bridge plateau is injected ONLY where there is actual coincident value
    drive. The floor is recomputed in-kernel from (slope, center) -- no extra parameter."""
    # GRADED, CENTERED, non-saturating logistic on the WEIGHTED coincident drive (the analog read-out),
    # with the no-drive (c_weighted=0) floor subtracted so V(0)=0 (no input -> no NMDA plateau).
    floor = 1.0 / (1.0 + cp.exp(slope * center))   # = sigmoid(-slope*center) = V at c_weighted=0
    V = cp.maximum(1.0 / (1.0 + cp.exp(-slope * (c_weighted - center))) - floor, 0.0)
    g_inc = plateau_strength * V
    # Dual-exponential plateau (g_slow - g_rise), mirroring fused_coincidence_plateau / NMDA kinetics.
    g_new = g * decay + g_inc
    g_rise_new = g_rise * decay_rise + g_inc
    g_eff = g_new - g_rise_new
    g_eff = cp.maximum(g_eff, 0.0)
    # Voltage-dependent Mg2+ block (Jahr & Stevens 1990) -- regenerative as V depolarizes.
    mg_block = 1.0 / (1.0 + (mg_conc / 3.57) * cp.exp(-0.062 * v))
    # Plateau current (conductance form: driving force toward E_e = 0 mV).
    I = g_eff * mg_block * (E_e - v)
    return g_new, g_rise_new, I

@fuse()
def fused_stp_decay_recovery(u, x, dt, tau_f, tau_d):
    """Fused kernel for STP u and x variable decay/recovery."""
    # Ensure tau_f and tau_d are not zero to prevent division by zero.
    tau_f_safe = cp.maximum(tau_f, 1e-9) # Use a small epsilon if tau_f is zero
    tau_d_safe = cp.maximum(tau_d, 1e-9) # Use a small epsilon if tau_d is zero

    # Decay of u (facilitation variable)
    u_decayed = u * cp.exp(-dt / tau_f_safe)
    # Recovery of x (depression variable)
    x_recovered_increment = (1.0 - x) * (dt / tau_d_safe) # dx/dt = (1-x)/tau_d
    x_recovered = x + x_recovered_increment
    x_clipped = cp.clip(x_recovered, 0.0, 1.0) # Ensure x stays within [0, 1]
    return u_decayed, x_clipped

@fuse()
def fused_homeostasis_update(neuron_activity_ema_in, fired_this_step_float, target_rate, alpha_ema, adapt_rate,
                             neuron_firing_thresholds_in, thresh_min, thresh_max):
    """Fused kernel for homeostatic threshold adaptation."""
    # Update Exponential Moving Average (EMA) of neuron activity
    new_neuron_activity_ema = (1.0 - alpha_ema) * neuron_activity_ema_in + alpha_ema * fired_this_step_float
    # Calculate error from target firing rate
    error = new_neuron_activity_ema - target_rate
    # Calculate change in threshold based on error and adaptation rate
    threshold_delta = error * adapt_rate
    # Update firing thresholds
    new_neuron_firing_thresholds = neuron_firing_thresholds_in + threshold_delta
    # Clip thresholds to min/max bounds
    new_neuron_firing_thresholds_clipped = cp.clip(new_neuron_firing_thresholds, thresh_min, thresh_max)
    return new_neuron_activity_ema, new_neuron_firing_thresholds_clipped

# --- Phase C2: STDP Kernels (Bi & Poo 1998, Caporale & Dan 2008) ---
@fuse()
def fused_stdp_weight_update(delta_t, w_current, A_plus, A_minus, tau_plus, tau_minus, w_min, w_max):
    """Fused kernel for STDP weight update based on spike timing difference.

    Implements classical asymmetric STDP window (Bi & Poo 1998):
    - delta_t > 0 (pre-before-post, causal): LTP (potentiation)
    - delta_t < 0 (post-before-pre, anti-causal): LTD (depression)

    Args:
        delta_t: Spike timing difference (t_post - t_pre) in ms
        w_current: Current synaptic weight
        A_plus: LTP amplitude
        A_minus: LTD amplitude
        tau_plus: LTP time constant (ms)
        tau_minus: LTD time constant (ms)
        w_min: Minimum weight
        w_max: Maximum weight

    Returns:
        Updated synaptic weight
    """
    # LTP: delta_t > 0 means post fired after pre -> strengthen synapse
    # Use soft-bound: delta_w = A_plus * (w_max - w) * exp(-delta_t / tau_plus)
    ltp_update = cp.where(
        delta_t > 0.0,
        A_plus * (w_max - w_current) * cp.exp(-delta_t / tau_plus),
        0.0
    )

    # LTD: delta_t < 0 means pre fired after post -> weaken synapse
    # Use soft-bound: delta_w = -A_minus * (w - w_min) * exp(delta_t / tau_minus)
    ltd_update = cp.where(
        delta_t < 0.0,
        -A_minus * (w_current - w_min) * cp.exp(delta_t / tau_minus),
        0.0
    )

    # Apply update and clip to bounds
    w_new = w_current + ltp_update + ltd_update
    w_new_clipped = cp.clip(w_new, w_min, w_max)
    return w_new_clipped


@fuse()
def fused_inhibitory_stdp_trace_update(trace, fired, decay):
    """Decay a local spike trace and add the current binary spike event."""

    return trace * decay + fired


@fuse()
def fused_inhibitory_stdp_weight_update(
    w_current,
    pre_trace,
    post_trace,
    pre_fired,
    post_fired,
    eta,
    alpha,
    w_min,
    w_max,
):
    """Vogels-style homeostatic iSTDP for positive GABA conductances.

    Traces are updated before this kernel is called. An inhibitory
    presynaptic spike contributes ``eta * (post_trace - alpha)`` and a
    postsynaptic spike contributes ``eta * pre_trace``. The caller performs
    anatomical and pathway-gate scoping before invoking this fused update.
    """

    pre_update = cp.where(pre_fired, eta * (post_trace - alpha), 0.0)
    post_update = cp.where(post_fired, eta * pre_trace, 0.0)
    return cp.clip(w_current + pre_update + post_update, w_min, w_max)

@fuse()
def fused_htm_permanence_update(w, pre_last, post_now, hfac_post, lam_pot, lam_dep, w_min, w_max):
    """Bouhadjar-Diesmann 2022 three-term HTM Temporal-Memory permanence update, per coincidence-routed distal
    synapse (rung-4 Stage C). The unsupervised, local, teacher-free learning rule that self-organizes context-
    specific high-order sequence prediction. All inputs are PER-SYNAPSE (gathered by the caller from the cached COO
    exactly like `fused_stdp_weight_update`):

      w         : current permanence (in [w_min, w_max] = [0, 1])
      pre_last  : 1.0 if the presynaptic cell fired on the PREVIOUS symbol (cp_prev_firing_states), else 0.0
      post_now  : 1.0 if the postsynaptic cell fired on THIS step (cp_firing_states — a sparse winner), else 0.0
      hfac_post : the post cell's dAP-rate homeostatic factor 0.5 + 0.5*max(0, z* - z_post), gathered per synapse

    Three terms (validated to reproduce EMERGE-9d on a flat permanence matrix, `_emerge13_stageC_flat_permanence`):
      (1) POTENTIATION — pre-before-post causal (a distal synapse from a prior WINNER onto a cell that just won),
          scaled by the homeostasis so over-used cells stop potentiating (fresh cells absorb new contexts).
      (2) PRESYNAPTIC DEPRESSION — the pre fired but the post did NOT win this step, so this synapse fails to predict
          the post and weakens (synapses to cells that keep firing without the post disconnect).
      (3) HOMEOSTASIS is carried in `hfac_post` (the caller maintains the per-cell low-pass dAP-rate z EMA).

    Soft-clamped to [w_min, w_max]. Pure math, no simulator deps; a no-op on any synapse where pre_last == 0."""
    pot = pre_last * post_now * lam_pot * hfac_post
    dep = pre_last * (1.0 - post_now) * lam_dep
    w_new = w + pot - dep
    return cp.clip(w_new, w_min, w_max)

@fuse()
def fused_htm_winner_inactive_depression(w, pre_active, post_win, lam_dep_wi, w_min, w_max):
    """HTM Spatial-Pooler WINNER-SELECTIVITY depression (Cui-Ahmad-Hawkins 2017 boosting SP; Diehl-Cook 2015 STDP +
    lateral inhibition). A WINNING column (`post_win` == 1) DEPRESSES the synapses from its INACTIVE inputs
    (`pre_active` == 0), so the column tunes to the features it actually needs and separates OVERLAPPING categories that
    a fixed projection cannot. This is the one term `fused_htm_permanence_update` structurally lacks: that kernel gates
    BOTH its terms on `pre_last`, so an inactive-presynapse synapse is a no-op there. This kernel is ADDITIVE + SEPARATE
    — existing callers of `fused_htm_permanence_update` are byte-unchanged. Per-synapse inputs (gathered from the cached
    COO exactly like `fused_stdp_weight_update`); a no-op on any synapse where `post_win` == 0 or `pre_active` == 1.

      w          : current permanence in [w_min, w_max] = [0, 1]
      pre_active : 1.0 if the presynaptic (input) cell is active in THIS input, else 0.0
      post_win   : 1.0 if the postsynaptic (column) cell is a WINNER this step, else 0.0
      lam_dep_wi : winner-inactive depression rate (0.0 = disabled = no effect)

    De-risked on-substrate at EMERGE-39 (the host version of this exact term lifted held-out inheritance 0.20 -> 0.96 on
    6 overlapping categories); this kernel makes the winner-selectivity learning fully-on-substrate. Pure math, no
    simulator deps; soft-clamped to [w_min, w_max]."""
    dep = (1.0 - pre_active) * post_win * lam_dep_wi
    return cp.clip(w - dep, w_min, w_max)

@fuse()
def fused_bdsp_update(w, etilde_pre, B_post, Pbar_post, E_post, eta, w_min, w_max):
    """Burst-Dependent Synaptic Plasticity (BDSP / Burstprop) feedforward weight update (D1 build, 2026-07-07;
    Payeur-Naud 2021 Nat Neurosci 10.1038/s41593-021-00857-x, eq. M1.2; Greedy-Naud 2022 BurstCCN). The spiking,
    LOCAL, three-factor deep-credit rule that the EMERGE-1b/EMERGE-3 rate result confirmed:

        dw_ij = eta * Etilde_j * ( B_i - Pbar_i * E_i )   ==   eta * Etilde_j * E_i * ( P_i - Pbar_i )

    where (all PER-SYNAPSE, gathered by the caller from the cached COO exactly like `fused_stdp_weight_update`):
      w         : current feedforward weight (in [w_min, w_max]; signed -- the apical credit can drive LTD)
      etilde_pre: the PRESYNAPTIC eligibility trace of source j (cp_eligibility_trace gathered on coo.row) = the
                  feedforward/event factor of the presynaptic partner (a decaying trace of its recent events).
      B_post    : the POSTsynaptic burst rate of target i (a low-pass of 2nd-spike-within-ISI events).
      Pbar_post : the POSTsynaptic slow EMA burst-probability baseline of i (init bdsp_p0) -- the SINGLE-PHASE
                  constant baseline (no teach/no-teach phase switch). At rest the apical is silent => P_i == Pbar_i
                  => the burst-deviation (B_i - Pbar_i*E_i) == 0 => dw == 0 (the P0 no-spurious-learning moat).
      E_post    : the POSTsynaptic event rate of i (a low-pass of its somatic events) -- the multiplexed
                  feedforward channel, INVARIANT to the apical (the multiplexing invariant).

    The burst-rate DEVIATION (B_i - Pbar_i*E_i) is POSITIVE when the apical drive raises target i's burst
    probability above its baseline (top-down says "be more active here" -> LTP) and NEGATIVE when the apical
    suppresses it (LTD) -- so the fixed-random apical feedback sets the LTP/LTD SIGN without changing E. This is a
    genuine three-factor rule: presynaptic activity (etilde_pre) x postsynaptic burst-deviation x eta. Fully local,
    no weight transport (the apical feedback that shapes P/B is a SEPARATE fixed-random pathway, a runner-side
    RegionPathway(plastic=False), never a transpose of a forward weight).

    Signed-clamped to [w_min, w_max]. Pure math, no simulator deps. Called ONLY from the guarded `if cfg.enable_bdsp`
    block in bridge._run_one_simulation_step (beside the STDP block), gated by cp_plasticity_rate_gain there, so its
    presence is byte-inert when enable_bdsp is False (the block is unreached and this kernel is never invoked).
    A near-no-op wherever the burst-deviation is ~0 (rest / self-predicting)."""
    dev = B_post - Pbar_post * E_post            # burst-rate deviation = E_post * (P_post - Pbar_post)  [M1.2]
    w_new = w + eta * etilde_pre * dev
    return cp.clip(w_new, w_min, w_max)

@fuse()
def fused_btsp_update(w, etilde_pre, is_post, eta, w_min, w_max):
    """Behavioral-Timescale Synaptic Plasticity (BTSP) plateau-gated ONE-SHOT weight update (gap#4, 2026-07-18;
    Bittner-Magee 2017 10.1126/science.aan3846; Milstein-Magee 2021 eLife 73046). The LOCAL, plateau-gated,
    no-weight-transport, no-global-loss credit rule the gap#4 rate de-risk confirmed 6-seed:

        dw_ij = eta * Etilde_j * IS_i * (w_max - w_ij)   (saturating one-shot potentiation)

    where (PER-SYNAPSE, gathered by the caller from the cached COO exactly like fused_bdsp_update):
      w         : current feedforward weight (in [w_min, w_max]).
      etilde_pre: the PRESYNAPTIC synaptic-eligibility trace of source j (cp_eligibility_trace gathered on coo.row) --
                  a SECONDS-long decaying trace of j's recent activity (Milstein's pre-side eligibility).
      is_post   : the POSTsynaptic INSTRUCTIVE signal of target i = max(v_apical_i - v_hold, 0), the dendritic PLATEAU
                  depolarization above threshold (gathered on coo.col). With the gap#5 BISTABLE apical (self-regen +
                  KIR) a triggered plateau LATCHES for SECONDS -> a seconds-long is_post -> a BEHAVIORAL-TIMESCALE
                  credit window (a pre-input active seconds before/after the plateau still potentiates); a transient
                  apical gives only a ms window. At rest the apical is silent (KIR down-state) => is_post == 0 =>
                  dw == 0 (the no-spurious-learning moat, by construction).

    Saturating (w_max - w) potentiation (Milstein's BTSP is dominated by potentiation of low-w synapses co-active with
    the plateau; the slower LTD arm is a separate process, omitted here). ONE-SHOT: a single plateau presentation
    shifts w. Fully local; NO weight transport (is_post is the cell's OWN apical plateau, not a transpose of any forward
    W); NO global loss. Pure math, no simulator deps. Called ONLY from the guarded `if cfg.enable_btsp` block in
    bridge._run_one_simulation_step, gated by cp_plasticity_rate_gain, so byte-inert when enable_btsp is False (the
    block is unreached, this kernel never invoked). A no-op wherever the plateau is silent (is_post ~ 0)."""
    w_new = w + eta * etilde_pre * is_post * (w_max - w)
    return cp.clip(w_new, w_min, w_max)

@fuse()
def fused_btsp_milstein_update(w, q_pot, q_dep, is_post, k_pot, k_dep, w_min, w_max):
    """gap#4: WEIGHT-DEPENDENT BIDIRECTIONAL BTSP (Milstein et al. 2021, eLife 10:e73046).

    Biology (measured, not modelled): "weak inputs potentiate, and strong inputs depress." dVm vs INITIAL Vm
    correlates at r = -0.91 while FINAL Vm vs initial Vm correlates at r = 0.04 -- the field converges to a
    target shape essentially independent of where it started. BTSP is "inherently stable, converting synaptic
    potentiation into depression when input strengths exceed a particular range."

    WHY THIS RULE AND NOT SEPARATION. Seven mechanisms were built to SEPARATE adjacent-lag from field-forming
    synapses; all failed. The literature says why: real CA1 field spacing is Poisson with a MODAL GAP OF ZERO
    while the potentiation window spans 75-150 cm, so biology never separates them -- it lets them collide and
    resolves the collision by the SIGN of the update. No separation is attempted here.

    Structurally immune to the three failure modes that killed the predecessors:
      * depression is MULTIPLICATIVE in (w - w_min), so it VANISHES at the floor -- the Miller-MacKay pathology
        (51% of weights pinned at w_min, surviving positive increments dragging the mean up) cannot arise;
      * potentiation is multiplicative in (w_max - w), so weights asymptote to an INTERIOR fixed point;
      * the fixed point w* = (k_pot*q_pot) / (k_pot*q_pot + k_dep*q_dep) is a RATIO, invariant to any common
        rescaling of both drives -- the zero-DC DoG died to exactly such an amplitude mismatch (0.36x).

    q_pot / q_dep are the caller's sigmoids of the NORMALIZED overlap (eligibility x instructive signal).
    Milstein's published thresholds (alpha_dep 0.09, alpha_pot 0.24) are in NORMALIZED units -- applying them to
    raw eligibility puts both sigmoids off the data and renders the rule inert, which is the DoG failure verbatim.
    Callers MUST normalize and unit-check before use.

    k_dep = 0 reduces to pure-potentiation BTSP.
    """
    pot = k_pot * q_pot * (w_max - w)
    dep = k_dep * q_dep * (w - w_min)
    return cp.clip(w + is_post * (pot - dep), w_min, w_max)


@fuse()
def fused_btsp_dog_update(w, e_fast_pre, e_slow_pre, is_post, eta, a_dep, w_min, w_max):
    """gap#4 Rank-2: ZERO-DC difference-of-exponentials BTSP.

    Why this exists. Every depression form tried so far is a function of the synapse's OWN eligibility
    MAGNITUDE, and two pre-registered attempts to select a lag BAND by magnitude both failed -- not
    because magnitude fails to encode lag (measured corr(eligibility, lag) = -0.9445, it encodes it
    cleanly) but because at the tested geometry the ADJACENT field and the lag WHERE THE FIELD FORMS
    are the same lag (field spacing 4 bins == measured backward shift 4-6 bins). No band can separate
    what lag space does not separate.

    This rule does not attempt to SELECT a lag at all. It drives the update with the SIGNED difference
    of a fast and a slow presynaptic eligibility trace. A kernel whose DC gain is zero CANNOT BUILD A
    PEDESTAL -- that is an algebraic guarantee, not a tuned one -- so the geometric collision above
    simply does not apply to it.

    Zero-DC condition over a field window W:
        a_dep = [tau_p * (1 - exp(-W/tau_p))] / [tau_d * (1 - exp(-W/tau_d))]

    a_dep = 0 reduces to pure-potentiation BTSP to within ONE float32 ULP (measured max|diff| =
    1.19e-07 over 2000 trials), NOT exactly: `e_fast - 0.0*e_slow` rounds differently than `e_fast`.
    An assertion caught this after the docstring had claimed "EXACTLY". The DEFAULT path is
    nevertheless byte-identical, for a different and stronger reason: the bridge allocates the slow
    trace only when btsp_dog_a_dep > 0 AND btsp_elig_tau_slow_ms > 0, so at defaults this kernel is
    UNREACHABLE (verified end-to-end: peaks [4,8,12,16], dw 2347.32 unchanged).
    """
    drive = e_fast_pre - a_dep * e_slow_pre
    pot = cp.maximum(drive, 0.0) * (w_max - w)
    dep = cp.maximum(-drive, 0.0) * (w - w_min)
    return cp.clip(w + eta * is_post * (pot - dep), w_min, w_max)


def fused_btsp_hetero_update(w, etilde_pre, is_post, eta, lam_dep, w_min, w_max,
                             dep_theta=0.0, inv_theta=0.0, use_thresh=0.0,
                             band_lo=0.0, band_hi=0.0, use_band=0.0):
    """Structured (heterosynaptic-COMPETITION) BTSP one-shot weight update -- the gap#4<->gap#5 UNIFICATION storing
    rule (2026-07-18). Extends fused_btsp_update with the Milstein-Magee 2021 (eLife 73046) BIDIRECTIONAL arm /
    Chistiakova-Volgushev heterosynaptic plasticity / Miller-MacKay/Chistiakova-Volgushev (NOT Oja: Oja is MULTIPLICATIVE and PRESERVES ratios rather than sharpening them -- that citation inverted the result it was invoked to support; caught by the 2026-07-20 research gate) competitive normalization: a plateauing cell POTENTIATES
    its plateau-coincident inputs AND DEPRESSES its non-coincident inputs -- the COMPETITION that sharpens the stored
    assembly into a stronger, more-specific recurrent attractor:

        dw_ij = eta * IS_i * [ Etilde_j * (w_max - w_ij)  -  lam_dep * (1 - Etilde_j) * (w_ij - w_min) ]

    where (PER-SYNAPSE, gathered by the caller from the cached COO exactly like fused_btsp_update):
      w          : current recurrent weight in [w_min, w_max].
      etilde_pre : the SECONDS-long presynaptic synaptic-eligibility of source j (Milstein pre-side, in ~[0,1]); high
                   for a plateau-coincident (assembly) input, ~0 for a non-coincident one.
      is_post    : the POSTsynaptic dendritic PLATEAU instructive signal max(v_apical - v_hold, 0) (the learning gate;
                   IS==0 => dw==0 at rest = the no-spurious-learning moat, by construction).
      lam_dep    : heterosynaptic-depression coefficient. lam_dep=0 => this reduces EXACTLY to the pure-potentiation
                   fused_btsp_update (the depression term vanishes) => byte-identical to the default BTSP path.

    WHY: the structured-vs-uniform head-to-head (structured Hebbian+competition cue 0.226 vs uniform BTSP 0.179, same
    assembly/recall/seed, 3-seed) proved the completion-magnitude residual is a property of the STORING RULE'S
    structure -- heterosynaptic competition sharpens the attractor; uniform (w_max-w) saturation leaves it diffuse.
    This rule keeps BTSP's one-shot behavioral-timescale plateau-gating (gap#4) AND adds the competition (gap#5
    completion strength) -- the SHARED fix for both residuals. Fully local; NO weight transport; NO global loss.
    Pure math, no simulator deps. Called ONLY from the guarded `if cfg.enable_btsp and btsp_hetero_dep>0` sub-branch
    of the BTSP block in bridge._run_one_simulation_step, gated by cp_plasticity_rate_gain -> byte-inert by default."""
    pot = etilde_pre * (w_max - w)
    # THRESHOLDED depression (2026-07-20 research gate). The LINEAR (1 - Etilde) gate is exactly the form theory says
    # must fail: Cone & Shouval 2021 give W_i(D) = I_p/(I_p+I_d), which for shared trace parameters is 0.5 for EVERY
    # delay -- i.e. provably UNIFORM potentiation; Milstein 2021 ran a linear-instead-of-sigmoidal variant as a control
    # and it "predicted a single value regardless of the timing". So the 2026-07-18 competition REFUTATION refutes the
    # LINEAR implementation, not heterosynaptic competition itself -- and this project's own adjacent HTM result fixed
    # the identical failure by THRESHOLDING (per-event keying fails; a cumulative/thresholded mask gives 5.2-8.9x).
    # Thresholded gate: depress only inputs whose eligibility is BELOW theta, PROTECTING strongly co-active pairs
    # (the salvage the burned finding itself named and deferred). This lowers the PEDESTAL without lowering the PEAK,
    # which is the only way out of the measured bind (a pedestal big enough to cross threshold destroys contrast).
    # use_thresh=0 => dep_gate == (1 - etilde_pre) EXACTLY => byte-identical to the committed linear behaviour.
    gate_lin = 1.0 - etilde_pre
    gate_thr = cp.maximum(dep_theta - etilde_pre, 0.0) * inv_theta
    # BAND gate (Milstein 2021): depression fires ONLY in a window BETWEEN two thresholds
    # (band_lo < Etilde < band_hi) -- i.e. the lags ADJACENT to the peak, NOT the far field.
    # The thresholded gate above depresses LOW-eligibility (= FAR) synapses, which lowers a
    # distant floor roughly uniformly and leaves the peak's NEIGHBOURS elevated; measured
    # 2026-07-20, that is exactly the observed defect (far contrast 2.60x, adjacent 1.21x).
    gate_band = cp.where((etilde_pre > band_lo) & (etilde_pre < band_hi), 1.0, 0.0)
    dep_gate = (use_band * gate_band
                + (1.0 - use_band) * (use_thresh * gate_thr + (1.0 - use_thresh) * gate_lin))
    dep = lam_dep * dep_gate * (w - w_min)
    w_new = w + eta * is_post * (pot - dep)
    return cp.clip(w_new, w_min, w_max)

@fuse()
def fused_eligibility_trace_decay(trace, decay_factor):
    """Fused kernel for eligibility trace exponential decay.

    Args:
        trace: Current eligibility trace value
        decay_factor: exp(-dt / tau)

    Returns:
        Decayed trace value
    """
    return trace * decay_factor
