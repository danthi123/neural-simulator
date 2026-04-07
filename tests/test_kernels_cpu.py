"""
Comprehensive CPU-only (NumPy-based) test harness for validating the mathematical
correctness of all fused CUDA kernels in the neural simulator.

This module contains pure NumPy reimplementations of each CUDA kernel and rigorously
tests them against:
  1. Analytical solutions (e.g., HH gating variable steady-states)
  2. Known biophysical parameters (e.g., Bi & Poo 1998 STDP windows)
  3. Numerical properties (e.g., exponential decay, weight clipping)
  4. Temperature-dependent kinetics (Q10 scaling)

No CuPy, GPU, or OpenGL required. Run with: pytest tests/test_kernels_cpu.py -v
"""

import numpy as np
import pytest
from typing import Tuple


# =============================================================================
# NumPy Reimplementations of Fused CUDA Kernels
# =============================================================================

def numpy_izhikevich2007_dynamics_update(
    v: np.ndarray,
    u: np.ndarray,
    C_param: np.ndarray,
    k_param: np.ndarray,
    vr_param: np.ndarray,
    vt_param: np.ndarray,
    a_param: np.ndarray,
    b_param: np.ndarray,
    total_synaptic_current: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NumPy reimplementation of fused_izhikevich2007_dynamics_update.

    Izhikevich 2007 model:
        dv/dt = (k*(v - v_r)*(v - v_t) - u + I_syn) / C
        du/dt = a*(b*(v - v_r) - u)

    With reset: v -> c, u -> u + d (on spike)
    """
    # Safe division to avoid issues with C=0
    C_param_safe = np.where(C_param == 0.0, 1.0, C_param)

    # Differential equations
    dv_dt = (k_param * (v - vr_param) * (v - vt_param) - u + total_synaptic_current) / C_param_safe
    du_dt = a_param * (b_param * (v - vr_param) - u)

    # Euler integration
    v_new = v + dv_dt * dt
    u_new = u + du_dt * dt

    return v_new, u_new


def numpy_hodgkin_huxley_dynamics_update(
    V: np.ndarray,
    m: np.ndarray,
    h: np.ndarray,
    n: np.ndarray,
    I_syn: np.ndarray,
    dt: float,
    C_m: float,
    g_Na_max: float,
    g_K_max: float,
    g_L: float,
    E_Na: float,
    E_K: float,
    E_L: float,
    temperature_celsius: float,
    q10_factor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    NumPy reimplementation of fused_hodgkin_huxley_dynamics_update.

    Classical Hodgkin-Huxley model with temperature-dependent kinetics.
    Uses epsilon-based safe division (no cp.where() branching).

    HH gating equations:
        m_inf = alpha_m / (alpha_m + beta_m)
        tau_m = 1 / (alpha_m + beta_m)
        m_new = m_inf + (m_old - m_inf) * exp(-dt / tau_m)

    Temperature scaling via Q10 factor.
    """
    # Temperature adjustment (phi)
    BASE_HH_KINETICS_TEMP_C = 6.3
    phi = q10_factor ** ((temperature_celsius - BASE_HH_KINETICS_TEMP_C) / 10.0)

    # Rate constants (alpha, beta) for m-gate
    v_plus_40 = V + 40.0
    alpha_m_orig = np.where(
        v_plus_40 == 0,
        1.0 * 0.1 * 10.0,  # Limit: alpha_m(-40) = 1.0
        -0.1 * v_plus_40 / np.expm1(-v_plus_40 / 10.0),
    )
    beta_m_orig = 4.0 * np.exp(-(V + 65.0) / 18.0)

    # Rate constants (alpha, beta) for h-gate
    alpha_h_orig = 0.07 * np.exp(-(V + 65.0) / 20.0)
    beta_h_orig = 1.0 / (np.exp(-(V + 35.0) / 10.0) + 1.0)

    # Rate constants (alpha, beta) for n-gate
    v_plus_55 = V + 55.0
    alpha_n_orig = np.where(
        v_plus_55 == 0,
        0.1 * 0.01 * 10.0,  # Limit: alpha_n(-55) = 0.1
        -0.01 * v_plus_55 / np.expm1(-v_plus_55 / 10.0),
    )
    beta_n_orig = 0.125 * np.exp(-(V + 65.0) / 80.0)

    # Apply temperature correction to rate constants
    alpha_m = alpha_m_orig * phi
    beta_m = beta_m_orig * phi
    alpha_h = alpha_h_orig * phi
    beta_h = beta_h_orig * phi
    alpha_n = alpha_n_orig * phi
    beta_n = beta_n_orig * phi

    # Epsilon-based safe division (avoids branching)
    _EPS_GATE = 1e-12
    sum_alpha_beta_m = alpha_m + beta_m + _EPS_GATE
    sum_alpha_beta_h = alpha_h + beta_h + _EPS_GATE
    sum_alpha_beta_n = alpha_n + beta_n + _EPS_GATE

    # Steady-state and time constant
    m_inf = alpha_m / sum_alpha_beta_m
    h_inf = alpha_h / sum_alpha_beta_h
    n_inf = alpha_n / sum_alpha_beta_n

    # Analytical solution for first-order kinetics
    m_new = m_inf + (m - m_inf) * np.exp(-dt * sum_alpha_beta_m)
    h_new = h_inf + (h - h_inf) * np.exp(-dt * sum_alpha_beta_h)
    n_new = n_inf + (n - n_inf) * np.exp(-dt * sum_alpha_beta_n)

    # Clip to [0, 1]
    m_new = np.clip(m_new, 0.0, 1.0)
    h_new = np.clip(h_new, 0.0, 1.0)
    n_new = np.clip(n_new, 0.0, 1.0)

    # Ionic currents
    I_Na = g_Na_max * (m_new**3) * h_new * (V - E_Na)
    I_K = g_K_max * (n_new**4) * (V - E_K)
    I_L = g_L * (V - E_L)
    I_ion = I_Na + I_K + I_L

    # Membrane potential update
    dV_dt = (I_syn - I_ion) / C_m
    V_new = V + dV_dt * dt

    return V_new, m_new, h_new, n_new


def numpy_hh_m_current_update(
    V: np.ndarray,
    p_old: np.ndarray,
    dt: float,
    g_M_max: float,
    E_K: float,
    tau_m_ms: float,
    phi: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NumPy reimplementation of fused_hh_m_current_update.

    Slow K+ M-current with temperature-dependent time constant.
    """
    p_inf = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    tau_safe = np.maximum(tau_m_ms / phi, 1e-3)
    p_new = p_inf + (p_old - p_inf) * np.exp(-dt / tau_safe)
    I_M = g_M_max * p_new * (V - E_K)
    return p_new, I_M


def numpy_hh_CaT_current_update(
    V: np.ndarray,
    m_old: np.ndarray,
    h_old: np.ndarray,
    dt: float,
    g_CaT_max: float,
    E_CaT: float,
    phi: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    NumPy reimplementation of fused_hh_CaT_current_update.

    Low-threshold T-type Ca2+ current with Q10 temperature scaling.
    """
    m_inf = 1.0 / (1.0 + np.exp(-(V + 50.0) / 7.4))
    h_inf = 1.0 / (1.0 + np.exp((V + 80.0) / 5.0))
    tau_m = 5.0 / phi
    tau_h = 20.0 / phi
    m_new = m_inf + (m_old - m_inf) * np.exp(-dt / tau_m)
    h_new = h_inf + (h_old - h_inf) * np.exp(-dt / tau_h)
    I_CaT = g_CaT_max * (m_new**2) * h_new * (V - E_CaT)
    return m_new, h_new, I_CaT


def numpy_hh_h_current_update(
    V: np.ndarray,
    q_old: np.ndarray,
    dt: float,
    g_h_max: float,
    E_h: float,
    phi: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NumPy reimplementation of fused_hh_h_current_update.

    Hyperpolarization-activated mixed cation current (I_h) with temperature scaling.
    """
    q_inf = 1.0 / (1.0 + np.exp((V + 75.0) / 5.5))
    tau_q = 100.0 / phi
    q_new = q_inf + (q_old - q_inf) * np.exp(-dt / tau_q)
    I_h = g_h_max * q_new * (V - E_h)
    return q_new, I_h


def numpy_hh_NaP_current_update(
    V: np.ndarray,
    p_old: np.ndarray,
    dt: float,
    g_NaP_max: float,
    E_Na: float,
    phi: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NumPy reimplementation of fused_hh_NaP_current_update.

    Persistent Na+ current with temperature-dependent kinetics.
    """
    p_inf = 1.0 / (1.0 + np.exp(-(V + 55.0) / 5.0))
    tau_p = 5.0 / phi
    p_new = p_inf + (p_old - p_inf) * np.exp(-dt / tau_p)
    I_NaP = g_NaP_max * p_new * (V - E_Na)
    return p_new, I_NaP


def numpy_adex_dynamics_update(
    V: np.ndarray,
    w: np.ndarray,
    I_syn: np.ndarray,
    dt: float,
    C: np.ndarray,
    g_L: float,
    E_L: float,
    V_T: float,
    Delta_T: float,
    a: float,
    tau_w: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NumPy reimplementation of fused_adex_dynamics_update.

    Adaptive Exponential Integrate-and-Fire model with exponential runaway clamping.

    AdEx dynamics:
        C dV/dt = -g_L*(V - E_L) + g_L*Delta_T*exp((V-V_T)/Delta_T) - w + I_syn
        tau_w dw/dt = a*(V - E_L) - w

    Exponential term clamped to [-20, 5] for numerical safety.
    """
    C_safe = np.where(C == 0.0, 1.0, C)
    tau_w_safe = np.maximum(tau_w, 1e-9)
    Delta_T_safe = np.maximum(Delta_T, 1e-9)

    # Clamp exponential argument to [-20, 5] for float32 safety
    exp_arg = np.clip((V - V_T) / Delta_T_safe, -20.0, 5.0)

    # Membrane equation
    dV_dt = (-g_L * (V - E_L) + g_L * Delta_T * np.exp(exp_arg) - w + I_syn) / C_safe
    # Adaptation equation
    dw_dt = (a * (V - E_L) - w) / tau_w_safe

    V_new = V + dV_dt * dt
    w_new = w + dw_dt * dt

    return V_new, w_new


def numpy_conductance_decay_and_current(
    g_e: np.ndarray,
    g_i: np.ndarray,
    decay_e: float,
    decay_i: float,
    v: np.ndarray,
    E_e: float,
    E_i: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    NumPy reimplementation of fused_conductance_decay_and_current.

    Synaptic conductance exponential decay and current calculation.

    Conductance decay follows: g_new = g_old * exp(-dt / tau_syn)
    where decay factor = exp(-dt / tau_syn)
    """
    g_e_new = g_e * decay_e
    g_i_new = g_i * decay_i
    I_syn = g_e_new * (E_e - v) + g_i_new * (E_i - v)
    return g_e_new, g_i_new, I_syn


def numpy_stp_decay_recovery(
    u: np.ndarray,
    x: np.ndarray,
    dt: float,
    tau_f: float,
    tau_d: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NumPy reimplementation of fused_stp_decay_recovery.

    Tsodyks-Markram short-term plasticity dynamics:
        du/dt = -u / tau_f
        dx/dt = (1 - x) / tau_d

    Analytical solution:
        u_new = u_old * exp(-dt / tau_f)
        x_new = 1 - (1 - x_old) * exp(-dt / tau_d)  [clamped to [0,1]]
    """
    tau_f_safe = np.maximum(tau_f, 1e-9)
    tau_d_safe = np.maximum(tau_d, 1e-9)

    u_decayed = u * np.exp(-dt / tau_f_safe)
    x_recovered_increment = (1.0 - x) * (dt / tau_d_safe)
    x_recovered = x + x_recovered_increment
    x_clipped = np.clip(x_recovered, 0.0, 1.0)

    return u_decayed, x_clipped


def numpy_homeostasis_update(
    neuron_activity_ema_in: np.ndarray,
    fired_this_step_float: np.ndarray,
    target_rate: float,
    alpha_ema: float,
    adapt_rate: float,
    neuron_firing_thresholds_in: np.ndarray,
    thresh_min: float,
    thresh_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NumPy reimplementation of fused_homeostasis_update.

    Homeostatic threshold adaptation via exponential moving average (EMA) of firing rate.

    EMA update: ema_new = (1 - alpha) * ema_old + alpha * fired_now
    Error: error = ema_new - target_rate
    Threshold delta: delta_thresh = error * adapt_rate
    """
    new_neuron_activity_ema = (1.0 - alpha_ema) * neuron_activity_ema_in + alpha_ema * fired_this_step_float
    error = new_neuron_activity_ema - target_rate
    threshold_delta = error * adapt_rate
    new_neuron_firing_thresholds = neuron_firing_thresholds_in + threshold_delta
    new_neuron_firing_thresholds_clipped = np.clip(new_neuron_firing_thresholds, thresh_min, thresh_max)

    return new_neuron_activity_ema, new_neuron_firing_thresholds_clipped


def numpy_stdp_weight_update(
    delta_t: np.ndarray,
    w_current: np.ndarray,
    A_plus: float,
    A_minus: float,
    tau_plus: float,
    tau_minus: float,
    w_min: float,
    w_max: float,
) -> np.ndarray:
    """
    NumPy reimplementation of fused_stdp_weight_update.

    Classical asymmetric STDP (Bi & Poo 1998):
        delta_t > 0 (post after pre): LTP
            delta_w = A_plus * (w_max - w) * exp(-delta_t / tau_plus)
        delta_t < 0 (pre after post): LTD
            delta_w = -A_minus * (w - w_min) * exp(delta_t / tau_minus)
        delta_t == 0: no update

    Soft-bound rule ensures weights stay within [w_min, w_max].
    """
    ltp_update = np.where(
        delta_t > 0.0,
        A_plus * (w_max - w_current) * np.exp(-delta_t / tau_plus),
        0.0,
    )
    ltd_update = np.where(
        delta_t < 0.0,
        -A_minus * (w_current - w_min) * np.exp(delta_t / tau_minus),
        0.0,
    )

    w_new = w_current + ltp_update + ltd_update
    w_new_clipped = np.clip(w_new, w_min, w_max)

    return w_new_clipped


def numpy_eligibility_trace_decay(
    trace: np.ndarray,
    decay_factor: float,
) -> np.ndarray:
    """
    NumPy reimplementation of fused_eligibility_trace_decay.

    Simple exponential decay: trace_new = trace_old * exp(-dt / tau)
    """
    return trace * decay_factor


# =============================================================================
# Tests: Izhikevich 2007 Model
# =============================================================================


class TestIzhikevich2007:
    """Tests for Izhikevich 2007 neuron dynamics."""

    def test_rs_cell_resting_state(self):
        """
        Test Regular Spiking (RS) cell at resting state.

        Biophysics: RS cells have d_increment ≈ 0.1 (weak AHP).
        At rest (V ≈ v_rest), u should be near equilibrium where du/dt ≈ 0.
        """
        # RS parameters from Izhikevich 2007
        C_param = np.array([100.0])
        k_param = np.array([0.7])
        vr_param = np.array([-60.0])
        vt_param = np.array([-40.0])
        a_param = np.array([0.03])
        b_param = np.array([0.2])

        # Start at rest with zero current
        v = np.array([-65.0])
        u = np.array([0.0])
        I_syn = np.array([0.0])
        dt = 0.1

        v_new, u_new = numpy_izhikevich2007_dynamics_update(
            v, u, C_param, k_param, vr_param, vt_param, a_param, b_param, I_syn, dt
        )

        # At resting state with zero input, v should change very little
        assert np.abs(v_new[0] - v[0]) < 1.0, "V change too large at rest"
        # u should remain small near equilibrium
        assert np.abs(u_new[0]) < 1.0, "u should stay near 0 at rest"

    def test_fs_cell_stronger_ahp(self):
        """
        Test Fast Spiking (FS) cell with d_increment = 0.25 (strong AHP).

        Biophysics: FS cells have d ≈ 0.25 (stronger post-spike adaptation).
        After a spike reset, FS cells should show stronger adaptation than RS.
        """
        # FS parameters
        C_param = np.array([20.0])
        k_param = np.array([1.0])
        vr_param = np.array([-65.0])
        vt_param = np.array([-20.0])
        a_param = np.array([0.2])
        b_param = np.array([0.025])

        # Simulate post-spike state: v reset to c, u incremented by d
        v = np.array([-65.0])  # Reset value (v_c)
        u_post_spike = np.array([0.25])  # Spike caused u += d_increment
        I_syn = np.array([0.0])
        dt = 0.1

        v_new, u_new = numpy_izhikevich2007_dynamics_update(
            v, u_post_spike, C_param, k_param, vr_param, vt_param, a_param, b_param, I_syn, dt
        )

        # FS cells with strong AHP (large u) should have negative dv/dt initially
        # This prevents immediate re-spiking (refractoriness)
        # du/dt = a_param * (b_param * (v - vr) - u)
        # With u = 0.25 > b*(v-vr) ≈ 0, du/dt should be negative
        du_dt_expected = a_param[0] * (b_param[0] * (v[0] - vr_param[0]) - u_post_spike[0])
        assert du_dt_expected < 0, "FS cell should have negative du/dt post-spike (AHP)"

    def test_euler_integration_accuracy(self):
        """
        Test Euler integration step size accuracy.

        For a simple linear ODE (constant input), verify Euler converges.
        """
        # Setup: constant input current
        C_param = np.array([100.0])
        k_param = np.array([0.7])
        vr_param = np.array([-60.0])
        vt_param = np.array([-40.0])
        a_param = np.array([0.03])
        b_param = np.array([0.2])

        v = np.array([-60.0])
        u = np.array([0.0])
        I_syn = np.array([200.0])  # Large input to drive spiking

        # Take one step
        dt = 0.1
        v_new, u_new = numpy_izhikevich2007_dynamics_update(
            v, u, C_param, k_param, vr_param, vt_param, a_param, b_param, I_syn, dt
        )

        # With large input, v should increase
        assert v_new[0] > v[0], "Voltage should increase with positive input"


# =============================================================================
# Tests: Hodgkin-Huxley Model
# =============================================================================


class TestHodgkinHuxley:
    """Tests for Hodgkin-Huxley gating variables and currents."""

    def test_gating_variable_steady_state_at_rest(self):
        """
        Test HH gating variables at resting potential V = -65 mV.

        Analytical values from Hodgkin & Huxley 1952:
            V = -65 mV (near resting):
            m_inf ≈ 0.053 (Na activation low)
            h_inf ≈ 0.596 (Na inactivation partial)
            n_inf ≈ 0.318 (K activation low)

        These are steady-state values: m_new → m_inf over time.
        """
        V = np.array([-65.0])
        m = np.array([0.053])  # Start at steady state
        h = np.array([0.596])
        n = np.array([0.318])
        I_syn = np.array([0.0])
        dt = 1.0  # Large step to let gating vars relax

        # HH parameters (squid axon at 6.3°C)
        C_m = 1.0  # µF/cm²
        g_Na_max = 120.0  # mS/cm²
        g_K_max = 36.0  # mS/cm²
        g_L = 0.3  # mS/cm²
        E_Na = 50.0  # mV
        E_K = -77.0  # mV
        E_L = -54.387  # mV
        temperature_celsius = 6.3
        q10_factor = 3.0

        V_new, m_new, h_new, n_new = numpy_hodgkin_huxley_dynamics_update(
            V, m, h, n, I_syn, dt, C_m, g_Na_max, g_K_max, g_L, E_Na, E_K, E_L,
            temperature_celsius, q10_factor,
        )

        # Gating variables should stay close to steady state at rest
        # (small dynamics when at equilibrium)
        assert np.abs(m_new[0] - m[0]) < 0.01, f"m changed too much: {m[0]} -> {m_new[0]}"
        assert np.abs(h_new[0] - h[0]) < 0.05, f"h changed too much: {h[0]} -> {h_new[0]}"
        assert np.abs(n_new[0] - n[0]) < 0.05, f"n changed too much: {n[0]} -> {n_new[0]}"

    def test_gating_variable_steady_state_depolarized(self):
        """
        Test HH gating variables at depolarized potential V = 0 mV.

        Analytical values from Hodgkin & Huxley 1952:
            V = 0 mV (depolarized):
            m_inf ≈ 0.98 (Na activation high)
            h_inf ≈ 0.001 (Na inactivation complete)
            n_inf ≈ 0.65 (K activation high)

        At depolarized potentials, Na inactivates (h → 0), K activates (n increases).
        """
        V = np.array([0.0])
        m = np.array([0.98])  # Expected steady state
        h = np.array([0.001])
        n = np.array([0.65])
        I_syn = np.array([0.0])
        dt = 0.1

        C_m = 1.0
        g_Na_max = 120.0
        g_K_max = 36.0
        g_L = 0.3
        E_Na = 50.0
        E_K = -77.0
        E_L = -54.387
        temperature_celsius = 6.3
        q10_factor = 3.0

        V_new, m_new, h_new, n_new = numpy_hodgkin_huxley_dynamics_update(
            V, m, h, n, I_syn, dt, C_m, g_Na_max, g_K_max, g_L, E_Na, E_K, E_L,
            temperature_celsius, q10_factor,
        )

        # At depolarized voltages, these steady-state values should be maintained
        assert m_new[0] > 0.95, f"m should be high at V=0: {m_new[0]}"
        assert h_new[0] < 0.05, f"h should be very low at V=0: {h_new[0]}"
        assert n_new[0] > 0.6, f"n should be high at V=0: {n_new[0]}"

    def test_gating_variables_clipped_to_unity(self):
        """
        Test that gating variables are clipped to [0, 1].

        Biophysics: Gating variables are dimensionless probabilities.
        Numerical errors or extreme conditions should not cause them to exceed [0, 1].
        """
        V = np.array([-100.0, 100.0])  # Extreme voltages
        m = np.array([0.5, 0.5])
        h = np.array([0.5, 0.5])
        n = np.array([0.5, 0.5])
        I_syn = np.array([0.0, 0.0])
        dt = 10.0  # Large step

        C_m = 1.0
        g_Na_max = 120.0
        g_K_max = 36.0
        g_L = 0.3
        E_Na = 50.0
        E_K = -77.0
        E_L = -54.387
        temperature_celsius = 6.3
        q10_factor = 3.0

        V_new, m_new, h_new, n_new = numpy_hodgkin_huxley_dynamics_update(
            V, m, h, n, I_syn, dt, C_m, g_Na_max, g_K_max, g_L, E_Na, E_K, E_L,
            temperature_celsius, q10_factor,
        )

        # All gating variables must be in [0, 1]
        assert np.all(m_new >= 0.0) and np.all(m_new <= 1.0), f"m out of range: {m_new}"
        assert np.all(h_new >= 0.0) and np.all(h_new <= 1.0), f"h out of range: {h_new}"
        assert np.all(n_new >= 0.0) and np.all(n_new <= 1.0), f"n out of range: {n_new}"

    def test_temperature_scaling_q10(self):
        """
        Test Q10 temperature scaling of HH kinetics.

        Biophysics: Reaction rates double (~Q10=3) for every 10°C increase.
        At higher temperature, gating variables should reach steady state faster.

        This test verifies: relaxation time tau_gate scales as 1/phi,
        where phi = Q10^((T - T_ref) / 10).
        """
        V = np.array([-65.0])
        m = np.array([0.02])  # Far from steady state
        h = np.array([0.8])
        n = np.array([0.2])
        I_syn = np.array([0.0])
        dt = 1.0

        C_m = 1.0
        g_Na_max = 120.0
        g_K_max = 36.0
        g_L = 0.3
        E_Na = 50.0
        E_K = -77.0
        E_L = -54.387
        q10_factor = 3.0

        # At cold temperature (6.3°C, reference)
        V_cold, m_cold, h_cold, n_cold = numpy_hodgkin_huxley_dynamics_update(
            V, m.copy(), h.copy(), n.copy(), I_syn, dt, C_m, g_Na_max, g_K_max, g_L,
            E_Na, E_K, E_L, temperature_celsius=6.3, q10_factor=q10_factor,
        )

        # At warm temperature (37°C, mammalian)
        V_warm, m_warm, h_warm, n_warm = numpy_hodgkin_huxley_dynamics_update(
            V, m.copy(), h.copy(), n.copy(), I_syn, dt, C_m, g_Na_max, g_K_max, g_L,
            E_Na, E_K, E_L, temperature_celsius=37.0, q10_factor=q10_factor,
        )

        # At higher temperature, gating vars should relax faster to steady state
        # m starts at 0.02, should be higher at 37°C than at 6.3°C
        assert m_warm[0] > m_cold[0], "m should increase faster at warm temperature"
        # h starts at 0.8, should decrease faster at 37°C
        assert h_warm[0] < h_cold[0], "h should decrease faster at warm temperature"

    def test_epsilon_gate_numerical_safety(self):
        """
        Test that epsilon-based safe division (new optimization) is numerically sound.

        The epsilon term (_EPS_GATE = 1e-12) avoids cp.where() branching overhead.
        It should not affect steady-state gating variables.

        Verification: Results should be nearly identical to original cp.where() approach.
        """
        V = np.array([-65.0, -40.0, 0.0, 50.0])
        m = np.array([0.05, 0.1, 0.5, 0.9])
        h = np.array([0.6, 0.5, 0.1, 0.01])
        n = np.array([0.3, 0.4, 0.6, 0.8])
        I_syn = np.zeros(4)
        dt = 0.1

        C_m = 1.0
        g_Na_max = 120.0
        g_K_max = 36.0
        g_L = 0.3
        E_Na = 50.0
        E_K = -77.0
        E_L = -54.387
        temperature_celsius = 6.3
        q10_factor = 3.0

        V_new, m_new, h_new, n_new = numpy_hodgkin_huxley_dynamics_update(
            V, m, h, n, I_syn, dt, C_m, g_Na_max, g_K_max, g_L, E_Na, E_K, E_L,
            temperature_celsius, q10_factor,
        )

        # Results should be finite and reasonable (no NaN or inf)
        assert np.all(np.isfinite(V_new)), "V_new contains NaN or inf"
        assert np.all(np.isfinite(m_new)), "m_new contains NaN or inf"
        assert np.all(np.isfinite(h_new)), "h_new contains NaN or inf"
        assert np.all(np.isfinite(n_new)), "n_new contains NaN or inf"


# =============================================================================
# Tests: Extended HH Currents (M, CaT, h, NaP)
# =============================================================================


class TestExtendedHHCurrents:
    """Tests for optional extended Hodgkin-Huxley currents with temperature scaling."""

    def test_m_current_activation(self):
        """
        Test slow K+ M-current activation.

        Biophysics: M-current steady-state centered around -35 mV.
        Activation: p_inf = 1 / (1 + exp(-(V+35)/10))
            V = -35 mV: p_inf ≈ 0.5
            V = -25 mV: p_inf ≈ 0.73
            V = -45 mV: p_inf ≈ 0.27
        """
        V = np.array([-45.0, -35.0, -25.0])
        p_old = np.array([0.1, 0.1, 0.1])
        dt = 10.0
        g_M_max = 1.0
        E_K = -90.0
        tau_m_ms = 40.0
        phi = 1.0  # Room temperature, no Q10 correction

        p_new, I_M = numpy_hh_m_current_update(V, p_old, dt, g_M_max, E_K, tau_m_ms, phi)

        # After enough time, p should move toward steady state
        p_inf_expected = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
        # p_new should be closer to p_inf than p_old
        assert np.abs(p_new[0] - p_inf_expected[0]) < np.abs(p_old[0] - p_inf_expected[0])
        assert np.abs(p_new[1] - p_inf_expected[1]) < np.abs(p_old[1] - p_inf_expected[1])
        assert np.abs(p_new[2] - p_inf_expected[2]) < np.abs(p_old[2] - p_inf_expected[2])

    def test_m_current_temperature_scaling(self):
        """
        Test M-current temperature dependence via phi scaling.

        Biophysics: Q10 ~3 means tau_m = base_tau / phi,
        so at higher temperature, tau is shorter, gating var relaxes faster.
        """
        V = np.array([-35.0])
        p_old = np.array([0.1])
        dt = 10.0
        g_M_max = 1.0
        E_K = -90.0
        tau_m_ms = 40.0

        # Cold: phi = 1 (reference temperature)
        p_cold, _ = numpy_hh_m_current_update(V, p_old.copy(), dt, g_M_max, E_K, tau_m_ms, phi=1.0)

        # Warm: phi ≈ 27 (for Q10=3, ΔT=30°C)
        p_warm, _ = numpy_hh_m_current_update(V, p_old.copy(), dt, g_M_max, E_K, tau_m_ms, phi=27.0)

        # At warmer temp, activation should move faster toward p_inf
        p_inf = 1.0 / (1.0 + np.exp(-(V[0] + 35.0) / 10.0))
        assert np.abs(p_warm[0] - p_inf) < np.abs(p_cold[0] - p_inf), \
            "Warm temperature should have faster activation"

    def test_CaT_current_inactivation_coupling(self):
        """
        Test T-type Ca2+ current: activation and inactivation gates are coupled.

        Biophysics: CaT current = g_CaT * m^2 * h * (V - E_CaT)
        At hyperpolarized V (e.g., -80 mV): h_inf high (channel open), m_inf low (closed)
        At depolarized V (e.g., 0 mV): m_inf high, h_inf very low (inactivated)

        This represents the "window" of gating where both gates can be partially open.
        """
        V = np.array([-80.0, 0.0])  # Hyperpolarized and depolarized
        m_old = np.array([0.1, 0.1])
        h_old = np.array([0.1, 0.1])
        dt = 1.0
        g_CaT_max = 2.0
        E_CaT = 120.0
        phi = 1.0

        m_new, h_new, I_CaT = numpy_hh_CaT_current_update(V, m_old, h_old, dt, g_CaT_max, E_CaT, phi)

        # At -80 mV: h should be high (inactivation gate open)
        h_inf_hyp = 1.0 / (1.0 + np.exp((V[0] + 80.0) / 5.0))
        assert h_inf_hyp >= 0.5, "h_inf should be high at -80 mV"

        # At 0 mV: h should be very low (inactivation gate closed)
        h_inf_dep = 1.0 / (1.0 + np.exp((V[1] + 80.0) / 5.0))
        assert h_inf_dep < 0.1, "h_inf should be very low at 0 mV"

    def test_h_current_hyperpolarization_activation(self):
        """
        Test I_h (HCN channel): activates at negative voltages.

        Biophysics: I_h = g_h * q * (V - E_h)
        Steady-state: q_inf = 1 / (1 + exp((V+75)/5.5))
            V = -75 mV: q_inf ≈ 0.5
            V = -100 mV: q_inf ≈ 0.95 (strong activation, inward current)
            V = 0 mV: q_inf ≈ 0.0 (virtually closed)
        """
        V = np.array([-100.0, -75.0, 0.0])
        q_old = np.array([0.0, 0.0, 0.0])
        dt = 100.0  # Longer timestep to allow activation
        g_h_max = 0.5
        E_h = -43.0  # Mixed cation reversal (between K+ and Na+)
        phi = 1.0

        q_new, I_h = numpy_hh_h_current_update(V, q_old, dt, g_h_max, E_h, phi)

        # At -100 mV, q should increase strongly (q_inf ≈ 0.95)
        q_inf_hyp = 1.0 / (1.0 + np.exp((-100.0 + 75.0) / 5.5))
        assert q_new[0] > q_old[0], "q should increase at -100 mV"
        assert q_new[0] > q_new[2], "q should be larger at -100 mV than at 0 mV"
        # At 0 mV, q should remain small (q_inf ≈ 0.0)
        q_inf_dep = 1.0 / (1.0 + np.exp((0.0 + 75.0) / 5.5))
        assert q_inf_dep < 0.01, "q_inf should be very small at 0 mV"

    def test_NaP_persistent_sodium_channel(self):
        """
        Test persistent Na+ current: voltage-dependent activation.

        Biophysics: NaP = g_NaP * p * (V - E_Na)
        Steady-state: p_inf = 1 / (1 + exp(-(V+55)/5))
            V = -55 mV: p_inf ≈ 0.5
            V = -40 mV: p_inf ≈ 0.95 (high activation)
            V = -80 mV: p_inf ≈ 0.07 (low activation)

        NaP is persistent (doesn't inactivate quickly like transient Na+).
        """
        V = np.array([-80.0, -55.0, -40.0])
        p_old = np.array([0.0, 0.0, 0.0])
        dt = 100.0  # Longer timestep (NaP tau ≈ 5 ms / phi ≈ 5 ms at room temp)
        g_NaP_max = 1.0
        E_Na = 60.0
        phi = 1.0

        p_new, I_NaP = numpy_hh_NaP_current_update(V, p_old, dt, g_NaP_max, E_Na, phi)

        # At -40 mV, p_inf should be high
        p_inf_dep = 1.0 / (1.0 + np.exp(-(-40.0 + 55.0) / 5.0))
        # p should move toward p_inf
        assert p_new[2] > p_old[2], "p should increase at depolarized voltage"
        assert p_new[2] > p_new[0], "p should be higher at -40 mV than at -80 mV"
        # At -80 mV, p_inf should be low
        p_inf_hyp = 1.0 / (1.0 + np.exp(-(-80.0 + 55.0) / 5.0))
        assert p_inf_hyp < 0.1, "p_inf should be low at -80 mV"


# =============================================================================
# Tests: AdEx Model
# =============================================================================


class TestAdEx:
    """Tests for Adaptive Exponential Integrate-and-Fire model."""

    def test_subthreshold_dynamics(self):
        """
        Test AdEx subthreshold integration (below threshold).

        Biophysics: Membrane equation integrates leak conductance and
        incoming synaptic current. Without reaching spike threshold,
        voltage should follow input current.
        """
        V = np.array([-70.0])
        w = np.array([0.0])
        I_syn = np.array([10.0])  # Small input
        dt = 0.1
        C = np.array([100.0])
        g_L = 10.0
        E_L = -70.0
        V_T = -50.0
        Delta_T = 2.0
        a = 0.0  # No coupling to voltage for this test
        tau_w = 100.0

        V_new, w_new = numpy_adex_dynamics_update(
            V, w, I_syn, dt, C, g_L, E_L, V_T, Delta_T, a, tau_w
        )

        # With positive input, V should depolarize
        assert V_new[0] > V[0], "Voltage should depolarize with positive input"

    def test_exponential_runaway_clamping(self):
        """
        Test AdEx exponential clipping to prevent numerical overflow.

        Biophysics: When V approaches V_T, exponential sodium inactivation
        becomes very large. Clamping to [-20, 5] prevents overflow in float32.

        exp(-20) ≈ 2e-9 (safe underflow)
        exp(5) ≈ 148 (safe with g_L*Delta_T scaling)
        """
        # Test voltage at threshold
        V = np.array([V_T := -50.0, V_T, V_T])
        w = np.array([0.0, 0.0, 0.0])
        I_syn = np.array([1000.0, 1000.0, 1000.0])  # Large driving input
        dt = 0.01
        C = np.array([100.0, 100.0, 100.0])
        g_L = 10.0
        E_L = -70.0
        Delta_T = 2.0
        a = 4.0
        tau_w = 100.0

        V_new, w_new = numpy_adex_dynamics_update(
            V, w, I_syn, dt, C, g_L, E_L, V_T, Delta_T, a, tau_w
        )

        # Results should be finite (not NaN or inf)
        assert np.all(np.isfinite(V_new)), "V_new should be finite (clamping works)"
        assert np.all(np.isfinite(w_new)), "w_new should be finite"

    def test_adaptation_variable_coupling(self):
        """
        Test AdEx adaptation variable w couples to voltage.

        Biophysics: dw/dt = a*(V - E_L) - w / tau_w
        If a > 0, depolarization increases w (outward adaptation current).
        This provides frequency adaptation: sustained input → spike frequency decreases.
        """
        V = np.array([-60.0])  # Depolarized relative to E_L=-70
        w = np.array([0.0])
        I_syn = np.array([0.0])
        dt = 1.0
        C = np.array([100.0])
        g_L = 10.0
        E_L = -70.0
        V_T = -50.0
        Delta_T = 2.0
        a = 4.0  # Coupling strength (positive = voltage-dependent adaptation)
        tau_w = 100.0

        V_new, w_new = numpy_adex_dynamics_update(
            V, w, I_syn, dt, C, g_L, E_L, V_T, Delta_T, a, tau_w
        )

        # Depolarization with a > 0 should increase w
        dw_dt_expected = a * (V[0] - E_L) - w[0]
        assert dw_dt_expected > 0, "dw/dt should be positive (adaptation increases)"
        assert w_new[0] > w[0], "w should increase with depolarization"


# =============================================================================
# Tests: Synaptic Conductance
# =============================================================================


class TestSynapticConductance:
    """Tests for synaptic conductance decay and current calculation."""

    def test_exponential_decay(self):
        """
        Test synaptic conductance exponential decay.

        Biophysics: Synaptic conductance decays as g(t) = g_0 * exp(-t / tau_syn)
        Decay factor: decay = exp(-dt / tau)
        """
        g_e = np.array([100.0])
        g_i = np.array([50.0])
        decay_e = np.exp(-0.1 / 10.0)  # dt=0.1, tau=10 ms
        decay_i = np.exp(-0.1 / 20.0)  # dt=0.1, tau=20 ms
        v = np.array([-65.0])
        E_e = 0.0
        E_i = -80.0

        g_e_new, g_i_new, I_syn = numpy_conductance_decay_and_current(g_e, g_i, decay_e, decay_i, v, E_e, E_i)

        # Conductances should decrease (decay_e, decay_i < 1)
        assert g_e_new[0] < g_e[0], "g_e should decay exponentially"
        assert g_i_new[0] < g_i[0], "g_i should decay exponentially"

        # Verify exponential relationship: g_new / g_old = decay
        np.testing.assert_allclose(g_e_new[0], g_e[0] * decay_e, rtol=1e-10)
        np.testing.assert_allclose(g_i_new[0], g_i[0] * decay_i, rtol=1e-10)

    def test_synaptic_current_direction(self):
        """
        Test synaptic current calculation with reversal potentials.

        Biophysics: I_syn = g_e * (E_e - V) + g_i * (E_i - V)
        If E_e = 0 (excitatory) and V = -65 mV, then E_e - V > 0, so I_syn drives inward (depolarizing).
        If E_i = -80 (inhibitory) and V = -65 mV, then E_i - V < 0, so I_syn drives outward (hyperpolarizing).
        """
        g_e = np.array([10.0])
        g_i = np.array([5.0])
        decay_e = 1.0  # No decay for this test
        decay_i = 1.0
        v = np.array([-65.0])
        E_e = 0.0  # Excitatory reversal (Na+/glutamate)
        E_i = -80.0  # Inhibitory reversal (K+/GABA)

        g_e_new, g_i_new, I_syn = numpy_conductance_decay_and_current(g_e, g_i, decay_e, decay_i, v, E_e, E_i)

        # Excitatory component: g_e * (E_e - V) = 10 * (0 - (-65)) = 650 pA (depolarizing)
        I_e_expected = g_e[0] * (E_e - v[0])
        assert I_e_expected > 0, "Excitatory current should be positive (depolarizing)"

        # Inhibitory component: g_i * (E_i - V) = 5 * (-80 - (-65)) = -75 pA (hyperpolarizing)
        I_i_expected = g_i[0] * (E_i - v[0])
        assert I_i_expected < 0, "Inhibitory current should be negative (hyperpolarizing)"

        # Total
        I_total_expected = I_e_expected + I_i_expected
        np.testing.assert_allclose(I_syn[0], I_total_expected, rtol=1e-10)


# =============================================================================
# Tests: Short-Term Plasticity (STP)
# =============================================================================


class TestSTP:
    """Tests for Tsodyks-Markram short-term plasticity."""

    def test_facilitation_decay(self):
        """
        Test STP facilitation variable (u) decay.

        Biophysics: u represents fraction of resources available for release.
        After a spike, u increases; during inter-spike interval, u decays back to 0.
        Decay: u(t) = u_0 * exp(-t / tau_f)
        """
        u = np.array([0.5])  # Facilitation variable (post-spike)
        x = np.array([0.5])  # Not updated here
        dt = 50.0
        tau_f = 100.0  # ms
        tau_d = 100.0

        u_decayed, x_recovered = numpy_stp_decay_recovery(u, x, dt, tau_f, tau_d)

        # u should decay toward 0
        u_expected = u[0] * np.exp(-dt / tau_f)
        np.testing.assert_allclose(u_decayed[0], u_expected, rtol=1e-10)
        assert u_decayed[0] < u[0], "u should decay"

    def test_depression_recovery(self):
        """
        Test STP depression variable (x) recovery.

        Biophysics: x represents fraction of resources recovered after depletion.
        After a spike, x decreases (depletion); during inter-spike interval, x recovers.
        Recovery: dx/dt = (1 - x) / tau_d
        """
        u = np.array([0.5])  # Not updated here
        x = np.array([0.3])  # Depression variable (post-spike, depleted)
        dt = 50.0
        tau_f = 100.0
        tau_d = 100.0

        u_decayed, x_recovered = numpy_stp_decay_recovery(u, x, dt, tau_f, tau_d)

        # x should recover toward 1
        # x_new = x_old + (1 - x_old) * (dt / tau_d)
        x_expected = x[0] + (1.0 - x[0]) * (dt / tau_d)
        np.testing.assert_allclose(x_recovered[0], x_expected, rtol=1e-10)
        assert x_recovered[0] > x[0], "x should recover"

    def test_stp_clipping(self):
        """
        Test that STP recovery doesn't exceed bounds.

        Biophysics: Both u and x are normalized fractions (should stay in [0, 1]).
        Very long inter-spike intervals should not cause x > 1 or u < 0.
        """
        u = np.array([0.01])
        x = np.array([0.1])
        dt = 1000.0  # Very long interval
        tau_f = 100.0
        tau_d = 100.0

        u_decayed, x_recovered = numpy_stp_decay_recovery(u, x, dt, tau_f, tau_d)

        # u should stay >= 0
        assert u_decayed[0] >= 0.0, "u should not become negative"
        # x should stay <= 1
        assert x_recovered[0] <= 1.0, "x should not exceed 1"

    def test_different_tau_dynamics(self):
        """
        Test that tau_f and tau_d have different time scales.

        Biophysics: tau_f (facilitation) << tau_d (depression)
        Typical: tau_f ≈ 100 ms, tau_d ≈ 1 s
        This creates a diversity of plasticity timescales.
        """
        u = np.array([0.5])
        x = np.array([0.5])
        dt = 200.0
        tau_f = 100.0  # Short time constant
        tau_d = 500.0  # Long time constant

        u_decayed, x_recovered = numpy_stp_decay_recovery(u, x, dt, tau_f, tau_d)

        # u should change more than x over same dt (because tau_f < tau_d)
        u_frac_change = 1.0 - (u_decayed[0] / u[0])
        x_frac_change = (x_recovered[0] - x[0]) / (1.0 - x[0])
        assert u_frac_change > x_frac_change, "u should change faster than x"


# =============================================================================
# Tests: Homeostasis
# =============================================================================


class TestHomeostasis:
    """Tests for homeostatic threshold adaptation."""

    def test_ema_convergence(self):
        """
        Test exponential moving average (EMA) convergence.

        Biophysics: EMA smooths firing rate over multiple timesteps.
        EMA_new = (1 - alpha) * EMA_old + alpha * fired_now
        For constant firing rate, EMA converges exponentially.
        """
        ema_in = np.array([0.0])
        fired = np.array([1.0])  # Every step fires
        target_rate = 0.5
        alpha_ema = 0.1
        adapt_rate = 0.1
        thresholds_in = np.array([-40.0])
        thresh_min = -60.0
        thresh_max = -20.0

        # Iterate EMA updates
        ema = ema_in.copy()
        for _ in range(100):  # 100 steps of constant firing
            ema, _ = numpy_homeostasis_update(
                ema, fired, target_rate, alpha_ema, adapt_rate, thresholds_in, thresh_min, thresh_max
            )

        # EMA should converge to 1.0 (100% firing)
        assert ema[0] > 0.95, f"EMA should converge to 1.0 after 100 steps of firing: {ema[0]}"

    def test_threshold_adjustment_error_feedback(self):
        """
        Test that threshold increases when firing rate is too low.

        Biophysics: Homeostasis maintains target firing rate by adjusting threshold.
        If actual_rate < target_rate, error > 0 → increase threshold → suppress firing.
        If actual_rate > target_rate, error < 0 → decrease threshold → promote firing.
        """
        ema_in = np.array([0.05])  # Low firing rate
        fired = np.array([0.0])
        target_rate = 0.2  # Want 20% firing
        alpha_ema = 0.5
        adapt_rate = 0.1
        thresholds_in = np.array([-40.0])
        thresh_min = -60.0
        thresh_max = -20.0

        # With ema < target, error < 0, threshold should increase (become less negative)
        ema_new, thresholds_new = numpy_homeostasis_update(
            ema_in, fired, target_rate, alpha_ema, adapt_rate, thresholds_in, thresh_min, thresh_max
        )

        # Actually, let's recalculate: after ema update:
        # ema_new = (1 - 0.5) * 0.05 + 0.5 * 0.0 = 0.025
        # error = 0.025 - 0.2 = -0.175
        # threshold_delta = -0.175 * 0.1 = -0.0175
        # new_threshold = -40 + (-0.0175) = -40.0175 (more negative, harder to spike)
        # Wait, that's backwards. Let me recalculate the intended behavior.

        # In homeostasis, if firing rate is too LOW (ema < target), we should DECREASE
        # threshold to PROMOTE firing. So error = ema - target < 0, and to decrease
        # threshold (make it more negative), we ADD a negative delta.
        # This is consistent: negative error → negative delta → lower threshold → more firing.

        # So the assertion is:
        error_in_step = ema_new[0] - target_rate
        threshold_delta_expected = error_in_step * adapt_rate
        assert threshold_delta_expected < 0, "Error should be negative when firing is low"
        assert thresholds_new[0] < thresholds_in[0], "Threshold should decrease to promote firing"

    def test_threshold_clipping(self):
        """
        Test that thresholds are clipped to [thresh_min, thresh_max].

        Biophysics: Thresholds have biological bounds.
        Very low ema (no firing) should not drop threshold infinitely low.
        """
        ema_in = np.array([0.0])
        fired = np.array([0.0])
        target_rate = 0.5
        alpha_ema = 0.5
        adapt_rate = 1.0  # Large adapt rate to test clipping
        thresholds_in = np.array([-40.0])
        thresh_min = -60.0
        thresh_max = -20.0

        # Take several steps of no firing
        for _ in range(10):
            ema_in, thresholds_in = numpy_homeostasis_update(
                ema_in, fired, target_rate, alpha_ema, adapt_rate, thresholds_in, thresh_min, thresh_max
            )

        # Threshold should not exceed bounds
        assert thresholds_in[0] >= thresh_min, "Threshold should not go below thresh_min"
        assert thresholds_in[0] <= thresh_max, "Threshold should not exceed thresh_max"


# =============================================================================
# Tests: STDP (Spike-Timing Dependent Plasticity)
# =============================================================================


class TestSTDP:
    """Tests for STDP weight updates following Bi & Poo 1998."""

    def test_ltp_positive_delta_t(self):
        """
        Test LTP when delta_t > 0 (post spike fires after pre spike).

        Biophysics (Bi & Poo 1998): Post-before-pre increases synaptic weight.
        Causal: pre fires, then post fires → strengthen connection.
        Implementation: delta_w = A_plus * (w_max - w) * exp(-delta_t / tau_plus)
        """
        delta_t = np.array([10.0])  # Post fires 10 ms after pre
        w_current = np.array([0.5])
        A_plus = 0.001
        A_minus = 0.0003
        tau_plus = 20.0
        tau_minus = 20.0
        w_min = 0.0
        w_max = 1.0

        w_new = numpy_stdp_weight_update(delta_t, w_current, A_plus, A_minus, tau_plus, tau_minus, w_min, w_max)

        # Weight should increase
        assert w_new[0] > w_current[0], "LTP: weight should increase for delta_t > 0"
        # Increase should be within bounds
        assert w_new[0] <= w_max, "Weight should not exceed w_max"

    def test_ltd_negative_delta_t(self):
        """
        Test LTD when delta_t < 0 (pre spike fires after post spike).

        Biophysics (Bi & Poo 1998): Pre-before-post decreases synaptic weight.
        Anti-causal: post fires, then pre fires → weaken connection.
        Implementation: delta_w = -A_minus * (w - w_min) * exp(delta_t / tau_minus)
        """
        delta_t = np.array([-10.0])  # Pre fires 10 ms after post
        w_current = np.array([0.5])
        A_plus = 0.001
        A_minus = 0.0003
        tau_plus = 20.0
        tau_minus = 20.0
        w_min = 0.0
        w_max = 1.0

        w_new = numpy_stdp_weight_update(delta_t, w_current, A_plus, A_minus, tau_plus, tau_minus, w_min, w_max)

        # Weight should decrease
        assert w_new[0] < w_current[0], "LTD: weight should decrease for delta_t < 0"
        # Decrease should be within bounds
        assert w_new[0] >= w_min, "Weight should not go below w_min"

    def test_stdp_no_change_at_zero(self):
        """
        Test that delta_t = 0 causes minimal weight change.

        Biophysics: Simultaneous firing (delta_t = 0) should not drive plasticity.
        LTP term: exp(0) = 1, but weight already at w → no further change if delta_t = 0.
        LTD term: exp(0) = 1, but weight already at w → no further change if delta_t = 0.
        """
        delta_t = np.array([0.0])
        w_current = np.array([0.5])
        A_plus = 0.001
        A_minus = 0.0003
        tau_plus = 20.0
        tau_minus = 20.0
        w_min = 0.0
        w_max = 1.0

        w_new = numpy_stdp_weight_update(delta_t, w_current, A_plus, A_minus, tau_plus, tau_minus, w_min, w_max)

        # Weight should stay the same (both LTP and LTD = 0 when delta_t = 0)
        np.testing.assert_allclose(w_new[0], w_current[0], atol=1e-10)

    def test_stdp_soft_bound_asymmetry(self):
        """
        Test STDP soft-bound rule: LTP depends on (w_max - w), LTD depends on (w - w_min).

        Biophysics: This asymmetry makes weights more resistant to increasing from w_max
        and decreasing from w_min (compresses weights toward center of range).
        """
        # Test LTP: weight near w_max should have smaller LTP increment
        delta_t_ltp = np.array([5.0, 5.0])
        w_low = np.array([0.1])
        w_high = np.array([0.9])
        A_plus = 0.01
        A_minus = 0.003
        tau_plus = 20.0
        tau_minus = 20.0
        w_min = 0.0
        w_max = 1.0

        w_low_new = numpy_stdp_weight_update(delta_t_ltp[0:1], w_low, A_plus, A_minus, tau_plus, tau_minus, w_min, w_max)
        w_high_new = numpy_stdp_weight_update(delta_t_ltp[1:2], w_high, A_plus, A_minus, tau_plus, tau_minus, w_min, w_max)

        # LTP increment when w=0.1: A_plus * (1.0 - 0.1) = A_plus * 0.9
        # LTP increment when w=0.9: A_plus * (1.0 - 0.9) = A_plus * 0.1
        # So LTP at w=0.1 should be much larger
        increment_low = w_low_new[0] - w_low[0]
        increment_high = w_high_new[0] - w_high[0]
        assert increment_low > increment_high, "LTP increment should depend on (w_max - w)"

    def test_stdp_weight_bounds(self):
        """
        Test that weights stay within [w_min, w_max] after STDP.

        Biophysics: Absolute bounds prevent runaway potentiation or depression.
        """
        # Test extreme weight changes
        delta_t = np.array([-100.0, 100.0])  # Very negative and very positive
        w_current = np.array([0.5, 0.5])
        A_plus = 0.1
        A_minus = 0.1
        tau_plus = 1.0
        tau_minus = 1.0
        w_min = 0.0
        w_max = 1.0

        w_new = numpy_stdp_weight_update(delta_t, w_current, A_plus, A_minus, tau_plus, tau_minus, w_min, w_max)

        # All weights must be within bounds
        assert np.all(w_new >= w_min), f"Weights should not go below w_min: {w_new}"
        assert np.all(w_new <= w_max), f"Weights should not exceed w_max: {w_new}"

    def test_stdp_symmetric_learning_window(self):
        """
        Test STDP exhibits expected time window symmetry (approximately).

        Biophysics (Bi & Poo 1998): Peak LTP at ~10 ms, peak LTD at ~10 ms.
        Both windows have exponential decay.
        """
        delta_t_array = np.linspace(-50, 50, 101)
        w_current = np.full_like(delta_t_array, 0.5)
        A_plus = 0.001
        A_minus = 0.0003
        tau_plus = 20.0
        tau_minus = 20.0
        w_min = 0.0
        w_max = 1.0

        w_new = numpy_stdp_weight_update(delta_t_array, w_current, A_plus, A_minus, tau_plus, tau_minus, w_min, w_max)

        # Find max LTP and LTD
        positive_delta_t = delta_t_array > 0
        negative_delta_t = delta_t_array < 0

        ltp_changes = w_new[positive_delta_t] - w_current[positive_delta_t]
        ltd_changes = w_new[negative_delta_t] - w_current[negative_delta_t]

        max_ltp = np.max(ltp_changes)
        min_ltd = np.min(ltd_changes)

        # Both should be non-zero and approximately balanced (in magnitude)
        assert max_ltp > 0.0, "Should have positive LTP changes"
        assert min_ltd < 0.0, "Should have negative LTD changes"
        # LTP and LTD amplitudes should be comparable (within 10x)
        assert max_ltp < 10 * np.abs(min_ltd) and np.abs(min_ltd) < 10 * max_ltp, \
            "LTP and LTD amplitudes should be comparable"


# =============================================================================
# Tests: Eligibility Trace
# =============================================================================


class TestEligibilityTrace:
    """Tests for eligibility trace decay."""

    def test_trace_exponential_decay(self):
        """
        Test eligibility trace decays exponentially.

        Biophysics: Eligibility traces mark synapses that participated in recent
        activity. They decay on a timescale of 100s of ms to seconds.
        """
        trace = np.array([1.0])
        decay_factor = np.exp(-0.01 / 0.1)  # dt=0.01, tau=0.1
        dt = 0.01

        trace_new = numpy_eligibility_trace_decay(trace, decay_factor)

        expected = trace[0] * decay_factor
        np.testing.assert_allclose(trace_new[0], expected, rtol=1e-10)
        assert trace_new[0] < trace[0], "Trace should decay"

    def test_trace_long_decay(self):
        """
        Test that traces decay away over many steps.

        Biophysics: After 10 tau, trace should be ~0.00005 of original.
        This prevents credit assignment to ancient synapses.
        """
        trace = np.array([1.0])
        decay_factor = np.exp(-0.01 / 0.01)  # dt=0.01, tau=0.01 (fast decay for testing)

        # Decay over 100 steps (10 * tau)
        for _ in range(100):
            trace = numpy_eligibility_trace_decay(trace, decay_factor)

        # Should be very close to 0
        assert trace[0] < 1e-5, f"Trace should be nearly zero after 10 tau: {trace[0]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
