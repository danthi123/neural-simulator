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

# Route through the backend abstraction so this module works on both
# CuPy and NumPy. `cp` is the active backend module; @fuse() is the
# backend-aware kernel-fusion decorator.
from sim.backend import get_backend, fuse
cp, _backend_name = get_backend()

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
def fused_eligibility_trace_decay(trace, decay_factor):
    """Fused kernel for eligibility trace exponential decay.

    Args:
        trace: Current eligibility trace value
        decay_factor: exp(-dt / tau)

    Returns:
        Decayed trace value
    """
    return trace * decay_factor
